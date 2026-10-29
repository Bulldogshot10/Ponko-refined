//! GGML Q8_0 block layout and FP16 scale conversion.

pub const QK8_0: usize = 32;

/// GGML Q8_0 layout: little-endian FP16 scale followed by 32 signed quantized values.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockQ8_0 {
    pub d_bits: u16,
    pub qs: [i8; QK8_0],
}

const _: [(); 34] = [(); core::mem::size_of::<BlockQ8_0>()];

pub(crate) fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;

    let out = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut mantissa = fraction;
            let mut exp = 113_u32;
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mantissa & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(out)
}

/// Round f32 to IEEE binary16 using round-to-nearest, ties-to-even.
pub(crate) fn f32_to_fp16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7f_ffff;

    if exponent == 0xff {
        if fraction == 0 {
            return sign | 0x7c00;
        }
        let payload = (fraction >> 13) as u16;
        return sign | 0x7c00 | payload | (if payload == 0 { 1 } else { 0 });
    }

    let exp16 = exponent - 127 + 15;
    if exp16 >= 31 {
        return sign | 0x7c00;
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        let mantissa = fraction | 0x80_0000;
        let shift = (14 - exp16) as u32;
        let truncated = mantissa >> shift;
        let remainder = mantissa & ((1_u32 << shift) - 1);
        let halfway = 1_u32 << (shift - 1);
        let round_up = remainder > halfway || (remainder == halfway && (truncated & 1) != 0);
        return sign | (truncated + u32::from(round_up)) as u16;
    }

    let truncated = fraction >> 13;
    let remainder = fraction & 0x1fff;
    let round_up = remainder > 0x1000 || (remainder == 0x1000 && (truncated & 1) != 0);
    let mut rounded_fraction = truncated + u32::from(round_up);
    let mut rounded_exp = exp16 as u32;
    if rounded_fraction == 0x400 {
        rounded_fraction = 0;
        rounded_exp += 1;
        if rounded_exp >= 31 {
            return sign | 0x7c00;
        }
    }
    sign | ((rounded_exp << 10) as u16) | (rounded_fraction as u16)
}

/// Quantize one contiguous row to GGML's reference Q8_0 representation.
pub(crate) fn quantize_row_q8_0(row: &[f32], out: &mut [BlockQ8_0]) -> bool {
    if row.len() % QK8_0 != 0 || out.len() != row.len() / QK8_0 {
        return false;
    }

    for (values, block) in row.chunks_exact(QK8_0).zip(out.iter_mut()) {
        let mut amax = 0.0_f32;
        for &value in values {
            amax = amax.max(value.abs());
        }
        let d = amax / 127.0;
        let inverse = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d_bits = f32_to_fp16_bits(d);
        for (q, &value) in block.qs.iter_mut().zip(values.iter()) {
            *q = (value * inverse).round().clamp(-127.0, 127.0) as i8;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{f32_to_fp16_bits, fp16_to_f32, quantize_row_q8_0, BlockQ8_0};

    #[test]
    fn converts_common_binary16_values() {
        assert_eq!(fp16_to_f32(0x0000), 0.0);
        assert_eq!(fp16_to_f32(0x8000), -0.0);
        assert_eq!(fp16_to_f32(0x3c00), 1.0);
        assert_eq!(fp16_to_f32(0xbc00), -1.0);
        assert_eq!(fp16_to_f32(0x3800), 0.5);
        assert_eq!(fp16_to_f32(0x7c00), f32::INFINITY);
    }

    #[test]
    fn fp16_roundtrip_covers_normal_and_subnormal_values() {
        for value in [0.0, -0.0, 1.0, -1.0, 0.5, 0.000_061_035_156_25, 0.333_251_95] {
            assert_eq!(fp16_to_f32(f32_to_fp16_bits(value)), value);
        }
    }

    #[test]
    fn q8_quantization_scales_to_127() {
        let input = [1.0_f32; 32];
        let mut blocks = [BlockQ8_0 { d_bits: 0, qs: [0; 32] }; 1];
        assert!(quantize_row_q8_0(&input, &mut blocks));
        assert_eq!(blocks[0].qs, [127; 32]);
        assert_eq!(fp16_to_f32(blocks[0].d_bits), 0.00787353515625_f32);
    }
}
