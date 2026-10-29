//! GGML Q4_0 block layout and scalar reference kernels.

use crate::matmul::KernelError;
use super::q8_0::{fp16_to_f32, quantize_row_q8_0, BlockQ8_0, QK8_0};

pub const QK4_0: usize = 32;

/// GGML Q4_0 layout: little-endian FP16 scale followed by 16 packed nibbles.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockQ4_0 {
    pub d_bits: u16,
    pub qs: [u8; QK4_0 / 2],
}

const _: [(); 18] = [(); core::mem::size_of::<BlockQ4_0>()];

pub(crate) fn unpack_q4_0(block: &BlockQ4_0, out: &mut [i8; QK4_0]) {
    for index in 0..QK4_0 / 2 {
        out[index] = (block.qs[index] & 0x0f) as i8 - 8;
        out[index + QK4_0 / 2] = (block.qs[index] >> 4) as i8 - 8;
    }
}

#[inline]
pub(crate) fn dot_block_q4_0_q8_0_scalar(a: &BlockQ4_0, b: &BlockQ8_0) -> i32 {
    let mut sum = 0_i32;
    for index in 0..QK4_0 / 2 {
        let low = i32::from(a.qs[index] & 0x0f) - 8;
        let high = i32::from(a.qs[index] >> 4) - 8;
        sum += low * i32::from(b.qs[index]);
        sum += high * i32::from(b.qs[index + QK8_0 / 2]);
    }
    sum
}

/// Reference kernel using already-quantized GGML Q8_0 activations.
pub fn matmul_q4_0_q8_0_reference(
    a: &[BlockQ4_0], b: &[BlockQ8_0], c: &mut [f32],
    m: usize, n: usize, k: usize,
) -> Result<(), KernelError> {
    matmul_q4_0_q8_0_reference_range(a, b, c, m, n, k, 0, n)
}

pub fn matmul_q4_0_q8_0_reference_range(
    a: &[BlockQ4_0], b: &[BlockQ8_0], c_rows: &mut [f32],
    m: usize, n: usize, k: usize, row_start: usize, row_end: usize,
) -> Result<(), KernelError> {
    if k % QK4_0 != 0 {
        return Err(KernelError::InvalidQuantizedDimension { name: "K", value: k, multiple: QK4_0 });
    }
    if row_start > row_end || row_end > n {
        return Err(KernelError::InvalidRowRange { start: row_start, end: row_end, total: n });
    }
    let blocks = k / QK4_0;
    let a_len = m.checked_mul(blocks).ok_or(KernelError::DimensionOverflow)?;
    let b_len = n.checked_mul(blocks).ok_or(KernelError::DimensionOverflow)?;
    let c_len = (row_end - row_start).checked_mul(m).ok_or(KernelError::DimensionOverflow)?;
    if c_len == 0 { return Ok(()); }

    for (name, actual, expected) in [
        ("Q4_0 A blocks", a.len(), a_len),
        ("Q8_0 B blocks", b.len(), b_len),
        ("C row partition", c_rows.len(), c_len),
    ] {
        if actual < expected {
            return Err(KernelError::InvalidInputLength { name, expected, actual });
        }
    }

    c_rows[..c_len].fill(0.0);
    for row_n in row_start..row_end {
        for row_m in 0..m {
            let mut sum = 0.0_f32;
            for block in 0..blocks {
                let qa = &a[row_m * blocks + block];
                let qb = &b[row_n * blocks + block];
                sum += (dot_block_q4_0_q8_0_scalar(qa, qb) as f32)
                    * fp16_to_f32(qa.d_bits)
                    * fp16_to_f32(qb.d_bits);
            }
            c_rows[(row_n - row_start) * m + row_m] = sum;
        }
    }
    Ok(())
}

/// Dispatch Q4_0 × F32 through the Exynos DOTPROD/I8MM path when built for that profile.
pub fn matmul_q4_0_f32(
    a: &[BlockQ4_0], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) -> Result<(), KernelError> {
    matmul_q4_0_f32_range(a, b, c, m, n, k, 0, n, false)
}

/// Scalar reference Q4_0 × F32 kernel using GGML-compatible Q8_0 row quantization.
pub fn matmul_q4_0_f32_reference(
    a: &[BlockQ4_0], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) -> Result<(), KernelError> {
    matmul_q4_0_f32_range(a, b, c, m, n, k, 0, n, true)
}

pub(crate) fn matmul_q4_0_f32_range(
    a: &[BlockQ4_0], b: &[f32], c_rows: &mut [f32],
    m: usize, n: usize, k: usize, row_start: usize, row_end: usize, use_reference: bool,
) -> Result<(), KernelError> {
    if k % QK4_0 != 0 {
        return Err(KernelError::InvalidQuantizedDimension { name: "K", value: k, multiple: QK4_0 });
    }
    if row_start > row_end || row_end > n {
        return Err(KernelError::InvalidRowRange { start: row_start, end: row_end, total: n });
    }

    let blocks = k / QK4_0;
    let a_len = m.checked_mul(blocks).ok_or(KernelError::DimensionOverflow)?;
    let b_len = n.checked_mul(k).ok_or(KernelError::DimensionOverflow)?;
    let c_len = (row_end - row_start).checked_mul(m).ok_or(KernelError::DimensionOverflow)?;
    if c_len == 0 { return Ok(()); }

    for (name, actual, expected) in [
        ("Q4_0 A blocks", a.len(), a_len),
        ("F32 B", b.len(), b_len),
        ("C row partition", c_rows.len(), c_len),
    ] {
        if actual < expected {
            return Err(KernelError::InvalidInputLength { name, expected, actual });
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "exynos2400e"))]
    if !use_reference {
        return super::aarch64::matmul_q4_0_f32_exynos_range(
            a, b, c_rows, m, n, k, row_start, row_end,
        );
    }

    let mut q8_row = Vec::new();
    q8_row.try_reserve_exact(blocks).map_err(|_| KernelError::AllocationFailure)?;
    q8_row.resize(blocks, BlockQ8_0 { d_bits: 0, qs: [0; QK8_0] });

    for row_n in row_start..row_end {
        let input = &b[row_n * k..(row_n + 1) * k];
        if !quantize_row_q8_0(input, &mut q8_row) {
            return Err(KernelError::InvalidInputLength {
                name: "Q8_0 scratch",
                expected: blocks,
                actual: q8_row.len(),
            });
        }
        let out_row = (row_n - row_start) * m;
        for row_m in 0..m {
            let mut sum = 0.0_f32;
            for block in 0..blocks {
                let qa = &a[row_m * blocks + block];
                let qb = &q8_row[block];
                sum += (dot_block_q4_0_q8_0_scalar(qa, qb) as f32)
                    * fp16_to_f32(qa.d_bits)
                    * fp16_to_f32(qb.d_bits);
            }
            c_rows[out_row + row_m] = sum;
        }
    }
    Ok(())
}
