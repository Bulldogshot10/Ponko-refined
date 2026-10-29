//! Exynos 2400E quantized kernels.
//!
//! DOTPROD handles standalone Q4_0 × Q8_0 dot products and odd matrix edges. I8MM computes 2x2
//! tiles using SMMLA. The crate is built with the exynos2400e Cargo feature for this device path.

use core::arch::aarch64::{
    vaddvq_s32, vdupq_n_s32, vdotq_s32, vld1q_s8, vst1q_s32,
};
use core::arch::asm;

use crate::matmul::KernelError;
use super::q4_0::{unpack_q4_0, BlockQ4_0, QK4_0};
use super::q8_0::{fp16_to_f32, quantize_row_q8_0, BlockQ8_0, QK8_0};

#[target_feature(enable = "dotprod")]
unsafe fn dot_block_dotprod(a: &BlockQ4_0, b: &BlockQ8_0) -> i32 {
    let mut unpacked = [0_i8; QK4_0];
    unpack_q4_0(a, &mut unpacked);
    let mut acc = unsafe { vdupq_n_s32(0) };
    for half in 0..2 {
        // SAFETY: unpacked and Q8_0 each contain 32 bytes; half selects a 16-byte region.
        let aq = unsafe { vld1q_s8(unpacked.as_ptr().add(half * 16)) };
        let bq = unsafe { vld1q_s8(b.qs.as_ptr().add(half * 16)) };
        acc = unsafe { vdotq_s32(acc, aq, bq) };
    }
    unsafe { vaddvq_s32(acc) }
}

/// The SMMLA instruction computes a 2x2 signed-int8 matrix tile accumulated in four int32 lanes.
#[target_feature(enable = "i8mm")]
unsafe fn smmla(
    acc: core::arch::aarch64::int32x4_t,
    lhs: core::arch::aarch64::int8x16_t,
    rhs: core::arch::aarch64::int8x16_t,
) -> core::arch::aarch64::int32x4_t {
    let mut out = acc;
    // SAFETY: callers are compiled for the known Exynos 2400E I8MM profile. The instruction
    // implements acc += (2x8 signed-int8 matrix) × (8x2 signed-int8 matrix).
    unsafe {
        asm!(
            "smmla {acc}.4s, {lhs}.16b, {rhs}.16b",
            acc = inout(vreg) out,
            lhs = in(vreg) lhs,
            rhs = in(vreg) rhs,
            options(pure, nomem, nostack)
        );
    }
    out
}

#[target_feature(enable = "i8mm")]
unsafe fn dot_2x2_i8mm(
    a0: &BlockQ4_0,
    a1: &BlockQ4_0,
    b0: &BlockQ8_0,
    b1: &BlockQ8_0,
) -> [i32; 4] {
    let mut q4_0 = [0_i8; QK4_0];
    let mut q4_1 = [0_i8; QK4_0];
    unpack_q4_0(a0, &mut q4_0);
    unpack_q4_0(a1, &mut q4_1);

    let mut acc = unsafe { vdupq_n_s32(0) };
    for chunk in 0..4 {
        let offset = chunk * 8;
        let mut lhs = [0_i8; 16];
        let mut rhs = [0_i8; 16];
        lhs[..8].copy_from_slice(&q4_0[offset..offset + 8]);
        lhs[8..].copy_from_slice(&q4_1[offset..offset + 8]);
        for lane in 0..8 {
            rhs[lane * 2] = b0.qs[offset + lane];
            rhs[lane * 2 + 1] = b1.qs[offset + lane];
        }

        // SAFETY: both temporary arrays contain exactly 16 initialized signed bytes.
        let lhs_v = unsafe { vld1q_s8(lhs.as_ptr()) };
        let rhs_v = unsafe { vld1q_s8(rhs.as_ptr()) };
        acc = unsafe { smmla(acc, lhs_v, rhs_v) };
    }

    let mut out = [0_i32; 4];
    // SAFETY: output contains four i32 values and acc is a four-lane vector.
    unsafe { vst1q_s32(out.as_mut_ptr(), acc) };
    out
}

pub(crate) fn matmul_q4_0_f32_exynos_range(
    a: &[BlockQ4_0],
    b: &[f32],
    c_rows: &mut [f32],
    m: usize,
    _n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
) -> Result<(), KernelError> {
    let blocks = k / QK4_0;
    let row_count = row_end - row_start;
    let quant_count = row_count.checked_mul(blocks).ok_or(KernelError::DimensionOverflow)?;

    let mut q8 = Vec::new();
    q8.try_reserve_exact(quant_count).map_err(|_| KernelError::AllocationFailure)?;
    q8.resize(quant_count, BlockQ8_0 { d_bits: 0, qs: [0; QK8_0] });

    for local_n in 0..row_count {
        let row = &b[(row_start + local_n) * k..(row_start + local_n + 1) * k];
        let start = local_n * blocks;
        if !quantize_row_q8_0(row, &mut q8[start..start + blocks]) {
            return Err(KernelError::InvalidInputLength {
                name: "Q8_0 row scratch",
                expected: blocks,
                actual: q8[start..start + blocks].len(),
            });
        }
    }

    let out_len = row_count.checked_mul(m).ok_or(KernelError::DimensionOverflow)?;
    if c_rows.len() < out_len {
        return Err(KernelError::InvalidInputLength {
            name: "C row partition",
            expected: out_len,
            actual: c_rows.len(),
        });
    }

    c_rows[..out_len].fill(0.0);
    let paired_n = row_count / 2 * 2;
    let paired_m = m / 2 * 2;

    // I8MM path: one SMMLA tile computes two weight rows against two activation rows.
    for local_n in (0..paired_n).step_by(2) {
        let b0_base = local_n * blocks;
        let b1_base = (local_n + 1) * blocks;
        for row_m in (0..paired_m).step_by(2) {
            let a0_base = row_m * blocks;
            let a1_base = (row_m + 1) * blocks;
            let mut s00 = 0.0_f32;
            let mut s01 = 0.0_f32;
            let mut s10 = 0.0_f32;
            let mut s11 = 0.0_f32;

            for block in 0..blocks {
                let a0 = &a[a0_base + block];
                let a1 = &a[a1_base + block];
                let b0 = &q8[b0_base + block];
                let b1 = &q8[b1_base + block];
                // SAFETY: this function only runs on the Exynos profile built with I8MM.
                let dots = unsafe { dot_2x2_i8mm(a0, a1, b0, b1) };
                let da0 = fp16_to_f32(a0.d_bits);
                let da1 = fp16_to_f32(a1.d_bits);
                let db0 = fp16_to_f32(b0.d_bits);
                let db1 = fp16_to_f32(b1.d_bits);
                s00 += dots[0] as f32 * da0 * db0;
                s01 += dots[1] as f32 * da0 * db1;
                s10 += dots[2] as f32 * da1 * db0;
                s11 += dots[3] as f32 * da1 * db1;
            }

            let o0 = local_n * m + row_m;
            let o1 = (local_n + 1) * m + row_m;
            c_rows[o0] = s00;
            c_rows[o0 + 1] = s10;
            c_rows[o1] = s01;
            c_rows[o1 + 1] = s11;
        }
    }

    // DOTPROD handles odd weight rows in paired activation rows.
    for local_n in (0..paired_n).step_by(2) {
        let b0_base = local_n * blocks;
        let b1_base = (local_n + 1) * blocks;
        for row_m in paired_m..m {
            let mut s0 = 0.0_f32;
            let mut s1 = 0.0_f32;
            for block in 0..blocks {
                let qa = &a[row_m * blocks + block];
                let qb0 = &q8[b0_base + block];
                let qb1 = &q8[b1_base + block];
                // SAFETY: this function only runs on the Exynos profile built with DOTPROD.
                let d0 = unsafe { dot_block_dotprod(qa, qb0) };
                let d1 = unsafe { dot_block_dotprod(qa, qb1) };
                s0 += d0 as f32 * fp16_to_f32(qa.d_bits) * fp16_to_f32(qb0.d_bits);
                s1 += d1 as f32 * fp16_to_f32(qa.d_bits) * fp16_to_f32(qb1.d_bits);
            }
            c_rows[local_n * m + row_m] = s0;
            c_rows[(local_n + 1) * m + row_m] = s1;
        }
    }

    // If N is odd, use DOTPROD for the last activation row.
    if paired_n < row_count {
        let local_n = paired_n;
        let b_base = local_n * blocks;
        for row_m in 0..m {
            let mut sum = 0.0_f32;
            for block in 0..blocks {
                let qa = &a[row_m * blocks + block];
                let qb = &q8[b_base + block];
                // SAFETY: this function only runs on the Exynos profile built with DOTPROD.
                let dot = unsafe { dot_block_dotprod(qa, qb) };
                sum += dot as f32 * fp16_to_f32(qa.d_bits) * fp16_to_f32(qb.d_bits);
            }
            c_rows[local_n * m + row_m] = sum;
        }
    }

    Ok(())
}
