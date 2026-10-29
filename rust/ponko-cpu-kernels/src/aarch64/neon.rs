use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

/// AArch64 NEON implementation for contiguous row-major FP32 matrices.
///
/// The caller must validate the dimensions and buffer lengths before calling this function.
pub(crate) fn matmul_f32_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // SAFETY: The dispatcher only reaches this function on AArch64/arm64-v8a and validates
    // m*k, k*n, and m*n with checked arithmetic before calling it.
    unsafe { matmul_f32_neon_impl(a, b, c, m, n, k) }
}

#[target_feature(enable = "neon")]
unsafe fn matmul_f32_neon_impl(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();
    let simd_cols = n / 4 * 4;

    for row in 0..m {
        let a_row = row * k;
        let c_row = row * n;

        // Four adjacent output columns are accumulated in one 128-bit NEON vector.
        for col in (0..simd_cols).step_by(4) {
            let mut acc = unsafe { vdupq_n_f32(0.0) };

            for p in 0..k {
                // SAFETY: validated extents guarantee a_row+p < m*k and p*n+col+4 <= k*n.
                let a_value = unsafe { *a_ptr.add(a_row + p) };
                let b_values = unsafe { vld1q_f32(b_ptr.add(p * n + col)) };
                acc = unsafe { vfmaq_f32(acc, vdupq_n_f32(a_value), b_values) };
            }

            // SAFETY: row*n+col+4 <= m*n because col is within simd_cols <= n.
            unsafe { vst1q_f32(c_ptr.add(c_row + col), acc) };
        }

        // Scalar tail for N values not covered by a full four-lane vector.
        for col in simd_cols..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                // SAFETY: all computed indices are within the validated A and B extents.
                let a_value = unsafe { *a_ptr.add(a_row + p) };
                let b_value = unsafe { *b_ptr.add(p * n + col) };
                sum += a_value * b_value;
            }
            // SAFETY: row*n+col is within C[M,N].
            unsafe { *c_ptr.add(c_row + col) = sum };
        }
    }
}
