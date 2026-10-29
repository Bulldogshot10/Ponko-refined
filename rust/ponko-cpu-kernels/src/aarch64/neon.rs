use core::arch::aarch64::{
    vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32,
};

pub(crate) fn matmul_f32_neon(
    a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize,
) {
    // SAFETY: dispatch validates the matrix extents before calling this implementation.
    unsafe { matmul_f32_neon_impl(a, b, c, m, n, k) }
}

#[target_feature(enable = "neon")]
unsafe fn matmul_f32_neon_impl(
    a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize,
) {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();
    let simd_cols = n / 4 * 4;

    for row in 0..m {
        for col in (0..simd_cols).step_by(4) {
            let mut acc = unsafe { vdupq_n_f32(0.0) };
            for p in 0..k {
                // SAFETY: dispatcher checked A, B and C shapes.
                let av = unsafe { *ap.add(row * k + p) };
                let bv = unsafe { vld1q_f32(bp.add(p * n + col)) };
                acc = unsafe { vfmaq_f32(acc, vdupq_n_f32(av), bv) };
            }
            // SAFETY: col+4 is within the validated output row.
            unsafe { vst1q_f32(cp.add(row * n + col), acc) };
        }
        for col in simd_cols..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                // SAFETY: indices are within the validated input matrices.
                sum += unsafe { *ap.add(row * k + p) } * unsafe { *bp.add(p * n + col) };
            }
            // SAFETY: output index is within C.
            unsafe { *cp.add(row * n + col) = sum };
        }
    }
}

pub(crate) fn matmul_f32_rhs_transposed_neon(
    a: &[f32], b: &[f32], c_rows: &mut [f32],
    m: usize, n: usize, k: usize, row_start: usize, row_end: usize,
) {
    // SAFETY: the dispatcher validates both matrices and the output row partition.
    unsafe { matmul_f32_rhs_transposed_neon_impl(a, b, c_rows, m, n, k, row_start, row_end) }
}

#[target_feature(enable = "neon")]
unsafe fn matmul_f32_rhs_transposed_neon_impl(
    a: &[f32], b: &[f32], c_rows: &mut [f32],
    m: usize, _n: usize, k: usize, row_start: usize, row_end: usize,
) {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c_rows.as_mut_ptr();
    let simd_k = k / 4 * 4;

    for row_n in row_start..row_end {
        for row_m in 0..m {
            let mut acc = unsafe { vdupq_n_f32(0.0) };
            for p in (0..simd_k).step_by(4) {
                // SAFETY: each four-element load lies within one validated K-sized row.
                let av = unsafe { vld1q_f32(ap.add(row_m * k + p)) };
                let bv = unsafe { vld1q_f32(bp.add(row_n * k + p)) };
                acc = unsafe { vfmaq_f32(acc, av, bv) };
            }
            let mut sum = unsafe { vaddvq_f32(acc) };
            for p in simd_k..k {
                // SAFETY: tail indices lie within the same validated rows.
                sum += unsafe { *ap.add(row_m * k + p) } * unsafe { *bp.add(row_n * k + p) };
            }
            // SAFETY: c_rows holds exactly the selected output rows.
            unsafe { *cp.add((row_n - row_start) * m + row_m) = sum };
        }
    }
}
