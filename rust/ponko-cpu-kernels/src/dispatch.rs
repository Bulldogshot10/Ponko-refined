use crate::matmul::{
    matmul_f32_reference, validate_lengths,
    validate_rhs_transposed_lengths, KernelError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    Reference,
    Aarch64NeonF32,
}

impl KernelKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference-f32",
            Self::Aarch64NeonF32 => "exynos-aarch64-neon-f32",
        }
    }
}

pub const fn selected_f32_kernel() -> KernelKind {
    #[cfg(target_arch = "aarch64")]
    { KernelKind::Aarch64NeonF32 }
    #[cfg(not(target_arch = "aarch64"))]
    { KernelKind::Reference }
}

pub fn matmul_f32(
    a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize,
) -> Result<(), KernelError> {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = validate_lengths(a, b, c, m, n, k)?;
        crate::aarch64::matmul_f32_neon(a, b, c, m, n, k);
        Ok(())
    }
    #[cfg(not(target_arch = "aarch64"))]
    { matmul_f32_reference(a, b, c, m, n, k) }
}

/// Select the GGML contiguous F32 MUL_MAT layout: A[M,K], B[N,K], C[N,M].
pub fn matmul_f32_rhs_transposed(
    a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize,
) -> Result<(), KernelError> {
    matmul_f32_rhs_transposed_range(a, b, c, m, n, k, 0, n)
}

pub(crate) fn matmul_f32_rhs_transposed_range(
    a: &[f32], b: &[f32], c_rows: &mut [f32],
    m: usize, n: usize, k: usize, row_start: usize, row_end: usize,
) -> Result<(), KernelError> {
    let (_, _, c_len) =
        validate_rhs_transposed_lengths(a, b, c_rows, m, n, k, row_start, row_end)?;
    if c_len == 0 { return Ok(()); }

    #[cfg(target_arch = "aarch64")]
    {
        crate::aarch64::matmul_f32_rhs_transposed_neon(
            a, b, c_rows, m, n, k, row_start, row_end,
        );
        Ok(())
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = c_len;
        // The reference takes full matrices, so run this partition directly without allocating.
        for row_n in row_start..row_end {
            let out_row = (row_n - row_start) * m;
            for row_m in 0..m {
                let mut sum = 0.0_f32;
                for p in 0..k {
                    sum += a[row_m * k + p] * b[row_n * k + p];
                }
                c_rows[out_row + row_m] = sum;
            }
        }
        Ok(())
    }
}
