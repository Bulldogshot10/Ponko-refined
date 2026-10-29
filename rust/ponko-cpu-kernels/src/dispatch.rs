use crate::matmul::{matmul_f32_reference_validated, validate_lengths, KernelError};

/// Kernel selected by the current compile target.
///
/// Optional features such as DOTPROD and I8MM are intentionally not listed yet: this first
/// milestone only implements FP32 matmul and does not claim to implement quantized GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    Reference,
    Aarch64NeonF32,
}

impl KernelKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference-f32",
            Self::Aarch64NeonF32 => "aarch64-neon-f32",
        }
    }
}

/// Report which FP32 kernel the dispatcher will use on this build target.
///
/// AArch64 Android devices provide Advanced SIMD (NEON) as part of the arm64-v8a ABI.
/// Other build targets use the portable reference kernel for now.
pub const fn selected_f32_kernel() -> KernelKind {
    #[cfg(target_arch = "aarch64")]
    {
        KernelKind::Aarch64NeonF32
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        KernelKind::Reference
    }
}

/// Compute C[M,N] = A[M,K] × B[K,N], selecting the available implementation.
///
/// The boundary validates shape arithmetic and slice lengths before entering any optimized
/// implementation. Matrices are contiguous, row-major FP32 buffers.
pub fn matmul_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), KernelError> {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = validate_lengths(a, b, c, m, n, k)?;
        // SAFETY: AArch64 Android's arm64-v8a ABI provides NEON. The safe boundary above
        // verifies the exact matrix extents, and the NEON implementation writes only C[M,N].
        crate::aarch64::matmul_f32_neon(a, b, c, m, n, k);
        Ok(())
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let (_, _, c_len) = validate_lengths(a, b, c, m, n, k)?;
        matmul_f32_reference_validated(a, b, c, m, n, k, c_len);
        Ok(())
    }
}
