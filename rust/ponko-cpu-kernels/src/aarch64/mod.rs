//! AArch64 kernels for the Exynos 2400E profile.
//!
//! NEON is used for the FP32 path. Quantized Q4_0 × Q8_0 kernels live under quant/aarch64.rs:
//! DOTPROD is the single-output fallback/tail and I8MM is the 2x2 integer matrix path.

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(target_arch = "aarch64")]
pub(crate) use neon::{matmul_f32_neon, matmul_f32_rhs_transposed_neon};
