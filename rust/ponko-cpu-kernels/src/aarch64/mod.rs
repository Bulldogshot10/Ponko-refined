//! AArch64-specific implementations.
//!
//! New optional ISA paths should live in separate functions/modules and be dispatched only
//! after the runtime feature check required by their instruction set. The initial NEON FP32
//! kernel uses the baseline arm64-v8a Advanced SIMD capability.

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(target_arch = "aarch64")]
pub(crate) use neon::matmul_f32_neon;
