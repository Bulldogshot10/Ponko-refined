//! GGML-compatible Q4_0 and Q8_0 block formats used by the CPU fast path.

pub mod q4_0;
pub mod q8_0;

#[cfg(all(target_arch = "aarch64", feature = "exynos2400e"))]
mod aarch64;

pub use q4_0::{
    matmul_q4_0_f32, matmul_q4_0_f32_reference, matmul_q4_0_q8_0_reference,
    matmul_q4_0_q8_0_reference_range, BlockQ4_0, QK4_0,
};
pub use q8_0::{BlockQ8_0, QK8_0};
