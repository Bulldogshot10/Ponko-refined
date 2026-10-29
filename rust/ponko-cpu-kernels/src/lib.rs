//! Standalone CPU kernels for the Exynos 2400E profile used by Ponko.
//!
//! The crate has a safe Rust test API and a narrow C ABI for opt-in GGML integration. It does
//! not own tensor allocation, graph execution, model loading, or thread scheduling.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod aarch64;
mod dispatch;
mod ffi;
mod matmul;
pub mod quant;

pub use dispatch::{
    matmul_f32, matmul_f32_rhs_transposed, selected_f32_kernel, KernelKind,
};
pub use matmul::{
    matmul_f32_reference, matmul_f32_rhs_transposed_reference, KernelError,
};
