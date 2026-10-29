//! Standalone CPU kernel experiments for Ponko.
//!
//! This crate deliberately stays independent from the inference graph and Kotlin/UniFFI
//! facade. It provides a correctness reference, a small dispatch surface, and architecture-
//! specific implementations that can be benchmarked before they are integrated into GGML.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod aarch64;
mod dispatch;
mod matmul;

pub use dispatch::{matmul_f32, selected_f32_kernel, KernelKind};
pub use matmul::{matmul_f32_reference, KernelError};
