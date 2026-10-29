# Ponko CPU Kernels

This crate is a small, independent laboratory for CPU kernels intended to be measured before
integration into Ponko's inference engine. It is deliberately not a tensor framework and does
not replace GGML, stable-diffusion.cpp, the graph executor, or model loading.

## Current scope

- A safe, row-major FP32 matrix multiplication API with checked dimension arithmetic.
- A scalar reference implementation for correctness comparisons.
- An AArch64 NEON FP32 matrix multiplication path with four-column vectorization and a scalar tail.
- Host tests for numerical agreement, non-square matrices, tail columns, invalid buffers, empty
  outputs, and dimension overflow.
- A dependency-free microbenchmark comparing the reference and dispatched FP32 paths.

Q4_0/Q8_0 block decoding, quantized dot products, DOTPROD/I8MM kernels, explicit strides,
threading, and GGML/SDCPP inference integration are not implemented yet. In particular, this crate
does not currently accelerate Stable Diffusion inference; it is a foundation to measure and extend.

## Matrix contract

The operation computes C[M,N] = A[M,K] × B[K,N]. A, B, and C must be contiguous row-major FP32
buffers with at least M×K, K×N, and M×N elements respectively. Additional elements are ignored.
All required buffer lengths are checked before indexing.

## Commands

From the repository root:

    cargo fmt --manifest-path rust/ponko-cpu-kernels/Cargo.toml -- --check
    cargo test --manifest-path rust/ponko-cpu-kernels/Cargo.toml
    cargo bench --manifest-path rust/ponko-cpu-kernels/Cargo.toml --bench matmul
    cargo check --manifest-path rust/ponko-cpu-kernels/Cargo.toml --target aarch64-linux-android

The final command checks that the architecture-specific code compiles for Android ARM64. It does
not link an Android shared library or exercise a physical device.

## Integration direction

Once a kernel is measurably correct and faster on a target workload, expose it behind a narrow C
ABI and prototype dispatch at a specific GGML operator boundary. Keep GGML's graph, tensor lifetime,
thread scheduling, and model loading unchanged during the first experiment. Only consider a custom
backend/runtime after benchmarks establish what GGML's existing path leaves on the table.

For performance comparisons, use identical shapes, thread counts, data layout, and output checks.
Include packing/dequantization overhead in end-to-end timings instead of reporting only the inner
multiply loop.
