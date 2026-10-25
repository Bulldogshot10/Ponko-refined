# PonkoCompute

A small Rust/Vulkan compute-runtime prototype for Android GPUs, including the Exynos 2400E / Xclipse 940. It uses the system Vulkan loader and installed device driver; it does not replace or patch Samsung's driver.

## Current prototype

- **ComputeDevice::open()** opens Vulkan and selects a physical device with a compute queue.
- **create_buffer(), upload(), and readback()** provide host-visible/coherent storage buffers for the first shared-memory experiment.
- **load_kernel()** loads a SPIR-V compute shader and builds a pipeline with storage-buffer bindings and optional push constants.
- **dispatch()** records a compute command buffer, binds buffers, dispatches workgroups, inserts a compute-to-host visibility barrier, submits work, and waits on a fence.
- **vector_add.comp** validates buffer transfer, dispatch, and readback against a CPU reference.
- **matmul.comp** adds a naive FP32 matrix multiplication baseline, (C[M,N] = A[M,K] \times B[K,N]), with bounds checks for arbitrary dimensions. Its runner compares every result against a CPU reference and prints a rough effective-GFLOP/s figure.

This is a correctness/measurement harness, not yet a tuned tensor library. Each dispatch currently allocates a fresh descriptor pool and command pool and uses host-visible/coherent buffers. The matrix kernel deliberately assigns one invocation to each output cell; it is a baseline to beat, not an optimized GEMM.

## Linux/Vulkan smoke test

Install Rust, a Vulkan loader/driver, and a GLSL-to-SPIR-V compiler such as `glslangValidator`. From this directory:

```sh
glslangValidator -V --target-env vulkan1.0 shaders/vector_add.comp -o shaders/vector_add.spv
glslangValidator -V --target-env vulkan1.0 shaders/matmul.comp -o shaders/matmul.spv
cargo fmt --all
cargo test --all-targets
cargo run --release --bin ponko-compute -- shaders/vector_add.spv 65536
cargo run --release --bin ponko-matmul -- shaders/matmul.spv 64 96 80
```

Each run should end with a `PASS` line once its GPU output matches the CPU reference. The reported Vulkan time includes queue submission and the completion-fence wait, but excludes shader compilation, pipeline creation, buffer allocation, and upload/readback. The effective-GFLOP/s value is therefore only a rough dispatch-level indicator, not a model-workload performance claim.

The `PonkoCompute Vulkan Prototype` GitHub Actions workflow compiles both shaders and runs both correctness probes against Mesa lavapipe. That validates the API/runtime path on a software Vulkan implementation; it does **not** establish the performance or compatibility of the Exynos vendor driver.

## Android / Termux bring-up

Compile the shaders with `glslangValidator`, then build for Android ARM64 using an Android Rust/NDK toolchain:

```sh
cargo build --release --target aarch64-linux-android
./target/aarch64-linux-android/release/ponko-compute shaders/vector_add.spv 65536
./target/aarch64-linux-android/release/ponko-matmul shaders/matmul.spv 64 64 64
```

The process must be able to access Android's Vulkan loader and the vendor ICD. If standalone Termux cannot enumerate the Samsung device, that is a loader/namespace integration issue; test from the Ponko APK process before treating the device's compute support as unavailable.

## Next stages

1. Run both correctness probes on the actual Exynos 2400E and keep device name, thermal state, and timing with every result.
2. Improve the measurement harness: warm-up iterations, repeated samples, median/p95 timing, and allocation/transfer timings reported separately.
3. Add tiled/shared-memory matrix multiplication, then compare it to the naive kernel.
4. Add device-local memory with staging buffers and reusable command/descriptor pools; measure whether those changes help this GPU rather than assuming they do.
5. Implement and benchmark a simple Conv2D kernel with representative shapes.
6. Only integrate kernels into GGML after they beat the existing backend on representative diffusion workloads and preserve numerical correctness.
