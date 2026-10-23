# PonkoCompute

A deliberately small Rust/Vulkan compute-runtime prototype for Android GPUs, including the Xclipse 940. It uses the system Vulkan loader and installed device driver; it does not replace or patch Samsung's driver.

## What this first slice contains

- **ComputeDevice::open()** opens Vulkan and selects a physical device with a compute queue.
- **create_buffer(), upload(), and readback()** provide host-visible/coherent storage buffers for the initial shared-memory experiment.
- **load_kernel()** loads a SPIR-V compute shader and builds a pipeline with storage-buffer bindings and optional push constants.
- **dispatch()** records a compute command buffer, binds buffers, dispatches workgroups, inserts a compute-to-host visibility barrier, submits work, and waits on a fence.
- **vector_add.comp** is the first correctness probe. The runner compares GPU output against a CPU reference and prints dispatch timing.

This is a bring-up/runtime prototype, not yet a tuned tensor library. Each dispatch currently allocates a fresh descriptor pool and command pool and uses host-visible/coherent buffers. That makes the first path easy to inspect, but is not the final performance design.

## Build and run on Linux with a Vulkan driver

Install Rust, a Vulkan loader/driver, and a GLSL-to-SPIR-V compiler such as glslangValidator. From this directory:

    glslangValidator -V --target-env vulkan1.0 shaders/vector_add.comp -o shaders/vector_add.spv
    cargo fmt --check
    cargo check
    cargo run --release -- shaders/vector_add.spv 65536

A successful run ends with the message **PASS: vector-add output matches CPU reference**. The CPU number is a reference time; the Vulkan number includes queue submission and fence wait but excludes shader compilation and pipeline creation. This is a correctness/timing smoke test, not proof that a tiny vector operation is faster on the GPU.

## Android / Termux bring-up

Build the shader to SPIR-V with glslangValidator available in the environment, then build the Rust binary with the environment's Android ARM64 Rust/NDK toolchain. Try launching it from the same environment that supplies the Vulkan loader:

    cargo build --release
    ./target/aarch64-linux-android/release/ponko-compute shaders/vector_add.spv 65536

The executable needs access to Android's Vulkan loader and the vendor ICD in its process namespace. If standalone Termux cannot enumerate the Samsung device, that's a loader/namespace integration issue, not a failed compute-shader correctness test. The next integration step would be calling this runtime from the Ponko APK process.

## Next experiments

1. Run this smoke test on the actual Exynos 2400E and record the reported device name.
2. Add device-local/staging memory and reusable command/descriptor pools only after this baseline works.
3. Add matrix multiply and Conv2D kernels, then benchmark dispatch overhead, throughput, memory use, and thermal behavior.
4. Integrate promising kernels into GGML only after they beat the existing backend on representative diffusion workloads.
