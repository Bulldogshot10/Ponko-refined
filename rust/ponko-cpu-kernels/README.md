# Ponko CPU Kernels: Exynos 2400E

This crate is a device-targeted experiment for Ponko, separate from the inference graph. The
Exynos build profile explicitly enables NEON, DOTPROD and I8MM code paths; it is not a portable
runtime-dispatched AArch64 backend. Do not enable the integrated build on other CPUs.

## Implemented paths

- NEON FP32 matrix multiplication, including the contiguous F32 MUL_MAT layout GGML uses.
- GGML Q4_0 weight blocks and Q8_0 activation quantization, with a scalar correctness reference.
- DOTPROD for single dot products and odd matrix edges.
- I8MM SMMLA for 2x2 signed-int8 tiles, with per-block scale application matching Q4_0 × Q8_0 math.
- A C ABI called directly from GGML's CPU MUL_MAT implementation when the opt-in build switch is enabled.
- Host tests, ARM64 cross-compilation checks, and a microbenchmark covering both FP32 and Q4_0 paths.

The integrated Q4_0 path accepts contiguous 2D Q4_0 weights, F32 activations, and F32 output with
K divisible by 32. It quantizes each activation row to Q8_0 and computes each selected output row.
Unsupported shapes stay on GGML's existing path. F16 and other quant formats are unchanged.

## Build and compare

The Android Gradle build defaults to the Exynos kernel integration on this experiment branch. To
build the GGML baseline APK instead, use:

    ./gradlew :app:assembleDebug -PponkoRustCpuKernels=OFF

The Rust static library is linked into GGML's CPU backend. The normal graph executor, model loading,
tensor lifetime, and GGML thread scheduler remain in place. The patched MUL_MAT path partitions
output rows across GGML workers, so each worker writes a separate output region.

Run the local microbenchmark on the device with:

    cargo bench --manifest-path rust/ponko-cpu-kernels/Cargo.toml --bench matmul --features exynos2400e

That crate-level benchmark isolates kernel timing. For real inference timing, use the same model,
seed, resolution, sampler, step count and thread count in two APKs, once with
ponkoRustCpuKernels=ON and once with it OFF. Only Q4_0 × F32 and F32 × F32 contiguous 2D GEMMs
take the Rust paths today, so choose a model/workload with those operators and profile how often
the fast path is used before drawing conclusions about overall SDXL speed.

The I8MM path uses inline SMMLA assembly because rustc's stable std::arch API still marks
vmmlaq_s32 experimental. DOTPROD uses the stable vdotq_s32 intrinsic. The function-specific target
feature annotations isolate optional instructions; this experiment must be built and run only on
the known Exynos 2400E profile with NEON, DOTPROD and I8MM.

Next work after validating on device: add counters for Rust-path hits, profile model operator types,
then evaluate additional weight formats and optimize packing/activation quantization costs.
