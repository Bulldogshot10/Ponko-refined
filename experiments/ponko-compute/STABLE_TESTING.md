# PonkoCompute stable-testing protocol

PonkoCompute is in **stable-testing preparation**, not yet declared stable for the Exynos 2400E. The software CI gate must pass first; the real-device gate needs repeatable results from the target phone.

## Gate A: source and shader checks

- `cargo fmt --all -- --check`
- `cargo test --all-targets`, including timing-summary tests.
- Compile both GLSL shaders to SPIR-V and validate them with `spirv-val`.

## Gate B: deterministic Vulkan correctness

The `PonkoCompute Vulkan Stable Testing` workflow runs against Mesa lavapipe with the Khronos validation layer enabled. It tests vector-add sizes 1, 65, and 16,387; matrix shapes 1×1×1, 17×9×3, 67×53×29, and 64×64×64. Odd dimensions deliberately exercise invocation bounds rather than only aligned workgroups.

Every warm-up and timed invocation is read back and checked against a CPU reference. A mismatch, non-finite output, Vulkan error, shader validation error, or Rust test failure must fail the workflow. Timings report minimum, median, p95, maximum, and mean. The dispatch time starts immediately before queue submission and ends after fence completion; it excludes CPU reference work, shader compilation, pipeline creation, buffer allocation, and readback. These are correctness diagnostics, not yet end-to-end workload timings.

## Gate C: Exynos 2400E device acceptance

Run the same kernels from a process that can load the device's Vulkan loader and vendor ICD. Record a fresh log for each session with:

- device name and Android build / vendor driver information;
- kernel, matrix/vector shape, warm-up count, and timed iteration count;
- max absolute error and PASS/FAIL;
- min/median/p95/max dispatch time;
- battery/charging state, approximate temperature/thermal state, and whether the run was cold or warmed up.

For each kernel and test shape, use at least 3 separate process launches and 100 timed iterations after 10 warm-ups for the device benchmark. Check that correctness passes in every process, no Vulkan validation/runtime errors or crashes occur, and there is no unexplained output drift. Compare timing distributions across launches; do not adopt a hard speed target until there is a stable baseline on this device.

Example Android ARM64 build (the executable still needs access to Android's Vulkan loader and vendor ICD):

```sh
cd experiments/ponko-compute
glslangValidator -V --target-env vulkan1.0 shaders/vector_add.comp -o shaders/vector_add.spv
glslangValidator -V --target-env vulkan1.0 shaders/matmul.comp -o shaders/matmul.spv
cargo build --release --target aarch64-linux-android --bin ponko-compute --bin ponko-matmul
./target/aarch64-linux-android/release/ponko-compute shaders/vector_add.spv 16387 10 100
./target/aarch64-linux-android/release/ponko-matmul shaders/matmul.spv 67 53 29 10 100
./target/aarch64-linux-android/release/ponko-matmul shaders/matmul.spv 64 64 64 10 100
```

If Termux cannot enumerate the vendor device, repeat from the Ponko APK process before concluding the hardware path is unavailable. Passing lavapipe CI does not prove the Samsung Vulkan driver path works.

## Current promotion rule

- **Prototype:** API bring-up and one vector-add smoke test.
- **Stable-testing candidate:** automated correctness, malformed/edge shapes, repeated samples, timing summaries, shader validation, and cleanup on dispatch setup failures.
- **Stable on Xclipse 940:** only after Gate C device logs pass and are reproducible across separate launches.

## Follow-on performance work

After correctness/stability gates pass, prototype tiled/shared-memory GEMM, staging/device-local memory, and reusable command/descriptor pools as separate changes. Keep the naive matmul kernel as the correctness baseline. Merge a faster path only if it remains correct and beats the baseline on representative shapes.
