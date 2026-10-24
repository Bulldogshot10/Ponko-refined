# Exynos 2400e inference workbench

This branch now has an opt-in CPU-affinity experiment, userspace CPU/topology diagnostics, process/RAM/thermal snapshots, VAE encode/decode timings, generation-level benchmark records, and a logcat-to-CSV summarizer.

## What is recorded

- `BENCHMARK cpu_topology`: OS-exposed CPU feature flags (NEON, DOTPROD, I8MM, FP16, SVE, SVE2), online CPU IDs, process-allowed CPU masks, sysfs capacity data, current clock and max clock where readable.
- `BENCHMARK generate_image`: wall time, txt2img/img2img, resolution, steps, requested backend, loaded thread count, direct VAE Conv2D flag, sampler, scheduler, CFG, LoRA count, seed, process CPU time, and success.
- `BENCHMARK vae_stage`: aggregate VAE encode/decode duration, worker-thread count, tiling status, and input/output tensor dimensions.
- `BENCHMARK device`: process RSS and high-water RSS, process thread count, readable per-core frequency values, and readable thermal-zone temperatures before and after generation.
- `BENCHMARK affinity`: whether the requested affinity was actually applied.

Telemetry is best-effort. Android kernels may hide sysfs files or thermal data. Missing readings are reported as unavailable; the code does not invent a CPU ordering from core IDs.

## CPU affinity controls

Open the drawing model settings. The affinity selector has three profiles:

- **Android scheduler (default)** restores the calling thread's previous scheduler-selected affinity.
- **Performance-capacity CPUs** builds a mask from readable sysfs CPU capacity/max-frequency values and selects the higher-capacity portion.
- **Efficiency-capacity CPUs** selects the lower-capacity portion.

The native API also accepts explicit lists such as `0,2-5`. The UI intentionally exposes only the two capacity-ranked profiles. If the kernel does not expose enough capacity data to rank clusters, the native layer rejects the profile rather than guessing.

Affinity currently applies to the thread calling `nativeGenerate`, and restores that thread's original mask after generation. It does **not** pin all existing OpenMP workers. Worker-pool affinity is a separate layer of work and should not be inferred from a successful caller-thread affinity log.

## SVE2 build experiment

The normal Exynos build keeps the existing `armv8.6-a+dotprod+i8mm+fp16` target. An opt-in SVE2 profile is available:

```sh
./gradlew --no-daemon :app:assembleDebug -PponkoExynosSve2=true
```

Only use that APK if the **Exynos CPU diagnostics** panel reports `sve2=1` on the actual Android kernel. The global SVE2 profile is an experimental build switch, not a runtime-dispatched per-kernel backend; using it when SVE2 is unavailable may execute unsupported instructions. For the normal build, omit the property or set `-PponkoExynosSve2=false`.

## Capture and analyze

On a host with Android platform-tools, connect the phone with USB debugging enabled, open Ponko-refined, and keep the app process running. Then:

```sh
sh tools/exynos2400e-benchmark.sh ponko-exynos-logcat.txt
```

Run fixed-settings cases and interrupt capture with Ctrl-C. The script creates:

- `ponko-exynos-generations.csv`
- `ponko-exynos-vae-stages.csv`

The Python parser prints median, minimum, maximum, and sample count for each exact generation configuration. It also tracks the last logged VAE convolution mode for VAE-stage rows when the upstream model-load log exposes it.

Use a fixed model, prompt, seed, dimensions, steps, sampler, scheduler, CFG, LoRA state, backend and thread count for a comparison. Do two warmups and at least five recorded runs per case, switch test order on a second pass, and allow the phone to cool between passes. Compare medians rather than the fastest sample.

## What the measurements mean

- The generation timer wraps `generate_image()`; model loading and Kotlin bitmap construction are outside it.
- VAE stage timing is the full VAE wrapper operation, including scaling and tiling where enabled. It is not per-operator/per-Conv2D kernel time.
- `backend_request` is the requested backend, not proof of the actual selected backend. Check model-load logs and Vulkan diagnostics for the resolved backend.
- Process RSS/HWM is not total system memory or a per-stage native peak allocation.
- Process CPU time includes CPU used by the app process's threads during the interval, not only the calling thread.

## Optimization direction

The code already builds the Exynos-specific DOTPROD/I8MM/FP16 profile and enables GGML KleidiAI and weight repacking. After the timings exist, inspect the dominant operations rather than changing every path blindly. If VAE dominates, continue into Conv2D/GroupNorm/attention kernels. If denoising dominates, inspect GGML op dispatch, KleidiAI eligibility/repacking, worker scaling, and the actually selected backend. If scaling stalls, compare wall time with process CPU time, clocks, thermal readings and memory before changing scheduling or allocation.

Huge pages, IRQ affinity, governors, cpuset isolation, and vendor driver modifications are system-level mechanisms, not guarantees available to a normal unrooted APK.
