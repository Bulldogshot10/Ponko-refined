# VAE direct-convolution A/B protocol

This experiment compares the standard VAE convolution path with GGML's direct Conv2D path. The current timing log measures the full native generation call, not the VAE decode stage by itself. It is useful for an A/B screen, but cannot yet attribute a speedup exclusively to VAE decoding.

## Keep the comparison controlled

Use the original VAE for the model family under test and leave TAESD disabled. Pick one backend per comparison (CPU and Vulkan are separate experiments). Keep the model file, prompt, seed, width/height, sampler, scheduler, step count, CFG, LoRA state, CPU thread count, and phone performance conditions fixed.

In the drawing settings, turn **Use direct Conv2D for VAE** ON, unload/reload the model, and run two warm-up generations followed by at least five measured generations. Then turn it OFF, unload/reload the model, and repeat. For a less biased result, repeat in the opposite order after the phone cools. Compare medians rather than the single fastest run.

The native log states the selected path when the model loads:

- `Using Conv2d direct in the vae model`
- `Using standard VAE convolution path (direct Conv2D disabled)`

## Capture timing

Filter Android logcat by the native tag:

```sh
adb logcat -c
adb logcat -s PonkoSd:I
```

Each generation prints a line like:

```text
BENCHMARK generate_image elapsed_ms=... mode=txt2img size=512x512 steps=4 backend=CPU vae_conv_direct=1 ok=1
```

Record the elapsed time, backend, mode, resolution, steps, and direct-convolution flag together. Only compare rows with matching settings. The number includes all work inside `generate_image` (including text conditioning, denoising, and VAE work where applicable); model loading and Kotlin bitmap conversion are outside this timer.

## Current limits

This first pass measures end-to-end native generation latency. It does not yet report VAE-only latency or peak RAM. If the end-to-end change is ambiguous, the next instrumentation step is a timer around the VAE decode runner, followed by peak-memory sampling. Do not infer a VAE-only speedup from the aggregate timer.
