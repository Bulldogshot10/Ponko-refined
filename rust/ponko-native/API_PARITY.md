# Hearth adapter support map

| Area | Current status |
|---|---|
| Kotlin/native API | Rust + UniFFI |
| Drawing engine | Hearth / Burn / WGPU Vulkan only |
| Checkpoint format | SafeTensors |
| Families | SD 1.5 and SDXL |
| Tokenizer | External vocab.json and merges.txt |
| SD 1.5 VAE | Embedded or separate SafeTensors |
| SDXL VAE | Separate FP16-safe SafeTensors |
| LoRA | Applies supported adapter weights to CLIP/UNet |
| Samplers | Euler, Euler a, DPM++ 2M, DPM++ SDE |
| Schedules | Normal, Karras |
| Progress/cancel | Step callbacks; cancellation checked between denoising steps |
| Request validation | Dimensions from 64 to 2048 px, divisible by 8; at most 4 LoRAs |
| Drawing GGUF / quantization | Not implemented |
| Img2img / HiRes Fix / ADetailer | Not implemented in this adapter |
| UNet quantization / ResAdapter / LoRA fusion | Not implemented in this adapter |
| Android Xclipse 940 validation | Pending device build and runtime smoke test |

## Required validation

1. Android ARM64 compile with recursive Hearth/Burn/CubeCL/Cubek submodules.
2. SD 1.5 SafeTensors text-to-image smoke test on Exynos 2400e / Xclipse 940.
3. SDXL load/generation test where device memory allows.
4. Verify model loading, generated dimensions, Vulkan diagnostics, memory use, and cancellation behavior.
5. Run request-validation tests for oversized dimensions, unsupported samplers/features, and the 4-LoRA limit.
