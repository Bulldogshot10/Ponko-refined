//! Typed Rust boundary for Ponko's Android native engine.
 //!
 //! This first migration slice deliberately keeps the legacy JNI engine intact.
 //! These records mirror the current Kotlin/JNI contract so later cutover can be
 //! performed with explicit parity tests rather than deleting features in one go.

uniffi::setup_scaffolding!();

/// Version of the Rust facade contract, independent of the app's versionCode.
pub const FACADE_API_VERSION: u32 = 1;

#[derive(Debug, Clone, uniffi::Record)]
pub struct BuildProfile {
    pub api_version: u32,
    pub abi: String,
    pub cpu_arch_flags: String,
    pub vulkan_build_default: bool,
    pub native_engine: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelSlots {
    pub model: Option<String>,
    pub diffusion: Option<String>,
    pub high_noise: Option<String>,
    pub uncond: Option<String>,
    pub clip_l: Option<String>,
    pub clip_g: Option<String>,
    pub clip_vision: Option<String>,
    pub t5xxl: Option<String>,
    pub llm: Option<String>,
    pub llm_vision: Option<String>,
    pub vae: Option<String>,
    pub audio_vae: Option<String>,
    pub taesd: Option<String>,
}

/// Mirrors the current nativeCreate inputs without relying on positional JNI arguments.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelLoadRequest {
    pub model_path: String,
    pub vae_path: Option<String>,
    pub n_threads: u32,
    /// -1 means preserve the model's original weight type.
    pub wtype: i32,
    pub flash_attn: bool,
    pub vae_conv_direct: bool,
    pub backend: Option<String>,
    pub slots: Option<ModelSlots>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct QuantizeUnetRequest {
    pub input_path: String,
    pub output_path: String,
    pub quant_type: String,
    pub diffusion_slot: bool,
    pub n_threads: u32,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ResAdapterRequest {
    pub input_model_path: String,
    pub vae_path: Option<String>,
    pub normalization_path: String,
    pub output_path: String,
    pub quant_type: String,
    pub n_threads: u32,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerationRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub lora_paths: Vec<String>,
    pub lora_scales: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg: f32,
    pub seed: i64,
    pub sample_method: i32,
    pub scheduler: i32,
    /// Negative values preserve the current Kotlin/JNI sentinel for automatic ETA.
    pub eta: f32,
    pub hires_enabled: bool,
    pub hires_upscaler: i32,
    pub hires_model_path: Option<String>,
    pub hires_scale: f32,
    pub hires_target_width: u32,
    pub hires_target_height: u32,
    pub hires_steps: u32,
    pub hires_denoising_strength: f32,
    pub hires_tile_size: u32,
    pub clip_skip: u32,
    /// Optional RGB888, row-major image bytes for img2img.
    pub init_image_rgb888: Option<Vec<u8>>,
    pub init_image_width: u32,
    pub init_image_height: u32,
    pub strength: f32,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ValidationReport {
    pub valid: bool,
    pub message: String,
}

#[uniffi::export]
pub fn engine_api_version() -> u32 {
    FACADE_API_VERSION
}

#[uniffi::export]
pub fn build_profile() -> BuildProfile {
    BuildProfile {
        api_version: FACADE_API_VERSION,
        abi: "arm64-v8a".to_owned(),
        // build.rs injects the Gradle/CMake profile; direct cargo builds use the Exynos default.
        cpu_arch_flags: option_env!("PONKO_CPU_ARCH_FLAGS")
            .unwrap_or("-march=armv8.6-a+dotprod+i8mm+fp16 -mtune=cortex-x4")
            .to_owned(),
        vulkan_build_default: !matches!(
            option_env!("PONKO_VULKAN_ENABLED"),
            Some("OFF") | Some("off") | Some("FALSE") | Some("false") | Some("0")
        ),
        native_engine: "stable-diffusion.cpp + GGML (legacy JNI retained during parity phase)"
            .to_owned(),
    }
}

#[uniffi::export]
pub fn validate_model_slots(slots: ModelSlots) -> ValidationReport {
    let has_path = [
        slots.model,
        slots.diffusion,
        slots.high_noise,
        slots.uncond,
        slots.clip_l,
        slots.clip_g,
        slots.clip_vision,
        slots.t5xxl,
        slots.llm,
        slots.llm_vision,
        slots.vae,
        slots.audio_vae,
        slots.taesd,
    ]
    .into_iter()
    .flatten()
    .any(|path| !path.trim().is_empty());

    if has_path {
        ValidationReport {
            valid: true,
            message: "At least one model slot is configured.".to_owned(),
        }
    } else {
        ValidationReport {
            valid: false,
            message: "At least one model file path is required.".to_owned(),
        }
    }
}

/// Check cross-field invariants before a request crosses the FFI boundary.
/// CFG=0 remains valid for TCD and other zero-CFG workflows.
/// Validate the current model loading contract before forwarding it to the native engine.
#[uniffi::export]
pub fn validate_model_load_request(request: ModelLoadRequest) -> ValidationReport {
    let explicit_model = !request.model_path.trim().is_empty();
    let slot_model = request.slots.as_ref().map(|slots| {
        slots.model.as_deref().map(str::trim).is_some_and(|path| !path.is_empty())
            || slots.diffusion.as_deref().map(str::trim).is_some_and(|path| !path.is_empty())
    }).unwrap_or(false);

    if explicit_model || slot_model {
        ValidationReport {
            valid: true,
            message: "A base model path is configured.".to_owned(),
        }
    } else {
        invalid("A base model or diffusion-model slot is required.")
    }
}

#[uniffi::export]
pub fn validate_unet_quantization_request(request: QuantizeUnetRequest) -> ValidationReport {
    if request.input_path.trim().is_empty() || request.output_path.trim().is_empty() {
        return invalid("Quantization input and output paths must not be empty.");
    }
    validate_quantization_type(request.quant_type)
}

#[uniffi::export]
pub fn validate_res_adapter_request(request: ResAdapterRequest) -> ValidationReport {
    if request.input_model_path.trim().is_empty()
        || request.normalization_path.trim().is_empty()
        || request.output_path.trim().is_empty()
    {
        return invalid("ResAdapter base, normalization, and output paths must not be empty.");
    }
    validate_quantization_type(request.quant_type)
}

#[uniffi::export]
pub fn validate_generation_request(request: GenerationRequest) -> ValidationReport {
    if request.width == 0 || request.height == 0 {
        return invalid("Width and height must be greater than zero.");
    }
    if !(1..=150).contains(&request.steps) {
        return invalid("Steps must be in the supported range 1..=150.");
    }
    if !request.cfg.is_finite() || request.cfg < 0.0 {
        return invalid("CFG must be finite and greater than or equal to zero.");
    }
    if !request.eta.is_finite() {
        return invalid("ETA must be finite; negative values select automatic ETA.");
    }
    if !(0..22).contains(&request.sample_method) {
        return invalid("Sampler code is not present in Ponko's current native sampler enum.");
    }
    if !(0..17).contains(&request.scheduler) {
        return invalid("Scheduler code is not present in Ponko's current native scheduler enum.");
    }
    if request.lora_paths.len() != request.lora_scales.len() {
        return invalid("LoRA paths and scales must have the same number of entries.");
    }
    if request.lora_scales.iter().any(|scale| !scale.is_finite()) {
        return invalid("Every LoRA scale must be finite.");
    }

    if let Some(rgb) = request.init_image_rgb888 {
        if request.init_image_width == 0 || request.init_image_height == 0 {
            return invalid("An img2img input requires non-zero image dimensions.");
        }
        let expected = (request.init_image_width as usize)
            .checked_mul(request.init_image_height as usize)
            .and_then(|pixels| pixels.checked_mul(3));
        if expected != Some(rgb.len()) {
            return invalid("The img2img buffer must contain exactly width * height * 3 RGB888 bytes.");
        }
    }

    ValidationReport {
        valid: true,
        message: "Request fields are internally consistent.".to_owned(),
    }
}

#[uniffi::export]
pub fn validate_quantization_type(quant_type: String) -> ValidationReport {
    match quant_type.as_str() {
        "q4_0" | "q5_0" | "q8_0" => ValidationReport {
            valid: true,
            message: "Quantization type is supported by the current native converter.".to_owned(),
        },
        _ => invalid("Supported UNet quantization types are q4_0, q5_0, and q8_0."),
    }
}

fn invalid(message: &str) -> ValidationReport {
    ValidationReport {
        valid: false,
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> GenerationRequest {
        GenerationRequest {
            prompt: "test".into(),
            negative_prompt: String::new(),
            lora_paths: vec!["one.safetensors".into()],
            lora_scales: vec![1.0],
            width: 512,
            height: 512,
            steps: 6,
            cfg: 0.0,
            seed: 42,
            sample_method: 11, // TCD
            scheduler: 16,     // TCD
            eta: -1.0,
            hires_enabled: false,
            hires_upscaler: 1,
            hires_model_path: None,
            hires_scale: 2.0,
            hires_target_width: 0,
            hires_target_height: 0,
            hires_steps: 0,
            hires_denoising_strength: 0.7,
            hires_tile_size: 128,
            clip_skip: 1,
            init_image_rgb888: None,
            init_image_width: 0,
            init_image_height: 0,
            strength: 1.0,
        }
    }

    #[test]
    fn preserves_zero_cfg_tcd_requests() {
        assert!(validate_generation_request(request()).valid);
    }

    #[test]
    fn rejects_mismatched_lora_arrays() {
        let mut input = request();
        input.lora_scales.clear();
        assert!(!validate_generation_request(input).valid);
    }

    #[test]
    fn validates_rgb888_img2img_buffer() {
        let mut input = request();
        input.init_image_width = 2;
        input.init_image_height = 2;
        input.init_image_rgb888 = Some(vec![0; 12]);
        assert!(validate_generation_request(input.clone()).valid);
        input.init_image_rgb888 = Some(vec![0; 11]);
        assert!(!validate_generation_request(input).valid);
    }

    #[test]
    fn does_not_impose_a_lora_count_limit() {
        let mut input = request();
        input.lora_paths = (0..8).map(|i| format!("{i}.safetensors")).collect();
        input.lora_scales = vec![0.5; 8];
        assert!(validate_generation_request(input).valid);
    }

    fn empty_slots() -> ModelSlots {
        ModelSlots {
            model: None,
            diffusion: None,
            high_noise: None,
            uncond: None,
            clip_l: None,
            clip_g: None,
            clip_vision: None,
            t5xxl: None,
            llm: None,
            llm_vision: None,
            vae: None,
            audio_vae: None,
            taesd: None,
        }
    }

    #[test]
    fn model_load_requires_a_model_or_diffusion_slot() {
        let request = ModelLoadRequest {
            model_path: String::new(),
            vae_path: None,
            n_threads: 4,
            wtype: -1,
            flash_attn: true,
            vae_conv_direct: true,
            backend: None,
            slots: Some(empty_slots()),
        };
        assert!(!validate_model_load_request(request.clone()).valid);
        let mut valid_request = request;
        valid_request.slots.as_mut().unwrap().diffusion = Some("unet.gguf".into());
        assert!(validate_model_load_request(valid_request).valid);
    }

    #[test]
    fn quantization_request_validates_paths_and_type() {
        let request = QuantizeUnetRequest {
            input_path: "input.gguf".into(),
            output_path: "output.gguf".into(),
            quant_type: "q5_0".into(),
            diffusion_slot: true,
            n_threads: 4,
        };
        assert!(validate_unet_quantization_request(request.clone()).valid);
        let mut bad_request = request;
        bad_request.quant_type = "q6_k".into();
        assert!(!validate_unet_quantization_request(bad_request).valid);
    }

    #[test]
    fn res_adapter_requires_normalization_and_output_paths() {
        let request = ResAdapterRequest {
            input_model_path: "base.gguf".into(),
            vae_path: None,
            normalization_path: "norm.safetensors".into(),
            output_path: "merged.gguf".into(),
            quant_type: "q4_0".into(),
            n_threads: 4,
        };
        assert!(validate_res_adapter_request(request.clone()).valid);
        let mut bad_request = request;
        bad_request.normalization_path.clear();
        assert!(!validate_res_adapter_request(bad_request).valid);
    }

    #[test]
    fn supports_only_current_converter_types() {
        assert!(validate_quantization_type("q4_0".into()).valid);
        assert!(validate_quantization_type("q5_0".into()).valid);
        assert!(validate_quantization_type("q8_0".into()).valid);
        assert!(!validate_quantization_type("q6_k".into()).valid);
    }
}
