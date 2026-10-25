//! Ponko native inference engine: Hearth (Rust) + Burn/WGPU Vulkan.
//! The diffusion path is fully Rust. No sd.cpp/GGML/C ABI loader is kept.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicI64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use burn::{
    backend::wgpu::{graphics::AutoGraphicsApi, init_setup, RuntimeOptions, WgpuDevice},
    prelude::Backend as _,
    tensor::{Device, Distribution, Tensor},
};
use hearth::{
    clip::{
        ClipTokenizer, OpenClipTextEncoder, Sd15ClipTextEncoder, Sd15Conditioning,
        SdxlClipLTextEncoder, SdxlConditioning,
    },
    lora::LoraFile,
    model_loader::SafeTensorsFile,
    sampling::{
        DpmPp2mSampler, DpmPpSdeSampler, EulerASampler, EulerSampler, NoiseSchedule,
        SchedulerKind,
    },
    types::Backend,
    unet::{Sd15Unet2D, SdxlUnet2D},
    vae::{Sd15VaeDecoder, SdxlVaeDecoder},
};

uniffi::setup_scaffolding!();

const FACADE_API_VERSION: i32 = 4;
static RUNTIME_INIT: OnceLock<Result<(), String>> = OnceLock::new();

fn native_error(error: impl std::fmt::Display) -> PonkoError {
    PonkoError::Native { details: error.to_string() }
}

fn init_runtime() -> Result<(), PonkoError> {
    let result = RUNTIME_INIT.get_or_init(|| {
        std::panic::catch_unwind(|| {
            init_setup::<AutoGraphicsApi>(
                &WgpuDevice::DefaultDevice,
                RuntimeOptions { tasks_max: 512, ..Default::default() },
            );
        })
        .map_err(|_| "Hearth could not initialise Burn/WGPU Vulkan. Check Android Vulkan driver support.".to_owned())
    });
    result.clone().map_err(|details| PonkoError::Native { details })
}

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum PonkoError {
    #[error("{details}")]
    InvalidArgument { details: String },
    #[error("{details}")]
    Native { details: String },
    #[error("{details}")]
    InvalidHandle { details: String },
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct BuildProfile {
    pub api_version: i32,
    pub abi: String,
    pub cpu_arch_flags: String,
    pub vulkan_build_default: bool,
    pub native_engine: String,
}

#[derive(Debug, Clone, uniffi::Record, Default)]
pub struct ModelSlots {
    pub model: Option<String>,
    pub diffusion: Option<String>,
    pub high_noise: Option<String>,
    pub uncond: Option<String>,
    /// Hearth convention in Ponko: CLIP-L slot holds tokenizer vocab.json.
    pub clip_l: Option<String>,
    /// Hearth convention in Ponko: CLIP-G slot holds tokenizer merges.txt.
    pub clip_g: Option<String>,
    pub clip_vision: Option<String>,
    pub t5xxl: Option<String>,
    pub llm: Option<String>,
    pub llm_vision: Option<String>,
    pub vae: Option<String>,
    pub audio_vae: Option<String>,
    pub taesd: Option<String>,
}

impl ModelSlots {
    fn ordered(&self) -> [Option<&str>; 13] {
        [
            self.model.as_deref(), self.diffusion.as_deref(), self.high_noise.as_deref(),
            self.uncond.as_deref(), self.clip_l.as_deref(), self.clip_g.as_deref(),
            self.clip_vision.as_deref(), self.t5xxl.as_deref(), self.llm.as_deref(),
            self.llm_vision.as_deref(), self.vae.as_deref(), self.audio_vae.as_deref(),
            self.taesd.as_deref(),
        ]
    }
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelLoadRequest {
    pub model_path: String,
    pub vae_path: Option<String>,
    pub n_threads: i32,
    pub wtype: i32,
    pub flash_attn: bool,
    pub vae_conv_direct: bool,
    pub backend: Option<String>,
    pub slots: Option<ModelSlots>,
    pub family: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerationRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub lora_paths: Vec<String>,
    pub lora_scales: Vec<f32>,
    pub width: i32,
    pub height: i32,
    pub steps: i32,
    pub cfg: f32,
    pub seed: i64,
    pub sample_method: i32,
    pub scheduler: i32,
    pub eta: f32,
    pub hires_enabled: bool,
    pub hires_upscaler: i32,
    pub hires_model_path: Option<String>,
    pub hires_scale: f32,
    pub hires_target_width: i32,
    pub hires_target_height: i32,
    pub hires_steps: i32,
    pub hires_denoising_strength: f32,
    pub hires_tile_size: i32,
    pub clip_skip: i32,
    pub init_image_rgb888: Option<Vec<u8>>,
    pub init_image_width: i32,
    pub init_image_height: i32,
    pub strength: f32,
    pub adetailer_enabled: bool,
    pub adetailer_detector_path: Option<String>,
    pub adetailer_prompt: String,
    pub adetailer_negative_prompt: String,
    pub adetailer_extra_args: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct GeneratedImage {
    pub width: i32,
    pub height: i32,
    pub rgb888: Vec<u8>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ValidationReport {
    pub valid: bool,
    pub message: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct QuantizeUnetRequest {
    pub input_path: String,
    pub output_path: String,
    pub quant_type: String,
    pub diffusion_slot: bool,
    pub n_threads: i32,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ResAdapterRequest {
    pub input_model_path: String,
    pub vae_path: Option<String>,
    pub normalization_path: String,
    pub output_path: String,
    pub quant_type: String,
    pub n_threads: i32,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct LoraFusionRequest {
    pub input_model_path: String,
    pub lora_paths: Vec<String>,
    pub lora_scales: Vec<f32>,
    pub output_path: String,
    pub quant_type: String,
    pub n_threads: i32,
}

#[uniffi::export(rust, foreign)]
pub trait StepCallback: Send + Sync {
    fn on_step(&self, step: i32, steps: i32);
}

#[uniffi::export(rust, foreign)]
pub trait PreviewCallback: Send + Sync {
    fn on_preview(&self, step: i32, width: i32, height: i32, rgb888: Vec<u8>);
}

#[derive(Clone)]
struct HearthContext {
    family: String,
    model_path: PathBuf,
    vae_path: Option<PathBuf>,
    vocab_path: PathBuf,
    merges_path: PathBuf,
}

#[derive(uniffi::Object)]
pub struct NativeEngine {
    next_id: AtomicI64,
    handles: Mutex<HashMap<i64, HearthContext>>,
    cancel_flags: Mutex<HashMap<i64, Arc<AtomicBool>>>,
    generation_lock: Mutex<()>,
    last_params: Mutex<String>,
}

#[uniffi::export]
impl NativeEngine {
    #[uniffi::constructor]
    pub fn new() -> Result<Arc<Self>, PonkoError> {
        Ok(Arc::new(Self {
            next_id: AtomicI64::new(1),
            handles: Mutex::new(HashMap::new()),
            cancel_flags: Mutex::new(HashMap::new()),
            generation_lock: Mutex::new(()),
            last_params: Mutex::new("Hearth not yet loaded".to_owned()),
        }))
    }

    pub fn create_context(&self, request: ModelLoadRequest) -> Result<i64, PonkoError> {
        let report = validate_model_load_request(request.clone());
        if !report.valid {
            return Err(PonkoError::InvalidArgument { details: report.message });
        }
        init_runtime()?;

        let slots = request.slots.as_ref();
        let vocab = slots.and_then(|s| s.clip_l.as_deref()).unwrap_or_default();
        let merges = slots.and_then(|s| s.clip_g.as_deref()).unwrap_or_default();
        let model_path = PathBuf::from(&request.model_path);
        let vae_path = request.vae_path.as_ref()
            .or_else(|| slots.and_then(|s| s.vae.as_ref()))
            .map(PathBuf::from);

        if !model_path.is_file() {
            return Err(PonkoError::InvalidArgument { details: format!("Checkpoint not found: {}", model_path.display()) });
        }
        if model_path.extension().and_then(|v| v.to_str()).is_none_or(|v| !v.eq_ignore_ascii_case("safetensors")) {
            return Err(PonkoError::InvalidArgument { details: "Hearth expects a SafeTensors checkpoint (.safetensors), not GGUF/CKPT.".to_owned() });
        }
        if vocab.trim().is_empty() || !Path::new(vocab).is_file() {
            return Err(PonkoError::InvalidArgument { details: "Choose tokenizer vocab.json in the Tokenizer vocab slot.".to_owned() });
        }
        if merges.trim().is_empty() || !Path::new(merges).is_file() {
            return Err(PonkoError::InvalidArgument { details: "Choose tokenizer merges.txt in the Tokenizer merges slot.".to_owned() });
        }
        if request.family == "sdxl" && vae_path.is_none() {
            return Err(PonkoError::InvalidArgument { details: "SDXL requires the FP16-safe VAE checkpoint in the VAE slot.".to_owned() });
        }
        if let Some(ref path) = vae_path {
            if !path.is_file() || path.extension().and_then(|v| v.to_str()).is_none_or(|v| !v.eq_ignore_ascii_case("safetensors")) {
                return Err(PonkoError::InvalidArgument { details: format!("VAE must be an existing .safetensors file: {}", path.display()) });
            }
        }

        let context = HearthContext {
            family: request.family.to_ascii_lowercase(),
            model_path,
            vae_path,
            vocab_path: PathBuf::from(vocab),
            merges_path: PathBuf::from(merges),
        };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.handles.lock().map_err(|_| native_error("Hearth handle registry lock poisoned"))?
            .insert(id, context);
        self.cancel_flags.lock().map_err(|_| native_error("Hearth cancellation registry lock poisoned"))?
            .insert(id, Arc::new(AtomicBool::new(false)));
        *self.last_params.lock().map_err(|_| native_error("Hearth status lock poisoned"))? =
            format!("Hearth context {id}: family={}, checkpoint={}", request.family, request.model_path);
        Ok(id)
    }

    pub fn cancel(&self, handle_id: i64) -> Result<(), PonkoError> {
        let flag = self.cancel_flags.lock().map_err(|_| native_error("Hearth cancellation registry lock poisoned"))?
            .get(&handle_id).cloned().ok_or_else(|| PonkoError::InvalidHandle {
                details: format!("unknown or unloaded Hearth handle {handle_id}"),
            })?;
        flag.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub fn free_context(&self, handle_id: i64) -> Result<(), PonkoError> {
        let removed = self.handles.lock().map_err(|_| native_error("Hearth handle registry lock poisoned"))?
            .remove(&handle_id);
        if removed.is_none() {
            return Err(PonkoError::InvalidHandle { details: format!("unknown or already freed Hearth handle {handle_id}") });
        }
        self.cancel_flags.lock().map_err(|_| native_error("Hearth cancellation registry lock poisoned"))?
            .remove(&handle_id);
        Ok(())
    }

    pub fn info(&self) -> Result<String, PonkoError> {
        Ok("Hearth Rust inference · Burn · WGPU Vulkan · SD 1.5 / SDXL".to_owned())
    }

    pub fn vulkan_diagnostics(&self) -> Result<String, PonkoError> {
        Ok("Burn/WGPU compiled with Vulkan backend. Adapter selection is performed by wgpu at runtime; Android/Xclipse adapter details are reported by the platform driver.".to_owned())
    }

    pub fn last_params(&self) -> Result<String, PonkoError> {
        Ok(self.last_params.lock().map_err(|_| native_error("Hearth status lock poisoned"))?.clone())
    }

    pub fn list_devices(&self) -> Result<String, PonkoError> {
        Ok("Vulkan\tHearth / Burn-WGPU Vulkan adapter (runtime-selected)".to_owned())
    }

    pub fn quantize_unet(&self, _request: QuantizeUnetRequest) -> Result<(), PonkoError> {
        Err(native_error("UNet quantization is not implemented in Hearth; import a supported SafeTensors checkpoint."))
    }

    pub fn fuse_loras(&self, _request: LoraFusionRequest) -> Result<(), PonkoError> {
        Err(native_error("LoRA fusion/export is not implemented in Hearth; adapters are applied at inference time."))
    }

    pub fn apply_resadapter(&self, _request: ResAdapterRequest) -> Result<(), PonkoError> {
        Err(native_error("ResAdapter checkpoint conversion is not implemented in Hearth."))
    }
}

impl NativeEngine {
    fn generate_internal(
        &self,
        handle_id: i64,
        request: GenerationRequest,
        step_callback: Option<Arc<dyn StepCallback>>,
        preview_callback: Option<Arc<dyn PreviewCallback>>,
    ) -> Result<GeneratedImage, PonkoError> {
        let report = validate_generation_request(request.clone());
        if !report.valid {
            return Err(PonkoError::InvalidArgument { details: report.message });
        }
        let context = self.handles.lock().map_err(|_| native_error("Hearth handle registry lock poisoned"))?
            .get(&handle_id).cloned().ok_or_else(|| PonkoError::InvalidHandle {
                details: format!("unknown or unloaded Hearth handle {handle_id}"),
            })?;
        let cancel = self.cancel_flags.lock().map_err(|_| native_error("Hearth cancellation registry lock poisoned"))?
            .get(&handle_id).cloned().ok_or_else(|| PonkoError::InvalidHandle {
                details: format!("unknown or unloaded Hearth handle {handle_id}"),
            })?;

        let _generation = self.generation_lock.lock().map_err(|_| native_error("Hearth generation lock poisoned"))?;
        cancel.store(false, Ordering::Relaxed);
        *self.last_params.lock().map_err(|_| native_error("Hearth status lock poisoned"))? = format!(
            "Hearth generate: family={}, {}x{}, steps={}, cfg={}, sampler={}, scheduler={}, LoRAs={}",
            context.family, request.width, request.height, request.steps, request.cfg,
            request.sample_method, request.scheduler, request.lora_paths.len()
        );

        match context.family.as_str() {
            "sd15" => render_sd15(&context, &request, step_callback.as_ref(), preview_callback.as_ref(), &cancel),
            "sdxl" => render_sdxl(&context, &request, step_callback.as_ref(), preview_callback.as_ref(), &cancel),
            other => Err(PonkoError::InvalidArgument { details: format!("Hearth backend supports SD 1.5 and SDXL; unsupported family '{other}'.") }),
        }
    }
}

#[uniffi::export]
pub fn generate_image(
    engine: Arc<NativeEngine>,
    handle_id: i64,
    request: GenerationRequest,
    step_callback: Option<Arc<dyn StepCallback>>,
    preview_callback: Option<Arc<dyn PreviewCallback>>,
) -> Result<GeneratedImage, PonkoError> {
    engine.generate_internal(handle_id, request, step_callback, preview_callback)
}

fn render_sd15(
    context: &HearthContext,
    request: &GenerationRequest,
    step_callback: Option<&Arc<dyn StepCallback>>,
    preview_callback: Option<&Arc<dyn PreviewCallback>>,
    cancel: &AtomicBool,
) -> Result<GeneratedImage, PonkoError> {
    init_runtime()?;
    let device: Device<Backend> = Default::default();
    let checkpoint = SafeTensorsFile::open(&context.model_path).map_err(native_error)?;
    let tensors = checkpoint.tensors().map_err(native_error)?;
    let tokenizer = ClipTokenizer::from_files(&context.vocab_path, &context.merges_path).map_err(native_error)?;

    let lora_files = request.lora_paths.iter().map(|path| LoraFile::open(path).map_err(native_error))
        .collect::<Result<Vec<_>, _>>()?;

    let mut clip = Sd15ClipTextEncoder::load(&tensors, &device).map_err(native_error)?;
    for (i, lora_file) in lora_files.iter().enumerate() {
        let lora = lora_file.tensors().map_err(native_error)?;
        let strength = request.lora_scales[i];
        clip.apply_lora("text_model", "lora_te", &lora, strength, &device).map_err(native_error)?;
    }
    let positive_tokens = tokenizer.encode(&request.prompt, &device);
    let negative_tokens = tokenizer.encode(&request.negative_prompt, &device);
    let positive = Sd15Conditioning::new(clip.forward(positive_tokens));
    let negative = Sd15Conditioning::new(clip.forward(negative_tokens));
    drop(clip);

    let mut unet = Sd15Unet2D::load(&tensors, &device).map_err(native_error)?;
    for (i, lora_file) in lora_files.iter().enumerate() {
        let lora = lora_file.tensors().map_err(native_error)?;
        unet.apply_lora("lora_unet", &lora, request.lora_scales[i], &device).map_err(native_error)?;
    }

    let noise_schedule = NoiseSchedule::load(&tensors, &device).map_err(native_error)?;
    let schedule = if request.scheduler == 1 {
        SchedulerKind::Karras.schedule(&noise_schedule, request.steps as usize)
    } else {
        SchedulerKind::Normal.schedule(&noise_schedule, request.steps as usize)
    };
    Backend::seed(&device, request.seed as u64);
    let sigma_max = schedule.sigmas[0];
    let latent: Tensor<Backend, 4> = Tensor::random(
        [1, 4, (request.height / 8) as usize, (request.width / 8) as usize],
        Distribution::Normal(0.0, sigma_max as f64), &device,
    );
    let progress = |step: usize, total: usize| -> bool {
        if let Some(callback) = step_callback {
            callback.on_step(step.min(i32::MAX as usize) as i32, total.min(i32::MAX as usize) as i32);
        }
        !cancel.load(Ordering::Relaxed)
    };
    let denoised = match request.sample_method {
        0 => EulerSampler::new(noise_schedule).sample_with_schedule(&unet, latent, &positive, &negative, &schedule, request.cfg, Some(&progress)),
        1 => EulerASampler::new(noise_schedule).sample_with_schedule(&unet, latent, &positive, &negative, &schedule, request.cfg, Some(&progress)),
        5 | 6 => DpmPp2mSampler::new(noise_schedule).sample_with_schedule(&unet, latent, &positive, &negative, &schedule, request.cfg, Some(&progress)),
        18 | 21 => DpmPpSdeSampler::new(noise_schedule).sample_with_schedule(&unet, latent, &positive, &negative, &schedule, request.cfg, Some(&progress)),
        _ => return Err(PonkoError::InvalidArgument { details: "Hearth supports Euler, Euler a, DPM++ 2M and DPM++ SDE samplers.".to_owned() }),
    };
    if cancel.load(Ordering::Relaxed) {
        return Err(native_error("Generation cancelled."));
    }
    drop(unet);
    drop(positive);
    drop(negative);
    let _ = Backend::sync(&device);
    Backend::memory_cleanup(&device);

    let decoded = if let Some(vae_path) = context.vae_path.as_ref() {
        let vae_file = SafeTensorsFile::open(vae_path).map_err(native_error)?;
        let vae_tensors = vae_file.tensors().map_err(native_error)?;
        let vae = Sd15VaeDecoder::load(&vae_tensors, None, &device).map_err(native_error)?;
        vae.forward(denoised)
    } else {
        let vae = Sd15VaeDecoder::load(&tensors, Some("first_stage_model"), &device).map_err(native_error)?;
        vae.forward(denoised)
    };
    tensor_to_rgb(decoded, request.width, request.height, request.steps, step_callback, preview_callback)
}

fn render_sdxl(
    context: &HearthContext,
    request: &GenerationRequest,
    step_callback: Option<&Arc<dyn StepCallback>>,
    preview_callback: Option<&Arc<dyn PreviewCallback>>,
    cancel: &AtomicBool,
) -> Result<GeneratedImage, PonkoError> {
    init_runtime()?;
    let device: Device<Backend> = Default::default();
    let checkpoint = SafeTensorsFile::open(&context.model_path).map_err(native_error)?;
    let tensors = checkpoint.tensors().map_err(native_error)?;
    let tokenizer = ClipTokenizer::from_files(&context.vocab_path, &context.merges_path).map_err(native_error)?;
    let lora_files = request.lora_paths.iter().map(|path| LoraFile::open(path).map_err(native_error))
        .collect::<Result<Vec<_>, _>>()?;

    let mut clip_l = SdxlClipLTextEncoder::load_with_prefix(&tensors, hearth::model_loader::prefix::SDXL_CLIP_L, &device).map_err(native_error)?;
    let mut clip_g = OpenClipTextEncoder::load(&tensors, hearth::model_loader::prefix::SDXL_CLIP_G, &device).map_err(native_error)?;
    for (i, lora_file) in lora_files.iter().enumerate() {
        let lora = lora_file.tensors().map_err(native_error)?;
        let strength = request.lora_scales[i];
        clip_l.apply_lora("text_model", "lora_te", &lora, strength, &device).map_err(native_error)?;
        clip_g.apply_lora("lora_te2", &lora, strength, &device).map_err(native_error)?;
    }

    let pos_l = tokenizer.encode(&request.prompt, &device);
    let neg_l = tokenizer.encode(&request.negative_prompt, &device);
    let pos_g = tokenizer.encode_open_clip(&request.prompt, &device);
    let neg_g = tokenizer.encode_open_clip(&request.negative_prompt, &device);
    let pos_l_hidden = clip_l.forward_hidden_layer(pos_l, Some(10));
    let neg_l_hidden = clip_l.forward_hidden_layer(neg_l, Some(10));
    let (pos_g_hidden, pos_pooled) = clip_g.forward(pos_g);
    let (neg_g_hidden, neg_pooled) = clip_g.forward(neg_g);
    let h = request.height as f32;
    let w = request.width as f32;
    let positive = SdxlConditioning::new(pos_l_hidden, pos_g_hidden, pos_pooled, (h, w), (0.0, 0.0), (h, w), &device);
    let negative = SdxlConditioning::new(neg_l_hidden, neg_g_hidden, neg_pooled, (h, w), (0.0, 0.0), (h, w), &device);
    drop(clip_l);
    drop(clip_g);

    let mut unet = SdxlUnet2D::load(&tensors, &device).map_err(native_error)?;
    for (i, lora_file) in lora_files.iter().enumerate() {
        let lora = lora_file.tensors().map_err(native_error)?;
        unet.apply_lora("lora_unet", &lora, request.lora_scales[i], &device).map_err(native_error)?;
    }

    let noise_schedule = NoiseSchedule::linear(1000, 0.00085, 0.012);
    let schedule = if request.scheduler == 1 {
        SchedulerKind::Karras.schedule(&noise_schedule, request.steps as usize)
    } else {
        SchedulerKind::Normal.schedule(&noise_schedule, request.steps as usize)
    };
    Backend::seed(&device, request.seed as u64);
    let sigma_max = schedule.sigmas[0];
    let latent: Tensor<Backend, 4> = Tensor::random(
        [1, 4, (request.height / 8) as usize, (request.width / 8) as usize],
        Distribution::Normal(0.0, sigma_max as f64), &device,
    );
    let progress = |step: usize, total: usize| -> bool {
        if let Some(callback) = step_callback {
            callback.on_step(step.min(i32::MAX as usize) as i32, total.min(i32::MAX as usize) as i32);
        }
        !cancel.load(Ordering::Relaxed)
    };
    let denoised = match request.sample_method {
        0 => EulerSampler::new(noise_schedule).sample_with_schedule(&unet, latent, &positive, &negative, &schedule, request.cfg, Some(&progress)),
        1 => EulerASampler::new(noise_schedule).sample_with_schedule(&unet, latent, &positive, &negative, &schedule, request.cfg, Some(&progress)),
        5 | 6 => DpmPp2mSampler::new(noise_schedule).sample_with_schedule(&unet, latent, &positive, &negative, &schedule, request.cfg, Some(&progress)),
        18 | 21 => DpmPpSdeSampler::new(noise_schedule).sample_with_schedule(&unet, latent, &positive, &negative, &schedule, request.cfg, Some(&progress)),
        _ => return Err(PonkoError::InvalidArgument { details: "Hearth supports Euler, Euler a, DPM++ 2M and DPM++ SDE samplers.".to_owned() }),
    };
    if cancel.load(Ordering::Relaxed) {
        return Err(native_error("Generation cancelled."));
    }
    drop(unet);
    drop(positive);
    drop(negative);
    let _ = Backend::sync(&device);
    Backend::memory_cleanup(&device);

    let vae_path = context.vae_path.as_ref().ok_or_else(|| PonkoError::InvalidArgument {
        details: "SDXL requires a separate FP16-safe VAE SafeTensors file.".to_owned(),
    })?;
    let vae_file = SafeTensorsFile::open(vae_path).map_err(native_error)?;
    let vae_tensors = vae_file.tensors().map_err(native_error)?;
    let vae = SdxlVaeDecoder::load(&vae_tensors, None, &device).map_err(native_error)?;
    let decoded = vae.forward_tiled(denoised, 64, 8);
    tensor_to_rgb(decoded, request.width, request.height, request.steps, step_callback, preview_callback)
}

fn tensor_to_rgb(
    image: Tensor<Backend, 4>,
    width: i32,
    height: i32,
    steps: i32,
    step_callback: Option<&Arc<dyn StepCallback>>,
    preview_callback: Option<&Arc<dyn PreviewCallback>>,
) -> Result<GeneratedImage, PonkoError> {
    let data: Vec<f32> = image.into_data().convert::<f32>().to_vec().map_err(native_error)?;
    let pixels = (width as usize).checked_mul(height as usize)
        .ok_or_else(|| native_error("Image dimensions overflowed"))?;
    if data.len() != pixels * 3 {
        return Err(native_error(format!("Hearth returned {} values for a {width}x{height} RGB image.", data.len())));
    }
    let mut rgb888 = vec![0u8; pixels * 3];
    let to_byte = |v: f32| ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
    // Burn tensor is NCHW. Convert each planar channel to row-major RGB888.
    for i in 0..pixels {
        rgb888[3 * i] = to_byte(data[i]);
        rgb888[3 * i + 1] = to_byte(data[pixels + i]);
        rgb888[3 * i + 2] = to_byte(data[2 * pixels + i]);
    }
    if let Some(callback) = step_callback {
        callback.on_step(steps, steps);
    }
    if let Some(callback) = preview_callback {
        callback.on_preview(steps, width, height, rgb888.clone());
    }
    Ok(GeneratedImage { width, height, rgb888 })
}

#[uniffi::export]
pub fn engine_api_version() -> i32 { FACADE_API_VERSION }

#[uniffi::export]
pub fn build_profile() -> BuildProfile {
    BuildProfile {
        api_version: FACADE_API_VERSION,
        abi: "arm64-v8a".to_owned(),
        cpu_arch_flags: "Rust stable · ARM64 · WGPU Vulkan".to_owned(),
        vulkan_build_default: true,
        native_engine: "Ponko Rust/UniFFI -> Hearth -> Burn/WGPU Vulkan".to_owned(),
    }
}

#[uniffi::export]
pub fn validate_model_slots(slots: ModelSlots) -> ValidationReport {
    if slots.ordered().iter().flatten().any(|path| !path.trim().is_empty()) {
        valid("At least one model slot is configured.")
    } else {
        invalid("At least one model file path is required.")
    }
}

#[uniffi::export]
pub fn validate_model_load_request(request: ModelLoadRequest) -> ValidationReport {
    if !["sd15", "sdxl"].contains(&request.family.to_ascii_lowercase().as_str()) {
        return invalid("Hearth supports only SD 1.5 and SDXL checkpoints.");
    }
    if request.model_path.trim().is_empty() {
        return invalid("A SafeTensors checkpoint path is required.");
    }
    if Path::new(&request.model_path).extension().and_then(|v| v.to_str()).is_none_or(|v| !v.eq_ignore_ascii_case("safetensors")) {
        return invalid("The diffusion checkpoint must be .safetensors.");
    }
    let slots = request.slots.as_ref();
    if slots.and_then(|s| s.clip_l.as_deref()).is_none_or(|v| v.trim().is_empty()) {
        return invalid("Tokenizer vocab.json path is required in the CLIP-L slot.");
    }
    if slots.and_then(|s| s.clip_g.as_deref()).is_none_or(|v| v.trim().is_empty()) {
        return invalid("Tokenizer merges.txt path is required in the CLIP-G slot.");
    }
    if request.family.eq_ignore_ascii_case("sdxl") && request.vae_path.as_deref()
        .or_else(|| slots.and_then(|s| s.vae.as_deref())).is_none_or(|v| v.trim().is_empty()) {
        return invalid("SDXL requires a separate FP16-safe VAE .safetensors file.");
    }
    valid("Hearth checkpoint and tokenizer paths are configured.")
}

#[uniffi::export]
pub fn validate_generation_request(request: GenerationRequest) -> ValidationReport {
    if request.width < 64 || request.height < 64 || request.width % 8 != 0 || request.height % 8 != 0 {
        return invalid("Width and height must be >=64 and divisible by 8.");
    }
    if !(1..=150).contains(&request.steps) { return invalid("Steps must be in 1..=150."); }
    if !request.cfg.is_finite() || request.cfg < 0.0 { return invalid("CFG must be finite and >=0."); }
    if request.lora_paths.len() != request.lora_scales.len() { return invalid("LoRA paths and scales must have equal lengths."); }
    if request.lora_scales.iter().any(|v| !v.is_finite()) { return invalid("Every LoRA scale must be finite."); }
    if ![0, 1, 5, 6, 18, 21].contains(&request.sample_method) {
        return invalid("Hearth supports Euler, Euler a, DPM++ 2M, and DPM++ SDE samplers.");
    }
    if ![0, 1].contains(&request.scheduler) { return invalid("Hearth supports Normal and Karras schedulers."); }
    if request.hires_enabled { return invalid("HiRes Fix is not implemented in the Hearth backend yet."); }
    if request.init_image_rgb888.is_some() { return invalid("Img2img is not implemented in the Ponko Hearth adapter yet."); }
    if request.adetailer_enabled { return invalid("ADetailer is not implemented in the Hearth backend yet."); }
    valid("Generation request is valid for Hearth.")
}

#[uniffi::export]
pub fn validate_quantization_type(_quant_type: String) -> ValidationReport {
    invalid("Hearth currently loads SafeTensors checkpoints and has no in-app quantization converter.")
}

#[uniffi::export]
pub fn validate_unet_quantization_request(_request: QuantizeUnetRequest) -> ValidationReport {
    invalid("UNet quantization is not implemented by the Hearth backend.")
}

#[uniffi::export]
pub fn validate_res_adapter_request(_request: ResAdapterRequest) -> ValidationReport {
    invalid("ResAdapter conversion is not implemented by the Hearth backend.")
}

#[uniffi::export]
pub fn validate_lora_fusion_request(_request: LoraFusionRequest) -> ValidationReport {
    invalid("LoRA fusion/export is not implemented by the Hearth backend.")
}

fn valid(message: &str) -> ValidationReport { ValidationReport { valid: true, message: message.to_owned() } }
fn invalid(message: &str) -> ValidationReport { ValidationReport { valid: false, message: message.to_owned() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_family_validation_is_limited_to_hearth_families() {
        let mut request = ModelLoadRequest {
            model_path: "model.safetensors".into(), vae_path: None, n_threads: 1,
            wtype: -1, flash_attn: false, vae_conv_direct: false, backend: Some("Vulkan".into()),
            slots: Some(ModelSlots { clip_l: Some("vocab.json".into()), clip_g: Some("merges.txt".into()), ..Default::default() }),
            family: "sd15".into(),
        };
        assert!(validate_model_load_request(request.clone()).valid);
        request.family = "flux".into();
        assert!(!validate_model_load_request(request).valid);
    }

    #[test]
    fn generation_validation_rejects_non_hearth_sampler_and_dual_backend_features() {
        let base = GenerationRequest {
            prompt: "test".into(), negative_prompt: String::new(), lora_paths: vec![],
            lora_scales: vec![], width: 512, height: 512, steps: 6, cfg: 7.0, seed: 42,
            sample_method: 1, scheduler: 0, eta: -1.0, hires_enabled: false,
            hires_upscaler: 1, hires_model_path: None, hires_scale: 2.0, hires_target_width: 0,
            hires_target_height: 0, hires_steps: 0, hires_denoising_strength: 0.7,
            hires_tile_size: 128, clip_skip: 1, init_image_rgb888: None, init_image_width: 0,
            init_image_height: 0, strength: 1.0, adetailer_enabled: false,
            adetailer_detector_path: None, adetailer_prompt: String::new(),
            adetailer_negative_prompt: String::new(), adetailer_extra_args: String::new(),
        };
        assert!(validate_generation_request(base.clone()).valid);
        let mut unsupported = base;
        unsupported.sample_method = 11;
        assert!(!validate_generation_request(unsupported).valid);
    }
}
