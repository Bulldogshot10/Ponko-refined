//! Rust/UniFFI boundary for Ponko's Android drawing engine.
//!
//! Kotlin calls this module through generated UniFFI bindings. The C++ inference
//! implementation remains stable-diffusion.cpp + GGML, reached through the
//! explicit Ponko C ABI bridge. No Kotlin-facing JNI entry points are required.

use libloading::Library;
use std::{
    collections::HashMap,
    ffi::{CStr, CString, NulError},
    os::raw::{c_char, c_void},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    slice,
    sync::{Arc, Mutex, RwLock},
};

uniffi::setup_scaffolding!();

pub const FACADE_API_VERSION: u32 = 2;

type StepFn = unsafe extern "C" fn(*mut c_void, i32, i32);
type PreviewFn = unsafe extern "C" fn(*mut c_void, i32, i32, i32, *const u8, usize);

#[repr(C)]
struct CGenerationParams {
    prompt: *const c_char,
    negative_prompt: *const c_char,
    lora_paths: *const *const c_char,
    lora_scales: *const f32,
    lora_count: usize,
    width: i32,
    height: i32,
    steps: i32,
    cfg: f32,
    seed: i64,
    sample_method: i32,
    scheduler: i32,
    eta: f32,
    hires_enabled: i32,
    hires_upscaler: i32,
    hires_model_path: *const c_char,
    hires_scale: f32,
    hires_target_width: i32,
    hires_target_height: i32,
    hires_steps: i32,
    hires_denoising_strength: f32,
    hires_tile_size: i32,
    clip_skip: i32,
    init_image_rgb888: *const u8,
    init_image_length: usize,
    init_image_width: i32,
    init_image_height: i32,
    strength: f32,
}

type LastErrorFn = unsafe extern "C" fn() -> *const c_char;
type CreateFn = unsafe extern "C" fn(
    *const c_char, *const c_char, i32, i32, i32, i32, *const c_char, *const *const c_char, usize,
) -> *mut c_void;
type GenerateFn = unsafe extern "C" fn(
    *mut c_void, *const CGenerationParams, Option<StepFn>, Option<PreviewFn>, *mut c_void,
    *mut *mut u8, *mut usize, *mut i32, *mut i32,
) -> i32;
type CancelFn = unsafe extern "C" fn(*mut c_void) -> i32;
type FreeHandleFn = unsafe extern "C" fn(*mut c_void);
type FreeBufferFn = unsafe extern "C" fn(*mut c_void);
type TextFn = unsafe extern "C" fn(*mut c_char, usize) -> usize;
type ListDevicesFn = TextFn;
type QuantizeFn = unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, i32, i32) -> i32;
type ResAdapterFn = unsafe extern "C" fn(
    *const c_char, *const c_char, *const c_char, *const c_char, *const c_char, i32,
) -> i32;

struct NativeApi {
    // Keep the shared library loaded for as long as any function pointer/context is alive.
    _library: Library,
    last_error: LastErrorFn,
    create: CreateFn,
    generate: GenerateFn,
    cancel: CancelFn,
    free_handle: FreeHandleFn,
    free_buffer: FreeBufferFn,
    info: TextFn,
    vulkan_diagnostics: TextFn,
    last_params: TextFn,
    list_devices: ListDevicesFn,
    quantize_unet: QuantizeFn,
    apply_resadapter: ResAdapterFn,
}

impl NativeApi {
    fn load() -> Result<Self, PonkoError> {
        let library_name = std::env::var("PONKO_NATIVE_BRIDGE_PATH")
            .unwrap_or_else(|_| "libponko_sd.so".to_owned());
        // SAFETY: The library is the in-app native bridge, packaged by Gradle in this APK.
        let library = unsafe { Library::new(&library_name) }.map_err(|error| PonkoError::Native {
            message: format!("could not load {library_name}: {error}"),
        })?;
        unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, PonkoError> {
            // SAFETY: Each symbol name and function type matches ponko_sd_c_api.h.
            unsafe { library.get::<T>(name) }
                .map(|symbol| *symbol)
                .map_err(|error| PonkoError::Native {
                    message: format!("native bridge symbol {} is unavailable: {error}",
                        String::from_utf8_lossy(&name[..name.len().saturating_sub(1)])),
                })
        }
        // SAFETY: All function signatures correspond to the exported C ABI header.
        let api = unsafe {
            Self {
                last_error: symbol(&library, b"ponko_sd_last_error\0")?,
                create: symbol(&library, b"ponko_sd_create\0")?,
                generate: symbol(&library, b"ponko_sd_generate\0")?,
                cancel: symbol(&library, b"ponko_sd_cancel\0")?,
                free_handle: symbol(&library, b"ponko_sd_free\0")?,
                free_buffer: symbol(&library, b"ponko_sd_free_buffer\0")?,
                info: symbol(&library, b"ponko_sd_info\0")?,
                vulkan_diagnostics: symbol(&library, b"ponko_sd_vulkan_diagnostics\0")?,
                last_params: symbol(&library, b"ponko_sd_last_params\0")?,
                list_devices: symbol(&library, b"ponko_sd_list_devices\0")?,
                quantize_unet: symbol(&library, b"ponko_sd_quantize_unet\0")?,
                apply_resadapter: symbol(&library, b"ponko_sd_apply_resadapter\0")?,
                _library: library,
            }
        };
        Ok(api)
    }

    fn last_error_message(&self) -> String {
        // SAFETY: The bridge returns a thread-local, NUL-terminated string valid on this thread.
        unsafe {
            let pointer = (self.last_error)();
            if pointer.is_null() {
                "native bridge returned no error detail".to_owned()
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        }
    }

    fn text(&self, function: TextFn) -> Result<String, PonkoError> {
        // First call asks for the number of bytes excluding the NUL terminator.
        let length = unsafe { function(ptr::null_mut(), 0) };
        let capacity = length.checked_add(1).ok_or_else(|| PonkoError::Native {
            message: "native text result length overflowed".to_owned(),
        })?;
        let mut bytes = vec![0u8; capacity];
        let written = unsafe { function(bytes.as_mut_ptr().cast::<c_char>(), bytes.len()) };
        if written > length {
            return Err(PonkoError::Native {
                message: "native text result changed size while being copied".to_owned(),
            });
        }
        let end = bytes.iter().position(|byte| *byte == 0).unwrap_or(bytes.len());
        Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
    }

    fn cstring(value: &str, field: &str) -> Result<CString, PonkoError> {
        CString::new(value).map_err(|error: NulError| PonkoError::InvalidArgument {
            message: format!("{field} contains an embedded NUL byte: {error}"),
        })
    }

    fn check_native(&self, success: i32, operation: &str) -> Result<(), PonkoError> {
        if success != 0 {
            Ok(())
        } else {
            Err(PonkoError::Native {
                message: format!("{operation}: {}", self.last_error_message()),
            })
        }
    }
}

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum PonkoError {
    #[error("{message}")]
    InvalidArgument { message: String },
    #[error("{message}")]
    Native { message: String },
    #[error("{message}")]
    InvalidHandle { message: String },
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct BuildProfile {
    pub api_version: i32,
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

impl ModelSlots {
    fn ordered(&self) -> [Option<&str>; 13] {
        [
            self.model.as_deref(),
            self.diffusion.as_deref(),
            self.high_noise.as_deref(),
            self.uncond.as_deref(),
            self.clip_l.as_deref(),
            self.clip_g.as_deref(),
            self.clip_vision.as_deref(),
            self.t5xxl.as_deref(),
            self.llm.as_deref(),
            self.llm_vision.as_deref(),
            self.vae.as_deref(),
            self.audio_vae.as_deref(),
            self.taesd.as_deref(),
        ]
    }
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelLoadRequest {
    pub model_path: String,
    pub vae_path: Option<String>,
    pub n_threads: i32,
    /// -1 preserves the checkpoint's original type.
    pub wtype: i32,
    pub flash_attn: bool,
    pub vae_conv_direct: bool,
    pub backend: Option<String>,
    pub slots: Option<ModelSlots>,
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
    /// Negative values select the sampler's automatic ETA, matching the old JNI contract.
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
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct GeneratedImage {
    pub width: i32,
    pub height: i32,
    /// Output is row-major RGB888, three bytes per pixel.
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

#[uniffi::export(callback_interface)]
pub trait StepCallback: Send + Sync {
    fn on_step(&self, step: i32, steps: i32);
}

#[uniffi::export(callback_interface)]
pub trait PreviewCallback: Send + Sync {
    fn on_preview(&self, step: i32, width: i32, height: i32, rgb888: Vec<u8>);
}

struct CallbackState {
    step: Option<Arc<dyn StepCallback>>,
    preview: Option<Arc<dyn PreviewCallback>>,
}

unsafe extern "C" fn step_trampoline(user_data: *mut c_void, step: i32, steps: i32) {
    if user_data.is_null() { return; }
    let state = unsafe { &*(user_data.cast::<CallbackState>()) };
    if let Some(callback) = &state.step {
        let _ = catch_unwind(AssertUnwindSafe(|| callback.on_step(step.max(0), steps.max(0))));
    }
}

unsafe extern "C" fn preview_trampoline(
    user_data: *mut c_void,
    step: i32,
    width: i32,
    height: i32,
    rgb: *const u8,
    length: usize,
) {
    if user_data.is_null() || rgb.is_null() || width <= 0 || height <= 0 { return; }
    let state = unsafe { &*(user_data.cast::<CallbackState>()) };
    if let Some(callback) = &state.preview {
        // The C++ callback owns this preview buffer only for the duration of this call.
        let bytes = unsafe { slice::from_raw_parts(rgb, length) }.to_vec();
        let _ = catch_unwind(AssertUnwindSafe(|| {
            callback.on_preview(step.max(0), width, height, bytes)
        }));
    }
}

struct NativeHandle {
    // Read locks allow generation and cancellation together. Free obtains a write lock,
    // so a native context can't be destroyed while any call is using its pointer.
    state: RwLock<usize>,
}

#[derive(uniffi::Object)]
pub struct NativeEngine {
    api: Arc<NativeApi>,
    handles: Mutex<HashMap<i64, Arc<NativeHandle>>>,
}

#[uniffi::export]
impl NativeEngine {
    #[uniffi::constructor]
    pub fn new() -> Result<Arc<Self>, PonkoError> {
        Ok(Arc::new(Self {
            api: Arc::new(NativeApi::load()?),
            handles: Mutex::new(HashMap::new()),
        }))
    }

    pub fn create_context(&self, request: ModelLoadRequest) -> Result<i64, PonkoError> {
        let report = validate_model_load_request(request.clone());
        if !report.valid {
            return Err(PonkoError::InvalidArgument { message: report.message });
        }
        let model = NativeApi::cstring(&request.model_path, "model_path")?;
        let vae = request.vae_path.as_deref()
            .map(|value| NativeApi::cstring(value, "vae_path"))
            .transpose()?;
        let backend = request.backend.as_deref()
            .map(|value| NativeApi::cstring(value, "backend"))
            .transpose()?;

        let slot_values = request.slots.as_ref().map(ModelSlots::ordered);
        let slot_storage: Vec<Option<CString>> = if let Some(values) = slot_values.as_ref() {
            values.iter().map(|value| {
                value.map(|text| NativeApi::cstring(text, "model slot")).transpose()
            }).collect::<Result<Vec<_>, PonkoError>>()?
        } else {
            vec![None; 13]
        };
        let slot_pointers: Vec<*const c_char> = slot_storage.iter()
            .map(|item| item.as_ref().map_or(ptr::null(), |value| value.as_ptr()))
            .collect();
        let slots_ptr = if request.slots.is_some() { slot_pointers.as_ptr() } else { ptr::null() };
        let slot_count = if request.slots.is_some() { slot_pointers.len() } else { 0 };

        let pointer = unsafe {
            (self.api.create)(
                model.as_ptr(),
                vae.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
                request.n_threads,
                request.wtype,
                request.flash_attn as i32,
                request.vae_conv_direct as i32,
                backend.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
                slots_ptr,
                slot_count,
            )
        };
        if pointer.is_null() {
            return Err(PonkoError::Native {
                message: format!("model context creation failed: {}", self.api.last_error_message()),
            });
        }
        let id = pointer as usize as i64;
        let handle = Arc::new(NativeHandle { state: RwLock::new(pointer as usize) });
        let mut handles = self.handles.lock().map_err(|_| PonkoError::Native {
            message: "native handle registry lock poisoned".to_owned(),
        })?;
        if handles.insert(id, handle).is_some() {
            unsafe { (self.api.free_handle)(pointer) };
            return Err(PonkoError::Native { message: "native handle identifier collision".to_owned() });
        }
        Ok(id)
    }

    pub fn cancel(&self, handle_id: i64) -> Result<(), PonkoError> {
        let handle = self.lookup_handle(handle_id)?;
        let pointer = handle.state.read().map_err(|_| PonkoError::Native {
            message: "native handle state lock poisoned".to_owned(),
        })?;
        if *pointer == 0 {
            return Err(PonkoError::InvalidHandle { message: "model handle has already been freed".to_owned() });
        }
        let success = unsafe { (self.api.cancel)(*pointer as *mut c_void) };
        self.api.check_native(success, "cancel generation")
    }

    pub fn free_context(&self, handle_id: i64) -> Result<(), PonkoError> {
        let handle = {
            let mut handles = self.handles.lock().map_err(|_| PonkoError::Native {
                message: "native handle registry lock poisoned".to_owned(),
            })?;
            handles.remove(&handle_id).ok_or_else(|| PonkoError::InvalidHandle {
                message: "unknown or already freed model handle".to_owned(),
            })?
        };
        let mut pointer = handle.state.write().map_err(|_| PonkoError::Native {
            message: "native handle state lock poisoned".to_owned(),
        })?;
        if *pointer != 0 {
            unsafe { (self.api.free_handle)(*pointer as *mut c_void) };
            *pointer = 0;
        }
        Ok(())
    }

    pub fn info(&self) -> Result<String, PonkoError> {
        self.api.text(self.api.info)
    }

    pub fn vulkan_diagnostics(&self) -> Result<String, PonkoError> {
        self.api.text(self.api.vulkan_diagnostics)
    }

    pub fn last_params(&self) -> Result<String, PonkoError> {
        self.api.text(self.api.last_params)
    }

    pub fn list_devices(&self) -> Result<String, PonkoError> {
        self.api.text(self.api.list_devices)
    }

    pub fn quantize_unet(&self, request: QuantizeUnetRequest) -> Result<(), PonkoError> {
        let report = validate_unet_quantization_request(request.clone());
        if !report.valid { return Err(PonkoError::InvalidArgument { message: report.message }); }
        let input = NativeApi::cstring(&request.input_path, "input_path")?;
        let output = NativeApi::cstring(&request.output_path, "output_path")?;
        let kind = NativeApi::cstring(&request.quant_type, "quant_type")?;
        let result = unsafe {
            (self.api.quantize_unet)(
                input.as_ptr(), output.as_ptr(), kind.as_ptr(), request.diffusion_slot as i32,
                request.n_threads,
            )
        };
        self.api.check_native(result, "UNet quantization")
    }

    pub fn apply_resadapter(&self, request: ResAdapterRequest) -> Result<(), PonkoError> {
        let report = validate_res_adapter_request(request.clone());
        if !report.valid { return Err(PonkoError::InvalidArgument { message: report.message }); }
        let input = NativeApi::cstring(&request.input_model_path, "input_model_path")?;
        let vae = request.vae_path.as_deref().map(|value| NativeApi::cstring(value, "vae_path")).transpose()?;
        let normalization = NativeApi::cstring(&request.normalization_path, "normalization_path")?;
        let output = NativeApi::cstring(&request.output_path, "output_path")?;
        let kind = NativeApi::cstring(&request.quant_type, "quant_type")?;
        let result = unsafe {
            (self.api.apply_resadapter)(
                input.as_ptr(),
                vae.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
                normalization.as_ptr(), output.as_ptr(), kind.as_ptr(),
                request.n_threads,
            )
        };
        self.api.check_native(result, "ResAdapter")
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
            return Err(PonkoError::InvalidArgument { message: report.message });
        }
        let handle = self.lookup_handle(handle_id)?;
        let pointer_guard = handle.state.read().map_err(|_| PonkoError::Native {
            message: "native handle state lock poisoned".to_owned(),
        })?;
        if *pointer_guard == 0 {
            return Err(PonkoError::InvalidHandle { message: "model handle has already been freed".to_owned() });
        }

        let prompt = NativeApi::cstring(&request.prompt, "prompt")?;
        let negative = NativeApi::cstring(&request.negative_prompt, "negative_prompt")?;
        let hires_path = request.hires_model_path.as_deref()
            .map(|value| NativeApi::cstring(value, "hires_model_path"))
            .transpose()?;
        let lora_paths = request.lora_paths.iter()
            .map(|path| NativeApi::cstring(path, "lora_path"))
            .collect::<Result<Vec<_>, _>>()?;
        let lora_pointers: Vec<*const c_char> = lora_paths.iter().map(|path| path.as_ptr()).collect();
        let image = request.init_image_rgb888.as_deref();
        let params = CGenerationParams {
            prompt: prompt.as_ptr(),
            negative_prompt: negative.as_ptr(),
            lora_paths: if lora_pointers.is_empty() { ptr::null() } else { lora_pointers.as_ptr() },
            lora_scales: if request.lora_scales.is_empty() { ptr::null() } else { request.lora_scales.as_ptr() },
            lora_count: request.lora_paths.len().min(request.lora_scales.len()),
            width: request.width,
            height: request.height,
            steps: request.steps,
            cfg: request.cfg,
            seed: request.seed,
            sample_method: request.sample_method,
            scheduler: request.scheduler,
            eta: request.eta,
            hires_enabled: i32::from(request.hires_enabled),
            hires_upscaler: request.hires_upscaler,
            hires_model_path: hires_path.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            hires_scale: request.hires_scale,
            hires_target_width: request.hires_target_width,
            hires_target_height: request.hires_target_height,
            hires_steps: request.hires_steps,
            hires_denoising_strength: request.hires_denoising_strength,
            hires_tile_size: request.hires_tile_size,
            clip_skip: request.clip_skip,
            init_image_rgb888: image.map_or(ptr::null(), |bytes| bytes.as_ptr()),
            init_image_length: image.map_or(0, |bytes| bytes.len()),
            init_image_width: request.init_image_width,
            init_image_height: request.init_image_height,
            strength: request.strength,
        };
        let mut callbacks = CallbackState { step: step_callback, preview: preview_callback };
        let callbacks_ptr = (&mut callbacks as *mut CallbackState).cast::<c_void>();
        let mut output: *mut u8 = ptr::null_mut();
        let mut output_length = 0usize;
        let mut width = 0i32;
        let mut height = 0i32;
        let success = unsafe {
            (self.api.generate)(
                (*pointer_guard) as *mut c_void,
                &params,
                if callbacks.step.is_some() { Some(step_trampoline) } else { None },
                if callbacks.preview.is_some() { Some(preview_trampoline) } else { None },
                callbacks_ptr,
                &mut output,
                &mut output_length,
                &mut width,
                &mut height,
            )
        };
        self.api.check_native(success, "image generation")?;
        if output.is_null() || width <= 0 || height <= 0 {
            if !output.is_null() { unsafe { (self.api.free_buffer)(output.cast::<c_void>()) }; }
            return Err(PonkoError::Native { message: "generation returned an empty image".to_owned() });
        }
        let expected = (width as usize).checked_mul(height as usize).and_then(|pixels| pixels.checked_mul(3));
        if expected != Some(output_length) {
            unsafe { (self.api.free_buffer)(output.cast::<c_void>()) };
            return Err(PonkoError::Native {
                message: "native RGB888 output length does not match the reported dimensions".to_owned(),
            });
        }
        let rgb888 = unsafe { slice::from_raw_parts(output, output_length) }.to_vec();
        unsafe { (self.api.free_buffer)(output.cast::<c_void>()) };
        Ok(GeneratedImage { width, height, rgb888 })
    }


    fn lookup_handle(&self, handle_id: i64) -> Result<Arc<NativeHandle>, PonkoError> {
        self.handles.lock().map_err(|_| PonkoError::Native {
            message: "native handle registry lock poisoned".to_owned(),
        })?.get(&handle_id).cloned().ok_or_else(|| PonkoError::InvalidHandle {
            message: format!("unknown or already freed model handle {handle_id}"),
        })
    }
}

impl Drop for NativeEngine {
    fn drop(&mut self) {
        let handles = match self.handles.get_mut() {
            Ok(handles) => std::mem::take(handles),
            Err(_) => return,
        };
        for (_, handle) in handles {
            if let Ok(mut pointer) = handle.state.write() {
                if *pointer != 0 {
                    unsafe { (self.api.free_handle)(*pointer as *mut c_void) };
                    *pointer = 0;
                }
            }
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

#[uniffi::export]
pub fn engine_api_version() -> i32 {
    FACADE_API_VERSION as i32
}

#[uniffi::export]
pub fn build_profile() -> BuildProfile {
    BuildProfile {
        api_version: FACADE_API_VERSION as i32,
        abi: "arm64-v8a".to_owned(),
        cpu_arch_flags: option_env!("PONKO_CPU_ARCH_FLAGS")
            .unwrap_or("-march=armv8.6-a+dotprod+i8mm+fp16 -mtune=cortex-x4")
            .to_owned(),
        vulkan_build_default: !matches!(
            option_env!("PONKO_VULKAN_ENABLED"),
            Some("OFF") | Some("off") | Some("FALSE") | Some("false") | Some("0")
        ),
        native_engine: "Rust/UniFFI -> Ponko C ABI -> stable-diffusion.cpp + GGML".to_owned(),
    }
}

#[uniffi::export]
pub fn validate_model_slots(slots: ModelSlots) -> ValidationReport {
    let has_path = slots.ordered().iter().flatten().any(|path| !path.trim().is_empty());
    if has_path {
        ValidationReport { valid: true, message: "At least one model slot is configured.".to_owned() }
    } else {
        invalid("At least one model file path is required.")
    }
}

#[uniffi::export]
pub fn validate_model_load_request(request: ModelLoadRequest) -> ValidationReport {
    let has_base_path = !request.model_path.trim().is_empty();
    let has_slot_path = request.slots.as_ref().map(|slots| {
        slots.model.as_deref().is_some_and(|path| !path.trim().is_empty())
            || slots.diffusion.as_deref().is_some_and(|path| !path.trim().is_empty())
    }).unwrap_or(false);
    if has_base_path || has_slot_path {
        ValidationReport { valid: true, message: "A base model path is configured.".to_owned() }
    } else {
        invalid("A base model or diffusion-model slot is required.")
    }
}

#[uniffi::export]
pub fn validate_generation_request(request: GenerationRequest) -> ValidationReport {
    if request.width <= 0 || request.height <= 0 {
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
    if !(0..10).contains(&request.hires_upscaler) {
        return invalid("HiRes upscaler code is not present in Ponko's current native enum.");
    }
    if request.lora_paths.len() != request.lora_scales.len() {
        return invalid("LoRA paths and scales must have the same number of entries.");
    }
    if request.lora_scales.iter().any(|scale| !scale.is_finite()) {
        return invalid("Every LoRA scale must be finite.");
    }
    if !request.hires_scale.is_finite() || !request.hires_denoising_strength.is_finite()
        || !request.strength.is_finite() {
        return invalid("HiRes and img2img strength parameters must be finite.");
    }
    if let Some(rgb) = request.init_image_rgb888 {
        if request.init_image_width <= 0 || request.init_image_height <= 0 {
            return invalid("An img2img input requires non-zero image dimensions.");
        }
        let expected = (request.init_image_width as usize).checked_mul(request.init_image_height as usize)
            .and_then(|pixels| pixels.checked_mul(3));
        if expected != Some(rgb.len()) {
            return invalid("The img2img buffer must contain exactly width * height * 3 RGB888 bytes.");
        }
    }
    ValidationReport { valid: true, message: "Request fields are internally consistent.".to_owned() }
}

#[uniffi::export]
pub fn validate_quantization_type(quant_type: String) -> ValidationReport {
    match quant_type.as_str() {
        "q4_0" | "q5_0" | "q8_0" => ValidationReport {
            valid: true, message: "Quantization type is supported by the current native converter.".to_owned(),
        },
        _ => invalid("Supported UNet quantization types are q4_0, q5_0, and q8_0."),
    }
}

#[uniffi::export]
pub fn validate_unet_quantization_request(request: QuantizeUnetRequest) -> ValidationReport {
    if request.input_path.trim().is_empty() || request.output_path.trim().is_empty() {
        return invalid("Quantization input and output paths must not be empty.");
    }
    if request.input_path == request.output_path {
        return invalid("Quantization input and output paths must be different.");
    }
    validate_quantization_type(request.quant_type)
}

#[uniffi::export]
pub fn validate_res_adapter_request(request: ResAdapterRequest) -> ValidationReport {
    if request.input_model_path.trim().is_empty() || request.normalization_path.trim().is_empty()
        || request.output_path.trim().is_empty() {
        return invalid("ResAdapter base, normalization, and output paths must not be empty.");
    }
    if request.output_path == request.input_model_path || request.output_path == request.normalization_path {
        return invalid("ResAdapter output path must not overwrite either input file.");
    }
    validate_quantization_type(request.quant_type)
}

fn invalid(message: &str) -> ValidationReport {
    ValidationReport { valid: false, message: message.to_owned() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> GenerationRequest {
        GenerationRequest {
            prompt: "test".into(), negative_prompt: String::new(),
            lora_paths: vec!["one.safetensors".into()], lora_scales: vec![1.0],
            width: 512, height: 512, steps: 6, cfg: 0.0, seed: 42,
            sample_method: 11, scheduler: 16, eta: -1.0,
            hires_enabled: false, hires_upscaler: 1, hires_model_path: None,
            hires_scale: 2.0, hires_target_width: 0, hires_target_height: 0,
            hires_steps: 0, hires_denoising_strength: 0.7, hires_tile_size: 128,
            clip_skip: 1, init_image_rgb888: None, init_image_width: 0,
            init_image_height: 0, strength: 1.0,
        }
    }

    fn empty_slots() -> ModelSlots {
        ModelSlots {
            model: None, diffusion: None, high_noise: None, uncond: None, clip_l: None,
            clip_g: None, clip_vision: None, t5xxl: None, llm: None, llm_vision: None,
            vae: None, audio_vae: None, taesd: None,
        }
    }

    #[test]
    fn preserves_zero_cfg_tcd_requests() { assert!(validate_generation_request(request()).valid); }

    #[test]
    fn rejects_mismatched_lora_arrays() {
        let mut input = request(); input.lora_scales.clear();
        assert!(!validate_generation_request(input).valid);
    }

    #[test]
    fn validates_rgb888_img2img_buffer() {
        let mut input = request();
        input.init_image_width = 2; input.init_image_height = 2;
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

    #[test]
    fn model_load_requires_a_model_or_diffusion_slot() {
        let request = ModelLoadRequest {
            model_path: String::new(), vae_path: None, n_threads: 4, wtype: -1,
            flash_attn: true, vae_conv_direct: true, backend: None, slots: Some(empty_slots()),
        };
        assert!(!validate_model_load_request(request.clone()).valid);
        let mut valid_request = request;
        valid_request.slots.as_mut().unwrap().diffusion = Some("unet.gguf".into());
        assert!(validate_model_load_request(valid_request).valid);
    }

    #[test]
    fn quantization_request_validates_paths_and_type() {
        let request = QuantizeUnetRequest {
            input_path: "input.gguf".into(), output_path: "output.gguf".into(),
            quant_type: "q5_0".into(), diffusion_slot: true, n_threads: 4,
        };
        assert!(validate_unet_quantization_request(request.clone()).valid);
        let mut bad = request; bad.quant_type = "q6_k".into();
        assert!(!validate_unet_quantization_request(bad).valid);
    }

    #[test]
    fn res_adapter_requires_normalization_and_output_paths() {
        let request = ResAdapterRequest {
            input_model_path: "base.gguf".into(), vae_path: None,
            normalization_path: "norm.safetensors".into(), output_path: "merged.gguf".into(),
            quant_type: "q4_0".into(), n_threads: 4,
        };
        assert!(validate_res_adapter_request(request.clone()).valid);
        let mut bad = request; bad.normalization_path.clear();
        assert!(!validate_res_adapter_request(bad).valid);
    }

    #[test]
    fn supports_only_current_converter_types() {
        assert!(validate_quantization_type("q4_0".into()).valid);
        assert!(validate_quantization_type("q5_0".into()).valid);
        assert!(validate_quantization_type("q8_0".into()).valid);
        assert!(!validate_quantization_type("q6_k".into()).valid);
    }

    #[test]
    fn backend_profile_preserves_exynos_2400e_flags() {
        assert!(build_profile().cpu_arch_flags.contains("-march=armv8.6-a+dotprod+i8mm+fp16"));
    }
}
