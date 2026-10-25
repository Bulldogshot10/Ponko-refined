# Ponko

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Release](https://img.shields.io/github/v/release/visitor257/Ponko?label=release&color=success) ![Platform](https://img.shields.io/badge/platform-Android-3DDC84?logo=android&logoColor=white) ![minSdk](https://img.shields.io/badge/minSdk-28-orange.svg) ![ABI](https://img.shields.io/badge/ABI-arm64--v8a-lightgrey.svg) ![Offline](https://img.shields.io/badge/offline-no%20telemetry-success.svg)

> The name comes from the Japanese 「ポンコツ」(ponkotsu) - "scrap heap". Not the sharpest brain, but you can swap in the best model whenever you like.

English · [中文](README.md)

An **offline-capable** Android app for local AI: chat with LLMs (**image, file and document input included**, and it can serve your phone as an API server), generate images, and tag them.

All inference runs on-device: apart from the optional in-app LoRA download, no feature needs the network.

## At a glance

| Module | In one line |
|---|---|
| **Chat** | `.litertlm` / `.gguf` dual backend; image, file and document input; separated reasoning, streaming Markdown |
| **Drawing** | Rust Hearth + Burn/WGPU Vulkan; SD 1.5 / SDXL SafeTensors; LoRA |
| **Tagging** | Fully offline ONNX (WD14 family); tags can be sent to T2I / I2I |
| **API server** | Serves loaded models over OpenAI-compatible endpoints; chat 8080 / drawing 8081, fully independent |
| **Local-first** | No telemetry, no account; no network needed except the manual LoRA download |

## Contents

- [Chat](#chat)
- [Drawing](#drawing)
- [API server (use the phone as a server)](#api-server-use-the-phone-as-a-server)
- [Tagging (Tagger)](#tagging-tagger)
- [Permissions](#permissions)
- [Languages](#languages)
- [Building](#building)
- [Where to get models](#where-to-get-models)
- [Known limitations](#known-limitations)
- [License](#license)
- [Third-party components](#third-party-components)
- [Credits](#credits)

## Chat

Two model formats are supported:

| Format | Runtime | Highlights |
|---|---|---|
| `.litertlm` | Google LiteRT-LM | CPU / GPU / NPU backends, thinking channel, multimodal extension |
| `.gguf` | llama.cpp | Multi-threaded CPU, in-session KV prefix reuse (no re-prefill on long chats) |

- **Model management**: pick a model file from local storage; it is copied into the app's private directory and can be reused anytime. The Models/API page has a run-mode selector (CPU / GPU) on top and three sub-pages below (**Chat model / Draw model / Tagger**, swipe to switch); both the **Chat model** and **Draw model** pages split into **Model / API** sub-tabs, where each kind of model can be viewed, selected or deleted.
  - The CPU/GPU selector applies to chat and tagging only: **Hearth drawing always uses Burn/WGPU Vulkan** and has no C++ or CPU fallback drawing engine; tagging can use NNAPI and GGUF chat uses CPU. Hearth's Vulkan path still needs a real-device test on Exynos 2400e / Xclipse 940
- **Multiple conversations**: create / switch / delete chats. History is persisted on-device (`sessions.json`).
- **Separate thinking**: reasoning and answer are shown apart and collapsible; the thinking toggle works for both formats.
- **Markdown rendering**: headings, bold/italic, strikethrough, ordered/unordered lists, inline code and code blocks, quotes, dividers, tables, tappable links.
- **Streaming UX**: token-by-token output; auto-follow scrolling that pauses the moment you scroll up, with a floating "back to bottom" button; interrupt anytime, and keep typing while generating.
- **Regenerate**: one tap to rerun, and it **carries the images and files of that turn along** (restored to the attachment strip above the input box, so you can see them); a random seed keeps results from repeating.
- **Image input (multimodal)**: the "+" button left of the input box opens an upward drawer with **Take photo / Gallery**, up to 4 images per message. Images appear inline in the bubble: **tap to view fullscreen (pinch-zoom / pan / double-tap)** and **long-press** for a menu: Save to gallery / Quote (put it into the input bar) / Send to Image-to-Image / Send to Tagger
  - `.litertlm`: needs a **multimodal model** (e.g. Gemma 3n). The app probes the model and tells you right away if it has no vision input
  - `.gguf`: needs a **vision model plus its matching `mmproj` file**. Import and select the mmproj on the Models/API page, Chat model, and **pick the mmproj before loading the model**
- **File and document input**: the third item in the same "+" drawer, **File**, up to 2 per message
  - **Plain text**: `.txt` / `.md` / `.json` / `.csv` / `.log` / `.xml` / source code and similar (UTF-8 / GBK auto-detected)
  - **Documents**: `.docx` / `.pptx` / `.xlsx` / `.odt` / `.ods` / `.odp` / `.epub` / `.rtf` / `.html` / **`.pdf`** - the text is extracted (table cells as tabs, paragraphs as lines, and the page / slide / sheet count is shown on the attachment chip) and trimmed to the context budget
  - `.doc` / `.xls` / `.ppt` (legacy 97-2003 binary formats) are rejected with a clear hint to save them as `.docx` / `.xlsx` / `.pptx` or PDF
  - The text is sent **with that one message only**, never added to the history; the chip shows the file name and character count (truncation is marked)
- **Chat parameters**: on the Models/API page, Chat model, you can tune context size, max output, temperature, Top-K, Top-P, repeat penalty, thinking budget and random seed; **saved per model**, as you type. Context size and max output take effect **after reloading the model**
- **Text-only chat models can borrow the tagger to "see" an image**: when the chat model cannot see images itself (no multimodal, no mmproj) but a tagger model is loaded, the model calls the tagger on its own (it emits a `<tag>` command, the app tags the image, the tags come back as a tool result) and then answers based on them; threshold and tag count are chosen by the model, BGR/RGB follows the Tagger setting on the drawing page. Models that can see images never take this path
- **API server (use the phone as a server)**: on the Models/API page → Chat model → the top **API** sub-tab, one tap turns the **already loaded** model into an **OpenAI-compatible** endpoint (`POST /v1/chat/completions` stream or not, `GET /v1/models`, `GET /health`); phones, computers and any OpenAI client on the same network can talk to it. Optional API key, custom port, ongoing notification, survives screen-off (see the section below). **The drawing model can be exposed too** - it has a complete set of its own settings (port / key / switch) and does not affect the chat side

## Drawing

The drawing engine has been fully replaced with [Hearth](https://github.com/jeff-hiner/hearth), a Rust inference engine using Burn/WGPU Vulkan. Ponko's drawing path no longer contains stable-diffusion.cpp, GGML, or a C ABI C++ bridge.

- **Models:** SD 1.5 and SDXL `.safetensors` checkpoints. SD 1.5 can use its embedded VAE or an imported VAE; SDXL requires a separate FP16-safe VAE.
- **Tokenizer:** each model set needs `vocab.json` and `merges.txt` from the matching CLIP tokenizer.
- **LoRA:** import `.safetensors` adapters, select up to four and tune their strengths independently. Hearth applies adapters to the supported CLIP and UNet layers.
- **Samplers:** Euler, Euler a, DPM++ 2M and DPM++ SDE. Schedulers: Normal and Karras.
- **GPU:** drawing uses Burn/WGPU Vulkan, with no sd.cpp CPU fallback. Initialization and performance depend on the Android Vulkan driver.
- **Not ported yet:** drawing GGUF/quantization, img2img, HiRes Fix, ADetailer, UNet quantization, ResAdapter conversion and LoRA fusion are not implemented in the current Hearth adapter, so their drawing UI entry points are hidden.

To import a model, open Models/API → Drawing Models → Add Model Set, choose SD 1.5 or SDXL, then import the checkpoint, `vocab.json` and `merges.txt`. Import the required VAE as well for SDXL. Chat `.gguf` models continue to use the separate llama.cpp path and are unaffected by the drawing-engine replacement.

## API server (use the phone as a server)

Serves the **already loaded** chat model over an OpenAI-compatible HTTP API, so other devices can use ready-made clients while the phone does the inference.

> **Chat and drawing are two separate things**, each with its own port, key, switch and stats
> (defaults 8080 / 8081). You can run either one, or both. They differ in who they serve, how much
> they cost and how much they expose - one image pins the CPU for tens of seconds and usually
> belongs on a trusted network only, while the chat endpoint is what you connect to daily.
> They **share a single foreground service and one notification** (a foreground service only exists
> to keep the process alive; two of them would just mean two persistent notifications).

- **How to start**: Models/API page → Chat model → switch the top **Model / API** tabs to **API** → tap "Start service". **Starting the service does not require a model to be loaded** - the service just opens the HTTP port; with no chat model the chat endpoint returns 503 (that one endpoint only), and it becomes usable as soon as a model is loaded
- **Endpoints**
  | Method | Path | Notes |
  |---|---|---|
  | POST | `/v1/chat/completions` | Chat; `"stream": true` switches to SSE (`data: {...}` chunks + `data: [DONE]`) |
  | GET | `/v1/models` | The loaded model, in OpenAI format |
  | GET | `/health` | Liveness: model name / ready / busy |
- **Client settings**: Base URL = `http://<phone-lan-ip>:8080/v1` (the app shows the address, tap it to copy), API key = the one you set (if you set none, clients still usually require a non-empty value - anything works)
- **Port and key**: port defaults to **8080** (changeable - that service is restarted for you so it applies right away). The API key is **optional** - leaving it unset means no check at all
- **How it runs**: a foreground service (`specialUse`, avoiding the 6-hour limit Android 15+ puts on `dataSync`) with an ongoing notification and a WifiLock, so it keeps serving in the background and with the screen off; the notification has a Stop action
- **One request at a time**: a local model cannot run concurrently, so a busy server answers 429 (the standard rate-limit semantic); clients just retry
- **Context reuse**: clients resend the whole history each turn; the server recognises "same history + new question" and continues the existing session (for gguf it is llama.cpp's KV reuse) instead of recomputing from scratch

### Drawing API (`/v1/images/generations`)

Serves the **already loaded** drawing model over an OpenAI-compatible text-to-image endpoint, so a client can render an image on the phone from a single prompt.
**It is a separate service**: its own port (default **8081**), its own key and its own switch, fully independent of the chat API.

- **How to enable**: Models/API page -> Draw model -> switch the top **Model / API** tabs to **API** -> tap "Start service". No need to go to the Chat model page
- **Off by default**: one image takes tens of seconds to minutes and pins the CPU, so it is not reachable from the network unless you ask for it
- **Client settings**: Base URL = `http://<phone-lan-ip>:8081/v1` (**note the different port**; the app shows both addresses and copies on tap)
- **Request**: `{"prompt": "..."}` is enough. Optional `n` (1-4), `size` (e.g. `"512x512"`), `seed`, `steps`, `cfg_scale`, `negative_prompt`; **anything you do not send comes from the drawing page's current settings** (steps / CFG / size / sampler / scheduler / LoRA)
- **Response**: `{"created": ..., "data": [{"b64_json": "..."}]}` - only `response_format=b64_json` is supported, as this server has no public URL to serve images from

- **Security**: the server listens on **every** network interface of the phone, including public ones. Without an API key, anyone who can reach the address can use your model and battery - **exposing it to the internet is your call** (set a key, or keep it on a trusted network)

## Tagging (Tagger)

- **On-device** image-to-tags: import a Danbooru-style ONNX tagger (`.onnx`, e.g. WD14) plus its matching tag list (`selected_tags.csv`) - nothing is bundled or downloaded
- The "Tagger" page: pick an image → set threshold (default 0.35) / max tags (default 40) / **channel order** (BGR by default) → get tags
  - "Channel order" exists for different preprocessing: the official WD-family pipeline is BGR; a few re-exported models baked BGR into their weights, so those need RGB (switch it if colours / hair colour come out wrong)
- Send the tags to **Text-to-Image / Image-to-Image** (with a confirmation prompt); from the Result view you can also send a generated image to **Image-to-Image** or to **Tagger**
- Multiple taggers / tag lists can be switched from the list on the Models/API page; "Load / Unload tagger" loads on demand and frees memory when you are done
- A loaded tagger can also be called as a **tool by chat models that cannot see images** (the model emits a `<tag>` command, the app tags the image, the tags are sent back and the model answers from them) - see the Chat section
- Backend follows the Models/API page run mode: CPU, or NNAPI when GPU is selected

## Permissions

Taking a photo, picking an image and saving to the gallery require **no storage access permission at all**; apart from the entries below the app declares no other permissions.

| What | Permission | Why |
| --- | --- | --- |
| Take photo | **No `CAMERA`** | Uses `ACTION_IMAGE_CAPTURE`, delegating to the system camera app: the photo is taken there and only the result comes back. Declaring `CAMERA` without holding the grant would actually make this throw `SecurityException`, so it is intentionally left undeclared |
| Pick an image (gallery / files) | **No storage read permission** | Uses the system file picker `ACTION_GET_CONTENT`: the app only receives a temporary read grant for the **single file the user taps** (`content://`), not for storage as a whole - hence no `READ_MEDIA_IMAGES` / `READ_EXTERNAL_STORAGE` |
| Save an image to the gallery | **Not needed on Android 10+** | Writes through `MediaStore` into the app's own media entry, which the platform allows without a permission |
| Save an image / take a photo | **Needed on Android 9 and below** | These need `WRITE_EXTERNAL_STORAGE` (declared with `maxSdkVersion="28"`, so it only applies on Android 9 or older), and the app **only prompts on Android 9, at the first save/photo**; on Android 10+ it is never requested |
| Downloading LoRA | `INTERNET`, `ACCESS_NETWORK_STATE` | Used only by the manual LCM-LoRA download (HuggingFace / hf-mirror) on the LoRA card. **No download, no network** - everything else needs no network |
| Models / chats / images | **No permission** | Everything lives in the app's private directory (`filesDir/`), the app sandbox |
| **API server** (optional, off by default) | `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_SPECIAL_USE`, `POST_NOTIFICATIONS`, `WAKE_LOCK`, `ACCESS_WIFI_STATE`, `CHANGE_WIFI_STATE` | Used only while the API server runs: a persistent foreground service (`specialUse`) + an ongoing notification + a WifiLock (survives screen-off). **Unused entirely when the feature is off**; denying notification access does not stop the service, it only hides the ongoing notification |

- The `content://` grant returned by the camera or gallery is **temporary**, so imported files are copied into the private directory right away (this is why picking a model copies it)
- The `IMAGE_CAPTURE` entry under `<queries>` is an Android 11+ package-visibility declaration, not a permission
- No location, contacts or microphone permission is requested; **`POST_NOTIFICATIONS` and the foreground-service permissions are only used while the API server is on**

## Languages

All UI strings are externalized (`res/values` = English, `res/values-zh` = Chinese) and **follow the system language** automatically - no setting needed.

The fourth tab is an **About** page with two collapsible sections: **Settings** (the three-position theme slider: Light / Follow system / Dark) and **About** (version and install time, app intro, notes on the thinking mode, license and third-party component list, author and project link).

## Building

The standard build requires JDK 17+, Android SDK (compileSdk 36), NDK 27.3, Rust stable (edition 2024), and `cargo-ndk` for `aarch64-linux-android`. Clone with recursive submodules so `rust/hearth` and Hearth's Burn, CubeCL and Cubek dependencies are available.

```bash
cargo fmt --manifest-path rust/ponko-native/Cargo.toml --all
cargo test --manifest-path rust/ponko-native/Cargo.toml
./gradlew :app:assembleRelease
# Output: app/build/outputs/apk/release/app-release-unsigned.apk (unsigned)
```

Cargo builds Hearth inference and `libponko_native.so`, while Gradle generates the UniFFI Kotlin bindings. The drawing path needs no CMake target and does not compile or package `libstable-diffusion.so` or `libponko_sd.so`.

Signing helpers are in `tools/`. `tools/sign-apk.ps1` reads the gitignored project-root `keystore.properties` or accepts the password from `PONKO_KS_PASS`.

## Where to get models

- `.litertlm`: the `litert-community` / `google` organizations on HuggingFace (Gemma family, etc.)
- `.gguf` (chat): any GGUF quantized model on HuggingFace; on a phone CPU, **1B-3B at Q4** is recommended
- Drawing: SD 1.5 / SDXL `.safetensors` checkpoints, matching `vocab.json` / `merges.txt`, and an FP16-safe VAE for SDXL
- **LoRA**: `lcm-lora-sdv1-5.safetensors` (~130 MB), downloadable in-app; a local `.safetensors` can also be imported directly
- **Tagger**: the `.onnx` of a Danbooru-style tagger plus its matching tag-list CSV (the two must go together; e.g. WD14's `wd-v1-4-swinv2-tagger-v2` and `selected_tags.csv`)

## Known limitations

- Chat context defaults to 4096 tokens (adjustable on the Models/API page; on the LiteRT-LM side it cannot exceed the limit built into the model); beyond that it relies on llama.cpp's context-shift sliding window
- The GGUF chat path currently uses CPU only (the binding used does not include GPU backends)
- "Disable thinking" depends on the model's own chat template; some models may still emit reasoning
- Drawing uses Hearth / Burn-WGPU Vulkan exclusively; real-device performance on Exynos 2400e has not yet been measured. Vulkan initialization errors are surfaced directly, without falling back to a C++ or CPU drawing engine
- Hearth's Vulkan path still needs a real-device test on Exynos 2400e / Xclipse 940. GGUF chat continues to use the CPU-only llama.cpp binding
- Hearth currently loads SafeTensors weights only; GGUF drawing models and in-app UNet quantization are not supported
- Android targets arm64-v8a. Hearth GPU compute is managed by Burn/WGPU Vulkan and no longer uses Ponko's custom Exynos C++ compile profile
- Tagging supports Danbooru-style ONNX taggers (WD14 and its derivatives; the tag list must be in `tag_id,name,category,count` format); the `.onnx` and the tag-list CSV must be imported as a matching pair
- If tagging looks wrong (e.g. wrong colours or hair colour), try switching the **channel order** (BGR / RGB): BGR is the official WD-family preprocessing, but a few re-exported models already baked BGR into their weights and need RGB; threshold and tag count also change the output a lot
- The tagger model is loaded lazily on first use (it holds hundreds of MB of memory) and can be unloaded manually from the Models/API page
- On **Android 9**, saving to the gallery / taking a photo requires the storage permission (prompted on first use); denying it makes those actions fail (Android 10+ is unaffected)
- Image input is capped at 4 images and 2 files per message; a `.gguf` vision model must be paired with its matching `mmproj`, selected before the model is loaded
- File input accepts plain text and the common document formats (`.docx` / `.pptx` / `.xlsx` / `.odt` / `.ods` / `.odp` / `.epub` / `.rtf` / `.html` / `.pdf`); `.doc` / `.xls` / `.ppt` have to be re-saved first; scanned (image-only) PDFs yield no text; binary files are rejected; per-file size cap (4 MB for plain text, 48 MB for documents); the text is trimmed to the context budget (about a quarter of the context by default)
- API server: **chat and drawing are fully independent** (own port / key / switch, defaults 8080 and 8081) sharing one foreground service and one notification; **starting a service does not require a model to be loaded** (the affected endpoint returns 503 until one is); the API only serves what is loaded; one request at a time; port / key changes restart that service automatically; with **no API key set there is no check at all**, so exposing it publicly is at your own risk
- The drawing endpoint is **text-to-image only** (no image-to-image), supports only `response_format=b64_json`, and once started a request cannot be cancelled midway (it finishes the current image, even if the client disconnects); while the app itself is generating the endpoint returns 429, and while the API is generating the app tells you to wait (one drawing engine, no concurrency)
- When the context does not fit, older turns and file text are trimmed automatically; if it still does not fit, the app tells you what to do: start a new chat, send a shorter file, or drop some old turns

## License

- **Code**: [MIT](LICENSE)
- **Artwork** (app icon, character portrait, original drawing): © the author, all rights reserved, **not covered by the MIT license** - see [NOTICE](NOTICE)
- **Third-party components**: distributed under their own licenses - see [NOTICE](NOTICE) for the full list

## Third-party components

| Component | License | Distributed as |
| --- | --- | --- |
| [Hearth](https://github.com/jeff-hiner/hearth) | MIT OR Apache-2.0 | Rust submodule (`rust/hearth/`) |
| Burn / CubeCL / Cubek | Their MIT / Apache-2.0 licenses | Hearth Rust submodules (`rust/hearth/vendor/`) |
| [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) | Apache-2.0 | Gradle dependency |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) (via java-llama.cpp binding) | MIT | Gradle dependency |
| [Markwon](https://github.com/noties/Markwon) | Apache-2.0 | Gradle dependency |
| [PDFBox-Android](https://github.com/TomRoush/PdfBox-Android) | Apache-2.0 | Gradle dependency (PDF text extraction) |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | Gradle dependency (Tagger) |
| Kotlin / kotlin-reflect / kotlinx.coroutines / AndroidX | Apache-2.0 | Gradle dependency |
| [commonmark-java](https://github.com/commonmark/commonmark-java) (via Markwon) | BSD-2-Clause | Gradle transitive dependency |
| [Jackson](https://github.com/FasterXML/jackson) (via the llama.cpp binding) | Apache-2.0 | Gradle transitive dependency |
| [Gson](https://github.com/google/gson) (via LiteRT-LM) | Apache-2.0 | Gradle transitive dependency |
| [SLF4J API](https://www.slf4j.org/) (via the llama.cpp binding) | MIT | Gradle transitive dependency |
| FastDoubleParser / Schubfach (bundled in Jackson) | MIT / Boost-1.0 | shipped inside Jackson (`META-INF/*-LICENSE`) |
| Annotation-only libs (jspecify / checker-qual / error_prone_annotations / JetBrains annotations) | Apache-2.0 / MIT | Gradle transitive dependency (annotations only, no runtime code) |
| Statically linked into LiteRT-LM's native library (LiteRT / TFLite, XNNPACK, sentencepiece, HF tokenizers, re2, Abseil, cpuinfo, protobuf, flatbuffers, zlib) | Apache-2.0 / BSD-3-Clause / BSD-2-Clause / zlib | built into `liblitertlm_jni.so` |
| Statically linked into ONNX Runtime's native library (XNNPACK, protobuf, ONNX, Abseil, flatbuffers, re2, cpuinfo, ...) | Apache-2.0 / BSD-3-Clause / BSD-2-Clause | built into `libonnxruntime.so` |
| LLVM libc++ / libomp (from the Android NDK) | Apache-2.0 with LLVM Exceptions | `app/src/main/jniLibs/` |

> Drawing models (SafeTensors checkpoints, tokenizer files, VAE and LoRA), chat models (`.litertlm` / `.gguf`) and tagging assets (`.onnx` + `selected_tags.csv`) are supplied by the user; no model weights are included.
> ONNX Runtime bundles additional third-party components; see [licenses/onnxruntime-ThirdPartyNotices.txt](licenses/onnxruntime-ThirdPartyNotices.txt).
> commonmark / Jackson / Gson / SLF4J are **transitive dependencies** pulled in automatically by Markwon, the llama.cpp binding, and LiteRT-LM.
> Components statically linked into the third-party native libraries (XNNPACK, protobuf, re2, cpuinfo, zlib, ...) and the full license texts are in [licenses/](licenses/) and [NOTICE](NOTICE).

## Credits

- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (Google)
- [java-llama.cpp](https://github.com/kherud/java-llama.cpp) / llama.cpp
- [Hearth](https://github.com/jeff-hiner/hearth) (Rust diffusion engine)
- [Burn](https://github.com/tracel-ai/burn) / [CubeCL](https://github.com/tracel-ai/cubecl)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (Microsoft)
- [Markwon](https://github.com/noties/Markwon)
- [commonmark-java](https://github.com/commonmark/commonmark-java)
