#include <stable-diffusion.h>

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

namespace {
std::mutex log_mutex;
std::string captured_logs;

void capture_log(enum sd_log_level_t, const char* text, void*) {
    if (text == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(log_mutex);
    captured_logs.append(text);
    captured_logs.push_back('\n');
}

std::string logs_snapshot() {
    std::lock_guard<std::mutex> lock(log_mutex);
    return captured_logs;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: taesd-runtime-smoke <sd15-model.gguf> <taesd.safetensors>\n";
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string taesd_path = argv[2];

    sd_set_log_callback(capture_log, nullptr);

    sd_ctx_params_t context_params;
    sd_ctx_params_init(&context_params);
    context_params.model_path = model_path.c_str();
    context_params.taesd_path = taesd_path.c_str();
    context_params.n_threads = 2;
    context_params.backend = "CPU";
    context_params.rng_type = STD_DEFAULT_RNG;
    context_params.sampler_rng_type = STD_DEFAULT_RNG;
    context_params.enable_mmap = true;
    context_params.flash_attn = false;
    context_params.diffusion_flash_attn = false;

    char* params_dump = sd_ctx_params_to_str(&context_params);
    if (params_dump == nullptr) {
        std::cerr << "FAIL: native parameter dump is null\n";
        return 3;
    }
    const std::string params_text(params_dump);
    std::free(params_dump);

    if (params_text.find("n_threads: 2") == std::string::npos ||
        params_text.find(taesd_path) == std::string::npos) {
        std::cerr << "FAIL: native context parameters lost thread count or TAESD path\n"
                  << params_text << '\n';
        return 4;
    }

    std::cout << "Creating SD1.5 context with n_threads=2 and TAESD...\n";
    sd_ctx_t* context = new_sd_ctx(&context_params);
    if (context == nullptr) {
        std::cerr << "FAIL: sd.cpp failed to create the model context\n"
                  << logs_snapshot() << '\n';
        return 5;
    }

    sd_img_gen_params_t generation_params;
    sd_img_gen_params_init(&generation_params);
    generation_params.prompt = "a small red ball on a plain table";
    generation_params.negative_prompt = "";
    generation_params.width = 64;
    generation_params.height = 64;
    generation_params.seed = 42;
    generation_params.batch_count = 1;
    generation_params.sample_params.sample_steps = 2;
    generation_params.sample_params.sample_method = EULER_SAMPLE_METHOD;
    generation_params.sample_params.scheduler = DISCRETE_SCHEDULER;
    generation_params.sample_params.guidance.txt_cfg = 7.0f;

    sd_image_t* images = nullptr;
    int image_count = 0;
    const bool generated = generate_image(context, &generation_params, &images, &image_count);
    const bool has_image = generated && images != nullptr && image_count > 0 &&
                           images[0].data != nullptr && images[0].width > 0 &&
                           images[0].height > 0;

    const std::string runtime_logs = logs_snapshot();
    const bool loaded_taesd =
        runtime_logs.find("loading tae from") != std::string::npos;
    const bool selected_tae_decoder =
        runtime_logs.find("using TAE for encoding / decoding") != std::string::npos;
    const bool selected_standard_vae =
        runtime_logs.find("using VAE for encoding / decoding") != std::string::npos;

    if (images != nullptr) {
        free_sd_images(images, image_count);
    }
    free_sd_ctx(context);

    if (!has_image) {
        std::cerr << "FAIL: image generation returned no valid image\n"
                  << runtime_logs << '\n';
        return 6;
    }
    if (!loaded_taesd || !selected_tae_decoder || selected_standard_vae) {
        std::cerr << "FAIL: TAESD did not exclusively override the standard VAE decoder\n"
                  << "loaded_taesd=" << loaded_taesd
                  << ", selected_tae_decoder=" << selected_tae_decoder
                  << ", selected_standard_vae=" << selected_standard_vae << '\n'
                  << runtime_logs << '\n';
        return 7;
    }

    std::cout << "PASS: native context accepted n_threads=2 and the TAESD path.\n"
              << "PASS: a model with baked-in VAE generated a valid image using only the final TAE decoder.\n";
    return 0;
}
