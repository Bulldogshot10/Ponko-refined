#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "model/adapter/lora.hpp"
#include "model_manager.h"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "model_io/streaming_writer.h"
#include "model_loader.h"

static std::string fusion_memory_snapshot() {
    std::ifstream status("/proc/self/status");
    std::string line;
    std::string snapshot;
    while (std::getline(status, line)) {
        if (starts_with(line, "VmRSS:") || starts_with(line, "VmHWM:") || starts_with(line, "VmSize:")) {
            if (!snapshot.empty()) snapshot += " ";
            snapshot += line;
        }
    }
    return snapshot.empty() ? "proc-status-unavailable" : snapshot;
}

static void log_fusion_memory(const char* stage) {
    const std::string memory = fusion_memory_snapshot();
    LOG_INFO("LoRA fusion diagnostics | stage=%s | %s", stage, memory.c_str());
}

struct TensorExportInfo {
    TensorStorage storage;
    ggml_type type;
};


struct LoraFusionContext {
    // Declare the backend owner before the runners so the LoRA runners are destroyed first.
    SDBackendManager backend_manager;
    std::shared_ptr<ModelManager> model_manager;
    std::vector<std::unique_ptr<LoraModel>> lora_models;
    // Keep only source tensor names resident during fusion. LoRA weight buffers are
    // loaded for one target tensor at a time and released immediately after applying it.
    std::vector<std::set<std::string>> lora_source_names;
    std::set<std::string> model_tensor_names;
    std::set<std::string> target_tensor_names;
    SDVersion model_version = VERSION_COUNT;
    int n_threads = 1;
    std::mutex apply_mutex;

    static bool has_weight_delta(const LoraModel& lora, const std::string& tensor_name) {
        static const char* suffixes[] = {
            ".lora_down", ".lora_up", ".lora_mid", ".diff", ".alpha", ".scale",
            ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
            ".lokr_w1", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2", ".lokr_w2_a", ".lokr_w2_b"
        };
        const std::string root = "lora." + tensor_name;
        // Some adapters split a target into multiple delta groups, suffixed .1, .2, ...
        for (int index = 0; index < 16; ++index) {
            const std::string key = root + (index == 0 ? "" : "." + std::to_string(index));
            for (const char* suffix : suffixes) {
                if (lora.lora_tensors.find(key + suffix) != lora.lora_tensors.end()) {
                    return true;
                }
            }
        }
        return false;
    }

    static std::set<std::string> matching_source_names(
        const std::set<std::string>& source_names, const std::string& tensor_name) {
        static const char* suffixes[] = {
            ".lora_down", ".lora_up", ".lora_mid", ".diff", ".alpha", ".scale",
            ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
            ".lokr_w1", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2", ".lokr_w2_a", ".lokr_w2_b"
        };
        std::set<std::string> matched;
        const std::string root = "lora." + tensor_name;
        for (int index = 0; index < 16; ++index) {
            const std::string key = root + (index == 0 ? "" : "." + std::to_string(index));
            for (const char* suffix : suffixes) {
                const std::string name = key + suffix;
                if (source_names.find(name) != source_names.end()) {
                    matched.insert(name);
                }
            }
        }
        return matched;
    }

    static bool has_source_weight_delta(const std::set<std::string>& source_names,
                                        const std::string& tensor_name) {
        return !matching_source_names(source_names, tensor_name).empty();
    }

    bool initialize(const std::vector<std::string>& paths,
                    const std::vector<float>& scales,
                    const std::set<std::string>& base_names,
                    SDVersion version,
                    int threads,
                    std::string* error) {
        if (paths.empty() || paths.size() != scales.size()) {
            if (error) *error = "at least one LoRA path and one matching scale are required";
            return false;
        }
        if (!sd_version_is_unet(version)) {
            if (error) *error = "LoRA fusion currently supports Stable Diffusion 1.x, 2.x, and SDXL checkpoints";
            return false;
        }

        std::string backend_error;
        if (!backend_manager.init("cpu", "cpu", nullptr, &backend_error)) {
            if (error) *error = "failed to initialize CPU backend for LoRA fusion: " + backend_error;
            return false;
        }
        ggml_backend_t compute_backend = backend_manager.runtime_backend(SDBackendModule::DIFFUSION);
        ggml_backend_t params_backend = backend_manager.params_backend(SDBackendModule::DIFFUSION);
        if (compute_backend == nullptr || params_backend == nullptr) {
            if (error) *error = "CPU backend did not provide a valid LoRA fusion backend";
            return false;
        }

        model_manager = std::make_shared<ModelManager>();
        model_manager->set_n_threads(std::max(1, threads));
        model_manager->set_enable_mmap(false);
        model_tensor_names = base_names;
        model_version = version;
        n_threads = std::max(1, threads);

        std::vector<ModelManager::LoraSpec> specs;
        specs.reserve(paths.size());
        for (size_t i = 0; i < paths.size(); ++i) {
            ModelManager::LoraSpec spec;
            spec.path = paths[i];
            spec.multiplier = scales[i];
            spec.required = true;
            specs.push_back(std::move(spec));
        }
        if (!model_manager->prepare_lora_sources(specs)) {
            if (error) *error = "failed to read or register one of the LoRA source files";
            return false;
        }

        lora_models.reserve(specs.size());
        lora_source_names.reserve(specs.size());
        for (const auto& spec : specs) {
            // file_tensors is a metadata-only view; do not allocate/register every LoRA
            // weight up front, because that makes peak RSS grow across the whole export.
            const auto source_tensors = model_manager->loader().file_tensors(spec.file_id, model_version);
            std::set<std::string> source_names;
            for (const auto& entry : source_tensors) {
                source_names.insert(entry.first);
            }

            auto model = std::make_unique<LoraModel>(
                spec.path, compute_backend, params_backend, model_manager, spec.file_id, model_version);
            model->multiplier = spec.multiplier;
            for (const auto& tensor_name : model_tensor_names) {
                if (has_source_weight_delta(source_names, tensor_name)) {
                    target_tensor_names.insert(tensor_name);
                }
            }
            lora_source_names.push_back(std::move(source_names));
            lora_models.push_back(std::move(model));
        }

        if (target_tensor_names.empty()) {
            if (error) *error = "none of the selected LoRAs matched tensors in this checkpoint";
            return false;
        }
        LOG_INFO("LoRA fusion initialized: %zu adapter(s), %zu matching model tensor(s), version=%d",
                 lora_models.size(), target_tensor_names.size(), static_cast<int>(model_version));
        return true;
    }

    bool targets(const std::string& tensor_name) const {
        return target_tensor_names.find(tensor_name) != target_tensor_names.end();
    }

    bool apply_tensor(const std::string& tensor_name, ggml_tensor* tensor, std::string* error) {
        std::lock_guard<std::mutex> lock(apply_mutex);
        std::map<std::string, ggml_tensor*> target{{tensor_name, tensor}};
        for (size_t i = 0; i < lora_models.size(); ++i) {
            auto& model = lora_models[i];
            const std::set<std::string> selected_names =
                matching_source_names(lora_source_names[i], tensor_name);
            if (selected_names.empty()) continue;

            // Register only this target's LoRA factors. ModelManager otherwise retains
            // previously used adapter tensors, causing RSS to climb over the full export.
            if (!model->init_params(n_threads, [&selected_names](const std::string& name) {
                    return selected_names.find(name) != selected_names.end();
                })) {
                model->release_loaded_tensors();
                if (error) *error = "failed to load LoRA tensors for target '" + tensor_name +
                                    "' from '" + model->file_path + "'";
                return false;
            }
            model->preprocess_lora_tensors(model_tensor_names);
            if (!has_weight_delta(*model, tensor_name)) {
                model->release_loaded_tensors();
                continue;
            }

            // Reuse the inference engine's LoRA/LoHA/LoKR math so fusion matches runtime
            // scaling, alpha/rank handling, tensor-name conversion, and shape checks.
            const bool applied = model->apply(target, model_tensor_names, model_version, n_threads, false);
            // apply() is synchronous. Unregister the selected parameter buffers before the
            // next adapter/tensor so peak memory is independent of the number of LoRA targets.
            model->release_loaded_tensors();
            if (!applied) {
                if (error) *error = "failed to apply LoRA delta to tensor '" + tensor_name + "'";
                return false;
            }
        }
        return true;
    }
};

static bool convert_fused_f32_tensor(const ggml_tensor* source,
                                     ggml_type destination_type,
                                     std::vector<uint8_t>* output,
                                     std::string* error) {
    if (source == nullptr || source->data == nullptr || output == nullptr ||
        source->type != GGML_TYPE_F32) {
        if (error) *error = "invalid F32 tensor supplied to LoRA fusion exporter";
        return false;
    }
    const int64_t elements = ggml_nelements(source);
    if (elements <= 0) {
        if (error) *error = "LoRA fusion encountered an empty model tensor";
        return false;
    }
    const size_t output_bytes = static_cast<size_t>(elements) *
        ggml_type_size(destination_type) / ggml_blck_size(destination_type);
    output->resize(output_bytes);
    const auto* src = static_cast<const float*>(source->data);

    if (destination_type == GGML_TYPE_F32) {
        std::memcpy(output->data(), src, output_bytes);
        return true;
    }
    if (destination_type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src, reinterpret_cast<ggml_fp16_t*>(output->data()),
                              static_cast<int64_t>(elements));
        return true;
    }
    if (destination_type == GGML_TYPE_Q4_0 || destination_type == GGML_TYPE_Q5_0 ||
        destination_type == GGML_TYPE_Q8_0) {
        const int64_t row_width = source->ne[0];
        if (row_width <= 0 || elements % row_width != 0) {
            if (error) *error = "invalid row shape while quantizing fused LoRA tensor";
            return false;
        }
        std::vector<float> importance(static_cast<size_t>(row_width), 1.0f);
        ggml_quantize_chunk(
            destination_type, src, output->data(), 0,
            static_cast<int64_t>(elements / row_width), static_cast<int64_t>(row_width),
            importance.data());
        return true;
    }
    if (error) *error = std::string("unsupported output tensor type for LoRA fusion: ") +
                        ggml_type_name(destination_type);
    return false;
}

struct TensorExportJob {
    TensorExportInfo info;
    std::vector<uint8_t> data;
    std::string error;
    bool success = false;
};

static ggml_type get_export_tensor_type(ModelLoader& model_loader,
                                        const TensorStorage& tensor_storage,
                                        ggml_type type,
                                        const TensorTypeRules& tensor_type_rules) {
    const std::string& name = tensor_storage.name;
    ggml_type tensor_type   = tensor_storage.type;
    ggml_type dst_type      = type;

    for (const auto& tensor_type_rule : tensor_type_rules) {
        std::regex pattern(tensor_type_rule.first);
        if (std::regex_search(name, pattern)) {
            dst_type = tensor_type_rule.second;
            break;
        }
    }

    if (model_loader.tensor_should_be_converted(tensor_storage, dst_type)) {
        tensor_type = dst_type;
    }

    return tensor_type;
}

static bool collect_tensors_for_export(ModelLoader& model_loader,
                                       ggml_type type,
                                       const TensorTypeRules& tensor_type_rules,
                                       std::vector<TensorExportInfo>& tensors) {
    tensors.clear();
    tensors.reserve(model_loader.get_tensor_storage_map().size());
    for (const auto& kv : model_loader.get_tensor_storage_map()) {
        const TensorStorage& tensor_storage = kv.second;
        TensorExportInfo info;
        info.storage = tensor_storage;
        info.type    = get_export_tensor_type(model_loader, tensor_storage, type, tensor_type_rules);
        tensors.push_back(std::move(info));
    }
    LOG_INFO("collected %zu tensors for export", tensors.size());
    return true;
}

static size_t export_tensor_nbytes(const TensorExportInfo& info) {
    TensorStorage output_storage = info.storage;
    output_storage.type          = info.type;
    return static_cast<size_t>(output_storage.nbytes());
}

static TensorWritePlan tensor_write_plan_from_export_info(const TensorExportInfo& info) {
    TensorWritePlan plan;
    plan.name   = info.storage.name;
    plan.type   = info.type;
    plan.n_dims = info.storage.n_dims;
    for (int i = 0; i < SD_MAX_DIMS; i++) {
        plan.ne[i] = info.storage.ne[i];
    }
    return plan;
}

static std::vector<TensorWritePlan> tensor_write_plans_from_export_infos(const std::vector<TensorExportInfo>& tensors) {
    std::vector<TensorWritePlan> plans;
    plans.reserve(tensors.size());
    for (const TensorExportInfo& info : tensors) {
        plans.push_back(tensor_write_plan_from_export_info(info));
    }
    return plans;
}

static bool preallocate_output_file(const std::string& output_path, uint64_t file_size, std::string* error) {
    if (file_size == 0) {
        return true;
    }

    std::fstream file(output_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file.is_open()) {
        if (error != nullptr) {
            *error = "failed to open output file '" + output_path + "' for preallocation";
        }
        return false;
    }

    // This portable fallback sets the final file size. A platform-specific
    // posix_fallocate/ftruncate path can replace it later.
    file.seekp(static_cast<std::streamoff>(file_size - 1), std::ios::beg);
    file.put('\0');
    file.flush();
    if (!file) {
        if (error != nullptr) {
            *error = "failed to preallocate output file '" + output_path + "'";
        }
        return false;
    }
    return true;
}

static bool load_tensor_for_export(ModelLoader& model_loader,
                                   TensorExportJob& job,
                                   LoraFusionContext* fusion = nullptr) {
    const TensorStorage& storage = job.info.storage;
    TensorStorage output_storage = storage;
    output_storage.type = job.info.type;

    if (fusion != nullptr && fusion->targets(storage.name)) {
        static std::atomic<size_t> fusion_tensor_sequence{0};
        const size_t sequence = fusion_tensor_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
        LOG_INFO("LoRA fusion tensor start | index=%zu | name=%s | shape=[%lld,%lld,%lld,%lld] | estimated_f32=%zu",
                 sequence, storage.name.c_str(),
                 static_cast<long long>(storage.ne[0]), static_cast<long long>(storage.ne[1]),
                 static_cast<long long>(storage.ne[2]), static_cast<long long>(storage.ne[3]),
                 static_cast<size_t>(storage.nelements()) * sizeof(float));
        if (sequence == 1 || sequence % 16 == 0) log_fusion_memory("before-fused-tensor");
        const size_t float_bytes = static_cast<size_t>(storage.nelements()) * sizeof(float);
        const size_t mem_size = 1 * 1024 * 1024 + ggml_tensor_overhead() + float_bytes;
        ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});
        if (ggml_ctx == nullptr) {
            job.error = "ggml_init failed for fused tensor '" + storage.name + "'";
            return false;
        }
        ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, storage.n_dims, storage.ne);
        if (tensor == nullptr) {
            ggml_free(ggml_ctx);
            job.error = "ggml_new_tensor failed for fused tensor '" + storage.name + "'";
            return false;
        }
        ggml_set_name(tensor, storage.name.c_str());
        std::vector<float> base_weights;
        // The loader's thread count was set before worker threads started; zero avoids
        // mutating shared loader state while other tensor-export workers are active.
        if (!model_loader.load_float_tensor(storage.name, base_weights, 0, false) ||
            base_weights.size() != static_cast<size_t>(storage.nelements())) {
            ggml_free(ggml_ctx);
            job.error = "failed to dequantize base weights for fused tensor '" + storage.name + "'";
            return false;
        }
        std::memcpy(tensor->data, base_weights.data(), base_weights.size() * sizeof(float));
        // The ggml tensor owns its F32 copy now. Free the vector before building the LoRA graph
        // so low-RAM Android devices do not keep two full-size F32 copies alive at once.
        std::vector<float>().swap(base_weights);
        if (sequence == 1 || sequence % 16 == 0) log_fusion_memory("after-base-vector-release");
        if (!fusion->apply_tensor(storage.name, tensor, &job.error)) {
            ggml_free(ggml_ctx);
            return false;
        }
        const bool converted = convert_fused_f32_tensor(tensor, job.info.type, &job.data, &job.error);
        if (sequence == 1 || sequence % 16 == 0) log_fusion_memory("after-lora-apply-and-export-convert");
        ggml_free(ggml_ctx);
        return converted;
    }

    size_t mem_size = 1 * 1024 * 1024;
    mem_size += ggml_tensor_overhead();
    mem_size += static_cast<size_t>(output_storage.nbytes());
    ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});
    if (ggml_ctx == nullptr) {
        job.error = "ggml_init failed for tensor '" + storage.name + "'";
        return false;
    }

    ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, job.info.type, storage.n_dims, storage.ne);
    if (tensor == nullptr) {
        ggml_free(ggml_ctx);
        job.error = "ggml_new_tensor failed for tensor '" + storage.name + "'";
        return false;
    }
    ggml_set_name(tensor, storage.name.c_str());
    const size_t tensor_nbytes = ggml_nbytes(tensor);
    if (tensor_nbytes > 0 && !model_loader.load_tensor(storage, tensor)) {
        ggml_free(ggml_ctx);
        job.error = "failed to load tensor '" + storage.name + "'";
        return false;
    }
    job.data.resize(tensor_nbytes);
    if (tensor_nbytes > 0) std::memcpy(job.data.data(), tensor->data, tensor_nbytes);
    ggml_free(ggml_ctx);
    return true;
}

static uint64_t read_proc_kib(const char* path, const char* key) {
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        if (line.compare(0, std::strlen(key), key) != 0) continue;
        const char* value = line.c_str() + std::strlen(key);
        while (*value == ' ') ++value;
        return static_cast<uint64_t>(std::strtoull(value, nullptr, 10));
    }
    return 0;
}

static void log_fusion_memory(const char* phase, const std::string& tensor_name, size_t estimate_bytes) {
    const uint64_t rss_kib = read_proc_kib("/proc/self/status", "VmRSS:");
    const uint64_t hwm_kib = read_proc_kib("/proc/self/status", "VmHWM:");
    const uint64_t available_kib = read_proc_kib("/proc/meminfo", "MemAvailable:");
    LOG_INFO("LoRA fusion memory phase=%s tensor=%s estimate=%llu MiB rss=%llu MiB hwm=%llu MiB system_available=%llu MiB",
             phase, tensor_name.c_str(),
             static_cast<unsigned long long>(estimate_bytes / (1024ull * 1024ull)),
             static_cast<unsigned long long>(rss_kib / 1024ull),
             static_cast<unsigned long long>(hwm_kib / 1024ull),
             static_cast<unsigned long long>(available_kib / 1024ull));
}

static bool stream_tensor_data(ModelLoader& model_loader,
                               const std::string& output_path,
                               const std::vector<TensorExportInfo>& tensors,
                               const StreamingModelWriter& writer,
                               int n_threads,
                               std::string* error,
                               LoraFusionContext* fusion = nullptr) {
    n_threads = n_threads > 0 ? n_threads : sd_get_num_physical_cores();
    // Fusion loads the base tensor in F32 and builds a separate LoRA graph. Serialize it on
    // memory-constrained Android devices instead of allowing several peak allocations at once.
    n_threads = fusion != nullptr ? 1 : std::max(1, n_threads);
    LOG_INFO("streaming convert with %d threads", n_threads);

    int64_t start_time       = ggml_time_ms();
    uint64_t bytes_written   = 0;
    size_t tensors_written   = 0;
    size_t next_tensor_index = 0;
    bool failed              = false;
    std::string failure;

    // Fusion's transient estimate includes several F32 copies but not the resident LoRA
    // parameters. Keep a hard ceiling for each fusion tensor; never allow the old
    // "single oversized tensor is okay" exception to the memory budget.
    const size_t memory_budget = fusion != nullptr
        ? 384ull * 1024ull * 1024ull
        : 1024ull * 1024ull * 1024ull;
    size_t reserved_bytes      = 0;

    std::mutex work_mutex;
    std::mutex progress_mutex;
    std::condition_variable memory_cv;
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    auto reserve_memory = [&](size_t bytes) -> bool {
        std::unique_lock<std::mutex> lock(work_mutex);
        memory_cv.wait(lock, [&]() {
            return failed || reserved_bytes == 0 ||
                   (bytes <= memory_budget && reserved_bytes <= memory_budget - bytes);
        });
        if (failed) {
            return false;
        }
        reserved_bytes += bytes;
        return true;
    };

    auto release_memory = [&](size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            reserved_bytes -= std::min(reserved_bytes, bytes);
        }
        memory_cv.notify_all();
    };

    auto fail = [&](const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            if (!failed) {
                failed  = true;
                failure = message;
            }
        }
        memory_cv.notify_all();
    };

    for (int worker = 0; worker < n_threads; worker++) {
        workers.emplace_back([&]() {
            std::fstream output_file(output_path, std::ios::binary | std::ios::in | std::ios::out);
            if (!output_file.is_open()) {
                fail("failed to open output file '" + output_path + "' for tensor writing");
                return;
            }

            while (true) {
                size_t tensor_index = 0;
                {
                    std::lock_guard<std::mutex> lock(work_mutex);
                    if (failed || next_tensor_index >= tensors.size()) {
                        return;
                    }
                    tensor_index = next_tensor_index++;
                }

                size_t tensor_bytes = export_tensor_nbytes(tensors[tensor_index]);
                const bool fusion_target = fusion != nullptr &&
                    fusion->targets(tensors[tensor_index].storage.name);
                if (fusion_target) {
                    const int64_t raw_elements = tensors[tensor_index].storage.nelements();
                    const size_t elements = raw_elements > 0 ? static_cast<size_t>(raw_elements) : 0;
                    const size_t float_bytes = elements > SIZE_MAX / sizeof(float)
                        ? SIZE_MAX
                        : elements * sizeof(float);
                    // Fusion keeps the base F32 tensor, a dequantized source copy, and output
                    // bytes alive together; the estimate is intentionally conservative.
                    const size_t fusion_bytes = float_bytes > SIZE_MAX / 3 ? SIZE_MAX : float_bytes * 3;
                    tensor_bytes = std::max(tensor_bytes, fusion_bytes);
                    log_fusion_memory("tensor-start", tensors[tensor_index].storage.name, tensor_bytes);
                    if (tensor_bytes > memory_budget) {
                        fail("LoRA fusion stopped before allocation: tensor '" +
                             tensors[tensor_index].storage.name + "' estimates " +
                             std::to_string(tensor_bytes / (1024ull * 1024ull)) +
                             " MiB transient memory, above the 384 MiB safe per-tensor limit");
                        return;
                    }
                }
                if (!reserve_memory(tensor_bytes)) {
                    return;
                }

                TensorExportJob job;
                job.info = tensors[tensor_index];
                try {
                    job.success = load_tensor_for_export(model_loader, job, fusion);
                } catch (const std::exception& e) {
                    job.error   = e.what();
                    job.success = false;
                }

                if (fusion_target) {
                    log_fusion_memory(job.success ? "tensor-computed" : "tensor-compute-failed",
                                      tensors[tensor_index].storage.name, tensor_bytes);
                }
                if (!job.success) {
                    release_memory(tensor_bytes);
                    fail(job.error.empty() ? "streaming conversion failed" : job.error);
                    return;
                }

                std::string write_error;
                if (!writer.write_tensor(output_file,
                                         tensor_index,
                                         job.data.empty() ? nullptr : job.data.data(),
                                         job.data.size(),
                                         &write_error)) {
                    release_memory(tensor_bytes);
                    fail(write_error.empty() ? "streaming conversion write failed" : write_error);
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    bytes_written += job.data.size();
                    tensors_written++;
                    float elapsed_seconds = (ggml_time_ms() - start_time) / 1000.0f;
                    pretty_bytes_progress(static_cast<int>(tensors_written),
                                          static_cast<int>(tensors.size()),
                                          bytes_written,
                                          elapsed_seconds);
                }
                release_memory(tensor_bytes);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    printf("\n");
    if (failed) {
        if (error != nullptr) {
            *error = failure;
        }
        if (fusion != nullptr) {
            LOG_ERROR("LoRA fusion tensor stream failed: %s", failure.c_str());
            log_fusion_memory("stream-failed", failure, reserved_bytes);
        }
        return false;
    }
    if (fusion != nullptr) {
        log_fusion_memory("stream-complete", "(all tensors)", static_cast<size_t>(bytes_written));
    }
    LOG_INFO("streaming conversion completed, taking %.2fs", (ggml_time_ms() - start_time) / 1000.f);
    return true;
}

static bool write_model_file_streaming(ModelLoader& model_loader,
                                       const std::string& output_path,
                                       const std::vector<TensorExportInfo>& tensors,
                                       StreamingModelWriter& writer,
                                       int n_threads,
                                       std::string* error,
                                       LoraFusionContext* fusion = nullptr) {
    std::vector<TensorWritePlan> plans = tensor_write_plans_from_export_infos(tensors);
    if (!writer.write_metadata(output_path, plans, error)) {
        return false;
    }
    if (!preallocate_output_file(output_path, writer.file_size(), error)) {
        return false;
    }
    model_loader.process_model_files(false, false);
    return stream_tensor_data(model_loader, output_path, tensors, writer, n_threads, error, fusion);
}

static bool init_convert_path(ModelLoader& model_loader, const char* path, const char* prefix, bool& loaded_any) {
    if (path == nullptr || strlen(path) == 0) {
        return true;
    }
    if (!model_loader.init_from_file(path, prefix)) {
        LOG_ERROR("init model loader from file failed: '%s'", path);
        return false;
    }
    loaded_any = true;
    return true;
}

static bool export_loaded_model(ModelLoader& model_loader,
                                const char* output_path,
                                sd_type_t output_type,
                                const char* tensor_type_rules,
                                int n_threads,
                                LoraFusionContext* fusion = nullptr) {
    ggml_type type             = sd_type_to_ggml_type(output_type);
    bool output_is_safetensors = ends_with(output_path, ".safetensors");
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules);

    std::vector<TensorExportInfo> tensors;
    bool success = collect_tensors_for_export(model_loader, type, type_rules, tensors);
    std::string error;
    if (success) {
        std::unique_ptr<StreamingModelWriter> writer;
        if (output_is_safetensors) {
            writer = std::make_unique<SafetensorsStreamingWriter>();
        } else {
            writer = std::make_unique<GGUFStreamingWriter>();
        }
        success = write_model_file_streaming(model_loader, output_path, tensors, *writer, n_threads, &error, fusion);
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    return success;
}

bool convert_with_components(const char* model_path,
                             const char* clip_l_path,
                             const char* clip_g_path,
                             const char* t5xxl_path,
                             const char* diffusion_model_path,
                             const char* vae_path,
                             const char* output_path,
                             sd_type_t output_type,
                             const char* tensor_type_rules,
                             bool convert_name,
                             int n_threads) {
    ModelLoader model_loader;
    bool loaded_any = false;

    if (!init_convert_path(model_loader, model_path, "", loaded_any) ||
        !init_convert_path(model_loader, clip_l_path, "text_encoders.clip_l.transformer.", loaded_any) ||
        !init_convert_path(model_loader, clip_g_path, "text_encoders.clip_g.transformer.", loaded_any) ||
        !init_convert_path(model_loader, t5xxl_path, "text_encoders.t5xxl.transformer.", loaded_any) ||
        !init_convert_path(model_loader, diffusion_model_path, "model.diffusion_model.", loaded_any) ||
        !init_convert_path(model_loader, vae_path, "vae.", loaded_any)) {
        return false;
    }

    if (!loaded_any) {
        LOG_ERROR("no input model path provided for convert");
        return false;
    }

    if (convert_name) {
        model_loader.convert_tensors_name();
    }

    return export_loaded_model(model_loader, output_path, output_type, tensor_type_rules, n_threads);
}

bool fuse_loras_to_file(const char* input_model_path,
                        const char* const* lora_paths,
                        const float* lora_scales,
                        size_t lora_count,
                        const char* output_path,
                        sd_type_t output_type,
                        const char* tensor_type_rules,
                        bool convert_name,
                        int n_threads) {
    if (input_model_path == nullptr || input_model_path[0] == '\0' ||
        output_path == nullptr || output_path[0] == '\0' ||
        lora_paths == nullptr || lora_scales == nullptr || lora_count == 0) {
        LOG_ERROR("LoRA fusion requires an input model, output path, and at least one LoRA");
        return false;
    }

    log_fusion_memory("fusion-start");
    LOG_INFO("LoRA fusion base checkpoint load start | path=%s | adapters=%zu | requested_threads=%d",
             input_model_path, lora_count, n_threads);
    ModelLoader model_loader;
    model_loader.set_n_threads(n_threads);
    if (!model_loader.init_from_file(input_model_path)) {
        LOG_ERROR("failed to load base checkpoint for LoRA fusion: '%s'", input_model_path);
        return false;
    }
    if (convert_name) model_loader.convert_tensors_name();
    LOG_INFO("LoRA fusion base checkpoint loaded | tensors=%zu | version=%d",
             model_loader.get_tensor_storage_map().size(), static_cast<int>(model_loader.get_sd_version()));
    log_fusion_memory("after-base-checkpoint-metadata");

    std::set<std::string> base_names;
    for (const auto& item : model_loader.get_tensor_storage_map()) {
        if (!starts_with(item.first, "lora.")) base_names.insert(item.first);
    }
    const SDVersion version = model_loader.get_sd_version();
    LoraFusionContext fusion;
    std::vector<std::string> paths;
    std::vector<float> scales;
    paths.reserve(lora_count);
    scales.reserve(lora_count);
    for (size_t i = 0; i < lora_count; ++i) {
        if (lora_paths[i] == nullptr || lora_paths[i][0] == '\0' || !std::isfinite(lora_scales[i])) {
            LOG_ERROR("invalid LoRA path or non-finite LoRA scale at index %zu", i);
            return false;
        }
        paths.emplace_back(lora_paths[i]);
        scales.push_back(lora_scales[i]);
    }
    std::string error;
    log_fusion_memory("before-load-selected-lora-tensors");
    if (!fusion.initialize(paths, scales, base_names, version, n_threads, &error)) {
        LOG_ERROR("LoRA fusion initialization failed: %s", error.c_str());
        return false;
    }
    log_fusion_memory("after-load-selected-lora-tensors");
    const bool export_success = export_loaded_model(model_loader, output_path, output_type, tensor_type_rules, n_threads, &fusion);
    log_fusion_memory(export_success ? "fusion-export-success" : "fusion-export-failure");
    return export_success;
}

bool convert(const char* input_path,
             const char* vae_path,
             const char* output_path,
             sd_type_t output_type,
             const char* tensor_type_rules,
             bool convert_name) {
    return convert_with_components(input_path,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   vae_path,
                                   output_path,
                                   output_type,
                                   tensor_type_rules,
                                   convert_name,
                                   0);
}
