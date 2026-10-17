#if defined(GGML_MAX_NAME) && GGML_MAX_NAME != 160
#error Native Direct WAI must use the exact linked SD/GGML tensor ABI
#endif
#ifndef GGML_MAX_NAME
#define GGML_MAX_NAME 160
#endif
#include "engine.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include "stable-diffusion.h"
#include "json.hpp"
#include "engine_config.h"
#include "v3_shutdown.h"
#include "pool.h"
#include "probe.h"
#include "memory_plan.h"
#include "ui_hooks.h"
static_assert(sizeof(ggml_tensor) == 432, "Exact linked MAX_NAME160 tensor ABI");

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cerrno>
#include <clocale>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

#if defined(PS5_AI_NATIVE_RADV) && defined(__PROSPERO__)
#include <vulkan/vulkan.h>
#endif

#if defined(__FreeBSD__) || defined(__linux__)
#include <sys/resource.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static std::mutex status_mutex;
static std::mutex log_mutex;
static ps5_ai_engine_status current_status = {};
static sd_ctx_t* model_context = nullptr;
static ggml_backend_dev_t gpu_device = nullptr;
static PS5AIEngineConfig engine_config;
static std::string log_path = "/app0/ai-runtime/logs/engine.jsonl";
static bool hardware_verified = false;
static bool partial_taesd_gpu_decode = false;
static std::atomic<bool> parameter_loading{false};
extern "C" void ps5_ai_native_config_mark(unsigned,const char*);
extern "C" void ps5_ai_native_locale_snapshot(uint64_t[11]);

static void set_error(char* error, size_t size, const char* message) {
    if (error && size) {
        std::snprintf(error, size, "%s", message);
    }
}

static nlohmann::json bounded_config(const char* path) {
    ps5_ai_native_config_mark(2,"config_FD_open_before");
    if (!path) throw std::runtime_error("Native Direct WAI config path is missing");
    const size_t length = strnlen(path, 4097);
    if (!length || length > 4096) throw std::runtime_error("Native Direct WAI config path is too long");
    const int descriptor = open(path, O_RDONLY | O_NOFOLLOW);
    if (descriptor < 0) throw std::runtime_error("Native Direct WAI config could not be opened read-only");
    ps5_ai_native_config_mark(3,"config_FD_open_returned");
    struct Close { int fd; ~Close() { close(fd); } } owned = {descriptor};
    struct stat before = {}, after = {};
    ps5_ai_native_config_mark(4,"config_FD_fstat_before");
    if (fstat(descriptor, &before) || !S_ISREG(before.st_mode) || before.st_size <= 0 ||
        static_cast<uint64_t>(before.st_size) > PS5_AI_MEMORY_PLAN_MAX_CONFIG_BYTES)
        throw std::runtime_error("Native Direct WAI config must be a regular file of1..65536 bytes");
    ps5_ai_ggml_report("NATIVE_CONFIG FD_metadata regular=1 bytes=%llu reported_UID_not_ownership_proof=1",static_cast<unsigned long long>(before.st_size));
    ps5_ai_native_config_mark(5,"config_FD_fstat_returned");
    std::string bytes(static_cast<size_t>(before.st_size), '\0');
    ps5_ai_native_config_mark(6,"config_FD_read_before");
    size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t count = read(descriptor, bytes.data() + done, bytes.size() - done);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("Native Direct WAI config read was incomplete");
        done += static_cast<size_t>(count);
    }
    ps5_ai_native_config_mark(7,"config_FD_read_complete");
    char extra;
    ssize_t eof;
    do { eof = read(descriptor, &extra, 1); } while (eof < 0 && errno == EINTR);
    if (eof != 0 || fstat(descriptor, &after) || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size ||
        before.st_mtime != after.st_mtime || before.st_ctime != after.st_ctime)
        throw std::runtime_error("Native Direct WAI config changed while being read");
    ps5_ai_native_config_mark(8,"config_FD_stability_complete");
    ps5_ai_native_config_mark(9,"public_localeconv_before_JSON");
    const auto* conventions=std::localeconv();
    uint64_t locale_stats[11]={};ps5_ai_native_locale_snapshot(locale_stats);
    double probe_value=0;std::memcpy(&probe_value,&locale_stats[6],sizeof(probe_value));
    ps5_ai_ggml_report("NATIVE_CONFIG localeconv effective_ptr=%p last_real_ptr=%#llx calls=%llu actual_NULL_fallbacks=%llu empty_decimal_observed=%llu empty_decimal_repairs=%llu original_decimal_first_byte=%llu point_byte_available=%d nonempty_real_unchanged=1",
        static_cast<const void*>(conventions),static_cast<unsigned long long>(locale_stats[2]),
        static_cast<unsigned long long>(locale_stats[0]),static_cast<unsigned long long>(locale_stats[1]),
        static_cast<unsigned long long>(locale_stats[3]),static_cast<unsigned long long>(locale_stats[4]),
        static_cast<unsigned long long>(locale_stats[5]),locale_stats[5]!=UINT64_MAX);
    ps5_ai_ggml_report("NATIVE_CONFIG dot_probe known_input=0.5 calls=%llu successes=%llu value=%.17g value_bits=%#llx end_offset=%llu errno=%llu expected_end_offset=3 expected_value_bits=0x3fe0000000000000 empty_repair_requires_full_consume_exact_half=1 actual_conversion_failure_claimed=0",
        static_cast<unsigned long long>(locale_stats[9]),static_cast<unsigned long long>(locale_stats[8]),probe_value,
        static_cast<unsigned long long>(locale_stats[6]),static_cast<unsigned long long>(locale_stats[7]),
        static_cast<unsigned long long>(locale_stats[10]));
    ps5_ai_native_config_mark(10,"public_localeconv_returned");
    ps5_ai_native_config_mark(11,"JSON_parse_before");
    auto config = nlohmann::json::parse(bytes);
    ps5_ai_native_config_mark(12,"JSON_parse_returned");
    if (!config.is_object()) throw std::runtime_error("Native Direct WAI config must be a JSON object");
    return config;
}

extern "C" int ps5_ai_v12_read_memory_plan(const char* path, ps5_ai_memory_plan* out,
                                          char* error, size_t error_size) noexcept {
    if (out) *out = {};
    try {
        if (!out) throw std::runtime_error("Native Direct WAI memory-plan output is missing");
        const auto config = bounded_config(path);
        ps5_ai_native_config_mark(13,"memory_plan_fields_before");
        const auto plan = PS5AIEngineConfig::memory_plan(config.value("engine", nlohmann::json::object()));
        *out = plan;
        ps5_ai_native_config_mark(14,"memory_plan_fields_returned");
        return 0;
    } catch (const std::exception& exception) {
        ps5_ai_native_config_mark(15,"memory_plan_exception_caught");
        set_error(error, error_size, exception.what());
    } catch (...) {
        ps5_ai_native_config_mark(15,"memory_plan_unknown_exception_caught");
        set_error(error, error_size, "Native Direct WAI memory-plan config failed");
    }
    return -1;
}

static uint64_t cpu_peak_bytes() {
#if defined(__FreeBSD__) || defined(__linux__)
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss > 0) {
        return static_cast<uint64_t>(usage.ru_maxrss) * 1024;
    }
#endif
    return 0;
}

static void fallback_log(const char* message) noexcept {
    std::fputs(message, stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

static void write_event(const nlohmann::json& event) noexcept {
    try {
        const std::string line = event.dump() + "\n";
        std::lock_guard<std::mutex> lock(log_mutex);
        FILE* file = std::fopen(log_path.c_str(), "ab");
        if(file){struct stat st;if(!stat(log_path.c_str(),&st) && st.st_size>=0 && (uint64_t)st.st_size+line.size()>UINT64_C(16777216)){std::fclose(file);file=nullptr;const std::string previous=log_path+".previous";if(std::rename(log_path.c_str(),previous.c_str())==0)file=std::fopen(log_path.c_str(),"ab");}}
        if (file) {
            std::fwrite(line.data(), 1, line.size(), file);
            std::fflush(file);
            std::fclose(file);
        }
        std::fwrite(line.data(), 1, line.size(), stderr);
        std::fflush(stderr);
    } catch (...) {
        fallback_log("ps5-ai: structured log unavailable");
    }
}

static void engine_log(enum sd_log_level_t level, const char* text, void*) noexcept {
    ps5_ai_native_ui_log(text);
    // The shared progress callback also receives tensor byte-loading progress.
    // These exact loader messages identify that phase without changing I/O.
    if (text && std::strncmp(text, "loading ", 8) == 0 && std::strstr(text, " tensors from "))
        parameter_loading.store(true);
    if (text && std::strncmp(text, "loading tensors completed", 25) == 0)
        parameter_loading.store(false);
    try {
        write_event({{"event", "sdcpp_log"}, {"level", static_cast<int>(level)}, {"text", text ? text : ""}});
    } catch (...) {
        fallback_log(text ? text : "ps5-ai: engine log unavailable");
    }
}

static void verify_native_hardware() {
#if defined(PS5_AI_NATIVE_RADV) && defined(__PROSPERO__)
    const auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!create_instance) {
        throw std::runtime_error("Native RADV vkCreateInstance is unavailable");
    }
    const VkApplicationInfo application = {VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
        "ps5-ai-hardware-gate", 1, "stable-diffusion.cpp", 1, VK_API_VERSION_1_3};
    const VkInstanceCreateInfo create = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr,
        0, &application, 0, nullptr, 0, nullptr};
    VkInstance instance = VK_NULL_HANDLE;
    const VkResult result = create_instance(&create, nullptr, &instance);
    if (result != VK_SUCCESS) {
        write_event({{"event", "native_hardware_gate_failed"}, {"vkCreateInstance_result", result}});
        throw std::runtime_error("Native RADV instance creation failed; inspect GPU probe/title context");
    }
    const auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(vkGetInstanceProcAddr(instance, "vkDestroyInstance"));
    struct InstanceCleanup {
        VkInstance instance;
        PFN_vkDestroyInstance destroy;
        ~InstanceCleanup() { if (destroy) destroy(instance, nullptr); }
    } cleanup = {instance, destroy};
    const auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(vkGetInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
    const auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties"));
    if (!destroy || !enumerate || !properties) {
        throw std::runtime_error("Native RADV physical-device query entry points are missing");
    }
    uint32_t count = 0;
    if (enumerate(instance, &count, nullptr) != VK_SUCCESS || count != 1) {
        throw std::runtime_error("Expected exactly one physical device in the native PS5 RADV ICD");
    }
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    if (enumerate(instance, &count, &physical) != VK_SUCCESS || count != 1) {
        throw std::runtime_error("Native RADV physical-device enumeration failed");
    }
    VkPhysicalDeviceProperties device = {};
    properties(physical, &device);
    write_event({{"event", "native_physical_device_properties"},
                 {"vendor_id", device.vendorID}, {"device_id", device.deviceID},
                 {"device_type", device.deviceType}, {"device_name", device.deviceName},
                 {"api_version", device.apiVersion}, {"driver_version", device.driverVersion}});
    if (device.vendorID != 0x1002 ||
        (device.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
         device.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) {
        throw std::runtime_error("Physical Vulkan device is not a hardware AMD GPU; software/unknown devices are rejected");
    }
    hardware_verified = true;
    std::snprintf(current_status.device, sizeof(current_status.device), "%s", device.deviceName);
    std::snprintf(current_status.driver, sizeof(current_status.driver), "native RADV driverVersion=%u", device.driverVersion);
#else
    throw std::runtime_error("PS5 native RADV hardware identity is unverified in this build; inference is disabled");
#endif
}

struct InferenceObservation {
    ps5_ai_progress_cb progress;
    void* user;
    ps5_ai_stats* stats;
    uint64_t gpu_math_nodes;
    uint64_t gpu_unet_math_nodes;
    uint64_t gpu_flash_attention_nodes;
    const ggml_tensor* requested_unet;
    const ggml_tensor* requested_flash;
    int requested_steps;
    int last_sampler_step;
};

static const char* unet_weight_name(const ggml_tensor* tensor) {
    if (!tensor || (tensor->op != GGML_OP_MUL_MAT && tensor->op != GGML_OP_CONV_2D)) return nullptr;
    const ggml_tensor* weight = tensor->src[0];
    for (unsigned depth = 0; weight && depth < 8; ++depth) {
        const char* name = ggml_get_name(weight);
        if (name && std::strncmp(name, "model.diffusion_model.", 22) == 0) return name;
        weight = weight->view_src;
    }
    return nullptr;
}

static bool is_gpu_math_tensor(const ggml_tensor* tensor) {
    if (!tensor || !tensor->buffer || !gpu_device || !hardware_verified) {
        return false;
    }
    const ggml_backend_buffer_type_t type = ggml_backend_buffer_get_type(tensor->buffer);
    if (ggml_backend_buft_get_device(type) != gpu_device) {
        return false;
    }
    // Views/copies/allocations alone do not demonstrate GPU inference.
    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_CONV_2D:
        case GGML_OP_FLASH_ATTN_EXT:
            return true;
        default:
            return false;
    }
}

static bool observe_graph(ggml_tensor* tensor, bool ask, void* data) noexcept {
    auto* observation = static_cast<InferenceObservation*>(data);
    if (ask) {
        if (!is_gpu_math_tensor(tensor)) return false;
        // Request only the first UNet-weight and first flash-attention witnesses.
        // Each true ask partitions and synchronizes the graph in SD.cpp; all
        // other nodes still execute normally, without per-node proof barriers.
        if (!observation->gpu_unet_math_nodes && !observation->requested_unet && unet_weight_name(tensor)) {
            observation->requested_unet = tensor;
            return true;
        }
        if (!observation->gpu_flash_attention_nodes && !observation->requested_flash && tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            observation->requested_flash = tensor;
            return true;
        }
        return false;
    }
    const bool unet = tensor == observation->requested_unet && !observation->gpu_unet_math_nodes;
    const bool flash = tensor == observation->requested_flash && !observation->gpu_flash_attention_nodes;
    if ((!unet && !flash) || !is_gpu_math_tensor(tensor) || (unet && !unet_weight_name(tensor))) return false;
    ++observation->gpu_math_nodes;
    if (unet) ++observation->gpu_unet_math_nodes;
    if (flash) ++observation->gpu_flash_attention_nodes;
    try {
        nlohmann::json event = {{"event", unet ? "gpu_unet_model_math_completed" : "gpu_flash_attention_math_completed"},
            {"op", ggml_op_name(tensor->op)}, {"tensor", ggml_get_name(tensor)},
            {"device", ggml_backend_dev_name(gpu_device)},
            {"scope", "sampled completed witness after backend synchronization; not total GPU operation count"}};
        if (unet) event["weight"] = unet_weight_name(tensor);
        write_event(event);
    } catch (...) { fallback_log("ps5-ai: completed GPU witness structured log unavailable"); }
    observation->stats->peak_cpu_bytes = std::max(observation->stats->peak_cpu_bytes, cpu_peak_bytes());
    return true;
}

static void observe_progress(int step, int steps, float, void* data) noexcept {
    auto* observation = static_cast<InferenceObservation*>(data);
    // Ignore model-loader byte progress and pre-UNet conditioning callbacks.
    // Tiling callbacks arrive after the terminal sampler step and are ignored.
    if (!observation->progress || !observation->gpu_unet_math_nodes || parameter_loading.load() ||
        steps != observation->requested_steps || step <= observation->last_sampler_step || step > steps) return;
    observation->last_sampler_step = step;
    /* Same accepted sampler phase scale as the API; only validated PNG reaches 100%. */
    ps5_ai_native_ui_sampler_progress(static_cast<unsigned>(10000.0*std::min(0.95,0.05+0.90*static_cast<double>(step)/steps)));
    try {
        observation->progress(std::min(0.95, 0.05 + 0.90 * static_cast<double>(step) / steps), observation->user);
    } catch (...) { fallback_log("ps5-ai: progress callback failed"); }
}

static void verify_flash_attention_support() {
    // Metadata-only capability query in the existing shared Vulkan device.
    // No allocation of tensor data, shader execution, second device or probe.
    ggml_context* context = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
    if (!context) throw std::runtime_error("Flash-attention capability metadata allocation failed");
    struct Cleanup { ggml_context* value; ~Cleanup() { ggml_free(value); } } cleanup{context};
    auto* q = ggml_new_tensor_3d(context, GGML_TYPE_F32, 64, 4096, 20);
    auto* k = ggml_new_tensor_3d(context, GGML_TYPE_F16, 64, 4096, 20);
    auto* v = ggml_new_tensor_3d(context, GGML_TYPE_F16, 64, 4096, 20);
    auto* attention = ggml_flash_attn_ext(context, q, k, v, nullptr, 0.125f, 0.f, 0.f);
    ggml_prec_set_acc(attention, GGML_PREC_F32);
    const bool supported = ggml_backend_dev_supports_op(gpu_device, attention);
    write_event({{"event", "native_flash_attention_capability"}, {"supported", supported},
        {"head_size", 64}, {"q_type", "F32"}, {"kv_type", "F16"},
        {"scope", "metadata capability only; completed generation witness still required"}});
    if (!supported) throw std::runtime_error("Native Vulkan UNet flash attention is unsupported; fallback is disabled");
}

static int engine_init_impl(const char* config_path, char* error, size_t error_size) {
    try {
        current_status = {};
        hardware_verified = false;
        gpu_device = nullptr;
        partial_taesd_gpu_decode = false;
        std::snprintf(current_status.backend, sizeof(current_status.backend), "stable-diffusion.cpp/Vulkan");
        if (!config_path) {
            throw std::runtime_error("Engine config path is missing");
        }
        const auto config = bounded_config(config_path);
        engine_config = PS5AIEngineConfig::parse(config.value("engine", nlohmann::json::object()));
        log_path = engine_config.log;
        std::snprintf(current_status.model, sizeof(current_status.model), "%s", engine_config.model.c_str());
        sd_set_log_callback(engine_log, nullptr);
        ps5_ai_flex_stats pool = {};
        ps5_ai_flex_snapshot(&pool);
        if (!pool.mapping_owned || !pool.gpu_access_granted || pool.fault_retained || !pool.log_healthy ||
            !pool.reserved_bytes || pool.free_bytes > pool.reserved_bytes)
            throw std::runtime_error("Owned native Direct data pool is not ready; no flexible-memory or CPU inference fallback");
        ps5_ai_flex_prepare_stats prepared = {};
        ps5_ai_flex_prepare_snapshot(&prepared);
        if (prepared.phase != PS5_AI_FLEX_GPU_READY || prepared.ownership_uncertain ||
            prepared.requested_bytes != engine_config.gpu_pool_bytes ||
            prepared.working_reserve_bytes != engine_config.working_reserve_bytes)
            throw std::runtime_error("Native Direct WAI configured memory plan differs from actual prepared pool admission");
        verify_native_hardware();
        // Backend registration is explicit; no remote/RPC fallback is compiled.
        sd_list_devices(nullptr, 0);
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto* device = ggml_backend_dev_get(i);
            auto* registry = ggml_backend_dev_backend_reg(device);
            if (registry && std::strcmp(ggml_backend_reg_name(registry), "Vulkan") == 0 &&
                ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                gpu_device = device;
                break;
            }
        }
        if (!gpu_device) {
            throw std::runtime_error("No Vulkan GPU backend was found; generation requires observed GPU model math");
        }
        const std::string gpu_name = ggml_backend_dev_name(gpu_device);
        engine_config.resolve_defaults(gpu_name);
        engine_config.validate_actual_pool(gpu_name, pool.reserved_bytes);
        verify_flash_attention_support();
        partial_taesd_gpu_decode = engine_config.partial_taesd_gpu_decode(gpu_name);
        std::snprintf(current_status.backend, sizeof(current_status.backend), "%s",
            "SD.cpp CPU CLIP; Vulkan UNet/TAESD owned Direct imports");
        current_status.gpu_available = 1;
        current_status.available = 1;
        write_event({{"event", "engine_configured"}, {"model", engine_config.model},
                     {"backend", engine_config.backend}, {"params_backend", engine_config.params_backend},
                     {"taesd_path", engine_config.taesd}, {"tae_preview_only", engine_config.tae_preview_only},
                     {"vae_tile_image_pixels", engine_config.vae_tile_size},
                     {"vae_tile_overlap", engine_config.vae_tile_overlap},
                     {"eager_load", engine_config.eager_load},
                      {"lora_apply_mode", "at_runtime"}, {"configured_LoRA_inputs", 0},
                      {"model_mmap_policy", "read-only; existing public mmap provider; no whole-file prefault"},
                     {"flash_attn_mode", engine_config.flash_attn_mode}, {"batched_cfg", engine_config.batched_cfg},
                     {"conditioning_cache_size", engine_config.conditioning_cache_size},
                     {"gpu_observation", "first UNet and first FLASH_ATTN completed witnesses; not total operation counts"},
                     {"partial_gpu_taesd_decode", partial_taesd_gpu_decode},
                     {"gpu_stage", "UNet denoising and learned TAESD decode on Vulkan; CLIP CPU"},
                     {"owned_direct_data_reserved_bytes", pool.reserved_bytes},
                     {"owned_direct_data_free_bytes", pool.free_bytes},
                     {"working_reserve_bytes", engine_config.working_reserve_bytes},
                     {"working_reserve_scope", "accounting admission outside GPU pool; not GPU workspace"},
                     {"max_vram_gib", engine_config.max_vram},
                     {"wai_v17_fp16_unet_reference_bytes", UINT64_C(5134927368)},
                     {"residency_scope", "GPU parameters retained if capacity fits; existing manager may demote to disk; workspace also required"},
                     {"whole_pool_GPU_verified", false},
                     {"direct_heap_report_unchanged", true},
                     {"commit", sd_commit()}, {"gpu_verified", false}});
        return 0;
    } catch (const std::exception& exception) {
        current_status.available = 0;
        std::snprintf(current_status.reason, sizeof(current_status.reason), "%s", exception.what());
        set_error(error, error_size, exception.what());
        return -1;
    }
}

static void engine_status_impl(ps5_ai_engine_status* status) {
    if (status) {
        std::lock_guard<std::mutex> lock(status_mutex);
        *status = current_status;
    }
}

/* One actual >Direct32 data-buffer gate in the same shared Vulkan device.
 * No model, scheduler, second device or inference on the development PC. */
extern "C" int ps5_ai_native_gpu_data_gate(char* error, size_t error_size) noexcept {
    static bool attempted = false;
    if (attempted || !current_status.available || !hardware_verified || !gpu_device || model_context) {
        set_error(error, error_size, "Native_Direct data memory gate requires one ready deferred-model context");
        return -1;
    }
    attempted = true;
    struct Resources {
        ggml_backend_t backend = nullptr;
        ggml_context* context = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        void release() noexcept {
            ps5_ai_ggml_dispatch_stats state = {};
            ps5_ai_ggml_dispatch_snapshot(&state);
            if (state.work_pending || !ps5_ai_ggml_wait_backend("Native_Direct_data_gate_cleanup", 0))
                ps5_ai_ggml_retain("Native_Direct_data_gate_cleanup_not_safe");
            if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (context) { ggml_free(context); context = nullptr; }
            if (backend) { ggml_backend_free(backend); backend = nullptr; }
        }
        ~Resources() noexcept { if (backend || context || buffer) release(); }
    } resources;
    try {
        resources.backend = ggml_backend_vk_init(0);
        if (!resources.backend || !ggml_backend_is_vk(resources.backend) ||
            ggml_backend_get_device(resources.backend) != gpu_device)
            throw std::runtime_error("Native_Direct memory gate did not obtain the shared hardware Vulkan device");
        constexpr size_t elements = size_t(16) * 1024 * 1024;
        constexpr size_t bytes = elements * sizeof(float), chunk_elements = 65536;
        ps5_ai_flex_stats before_pool = {};
        ps5_ai_flex_snapshot(&before_pool);
        ggml_init_params init = {1 << 20, nullptr, true};
        resources.context = ggml_init(init);
        if (!resources.context) throw std::runtime_error("Native_Direct memory gate metadata allocation failed");
        ggml_tensor* input = ggml_new_tensor_1d(resources.context, GGML_TYPE_F32, elements);
        ggml_tensor* output = ggml_scale(resources.context, input, 2.0f);
        ggml_set_name(input, "Native_Direct_owned_Direct_input_64MiB");
        ggml_set_name(output, "Native_Direct_owned_Direct_scale_output_64MiB");
        ggml_set_input(input); ggml_set_output(output);
        if (!ggml_backend_supports_op(resources.backend, output))
            throw std::runtime_error("Vulkan SCALE unsupported; CPU fallback forbidden");
        ggml_cgraph* graph = ggml_new_graph_custom(resources.context, 8, false);
        ggml_build_forward_expand(graph, output);
        resources.buffer = ggml_backend_alloc_ctx_tensors(resources.context, resources.backend);
        if (!resources.buffer || ggml_backend_buffer_get_size(resources.buffer) < bytes * 2)
            throw std::runtime_error("Native_Direct memory gate large input/output buffers were not allocated");
        ps5_ai_ggml_report("Native_Direct_data_gate_buffers elements=%llu tensor_bytes=%llu combined_bytes=%zu CPU_scheduler=0 model_loaded=0 whole_pool_GPU_verified=0",
            (unsigned long long)elements, (unsigned long long)bytes, ggml_backend_buffer_get_size(resources.buffer));
        std::vector<float> chunk(chunk_elements);
        for (size_t offset = 0; offset < elements; offset += chunk_elements) {
            for (size_t i = 0; i < chunk_elements; ++i)
                chunk[i] = float((offset + i) % 4093) * 0.25f - 511.0f;
            ggml_backend_tensor_set(input, chunk.data(), offset * sizeof(float), chunk.size() * sizeof(float));
        }
        ggml_backend_synchronize(resources.backend);
        if (!ps5_ai_ggml_wait_backend("Native_Direct_data_gate_before_graph", 0))
            ps5_ai_ggml_retain("Native_Direct_data_gate_upload_not_quiescent");
        ps5_ai_ggml_dispatch_stats before = {}, after = {};
        ps5_ai_ggml_dispatch_snapshot(&before);
        if (before.work_pending) ps5_ai_ggml_retain("Native_Direct_data_gate_pre_graph_pending");
        const ggml_status status = ggml_backend_graph_compute(resources.backend, graph);
        ggml_backend_synchronize(resources.backend);
        if (status != GGML_STATUS_SUCCESS || !ps5_ai_ggml_wait_backend("Native_Direct_data_gate_after_graph", 1))
            ps5_ai_ggml_retain("Native_Direct_data_gate_compute_completion_unknown");
        ps5_ai_ggml_dispatch_snapshot(&after);
        if (after.work_pending || !after.hardware_verified || after.dispatches <= before.dispatches ||
            after.submits <= before.submits || after.fenced_completions <= before.fenced_completions)
            throw std::runtime_error("Native_Direct memory gate lacks actual hardware dispatch/submit/fence proof");
        uint64_t mismatches = 0, checksum = UINT64_C(1469598103934665603), expected_checksum = checksum;
        for (size_t offset = 0; offset < elements; offset += chunk_elements) {
            ggml_backend_tensor_get(output, chunk.data(), offset * sizeof(float), chunk.size() * sizeof(float));
            for (size_t i = 0; i < chunk_elements; ++i) {
                const float expected = float((offset + i) % 4093) * 0.5f - 1022.0f;
                uint32_t actual_bits, expected_bits;
                std::memcpy(&actual_bits, &chunk[i], sizeof(actual_bits));
                std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
                mismatches += chunk[i] != expected;
                checksum = (checksum ^ actual_bits) * UINT64_C(1099511628211);
                expected_checksum = (expected_checksum ^ expected_bits) * UINT64_C(1099511628211);
            }
        }
        ggml_backend_synchronize(resources.backend);
        if (!ps5_ai_ggml_wait_backend("Native_Direct_data_gate_after_readback", 1))
            ps5_ai_ggml_retain("Native_Direct_data_gate_readback_not_quiescent");
        ps5_ai_ggml_report("Native_Direct_data_gate_readback compared_values=%llu mismatches=%llu checksum=0x%016llx expected_checksum=0x%016llx dispatch_delta=%llu submit_delta=%llu fence_delta=%llu whole_pool_GPU_verified=0",
            (unsigned long long)elements, (unsigned long long)mismatches,
            (unsigned long long)checksum, (unsigned long long)expected_checksum,
            (unsigned long long)(after.dispatches - before.dispatches),
            (unsigned long long)(after.submits - before.submits),
            (unsigned long long)(after.fenced_completions - before.fenced_completions));
        if (mismatches || checksum != expected_checksum || !ps5_ai_ggml_log_healthy())
            throw std::runtime_error("Native_Direct large Direct GPU buffer readback mismatch or missing durable log");
        resources.release();
        ps5_ai_flex_stats final_pool = {};
        ps5_ai_flex_snapshot(&final_pool);
        ps5_ai_ggml_report("Native_Direct_data_gate_completed success=1 verified_tensor_bytes=67108864 before_used=%llu after_used=%llu peak_used=%llu reserved=%llu active_imports=%llu scope=single_SCALE_graph_not_model_success",
            (unsigned long long)before_pool.used_bytes, (unsigned long long)final_pool.used_bytes,
            (unsigned long long)final_pool.peak_used_bytes, (unsigned long long)final_pool.reserved_bytes,
            (unsigned long long)final_pool.active_imports);
        if (!ps5_ai_ggml_log_healthy() || final_pool.fault_retained)
            ps5_ai_ggml_retain("Native_Direct_data_gate_final_receipt_failed");
        return 0;
    } catch (const std::exception& exception) {
        set_error(error, error_size, exception.what());
    } catch (...) {
        set_error(error, error_size, "Native_Direct memory gate failed with unknown CPU exception");
    }
    std::lock_guard<std::mutex> lock(status_mutex);
    current_status.available = 0;
    std::snprintf(current_status.reason, sizeof(current_status.reason), "%s", "Native_Direct large Direct GPU buffer gate failed");
    return -1;
}

static int engine_generate_impl(const ps5_ai_request* request, const char* output_path,
                                       ps5_ai_stats* stats, ps5_ai_progress_cb progress, void* user,
                                       char* error, size_t error_size) {
    if (!request || !output_path || !stats || !current_status.available || !gpu_device || !hardware_verified) {
        set_error(error, error_size, "GPU engine is unavailable or generation arguments are invalid");
        return -1;
    }
    *stats = {};
    InferenceObservation observation = {progress, user, stats, 0, 0, 0, nullptr, nullptr, request->steps, 0};
    parameter_loading.store(false);
    sd_image_t* images = nullptr;
    int image_count = 0;
    struct GenerationCleanup {
        sd_image_t** images;
        int* count;
        ~GenerationCleanup() noexcept {
            try {
                sd_set_backend_eval_callback(nullptr, nullptr);
                sd_set_progress_callback(nullptr, nullptr);
                if (*images) free_sd_images(*images, *count);
            } catch (...) {
                fallback_log("ps5-ai: generation cleanup failed; reload title");
            }
        }
    } cleanup = {&images, &image_count};
    const auto start = std::chrono::steady_clock::now();
    int result = -1;
    try {
        if (!model_context) {
            sd_ctx_params_t context_params;
            sd_ctx_params_init(&context_params);
            engine_config.apply_context(context_params);
            ps5_ai_native_ui_phase(PS5_AI_STATUS_LOADING);
            ps5_ai_native_ui_log("Loading WAI-Illustrious-SDXL v17.0 checkpoint");
            model_context = new_sd_ctx(&context_params);
            if (!model_context) {
                throw std::runtime_error("Checkpoint load failed; see engine-native-wai.jsonl");
            }
            {
                std::lock_guard<std::mutex> lock(status_mutex);
                current_status.model_loaded = 1;
            }
            write_event({{"event", "model_loaded"}, {"path", engine_config.model},
                         {"architecture", sd_get_model_version_name(model_context)}});
        }
        sd_img_gen_params_t params;
        sd_img_gen_params_init(&params);
        params.prompt = request->prompt;
        params.negative_prompt = request->negative_prompt;
        params.width = request->width;
        params.height = request->height;
        params.seed = request->seed;
        params.batch_count = 1;
        params.sample_params.sample_steps = request->steps;
        params.sample_params.guidance.txt_cfg = static_cast<float>(request->cfg_scale);
        params.sample_params.sample_method = str_to_sample_method(request->sampler);
        if (params.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            throw std::runtime_error("Unsupported sampler");
        }
        params.sample_params.scheduler = sd_get_default_scheduler(model_context, params.sample_params.sample_method);
        engine_config.apply_image(params);
        sd_set_backend_eval_callback(observe_graph, &observation);
        sd_set_progress_callback(observe_progress, &observation);
        if (progress) {
            progress(0.05, user);
        }
        ps5_ai_native_ui_phase(PS5_AI_STATUS_GENERATING);
        if (!generate_image(model_context, &params, &images, &image_count) || image_count != 1 || !images || !images[0].data) {
            throw std::runtime_error("Image generation failed; see engine-native-wai.jsonl");
        }
        stats->gpu_used = observation.gpu_unet_math_nodes > 0 ? 1 : 0;
        if (!stats->gpu_used || !observation.gpu_flash_attention_nodes) {
            throw std::runtime_error("Image rejected: required completed native UNet/flash-attention witness is missing");
        }
        const auto& image = images[0];
        const std::string temporary_path = std::string(output_path) + ".partial";
        if (!stbi_write_png(temporary_path.c_str(), image.width, image.height, image.channel,
                            image.data, image.width * image.channel)) {
            throw std::runtime_error("PNG write failed");
        }
        if (std::rename(temporary_path.c_str(), output_path) != 0) {
            std::remove(temporary_path.c_str());
            throw std::runtime_error("PNG rename failed");
        }
        {
            std::lock_guard<std::mutex> lock(status_mutex);
            current_status.gpu_verified = 1;
            current_status.reason[0] = '\0';
        }
        result = 0;
    } catch (const std::exception& exception) {
        set_error(error, error_size, exception.what());
        {
            std::lock_guard<std::mutex> lock(status_mutex);
            std::snprintf(current_status.reason, sizeof(current_status.reason), "%s", exception.what());
        }
        try {
            write_event({{"event", "generation_failed"}, {"error", exception.what()}});
        } catch (...) {
            fallback_log(exception.what());
        }
    }
    sd_set_backend_eval_callback(nullptr, nullptr);
    sd_set_progress_callback(nullptr, nullptr);
    if (images) {
        free_sd_images(images, image_count);
        images = nullptr;
    }
    stats->gpu_used = observation.gpu_unet_math_nodes > 0 ? 1 : 0;
    stats->generation_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    stats->peak_cpu_bytes = std::max(stats->peak_cpu_bytes, cpu_peak_bytes());
    ps5_ai_flex_stats pool = {};
    ps5_ai_flex_snapshot(&pool);
    write_event({{"event", "generation_result"}, {"success", result == 0},
                 {"gpu_used", stats->gpu_used != 0}, {"gpu_math_nodes", observation.gpu_math_nodes},
                 {"gpu_unet_math_nodes", observation.gpu_unet_math_nodes},
                 {"gpu_flash_attention_nodes", observation.gpu_flash_attention_nodes},
                 {"gpu_observation", "sampled completed witnesses; not total GPU operation counts"},
                 {"unet_witness_requested", observation.requested_unet != nullptr},
                 {"flash_witness_requested", observation.requested_flash != nullptr},
                 {"owned_direct_data_reserved_bytes", pool.reserved_bytes},
                 {"owned_direct_data_used_bytes", pool.used_bytes},
                 {"owned_direct_data_peak_used_bytes", pool.peak_used_bytes},
                 {"owned_direct_data_free_bytes", pool.free_bytes},
                 {"owned_direct_data_fault_retained", pool.fault_retained != 0},
                 {"partial_gpu_taesd_decode", partial_taesd_gpu_decode},
                 {"backend", engine_config.backend},
                 {"seconds_including_initial_model_load", stats->generation_seconds},
                 {"process_peak_rss_bytes", stats->peak_cpu_bytes ? nlohmann::json(stats->peak_cpu_bytes) : nlohmann::json(nullptr)},
                 {"peak_gpu_bytes", nullptr}, {"gpu_memory_measurement", "unavailable"}});
    if (result == 0 && progress) {
        progress(1.0, user);
    }
    return result;
}

static void engine_shutdown_impl() {
    if (model_context) {
        free_sd_ctx(model_context);
        model_context = nullptr;
    }
    std::lock_guard<std::mutex> lock(status_mutex);
    current_status.model_loaded = 0;
    current_status.available = 0;
}

extern "C" int ps5_ai_engine_init(const char* config_path, char* error, size_t error_size) {
    try {
        return engine_init_impl(config_path, error, error_size);
    } catch (const std::exception& exception) {
        set_error(error, error_size, exception.what());
        fallback_log(exception.what());
    } catch (...) {
        set_error(error, error_size, "Unknown engine initialization failure");
        fallback_log("ps5-ai: unknown engine initialization failure");
    }
    return -1;
}

extern "C" void ps5_ai_engine_get_status(ps5_ai_engine_status* status) {
    try {
        engine_status_impl(status);
    } catch (...) {
        if (status) {
            *status = {};
            std::snprintf(status->reason, sizeof(status->reason), "Engine status unavailable");
        }
        fallback_log("ps5-ai: engine status unavailable");
    }
}

extern "C" int ps5_ai_engine_generate(const ps5_ai_request* request, const char* output_path,
                                       ps5_ai_stats* stats, ps5_ai_progress_cb progress, void* user,
                                       char* error, size_t error_size) {
    try {
        return engine_generate_impl(request, output_path, stats, progress, user, error, error_size);
    } catch (const std::exception& exception) {
        set_error(error, error_size, exception.what());
        fallback_log(exception.what());
    } catch (...) {
        set_error(error, error_size, "Unknown engine generation failure; reload title");
        fallback_log("ps5-ai: unknown generation failure; reload title");
    }
    return -1;
}

extern "C" void ps5_ai_engine_shutdown(void) {
    try {
        engine_shutdown_impl();
    } catch (...) {
        fallback_log("ps5-ai: shutdown failed; reload title");
    }
}

extern "C" int ps5_ai_engine_shutdown_checked(char* error, size_t error_size) {
    try {
        sd_set_backend_eval_callback(nullptr, nullptr);
        sd_set_progress_callback(nullptr, nullptr);
        engine_shutdown_impl();
        return 0;
    } catch (const std::exception& exception) {
        set_error(error, error_size, exception.what());
        fallback_log("ps5-ai: checked shutdown failed");
    } catch (...) {
        set_error(error, error_size, "Unknown engine shutdown failure");
        fallback_log("ps5-ai: checked shutdown failed");
    }
    return -1;
}
