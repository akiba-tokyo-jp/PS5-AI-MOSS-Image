#include "phase_profile.h"
#include "backend_precision_log.hpp"
#if defined(GGML_MAX_NAME) && GGML_MAX_NAME != 160
#error Native Direct MOSS-Image must use the exact linked SD/GGML tensor ABI
#endif
#ifndef GGML_MAX_NAME
#define GGML_MAX_NAME 160
#endif
#include "engine.h"
#include "reference_image.h"
#include "precision_mode.h"
#include <cmath>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include "stable-diffusion.h"
#include "json.hpp"
#include "engine_config.h"
#include "moss_runtime_admission.h"
#include "moss_trial_contract.hpp"
#include "moss_spectrum_trial.h"
#if PS5_AI_MOSS_SPECTRUM_TRIAL
#include "runtime/moss_spectrum.hpp"
#endif
#include "moss_accumulation_policy.h"
#include "moss_gpu_profile.h"
#include "moss_q4_bk64.h"
#include "moss_bounded_weight_upload.h"
#include "moss_ffn_chunk_trial.h"
#include "moss_ffn_chunk_observation.hpp"
#include "moss_lora_contract.hpp"
#include "moss_lora_receipt.h"
#include "v3_shutdown.h"
#include "pool.h"
#include "cpu_clip_pool_buffer.hpp"
#include "probe.h"
#include "memory_plan.h"
#include "ui_hooks.h"
#include "journal_metrics.h"
#include "log_policy.h"
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
#include "init_stage_impl.hpp"

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
static std::atomic<uint64_t> cpu_clip_pool_live_bytes{0};
static std::atomic<uint64_t> cpu_clip_pool_peak_bytes{0};
static std::atomic<unsigned> moss_lora_loaded{0}, moss_lora_matched{0}, moss_lora_used{0}, moss_lora_skipped{0};
static std::atomic<uint32_t> moss_lora_scale_bits{0};
static std::atomic<bool> moss_lora_configured{false}, moss_lora_verified{false}, moss_lora_invalid{false};
#if PS5_AI_MOSS_TURBO_TRIAL
static std::atomic<bool> moss_turbo_small_image_completed{false};
#endif
static nlohmann::json moss_effective_lora_identity() {
#if PS5_AI_MOSS_TURBO_TRIAL
    return {{"schema",PS5_AI_MOSS_TURBO_ADAPTER_SCHEMA},{"adapter",PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL?"NSFW0.7+Turbo1.0 lossy F16 storage":"NSFW0.7+Turbo1.0 composed F32"},
        {"lossy_storage",bool(PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL)},
        {"source_composed_F32_sha256",PS5_AI_MOSS_LORA_F16_STORAGE_SOURCE_SHA256},
        {"path",engine_config.moss_lora_path},{"sha256",engine_config.moss_lora_sha256},{"rank",144},
        {"sha256_observation","configured whole-file identity; external preparation/staging rehash required, not a native runtime rehash"},
        {"runtime_multiplier",1.0},{"base_merged",false},
        {"configured",moss_lora_configured.load()},{"strength",1.0},
        {"loaded_tensors",moss_lora_loaded.load()},{"matched_tensors",moss_lora_matched.load()},
        {"used_tensors",moss_lora_used.load()},{"skipped_tensors",moss_lora_skipped.load()},
        {"application_verified",moss_lora_verified.load()},
        {"components",nlohmann::json::array({
            {{"adapter","axiomofmind/Moss-image-nsfw-uncensored-lora"},{"F32_sha256",PS5_AI_MOSS_TURBO_NSFW_SHA256},{"rank",16},{"strength",0.7}},
            {{"adapter","akhaliq/MOSS-Image-SFT-4-steps"},{"sha256",PS5_AI_MOSS_TURBO_SOURCE_SHA256},{"rank",128},{"strength",1.0}}})}};
#else
    return {{"adapter","axiomofmind/Moss-image-nsfw-uncensored-lora"},{"path",engine_config.moss_lora_path},
            {"rank",16},{"strength",engine_config.moss_lora_strength}};
#endif
}
#if PS5_AI_MOSS_TURBO_TRIAL
extern "C" int ps5_ai_moss_turbo_identity_json(char* out,size_t capacity) {
    if(!out||!capacity)return -1;
    try {
        const std::string body=moss_effective_lora_identity().dump();
        if(body.size()>=capacity)return -1;
        std::memcpy(out,body.c_str(),body.size()+1);return 0;
    }catch(...){out[0]=0;return -1;}
}
#endif
extern "C" void ps5_ai_moss_lora_contract_record(unsigned loaded, unsigned matched, unsigned used, unsigned skipped, float strength) {
    if (loaded != 504 || matched != 504 || used > 504 || skipped || !std::isfinite(strength) ||
        strength != static_cast<float>(engine_config.moss_lora_strength)) moss_lora_invalid = true;
    moss_lora_loaded = loaded; moss_lora_matched = matched; moss_lora_used = used; moss_lora_skipped = skipped;
    if (!std::isfinite(strength)) strength = 0.f;
    uint32_t bits; std::memcpy(&bits, &strength, sizeof(bits)); moss_lora_scale_bits = bits;
}
extern "C" int ps5_ai_moss_lora_snapshot(struct ps5_ai_moss_lora_receipt *out) {
    if (!out) return -1;
    *out = {};
    out->loaded = moss_lora_loaded.load(); out->matched = moss_lora_matched.load();
    out->used = moss_lora_used.load(); out->skipped = moss_lora_skipped.load();
    uint32_t bits = moss_lora_scale_bits.load(); std::memcpy(&out->strength, &bits, sizeof(bits));
    out->configured = moss_lora_configured.load(); out->application_verified = moss_lora_verified.load();
    return 0;
}
extern "C" uint64_t ps5_ai_cpu_clip_shared_direct_live_bytes(void) {
    return cpu_clip_pool_live_bytes.load();
}
extern "C" void ps5_ai_native_config_mark(unsigned,const char*);
extern "C" void ps5_ai_native_locale_snapshot(uint64_t[11]);
extern "C" void ps5_ai_native_capacity_snapshot(const char*);

static void set_error(char* error, size_t size, const char* message) {
    if (error && size) {
        std::snprintf(error, size, "%s", message);
    }
}

static nlohmann::json bounded_config(const char* path) {
    ps5_ai_native_config_mark(2,"config_FD_open_before");
    if (!path) throw std::runtime_error("Native Direct MOSS-Image config path is missing");
    const size_t length = strnlen(path, 4097);
    if (!length || length > 4096) throw std::runtime_error("Native Direct MOSS-Image config path is too long");
    const int descriptor = open(path, O_RDONLY | O_NOFOLLOW);
    if (descriptor < 0) throw std::runtime_error("Native Direct MOSS-Image config could not be opened read-only");
    ps5_ai_native_config_mark(3,"config_FD_open_returned");
    struct Close { int fd; ~Close() { close(fd); } } owned = {descriptor};
    struct stat before = {}, after = {};
    ps5_ai_native_config_mark(4,"config_FD_fstat_before");
    if (fstat(descriptor, &before) || !S_ISREG(before.st_mode) || before.st_size <= 0 ||
        static_cast<uint64_t>(before.st_size) > PS5_AI_MEMORY_PLAN_MAX_CONFIG_BYTES)
        throw std::runtime_error("Native Direct MOSS-Image config must be a regular file of1..65536 bytes");
    ps5_ai_ggml_report("NATIVE_CONFIG FD_metadata regular=1 bytes=%llu reported_UID_not_ownership_proof=1",static_cast<unsigned long long>(before.st_size));
    ps5_ai_native_config_mark(5,"config_FD_fstat_returned");
    std::string bytes(static_cast<size_t>(before.st_size), '\0');
    ps5_ai_native_config_mark(6,"config_FD_read_before");
    size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t count = read(descriptor, bytes.data() + done, bytes.size() - done);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("Native Direct MOSS-Image config read was incomplete");
        done += static_cast<size_t>(count);
    }
    ps5_ai_native_config_mark(7,"config_FD_read_complete");
    char extra;
    ssize_t eof;
    do { eof = read(descriptor, &extra, 1); } while (eof < 0 && errno == EINTR);
    if (eof != 0 || fstat(descriptor, &after) || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size ||
        before.st_mtime != after.st_mtime || before.st_ctime != after.st_ctime)
        throw std::runtime_error("Native Direct MOSS-Image config changed while being read");
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
    (void)PS5AIEngineConfig::engine_object(config);
    return config;
}

extern "C" int ps5_ai_v12_read_memory_plan(const char* path, ps5_ai_memory_plan* out,
                                          char* error, size_t error_size) noexcept {
    if (out) *out = {};
#if !PS5_AI_MOSS_CAN_RUN_ADMISSION
    (void)path;
    {
        std::lock_guard<std::mutex> lock(status_mutex);
        current_status.available = 0;
        current_status.gpu_available = 0;
        std::snprintf(current_status.backend, sizeof(current_status.backend), "%s", "SD.cpp MOSS preparation; native admission pending");
        std::snprintf(current_status.reason, sizeof(current_status.reason), "%s", PS5_AI_MOSS_UNADMITTED_REASON);
    }
    set_error(error, error_size, PS5_AI_MOSS_UNADMITTED_REASON);
    return -1;
#endif
    try {
        if (!out) throw std::runtime_error("Native Direct MOSS-Image memory-plan output is missing");
        const auto config = bounded_config(path);
        if(!ps5_ai_native_journal_configure_debug(ps5_ai_log_policy_debug(config)))
            throw std::runtime_error("Native journal policy must be fixed once before AGC initialization");
        ps5_ai_native_config_mark(13,"memory_plan_fields_before");
        const auto plan = PS5AIEngineConfig::memory_plan(PS5AIEngineConfig::engine_object(config));
        *out = plan;
        ps5_ai_native_config_mark(14,"memory_plan_fields_returned");
        return 0;
    } catch (const std::exception& exception) {
        ps5_ai_native_config_mark(15,"memory_plan_exception_caught");
        set_error(error, error_size, exception.what());
    } catch (...) {
        ps5_ai_native_config_mark(15,"memory_plan_unknown_exception_caught");
        set_error(error, error_size, "Native Direct MOSS-Image memory-plan config failed");
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
    PS5AIProfileScope timing(28);
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


#if PS5_AI_MOSS_SPECTRUM_TRIAL
// The existing single model worker owns this CPU-only per-generation receipt.
static sd_moss_spectrum::Receipt moss_spectrum_last{};
static std::atomic<unsigned> moss_spectrum_completions{0};
extern "C" void ps5_ai_moss_spectrum_complete(const sd_moss_spectrum::Receipt* r) {
    if(!r || r->initialized!=1 || r->completed_steps!=50 || r->real_steps!=30 ||
       r->forecast_steps!=20 || r->anchor_count!=6 || r->errors ||
       r->history_payload_bytes>sd_moss_spectrum::State::max_history_bytes ||
       r->peak_payload_bytes>sd_moss_spectrum::State::max_peak_payload_bytes ||
       moss_spectrum_completions.load(std::memory_order_acquire)!=0)
        throw std::runtime_error("MOSS Spectrum core completion receipt refused");
    moss_spectrum_last=*r;
    moss_spectrum_completions.store(1,std::memory_order_release);
    write_event({{"event","moss_spectrum_completed"},{"policy",PS5_AI_MOSS_SPECTRUM_POLICY},
        {"sampler_steps",r->completed_steps},{"actual_cfg_steps",r->real_steps},
        {"forecast_cfg_steps",r->forecast_steps},{"actual_dit_forwards",r->real_steps*2},
        {"history_host_payload_bytes",r->history_payload_bytes},{"peak_host_payload_bytes",r->peak_payload_bytes},
        {"memory_scope","F32 vector payload only, not physical RAM or Direct/VRAM"},
        {"forecast_device","PS5 CPU"},{"forecast_space","post_CFG_denoised_latent"},
        {"nonfinite_or_state_errors",r->errors},{"NaN_clamping",false}});
}
#endif
static nlohmann::json moss_spectrum_json() {
#if PS5_AI_MOSS_SPECTRUM_TRIAL
    const auto r=moss_spectrum_last;
    return {{"enabled",true},{"policy",PS5_AI_MOSS_SPECTRUM_POLICY},
        {"completion_events",moss_spectrum_completions.load(std::memory_order_acquire)},
        {"sampler_steps",r.completed_steps},{"actual_cfg_steps",r.real_steps},
        {"forecast_cfg_steps",r.forecast_steps},{"actual_dit_forwards",r.real_steps*2},
        {"history_host_payload_bytes",r.history_payload_bytes},{"peak_host_payload_bytes",r.peak_payload_bytes},
        {"memory_scope","F32 vector payload; existing host reserve/actual process peak remain separate"},
        {"nonfinite_or_state_errors",r.errors},{"NaN_clamping",false}};
#else
    return {{"enabled",false},{"policy",PS5_AI_MOSS_SPECTRUM_POLICY}};
#endif
}

static std::atomic<bool> phase_profile_active{false};
struct PhaseCounter{std::atomic<uint64_t> ns{0},calls{0},bytes{0},invalid{0};};
static PhaseCounter phase_counters[PS5_AI_PROFILE_METRICS];
static thread_local unsigned phase_runner_group=2;
extern "C" unsigned ps5_ai_profile_set_runner(unsigned group){const unsigned previous=phase_runner_group;phase_runner_group=group;return previous;}
extern "C" unsigned ps5_ai_profile_get_runner(void){return phase_runner_group;}
extern "C" int ps5_ai_profile_enabled(void){return phase_profile_active.load(std::memory_order_relaxed);}
extern "C" void ps5_ai_profile_add(unsigned id,uint64_t ns,uint64_t bytes,int valid){
    if(id>=PS5_AI_PROFILE_METRICS || !ps5_ai_profile_enabled())return;
    auto& c=phase_counters[id];c.ns.fetch_add(ns,std::memory_order_relaxed);c.calls.fetch_add(1,std::memory_order_relaxed);c.bytes.fetch_add(bytes,std::memory_order_relaxed);if(!valid)c.invalid.fetch_add(1,std::memory_order_relaxed);
}
static void phase_profile_begin(){for(auto& c:phase_counters){c.ns=0;c.calls=0;c.bytes=0;c.invalid=0;}phase_profile_active=true;}
static void phase_profile_report(){
    phase_profile_active=false;
    const char* groups[]={"dit","controlnet_unused","other_runners"};
    const char* phases[]={"graph_total","weight_prepare_including_capacity","workspace_prepare_allocate","activation_upload","graph_call_including_internal_wait","explicit_workspace_wait","output_capture_readback","cleanup_including_wait"};
    const char* extras[]={"weight_tensor_copy_set","weight_transfer_explicit_wait","startup_log_fsync","Direct_pool_log_fsync","engine_structured_log","startup_stderr_mirror","backend_quiescence_call_including_logging","backend_idle_poll_without_logging",
        "dit.vulkan_submit_driver_call","dit.vulkan_fence_wait_driver_call","dit.vulkan_dispatch_record_call","dit.vulkan_bind_pipeline_record_call","dit.vulkan_backend_graph_call",
        "other_runners.vulkan_submit_driver_call","other_runners.vulkan_fence_wait_driver_call","other_runners.vulkan_dispatch_record_call","other_runners.vulkan_bind_pipeline_record_call","other_runners.vulkan_backend_graph_call"};
    nlohmann::json metrics=nlohmann::json::object();
    for(unsigned i=0;i<PS5_AI_PROFILE_METRICS;i++){
        auto& c=phase_counters[i];std::string name=i<24?std::string(groups[i/8])+"."+phases[i%8]:extras[i-24];
        metrics[name]={{"seconds",c.ns.load()/1e9},{"calls",c.calls.load()},{"bytes",c.bytes.load()},{"invalid_clock_samples",c.invalid.load()}};
    }
    write_event({{"event","native_phase_profile"},{"schema",2},{"CPU_wall_time",true},{"pure_GPU_timestamp",false},{"nested_times_do_not_sum",true},{"scope","engine generation only; log summary excluded; Vulkan wait includes pending GPU execution"},{"metrics",metrics}});
}

static nlohmann::json moss_gpu_profile_json(const ps5_ai_moss_gpu_profile_report& r) {
    if(!r.enabled)return nullptr;
    nlohmann::json graphs=nlohmann::json::array();
    for(unsigned i=0;i<PS5_AI_MOSS_GPU_PROFILE_GRAPHS && i<r.graphs_sampled;i++) {
        const auto& g=r.graphs[i];nlohmann::json classes=nlohmann::json::object();
        for(unsigned c=0;c<PS5_AI_MOSS_GPU_PROFILE_CLASSES;c++) {
            const auto& s=g.classes[c];
            classes[ps5_ai_moss_gpu_class_name(c)]={{"intervals",s.intervals},{"ticks",s.ticks},
                {"interval_seconds",double(s.ticks)*r.timestamp_period_ns*1e-9},
                {"minimum_ticks",s.minimum_ticks},{"maximum_ticks",s.maximum_ticks}};
#if PS5_AI_MOSS_DIT_GPU_PROFILE
            nlohmann::json shaders=nlohmann::json::object();
            for(unsigned k=0;k<PS5_AI_MOSS_GPU_SHADER_CLASSES;k++){
                const auto& t=g.shader_classes[c][k];if(!t.intervals)continue;
                shaders[ps5_ai_moss_gpu_shader_name(k)]={{"intervals",t.intervals},{"ticks",t.ticks},
                    {"interval_seconds",double(t.ticks)*r.timestamp_period_ns*1e-9},
                    {"minimum_ticks",t.minimum_ticks},{"maximum_ticks",t.maximum_ticks}};
            }
            classes[ps5_ai_moss_gpu_class_name(c)]["shader_groups"]=shaders;
#endif
        }
        graphs.push_back({{"ordinal",g.ordinal},{"nodes",g.nodes},{"recorded_nodes",g.recorded_nodes},
            {"complete_main_graph",bool(g.complete_main_graph)},{"canonical_layer_count",g.canonical_layer_count},
            {"canonical_layer_mask",g.canonical_layer_mask},
            {"q4_main_nodes",{g.q4_main_nodes[0],g.q4_main_nodes[1],g.q4_main_nodes[2],g.q4_main_nodes[3]}},
            {"lora_down_nodes",g.lora_down_nodes},{"lora_up_nodes",g.lora_up_nodes},{"tokens",g.tokens},
            {"intervals",g.intervals},{"queries_written",g.queries_written},
            {"queries_available",g.queries_available},{"zero_intervals",g.zero_intervals},
            {"invalid_intervals",g.invalid_intervals},{"overflow",g.overflow},{"dispatches",g.dispatches},
            {"submits",g.submits},{"fences",g.fences},{"CPU_wall_seconds",double(g.cpu_wall_ns)*1e-9},
            {"classes",classes}});
    }
    return {{"schema",r.schema},{"enabled",bool(r.enabled)},{"valid",bool(r.valid)},
        {"dit_mask15_rank144_diagnostic",bool(PS5_AI_MOSS_DIT_GPU_PROFILE)},
        {"startup_gate",bool(r.startup_gate)},{"active",bool(r.active)},{"gate_verified",bool(r.gate_verified)},
        {"generation_ordinal",r.generation_ordinal},{"graphs_seen",r.graphs_seen},
        {"full_graphs_seen",r.full_graphs_seen},{"skipped_partial_views",r.skipped_partial_views},
        {"graphs_sampled",r.graphs_sampled},{"unprofiled_graphs",r.unprofiled_graphs},
        {"errors",r.errors},{"capacity_errors",r.capacity_errors},{"unavailable_queries",r.unavailable_queries},
        {"invalid_intervals",r.invalid_intervals},{"result_calls",r.result_calls},
        {"cache_flush_calls",r.cache_flush_calls},{"query_reset_commands",r.query_reset_commands},
        {"query_pool_queries",r.query_pool_queries},
        {"query_pool_bytes",r.query_pool_bytes},{"cpu_fixed_bytes",r.cpu_fixed_bytes},
        {"device_generation",r.device_generation},{"timestamp_valid_bits",r.timestamp_valid_bits},
        {"timestamp_period_ns",r.timestamp_period_ns},{"graphs",graphs},
        {"scope","bounded sampled dispatch timestamp intervals attributed to graph node classes, including conversion/reduction dispatches; inserted timestamps may perturb scheduling; not an exclusive whole-model decomposition"},
        {"extra_CPU_waits",0},{"extra_submissions",0},{"extrapolated_whole_generation_seconds",nullptr}};
}

extern "C" ggml_backend_buffer_t ps5_ai_cpu_clip_params_buffer(size_t size) {
    return ps5_ai_cpu_clip_pool_buffer(size);
}

extern "C" void ps5_ai_cpu_clip_pool_event(const char* action, uint64_t bytes, int success) noexcept {
    const uint64_t rounded = (bytes + PS5_AI_FLEX_PAGE_BYTES - 1) & ~(PS5_AI_FLEX_PAGE_BYTES - 1);
    if (success && !std::strcmp(action, "allocated")) {
        const uint64_t live = cpu_clip_pool_live_bytes.fetch_add(rounded) + rounded;
        uint64_t peak = cpu_clip_pool_peak_bytes.load();
        while (peak < live && !cpu_clip_pool_peak_bytes.compare_exchange_weak(peak, live)) {}
    } else if (success && !std::strcmp(action, "free")) {
        const uint64_t before = cpu_clip_pool_live_bytes.load();
        if (before < rounded) ps5_ai_flex_retain_fault("CPU_CLIP_pool_counter_underflow");
        else cpu_clip_pool_live_bytes.fetch_sub(rounded);
    }
    try { write_event({{"event","cpu_clip_shared_direct_params"},{"action",action},{"bytes",bytes},
        {"success",success!=0},{"live_bytes",cpu_clip_pool_live_bytes.load()},
        {"peak_bytes",cpu_clip_pool_peak_bytes.load()},{"compute_backend","CPU"},
        {"physical_allocation_calls",0},{"Vulkan_import",false}}); }
    catch (...) { fallback_log("CPU CLIP pool accounting log unavailable"); }
}

static void engine_log(enum sd_log_level_t level, const char* text, void*) noexcept {
    // The shared progress callback also receives tensor byte-loading progress.
    // These exact loader messages identify that phase without changing I/O.
    if (text && std::strncmp(text, "loading ", 8) == 0 && std::strstr(text, " tensors from "))
        parameter_loading.store(true);
    if (text && std::strncmp(text, "loading tensors completed", 25) == 0)
        parameter_loading.store(false);
    // Preserve loading/progress state, warnings, errors and INFO phase times.
    // Detailed tensor/token diagnostics and full split prompts are debug-only.
    if(!ps5_ai_native_journal_debug_enabled() &&
       (level==SD_LOG_DEBUG || level==SD_LOG_VERBOSE ||
        (text && std::strncmp(text,"split prompt ",13)==0)))return;
    ps5_ai_native_ui_log(text);
    try {
        write_event({{"event", "sdcpp_log"}, {"level", static_cast<int>(level)}, {"text", text ? text : ""}});
    } catch (...) {
        fallback_log(text ? text : "ps5-ai: engine log unavailable");
    }
}

static std::atomic<uint64_t> precision_outputs{0}, precision_values{0}, precision_nonfinite{0}, precision_dit_outputs{0}, precision_vae_outputs{0};
static bool precision_announced=false,precision_fp16=false;
static std::atomic<unsigned> perf_submit_multiplier{1},perf_max_nodes{100},perf_limited_fusion{0};
static std::atomic<uint64_t> perf_fused_selected{0};
static std::atomic<uint64_t> q4_matrix_f32acc{0},q4_matrix_f16acc{0},q4_matrix_invalid{0};
extern "C" void ps5_ai_vk_quant_accumulation_selected(unsigned lhs_type,unsigned f16acc) {
    if(f16acc>1){q4_matrix_invalid.fetch_add(1,std::memory_order_relaxed);return;}
    if(lhs_type!=unsigned(GGML_TYPE_Q4_K))return;
    (f16acc?q4_matrix_f16acc:q4_matrix_f32acc).fetch_add(1,std::memory_order_relaxed);
}
extern "C" void ps5_ai_moss_quant_snapshot(struct ps5_ai_moss_quant_receipt *out) {
    if(out)*out={q4_matrix_f32acc.load(),q4_matrix_f16acc.load(),q4_matrix_invalid.load()};
}
extern "C" void ps5_ai_vk_optimization_profile(unsigned multiplier,unsigned max_nodes,unsigned limited_fusion,
                                              uint64_t base_cap,uint64_t ramped_cap) {
    perf_submit_multiplier=multiplier;perf_max_nodes=max_nodes;perf_limited_fusion=limited_fusion;
    write_event({{"event","native_moss_optimization_profile"},{"performance_trial",bool(PS5_AI_MOSS_PERFORMANCE_TRIAL)},
        {"variant",PS5_AI_MOSS_PERF_VARIANT},{"submit_multiplier",multiplier},{"max_nodes_per_submit",max_nodes},
        {"limited_elementwise_fusion",limited_fusion!=0},{"adaptive_flops_cap_upper",base_cap},
        {"maximum_ramped_flops_cap_upper",ramped_cap},{"serial_fenced_ownership_preserved",true}});
}
extern "C" void ps5_ai_vk_limited_fusion_selected(unsigned unary_op) {
    (void)unary_op;perf_fused_selected.fetch_add(1,std::memory_order_relaxed);
}
static ggml_log_callback precision_previous_log=nullptr;
static void *precision_previous_log_data=nullptr;
static void precision_log(enum ggml_log_level level,const char *text,void *) {
    if (precision_previous_log) precision_previous_log(level,text,precision_previous_log_data);
    if (text && std::strstr(text,"ggml_vulkan:") && std::strstr(text," | fp16: ")) {
        const int parsed=ps5_ai_backend_precision_token(text);
        precision_announced=parsed>=0;precision_fp16=parsed==1;
        write_event({{"event","native_backend_precision"},{"fp16",precision_fp16},{"GGML_VK_DISABLE_F16_present",std::getenv("GGML_VK_DISABLE_F16")!=nullptr},{"startup_log",text}});
    }
}
extern "C" int ps5_ai_precision_check_output(const float *data,unsigned long long count,const char *stage) {
    uint64_t invalid=0;
    if(!data && count) invalid=count;
    else for(unsigned long long i=0;i<count;++i)invalid+=!std::isfinite(data[i]);
    precision_outputs.fetch_add(1);precision_values.fetch_add(count);precision_nonfinite.fetch_add(invalid);
    if(stage && std::strcmp(stage,"MOSS-Image")==0)precision_dit_outputs.fetch_add(1);
    if(stage && std::strcmp(stage,"vae")==0)precision_vae_outputs.fetch_add(1);
    return invalid==0;
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
    const auto get_features=reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(vkGetInstanceProcAddr(instance,"vkGetPhysicalDeviceFeatures2"));
    if(!get_features)throw std::runtime_error("Physical feature query unavailable");
    VkPhysicalDeviceShaderFloat16Int8Features arithmetic{};arithmetic.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    VkPhysicalDevice16BitStorageFeatures storage{};storage.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;storage.pNext=&arithmetic;
    VkPhysicalDeviceFeatures2 features{};features.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;features.pNext=&storage;
    get_features(physical,&features);
    write_event({{"event","native_fp16_capabilities"},{"shaderFloat16",bool(arithmetic.shaderFloat16)},
        {"storageBuffer16BitAccess",bool(storage.storageBuffer16BitAccess)},
        {"uniformAndStorageBuffer16BitAccess",bool(storage.uniformAndStorageBuffer16BitAccess)},
        {"storagePushConstant16",bool(storage.storagePushConstant16)},{"storageInputOutput16",bool(storage.storageInputOutput16)},
        {"requested_fp16_math",bool(PS5_AI_ENABLE_FP16)}});
    if(PS5_AI_ENABLE_FP16 && (!arithmetic.shaderFloat16 || !storage.storageBuffer16BitAccess || !storage.uniformAndStorageBuffer16BitAccess))
        throw std::runtime_error("Required physical FP16 arithmetic/storage features missing; no feature override");
    hardware_verified = true;
    std::snprintf(current_status.device, sizeof(current_status.device), "%s", device.deviceName);
    std::snprintf(current_status.driver, sizeof(current_status.driver), "native RADV driverVersion=%u", device.driverVersion);
#else
    throw std::runtime_error("PS5 native RADV hardware identity is unverified in this build; inference is disabled");
#endif
}

struct InferenceObservation {
#if PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL
    moss_ffn_chunk_observation::State ffn_chunk;
#endif
    ps5_ai_progress_cb progress;
    void* user;
    ps5_ai_stats* stats;
    uint64_t gpu_math_nodes;
    uint64_t gpu_dit_math_nodes;
    uint64_t gpu_flash_attention_nodes;
    const ggml_tensor* requested_dit;
    const ggml_tensor* requested_flash;
    int requested_steps;
    int last_sampler_step;
    bool wants_ip;
    uint64_t gpu_ip_math_nodes;
    const ggml_tensor* requested_ip;
    bool init_encoded;
    bool wants_controlnet=false;
    uint64_t gpu_controlnet_math_nodes=0;
    const ggml_tensor* requested_controlnet=nullptr;
    uint64_t gpu_vae_math_nodes=0;
    const ggml_tensor* requested_vae=nullptr;
};

static InferenceObservation* active_reference_observation = nullptr;
extern "C" void ps5_ai_controlnet_phase(int tensors,uint64_t values,double seconds){
    auto* o=active_reference_observation;
    if(!o || !o->wants_controlnet || tensors!=10 || !values || !std::isfinite(seconds) || seconds<0)
        throw std::runtime_error("Invalid ControlNet completion evidence");
    o->stats->controlnet_evals++;o->stats->controlnet_tensors=tensors;o->stats->controlnet_seconds+=seconds;
    if(o->stats->controlnet_evals==1)write_event({{"event","controlnet_residuals_completed"},{"tensors",tensors},{"finite_values",values},{"seconds",seconds}});
}
extern "C" void ps5_ai_reference_phase(int phase, int value, double seconds) {
    auto* o=active_reference_observation;
    if(!o || !std::isfinite(seconds) || seconds<0)return;
    if(phase==0 && value>0 && value<=o->requested_steps){
        o->requested_steps=value;o->stats->actual_steps=value;
    }else if(phase==1 && value==1){o->init_encoded=true;o->stats->init_encode_seconds=seconds;}
    else if(phase==2 && value==16){o->stats->ip_adapter_tokens=value;o->stats->ip_adapter_seconds=seconds;}
    else if(phase==3 && value>=0){o->stats->mask_denoise_evaluations=value;}
    else if(phase==4 && value==1){o->stats->mask_prepared=1;}
    try{write_event({{"event","reference_phase"},{"phase",phase},{"value",value},{"seconds",seconds}});}
    catch(...){fallback_log("reference phase logging unavailable");}
}

static const char* dit_weight_name(const ggml_tensor* tensor) {
    if (!tensor || (tensor->op != GGML_OP_MUL_MAT && tensor->op != GGML_OP_CONV_2D)) return nullptr;
    const ggml_tensor* weight = tensor->src[0];
    for (unsigned depth = 0; weight && depth < 8; ++depth) {
        const char* name = ggml_get_name(weight);
        constexpr char prefix[] = "model.diffusion_model.layers.";
        if (name && std::strncmp(name, prefix, sizeof(prefix)-1) == 0) return name;
        weight = weight->view_src;
    }
    return nullptr;
}

static const char* vae_weight_name(const ggml_tensor* tensor) {
    if (!tensor || (tensor->op != GGML_OP_MUL_MAT && tensor->op != GGML_OP_CONV_2D &&
                    tensor->op != GGML_OP_CONV_3D)) return nullptr;
    const ggml_tensor* weight = tensor->src[0];
    for (unsigned depth = 0; weight && depth < 8; ++depth) {
        const char* name = ggml_get_name(weight);
        constexpr char prefix[] = "first_stage_model.decoder.";
        if (name && std::strncmp(name, prefix, sizeof(prefix)-1) == 0) return name;
        weight = weight->view_src;
    }
    return nullptr;
}

static const char* controlnet_weight_name(const ggml_tensor* tensor){
    if(!tensor || (tensor->op!=GGML_OP_MUL_MAT && tensor->op!=GGML_OP_CONV_2D))return nullptr;
    const ggml_tensor* weight=tensor->src[0];
    for(unsigned depth=0;weight && depth<8;depth++){
        const char* n=ggml_get_name(weight);
        if(n && (!std::strncmp(n,"input_blocks.",13) || !std::strncmp(n,"input_hint_block.",17) || !std::strncmp(n,"zero_convs.",11)))return n;
        weight=weight->view_src;
    }return nullptr;
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
        case GGML_OP_CONV_3D:
        case GGML_OP_FLASH_ATTN_EXT:
            return true;
        default:
            return false;
    }
}

static uint64_t moss_ffn_chunk_expected(const InferenceObservation& o) noexcept {
#if PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL
    return o.ffn_chunk.proof(moss_ffn_token_chunk::configured_tokens).expected_q4;
#else
    (void)o;return 672;
#endif
}

static bool observe_graph(ggml_tensor* tensor, bool ask, void* data) noexcept {
    auto* observation = static_cast<InferenceObservation*>(data);
    if (ask) {
#if PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL
        if(tensor&&tensor->op==GGML_OP_MUL_MAT&&tensor->src[0]&&tensor->src[1]&&tensor->src[0]->type==GGML_TYPE_Q4_K) {
            const auto* w=tensor->src[0];const auto* x=tensor->src[1];
            const bool precise=tensor->type==GGML_TYPE_F32&&x->type==GGML_TYPE_F32&&
                w->ne[2]==1&&w->ne[3]==1&&x->ne[2]==1&&x->ne[3]==1&&
                x->ne[0]==w->ne[0]&&tensor->ne[0]==w->ne[1]&&tensor->ne[1]==x->ne[1]&&
                (ggml_prec)tensor->op_params[0]==GGML_PREC_F32;
            observation->ffn_chunk.add(w->name,w->ne[0],w->ne[1],x->ne[1],precise);
        }
#endif
        if (!is_gpu_math_tensor(tensor)) return false;
        if(observation->wants_controlnet && !observation->gpu_controlnet_math_nodes && !observation->requested_controlnet && controlnet_weight_name(tensor)){
            observation->requested_controlnet=tensor;return true;
        }
        // Request one DiT-weight, one VAE decoder-weight and one attention witness.
        // Each true ask partitions and synchronizes the graph in SD.cpp; all
        // other nodes still execute normally, without per-node proof barriers.
        if (!observation->gpu_dit_math_nodes && !observation->requested_dit && dit_weight_name(tensor)) {
            observation->requested_dit = tensor;
            return true;
        }
        if (!observation->gpu_vae_math_nodes && !observation->requested_vae && vae_weight_name(tensor)) {
            observation->requested_vae = tensor;
            return true;
        }
        const char* weight=dit_weight_name(tensor);
        if(observation->wants_ip && !observation->gpu_ip_math_nodes && !observation->requested_ip &&
           weight && std::strstr(weight,".to_k_ip.weight")){
            observation->requested_ip=tensor;return true;
        }
        if (!observation->gpu_flash_attention_nodes && !observation->requested_flash && tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            observation->requested_flash = tensor;
            return true;
        }
        return false;
    }
    const bool dit = tensor == observation->requested_dit && !observation->gpu_dit_math_nodes;
    const bool vae = tensor == observation->requested_vae && !observation->gpu_vae_math_nodes;
    const bool flash = tensor == observation->requested_flash && !observation->gpu_flash_attention_nodes;
    const bool ip=tensor==observation->requested_ip && !observation->gpu_ip_math_nodes;
    const bool control=tensor==observation->requested_controlnet && !observation->gpu_controlnet_math_nodes;
    if ((!dit && !vae && !flash && !ip && !control) || !is_gpu_math_tensor(tensor) ||
        ((dit || ip) && !dit_weight_name(tensor)) || (vae && !vae_weight_name(tensor)) ||
        (control && !controlnet_weight_name(tensor))) return false;
    ++observation->gpu_math_nodes;
    if (dit) ++observation->gpu_dit_math_nodes;
    if (vae) ++observation->gpu_vae_math_nodes;
    if (flash) ++observation->gpu_flash_attention_nodes;
    if(ip){++observation->gpu_ip_math_nodes;observation->stats->gpu_ip_attention_verified=1;}
    if(control){++observation->gpu_controlnet_math_nodes;observation->stats->gpu_controlnet_verified=1;}
    try {
        nlohmann::json event = {{"event", control ? "gpu_controlnet_math_completed" : ip ? "gpu_ip_attention_math_completed" : dit ? "gpu_dit_model_math_completed" : vae ? "gpu_vae_decoder_math_completed" : "gpu_flash_attention_math_completed"},
            {"op", ggml_op_name(tensor->op)}, {"tensor", ggml_get_name(tensor)},
            {"device", ggml_backend_dev_name(gpu_device)},
            {"scope", "sampled completed witness after backend synchronization; not total GPU operation count"}};
        if (dit || ip) event["weight"] = dit_weight_name(tensor);
        if (vae) event["weight"] = vae_weight_name(tensor);
        if(control)event["weight"]=controlnet_weight_name(tensor);
        write_event(event);
    } catch (...) { fallback_log("ps5-ai: completed GPU witness structured log unavailable"); }
    observation->stats->peak_cpu_bytes = std::max(observation->stats->peak_cpu_bytes, cpu_peak_bytes());
    return true;
}

static void observe_progress(int step, int steps, float, void* data) noexcept {
    auto* observation = static_cast<InferenceObservation*>(data);
    // Ignore model-loader byte progress and pre-DiT conditioning callbacks.
    // Tiling callbacks arrive after the terminal sampler step and are ignored.
    if (!observation->progress || !observation->gpu_dit_math_nodes || parameter_loading.load() ||
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
    auto* q = ggml_new_tensor_3d(context, GGML_TYPE_F32, 128, 1024, 24);
    auto* k = ggml_new_tensor_3d(context, GGML_TYPE_F16, 128, 1024, 24);
    auto* v = ggml_new_tensor_3d(context, GGML_TYPE_F16, 128, 1024, 24);
    auto* attention = ggml_flash_attn_ext(context, q, k, v, nullptr, 1.f/std::sqrt(128.f), 0.f, 0.f);
    ggml_prec_set_acc(attention, GGML_PREC_F32);
    const bool supported = ggml_backend_dev_supports_op(gpu_device, attention);
    write_event({{"event", "native_flash_attention_capability"}, {"supported", supported},
        {"model", "MOSS-Image"}, {"head_size", 128}, {"heads", 24}, {"q_type", "F32"}, {"kv_type", "F16"},
        {"scope", "metadata capability only; completed generation witness still required"}});
    if (!supported && engine_config.requires_dit_flash_attention()) throw std::runtime_error("Native Vulkan MOSS-Image DiT flash attention is unsupported; fallback is disabled");
}

static int engine_init_impl(const char* config_path, char* error, size_t error_size) {
#if !PS5_AI_MOSS_CAN_RUN_ADMISSION
    return ps5_ai_v12_read_memory_plan(config_path, nullptr, error, error_size);
#endif
    try {
        // Preserve the bounded GGML assertion reason before the existing abort
        // boundary retains GPU resources. This callback never unwinds or frees.
        ggml_set_abort_callback([](const char* message) {
            ps5_ai_ggml_report("GGML_ABORT_DIAGNOSTIC message=%.1800s", message ? message : "unavailable");
        });
        current_status = {};
        hardware_verified = false;
        gpu_device = nullptr;
        partial_taesd_gpu_decode = false;
        std::snprintf(current_status.backend, sizeof(current_status.backend), "stable-diffusion.cpp/Vulkan");
#if PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL
        ps5_ai_ggml_report("MOSS_EXPERIMENTAL_SMALL_TRIAL enabled=1 production_runtime_admitted=0 capacity_readback_verification_pending=1 image_size=256x256 image_attempt_limit=1 prior_GPU_or_model_success_claimed=0");
#endif
        if (!config_path) {
            throw std::runtime_error("Engine config path is missing");
        }
        const auto config = bounded_config(config_path);
        engine_config = PS5AIEngineConfig::parse(PS5AIEngineConfig::engine_object(config));
        const float configured_scale = static_cast<float>(engine_config.moss_lora_strength);
        uint32_t configured_scale_bits; std::memcpy(&configured_scale_bits, &configured_scale, sizeof(configured_scale_bits));
        moss_lora_scale_bits = configured_scale_bits;
        moss_lora_configured = true;
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
            throw std::runtime_error("Native Direct MOSS-Image configured memory plan differs from actual prepared pool admission");
        verify_native_hardware();
        // Backend registration is explicit; no remote/RPC fallback is compiled.
        ggml_log_get(&precision_previous_log,&precision_previous_log_data);
        ggml_log_set(precision_log,nullptr);
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
        if(!precision_announced || precision_fp16!=bool(PS5_AI_ENABLE_FP16))
            throw std::runtime_error("Actual GGML fp16 startup log does not match requested precision");
        const std::string gpu_name = ggml_backend_dev_name(gpu_device);
        engine_config.resolve_defaults(gpu_name);
        engine_config.validate_actual_pool(gpu_name, pool.reserved_bytes);
        verify_flash_attention_support();
        partial_taesd_gpu_decode = engine_config.partial_taesd_gpu_decode(gpu_name);
        std::snprintf(current_status.backend, sizeof(current_status.backend), "%s",
            "SD.cpp CPU Qwen3.5; Vulkan MOSS DiT/FLUX2 VAE");
        current_status.gpu_available = 1;
        current_status.available = 1;
        write_event({{"event", "engine_configured"}, {"model", engine_config.model},
                     {"moss_spectrum_trial",PS5_AI_MOSS_SPECTRUM_TRIAL!=0},
                     {"moss_spectrum_policy",PS5_AI_MOSS_SPECTRUM_POLICY},
                     {"moss_sampler_trial",bool(PS5_AI_MOSS_SAMPLER_TRIAL)},
                     {"moss_bounded_weight_upload_trial",PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL!=0},
                     {"moss_bounded_weight_upload_policy",PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_POLICY},
                     {"bounded_weight_upload_enabled",ps5_ai_moss_bounded_weight_upload_enabled()!=0},
                     {"moss_sampler_policy",PS5_AI_MOSS_SAMPLER_POLICY},
                     {"moss_turbo_trial",bool(PS5_AI_MOSS_TURBO_TRIAL)},
                     {"moss_turbo_policy",PS5_AI_MOSS_TURBO_POLICY},
                     {"moss_lora_fp32_shared_trial",bool(PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL)},
                     {"moss_lora_f16_storage_trial",bool(PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL)},
                     {"moss_lora_precision_policy",PS5_AI_MOSS_LORA_PRECISION_POLICY},
                     {"moss_turbo_bounded_upload_trial",bool(PS5_AI_MOSS_TURBO_BOUNDED_UPLOAD_TRIAL)},
                     {"moss_turbo_bounded_upload_policy",PS5_AI_MOSS_TURBO_BOUNDED_UPLOAD_POLICY},
                     {"moss_ffn_token_chunk_trial",bool(PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL)},{"moss_ffn_chunk_tokens",PS5_AI_MOSS_FFN_CHUNK_TOKENS},
                     {"moss_ffn_chunk_policy",PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL?PS5_AI_MOSS_FFN_CHUNK_POLICY:"disabled"},
                     {"effective_adapter",moss_effective_lora_identity()},
                     {"trial_sampler",PS5_AI_MOSS_DEFAULT_REQUEST_SAMPLER},{"trial_scheduler",PS5_AI_MOSS_REQUEST_SCHEDULER},
                     {"trial_steps",PS5_AI_MOSS_DEFAULT_REQUEST_STEPS},{"trial_cfg_scale",PS5_AI_MOSS_DEFAULT_REQUEST_CFG},
                     {"negative_cfg_skip_scope","all_negative_forwards_existing_MOSS_policy"},
                     {"moss_gpu_profile",bool(PS5_AI_MOSS_GPU_PROFILE)},
                     {"moss_dit_gpu_profile",bool(PS5_AI_MOSS_DIT_GPU_PROFILE)},
                     {"moss_q4_bk64_raw64",bool(PS5_AI_MOSS_Q4_BK64_RAW64)},
                     {"moss_q4_bk64_policy",PS5_AI_MOSS_Q4_BK64_RAW64?"canonical_moss_ffn_up_q4k_f32acc_bk64_raw64_v1":"disabled"},
                     {"moss_gpu_profile_policy",PS5_AI_MOSS_DIT_GPU_PROFILE?"mask15_Turbo4_rank144_first_two_complete_cached_DiT_graphs_parent_and_shader_v1":PS5_AI_MOSS_GPU_PROFILE?"first_two_complete_dit_graphs_per_generation":"disabled"},
                     {"moss_f32_accumulation",bool(PS5_AI_MOSS_F32_ACCUMULATION)},
                     {"MOSS_linear_accumulation",PS5_AI_MOSS_F32_ACCUMULATION?"f32":"default"},
                     {"architecture", "MOSS-Image"}, {"llm_path", engine_config.llm}, {"vae_path", engine_config.vae}, {"tokenizer_path", engine_config.tokenizer},
                     {"backend", engine_config.backend}, {"params_backend", engine_config.params_backend},
                     {"packed_latent_channels",128}, {"taesd_enabled",false},
                     {"vae_tile_image_pixels", engine_config.vae_tile_size},
                     {"vae_tile_overlap", engine_config.vae_tile_overlap},
                     {"eager_load", engine_config.eager_load},
                      {"configured_LoRA_inputs", 1}, {"moss_lora_path", engine_config.moss_lora_path},
                      {"moss_lora_strength", engine_config.moss_lora_strength}, {"lora_apply_mode", "at_runtime"}, {"mode", "text2img"},
                      {"model_mmap_policy", "disabled; explicit file I/O; no whole-file prefault"},
                     {"flash_attn_mode", engine_config.flash_attn_mode}, {"batched_cfg", engine_config.batched_cfg},
                     {"conditioning_cache_size", engine_config.conditioning_cache_size},
                     {"gpu_observation", "first MOSS DiT and FLUX2 VAE decoder completed witnesses; FLASH_ATTN when selected; not total operation counts"},
                     {"partial_gpu_taesd_decode", partial_taesd_gpu_decode},
                     {"gpu_stage", "MOSS DiT denoising and FLUX2 VAE decode on Vulkan; Qwen3.5 text encoding CPU"},
                     {"owned_direct_data_reserved_bytes", pool.reserved_bytes},
                     {"owned_direct_data_free_bytes", pool.free_bytes},
                     {"working_reserve_bytes", engine_config.working_reserve_bytes},
                     {"working_reserve_scope", "accounting admission outside GPU pool; not GPU workspace"},
                     {"max_vram_gib", engine_config.max_vram},
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
        { std::lock_guard<std::mutex> lock(status_mutex); *status = current_status; }
        (void)ps5_ai_init_stage_json(status->model_initialization,sizeof(status->model_initialization));
    }
}

#include "precision_gate.inc"
#include "precision_conv_gate.inc"
#include "moss_lora_precision_receipt.inc"
#if PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL
#include "moss_precision_gate.inc"
#include "moss_perf_gate.inc"
#include "moss_q4_bk64_gate.inc"
#if PS5_AI_MOSS_TURBO_TRIAL
#include "moss_turbo_lora_gate.inc"
#include "moss_lora_f16_storage_gate.inc"
#include "moss_ffn_chunk_gate.inc"
#endif
#endif
#include "moss_bounded_weight_upload_gate.inc"
#include "moss_dit_gate.inc"
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
#if PS5_AI_MOSS_GPU_PROFILE
        if(!ps5_ai_moss_gpu_profile_gate_begin())
            throw std::runtime_error("GPU timestamp startup gate could not be armed");
#endif
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
#if PS5_AI_MOSS_GPU_PROFILE
        ps5_ai_moss_gpu_profile_report gpu_gate{};
        const bool gpu_gate_ok=ps5_ai_moss_gpu_profile_gate_end(&gpu_gate)!=0;
        write_event({{"event","native_moss_gpu_timestamp_gate"},{"report",moss_gpu_profile_json(gpu_gate)}});
        if(!gpu_gate_ok)
            throw std::runtime_error("GPU timestamp startup readback unavailable, invalid or incomplete");
#endif
        ps5_ai_ggml_report("Native_Direct_data_gate_readback compared_values=%llu mismatches=%llu checksum=0x%016llx expected_checksum=0x%016llx dispatch_delta=%llu submit_delta=%llu fence_delta=%llu whole_pool_GPU_verified=0",
            (unsigned long long)elements, (unsigned long long)mismatches,
            (unsigned long long)checksum, (unsigned long long)expected_checksum,
            (unsigned long long)(after.dispatches - before.dispatches),
            (unsigned long long)(after.submits - before.submits),
            (unsigned long long)(after.fenced_completions - before.fenced_completions));
        if (mismatches || checksum != expected_checksum || !ps5_ai_ggml_log_healthy())
            throw std::runtime_error("Native_Direct large Direct GPU buffer readback mismatch or missing durable log");
        resources.release();
        precision_readback_gate();
        precision_conv_readback_gate();
#if PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL
        moss_precision_readback_gate();
        moss_performance_readback_gate();
#if PS5_AI_MOSS_TURBO_TRIAL
        moss_turbo_lora_readback_gate();
#if PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL
        moss_lora_f16_storage_readback_gate();
#endif
#if PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL
        moss_ffn_chunk_readback_gate();
#endif
#endif
#if PS5_AI_MOSS_Q4_BK64_RAW64
        const auto candidate_gate=moss_q4_bk64_readback_gate();
        if(!candidate_gate.numerical_pass)throw std::runtime_error("MOSS Q4 BK64/raw64 candidate B gate did not pass");
#endif
#endif
#if PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL
        moss_bounded_weight_upload_readback_gate();
#endif
#if PS5_AI_MOSS_TURBO_BOUNDED_UPLOAD_TRIAL
        if(!moss_turbo_lora_gate_admitted() || !ps5_ai_moss_bounded_weight_upload_enabled())
            throw std::runtime_error("Combined Turbo/upload initialization requires both unchanged GPU gates");
        write_event({{"event","native_moss_turbo_bounded_upload_gates"},
                     {"policy",PS5_AI_MOSS_TURBO_BOUNDED_UPLOAD_POLICY},
                     {"rank144_gate_admitted",true},{"bounded_upload_gate_admitted",true},
                     {"model_loaded",false},{"GPU_inference_claimed",false}});
#endif
#if PS5_AI_MOSS_DIT_OPT_TRIAL
        moss_dit_readback_gate();
#endif
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
        ps5_ai_ggml_report("FAIL Native_Direct_data_gate_exception reason=%s",exception.what());
        set_error(error, error_size, exception.what());
    } catch (...) {
        ps5_ai_ggml_report("FAIL Native_Direct_data_gate_exception reason=unknown_CPU_exception");
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
    if (!ps5_ai_moss_trial_request_valid(request->width,request->height,request->steps,
                                        request->cfg_scale,request->flow_shift,request->sampler,request->seed)) {
        set_error(error,error_size,PS5_AI_MOSS_TURBO_TRIAL
            ? "MOSS Turbo trial requires256/1024 square,4steps,CFG1,shift5,Euler/custom sigmas,seed20261020"
            : PS5_AI_MOSS_SAMPLER_TRIAL
            ? "MOSS sampler trial requires256/1024 square,30steps,CFG4,shift5,res_multistep/simple,seed20261020; production admission is not asserted"
            : PS5_AI_MOSS_PERFORMANCE_TRIAL
            ? "MOSS performance trial requires256/1024 square,50steps,CFG3,shift5,Euler,seed20261020; production admission is not asserted"
            : PS5_AI_MOSS_1024_COLD_WARM_TRIAL
            ? "MOSS benchmark requires1024x1024,50steps,CFG3,shift5,Euler,seed20261020; production admission is not asserted"
            : PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL
            ? "MOSS quality trial requires 256x256,4 or50 steps,CFG3,flow shift5,Euler and seed20261019; production admission is not asserted"
            : "MOSS trial requires 256x256,4 steps,CFG3,flow shift5 and Euler; production admission is not asserted");
        return -1;
    }
#if PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL || PS5_AI_MOSS_1024_COLD_WARM_TRIAL || PS5_AI_MOSS_PERFORMANCE_TRIAL
    if (engine_config.moss_lora_strength!=(PS5_AI_MOSS_TURBO_TRIAL?1.0:0.7)) {
        set_error(error,error_size,PS5_AI_MOSS_TURBO_TRIAL?
            "MOSS Turbo composed adapter requires global1.0, containing NSFW0.7+Turbo1.0":
            "MOSS controlled trial requires unchanged configured LoRA strength0.7");
        return -1;
    }
#endif
#if PS5_AI_MOSS_PERFORMANCE_TRIAL
#if PS5_AI_MOSS_TURBO_TRIAL
    if(request->width==1024 && !moss_turbo_small_image_completed.load(std::memory_order_acquire)) {
        set_error(error,error_size,"Turbo1024 requires one completed native256 image first; manual quality review is additionally required by the client");
        return -1;
    }
#endif
    static std::atomic<unsigned> performance_attempts{0};
    if(!ps5_ai_moss_performance_trial_claim(performance_attempts,request->width,request->height,request->steps,
                                          request->cfg_scale,request->flow_shift,request->sampler,request->seed)) {
        set_error(error,error_size,"MOSS performance trial permits four attempts per process; failures consume claims and the fifth is refused");
        return -1;
    }
#elif PS5_AI_MOSS_1024_COLD_WARM_TRIAL
    static std::atomic<unsigned> experimental_attempts{0};
    if(!ps5_ai_moss_benchmark_trial_claim(experimental_attempts,request->width,request->height)) {
        set_error(error,error_size,"MOSS1024 cold/warm benchmark permits two attempts per process; failures consume claims and the third is refused");
        return -1;
    }
#else
    static std::atomic<bool> experimental_attempt_claimed{false};
    if (!ps5_ai_moss_trial_claim(experimental_attempt_claimed,request->width,request->height)) {
        set_error(error,error_size,"MOSS experimental verification permits one256x256 generation attempt per process; failures do not reset this claim");
        return -1;
    }
#endif
    phase_profile_begin();
#if PS5_AI_MOSS_SPECTRUM_TRIAL
    moss_spectrum_last={};
    moss_spectrum_completions.store(0,std::memory_order_release);
#endif
    const uint64_t fused_before=perf_fused_selected.load(std::memory_order_relaxed);
    ps5_ai_moss_quant_receipt quant_before{};ps5_ai_moss_quant_snapshot(&quant_before);
    ps5_ai_moss_q4_bk64_report candidate_before{};ps5_ai_moss_q4_bk64_snapshot(&candidate_before);
    ps5_ai_moss_lora_storage_receipt storage_before{};ps5_ai_moss_lora_storage_snapshot(&storage_before);
    InferenceObservation observation = {};
    observation.progress=progress;observation.user=user;observation.stats=stats;
    observation.requested_steps=request->steps;
    active_reference_observation=&observation;
    parameter_loading.store(false);
    precision_outputs.store(0);precision_values.store(0);precision_nonfinite.store(0);
    precision_dit_outputs.store(0);precision_vae_outputs.store(0);
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
                active_reference_observation=nullptr;
            } catch (...) {
                fallback_log("ps5-ai: generation cleanup failed; reload title");
            }
        }
    } cleanup = {&images, &image_count};
    const auto start = std::chrono::steady_clock::now();
    int result = -1;
    ps5_ai_moss_gpu_profile_report gpu_profile{};
#if PS5_AI_MOSS_GPU_PROFILE
    bool gpu_profile_open=false;
#endif
    moss_lora_loaded = 0; moss_lora_matched = 0; moss_lora_used = 0; moss_lora_skipped = 0;
    moss_lora_verified = false; moss_lora_invalid = false;
    try {
#if PS5_AI_MOSS_Q4_BK64_RAW64
        if(!moss_q4_bk64_admission_passed || !candidate_before.enabled || candidate_before.active!=1 || !candidate_before.device_generation)
            throw std::runtime_error("MOSS Q4 BK64/raw64 candidate B was not admitted by strict numerical and warmed microtime gates; no image inference permitted");
#endif
#if PS5_AI_MOSS_GPU_PROFILE
        if(!ps5_ai_moss_gpu_profile_job_begin())
            throw std::runtime_error("Bounded GPU operator profile could not be armed");
        gpu_profile_open=true;
#endif
        if (!std::isfinite(request->manga_strength) || request->manga_strength != 0 || request->manga_monochrome)
            throw std::runtime_error("MOSS-Image-SFT text2img does not support the WAI-specific manga LoRA or monochrome option");
        if (request->init_image_id[0] || request->reference_image_id[0] || request->mask_image_id[0] ||
            request->controlnet[0] || request->control_image_id[0])
            throw std::runtime_error("MOSS-Image-SFT currently supports text2img only; SDXL IP-Adapter/ControlNet, img2img and inpainting were not run");
#if PS5_AI_MOSS_TURBO_TRIAL
        if(!moss_turbo_lora_gate_admitted())
            throw std::runtime_error("Turbo model loading requires all rank144 F32 GPU numerical gates first");
#endif
#if PS5_AI_MOSS_TURBO_BOUNDED_UPLOAD_TRIAL
        if(!ps5_ai_moss_bounded_weight_upload_enabled())
            throw std::runtime_error("Combined model loading requires the bounded upload GPU readback gate");
#endif
#if PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL
        if(!moss_lora_f16_storage_gate_admitted())
            throw std::runtime_error("F16 LoRA storage model loading requires all8 fixed F32/F16 private-math GPU numerical gates");
#endif
        if (!model_context)
            ps5_ai_init_stage_call(PS5_AI_INIT_PREAMBLE, [&]() { ps5_ai_native_capacity_snapshot("before_moss_generation"); });
        else
            ps5_ai_native_capacity_snapshot("before_moss_generation");
        if (!model_context) {
            sd_ctx_params_t context_params;
            sd_ctx_params_init(&context_params);
            engine_config.apply_context(context_params);
            // Same direct-convolution path as the verified FP32 auto selection.
            // It is admitted only after the separate startup CPU-oracle gate.
            // FP16 im2col is not used as the numerical reference for this gate.
            context_params.conv_direct_mode="on";
            ps5_ai_init_stage_call(PS5_AI_INIT_PRECONTEXT_EVENT, [&]() {
                write_event({{"event","native_conv_policy"},{"mode","on"},{"basis","direct GPU convolution passed independent CPU oracle"}});
            });
            ps5_ai_native_ui_phase(PS5_AI_STATUS_LOADING);
            ps5_ai_native_ui_log("Loading MOSS-Image-SFT + Qwen3.5 + FLUX2 VAE");
            model_context = ps5_ai_init_stage_call(PS5_AI_INIT_CONTEXT, [&]() { return new_sd_ctx(&context_params); });
            if (!model_context) {
                throw std::runtime_error("MOSS-Image split model load failed; see engine.jsonl");
            }
            if (std::strcmp(sd_get_model_version_name(model_context), "MOSS-Image") != 0) {
                free_sd_ctx(model_context);model_context=nullptr;
                throw std::runtime_error("Loaded diffusion architecture is not MOSS-Image; no WAI/SDXL fallback was run");
            }
            {
                std::lock_guard<std::mutex> lock(status_mutex);
                current_status.model_loaded = 1;
            }
            write_event({{"event", "model_loaded"}, {"path", engine_config.model},
                         {"architecture", sd_get_model_version_name(model_context)}});
            ps5_ai_native_capacity_snapshot("after_MOSS-Image_model_load");
        }
        sd_img_gen_params_t params;
        sd_img_gen_params_init(&params);
        // MOSS has its own Qwen3.5 conditioning; do not apply SDXL clip_skip=2
        // or the WAI-specific adapter/prompt rewriting path.
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
        if (!std::isfinite(request->flow_shift) || !std::isfinite(static_cast<float>(request->flow_shift)) ||
            static_cast<float>(request->flow_shift) <= 0.f)
            throw std::runtime_error("MOSS flow_shift must be finite, positive and representable as float");
        if (request->width <= 0 || request->height <= 0 || request->width % 16 || request->height % 16)
            throw std::runtime_error("MOSS image dimensions must be positive multiples of 16");
        params.sample_params.scheduler = PS5_AI_MOSS_SAMPLER_TRIAL ? SIMPLE_SCHEDULER : DISCRETE_SCHEDULER;
        params.sample_params.flow_shift = static_cast<float>(request->flow_shift);
        // The real upstream MOSS branch applies zero-based layer22 skipping
        // only to the negative CFG evaluation; positive conditioning is intact.
        engine_config.apply_image(params);
#if PS5_AI_MOSS_SPECTRUM_TRIAL
        params.cache.mode=SD_CACHE_SPECTRUM;
        params.cache.spectrum_w=0.4f;params.cache.spectrum_m=3;params.cache.spectrum_lam=1.0f;
        params.cache.spectrum_window_size=2;params.cache.spectrum_flex_window=0.0f;
        params.cache.spectrum_warmup_steps=8;params.cache.spectrum_stop_percent=0.96f;
#endif
        sd_lora_t selected_lora = {};
        selected_lora.path = engine_config.moss_lora_path.c_str();
        selected_lora.multiplier = static_cast<float>(engine_config.moss_lora_strength);
        selected_lora.is_high_noise = false;
#if PS5_AI_MOSS_TURBO_TRIAL
        if(!moss_turbo_lora_gate_admitted())
            throw std::runtime_error("Turbo generation requires all new rank144 F32 GPU numerical gates");
        float turbo_sigmas[PS5_AI_MOSS_TURBO_SIGMA_COUNT];
        ps5_ai_moss_turbo_sigmas(turbo_sigmas);
        params.sample_params.sample_method=EULER_SAMPLE_METHOD;
        params.sample_params.sample_steps=4;
        params.sample_params.scheduler=DISCRETE_SCHEDULER;
        params.sample_params.custom_sigmas=turbo_sigmas;
        params.sample_params.custom_sigmas_count=5;
        params.sample_params.guidance.txt_cfg=1.0f;
        params.sample_params.guidance.img_cfg=1.0f;
        params.sample_params.guidance.distilled_guidance=0.0f;
        params.sample_params.guidance.slg=sd_slg_params_t{};
        params.sample_params.extra_sample_args=nullptr;
#endif
        params.loras = &selected_lora;
        params.lora_count = 1;
        write_event({{"event", "moss_selected_lora_request"}, {"path", engine_config.moss_lora_path},
                     {"strength", engine_config.moss_lora_strength}, {"required_tensors", 504},
                     {"rank", PS5_AI_MOSS_TURBO_TRIAL?144:16}, {"alpha", PS5_AI_MOSS_TURBO_TRIAL?144:16},
                     {"source_storage", PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL?"lossy F16 rounded independent rank16+128 composition":PS5_AI_MOSS_TURBO_TRIAL?"F32 independent rank16+128 coefficient composition":"F32 exact from BF16"},
                     {"effective_adapter",moss_effective_lora_identity()},
                     {"qkv_layout", "independent A/B pairs in ordered q,k,v output slices"}, {"base_merge", false}});
        sd_set_backend_eval_callback(observe_graph, &observation);
        sd_set_progress_callback(observe_progress, &observation);
        if (progress) {
            progress(0.05, user);
        }
        ps5_ai_native_ui_phase(PS5_AI_STATUS_GENERATING);
#if PS5_AI_MOSS_DIT_OPT_TRIAL
        if(!moss_dit_gate_passed)throw std::runtime_error("DiT optimization startup admission lost");
        moss_dit_opt::begin(PS5_AI_MOSS_DIT_OPT_MASK);
#endif
        if (!generate_image(model_context, &params, &images, &image_count) || image_count != 1 || !images || !images[0].data) {
            throw std::runtime_error("MOSS-Image image generation failed; see engine.jsonl");
        }
        ps5_ai_moss_lora::validate_counts(moss_lora_loaded.load(), moss_lora_matched.load(),
            moss_lora_used.load(), moss_lora_skipped.load(), true);
        if (moss_lora_invalid.load())
            throw std::runtime_error("Image rejected: selected MOSS LoRA application/strength receipt is invalid");
#if PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL
        ps5_ai_moss_lora_storage_receipt storage_completed{};ps5_ai_moss_lora_storage_snapshot(&storage_completed);
        if(!moss_lora_f16_storage_gate_admitted()||!moss_lora_storage_generation_delta_valid(storage_before,storage_completed))
            throw std::runtime_error("Image rejected: exact2016 canonical F16/F32acc LoRA pipeline recordings, four per504 tensors, required");
#endif
        stats->gpu_used = observation.gpu_dit_math_nodes > 0 ? 1 : 0;
        if (!stats->gpu_used || (engine_config.requires_dit_flash_attention() && !observation.gpu_flash_attention_nodes)) {
            throw std::runtime_error("Image rejected: required completed native DiT/flash-attention witness is missing");
        }
        if(!stats->actual_steps || observation.last_sampler_step!=stats->actual_steps)
            throw std::runtime_error("Image rejected: sampler completion evidence is missing");
        if (!observation.gpu_vae_math_nodes || !precision_dit_outputs.load() ||
            !precision_vae_outputs.load() || precision_nonfinite.load())
            throw std::runtime_error("Image rejected: completed GPU DiT/VAE and finite readback evidence is required");
        if (cpu_clip_pool_live_bytes.load() != 0)
            throw std::runtime_error("CPU Qwen shared Direct weights were not returned after conditioning");
#if PS5_AI_MOSS_F32_ACCUMULATION
        ps5_ai_moss_quant_receipt quant_completed{};ps5_ai_moss_quant_snapshot(&quant_completed);
        if(quant_completed.q4_f32acc<=quant_before.q4_f32acc ||
           quant_completed.q4_f16acc!=quant_before.q4_f16acc || quant_completed.invalid!=quant_before.invalid)
            throw std::runtime_error("Image rejected: MOSS Q4 matrix F32 accumulation policy not observed");
#if PS5_AI_MOSS_SPECTRUM_TRIAL
        const uint64_t latent_values=uint64_t(request->width/16)*uint64_t(request->height/16)*128;
        const auto sr=moss_spectrum_last;
        if(request->steps!=50 || stats->actual_steps!=50 || request->cfg_scale!=3.0 ||
           std::strcmp(request->sampler,"euler") || request->flow_shift!=5.0 ||
           moss_spectrum_completions.load(std::memory_order_acquire)!=1 ||
           sr.completed_steps!=50 || sr.real_steps!=30 || sr.forecast_steps!=20 || sr.errors ||
           sr.history_payload_bytes!=6*latent_values*sizeof(float) ||
           sr.peak_payload_bytes!=7*latent_values*sizeof(float) ||
           precision_dit_outputs.load()!=60 ||
           quant_completed.q4_f32acc-quant_before.q4_f32acc!=9960)
            throw std::runtime_error("Image rejected: MOSS Spectrum requires30actual CFG steps,20forecasts,60completed DiT forwards and9960 F32 Q4 recordings");
#endif
#if PS5_AI_MOSS_SAMPLER_TRIAL
        // One positive42-layer and one negative41-layer forward per sampler
        // step, with four Q4 matrix operations per retained main layer.
        const uint64_t expected_forwards=uint64_t(request->steps)*2;
        const uint64_t expected_q4=uint64_t(request->steps)*(42+41)*4;
        if(precision_dit_outputs.load()!=expected_forwards ||
           quant_completed.q4_f32acc-quant_before.q4_f32acc!=expected_q4)
            throw std::runtime_error("Image rejected: sampler trial requires exact60 completed DiT forwards and9960 F32 Q4 recordings");
#endif
#if PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL
        const auto ffn_proof=observation.ffn_chunk.proof(moss_ffn_token_chunk::configured_tokens);
        if(!moss_ffn_chunk_gate_admitted()||!ffn_proof.valid)
            throw std::runtime_error("FFN image rejected: all42 layers/four positive evaluations/token partition metadata required");
        write_event({{"event","moss_FFN_chunk_completed_metadata"},{"policy",PS5_AI_MOSS_FFN_CHUNK_POLICY},
            {"chunk_tokens",moss_ffn_token_chunk::configured_tokens},{"actual_main_tokens",ffn_proof.main_tokens},
            {"pieces_per_FFN",ffn_proof.pieces},{"layers",42},{"positive_evaluations",4},
            {"expected_Q4_F32_recordings",ffn_proof.expected_q4},{"metadata_is_not_GPU_completion_proof",true}});
#endif
#if PS5_AI_MOSS_TURBO_TRIAL
        if(request->steps!=4 || stats->actual_steps!=4 || request->cfg_scale!=1.0 ||
           precision_dit_outputs.load()!=4 || quant_completed.q4_f32acc-quant_before.q4_f32acc!=moss_ffn_chunk_expected(observation) ||
           !moss_turbo_lora_gate_admitted())
            throw std::runtime_error("Image rejected: Turbo requires actual4 positive DiT outputs and exact compiled/observed FFN F32 Q4 count");
#endif
#if PS5_AI_MOSS_TURBO_BOUNDED_UPLOAD_TRIAL
        if(!ps5_ai_moss_bounded_weight_upload_enabled())
            throw std::runtime_error("Image rejected: combined bounded upload admission was lost");
#endif
#endif
#if PS5_AI_MOSS_Q4_BK64_RAW64
        ps5_ai_moss_q4_bk64_report candidate_completed{};ps5_ai_moss_q4_bk64_snapshot(&candidate_completed);
        if(request->steps!=50 || stats->actual_steps!=50 || !moss_q4_bk64_admission_passed ||
           candidate_completed.schema!=1 || candidate_completed.enabled!=1 || candidate_completed.active!=1 ||
           candidate_completed.device_generation!=candidate_before.device_generation ||
           candidate_completed.selected_calls-candidate_before.selected_calls!=4150 ||
           candidate_completed.baseline_calls!=candidate_before.baseline_calls ||
           candidate_completed.invalid_receipts!=candidate_before.invalid_receipts)
            throw std::runtime_error("Image rejected: MOSS Q4 BK64/raw64 candidate B actual50-step4150FFN-up pipeline selection proof missing");
        for(unsigned layer=0;layer<42;layer++)
            if(candidate_completed.layer_calls[layer]-candidate_before.layer_calls[layer]!=(layer==22?50u:100u) ||
               candidate_completed.baseline_layer_calls[layer]!=candidate_before.baseline_layer_calls[layer])
                throw std::runtime_error("Image rejected: MOSS Q4 BK64/raw64 candidate B positive42/negative41 layer selection differs");
        const bool candidate_small=request->width==256 && request->height==256;
        const bool candidate_large=request->width==1024 && request->height==1024;
        if(!candidate_small && !candidate_large)
            throw std::runtime_error("Image rejected: MOSS Q4 BK64/raw64 candidate B exact real-token shape size not admitted");
        for(unsigned token=0;token<4;token++) {
            const uint64_t expected=(token==(candidate_small?0u:2u)?2050u:token==(candidate_small?1u:3u)?2100u:0u);
            if(candidate_completed.token_calls[token]-candidate_before.token_calls[token]!=expected ||
               candidate_completed.baseline_token_calls[token]!=candidate_before.baseline_token_calls[token])
                throw std::runtime_error("Image rejected: MOSS Q4 BK64/raw64 candidate B actual negative/positive input-token receipts differ");
        }
#endif
#if PS5_AI_MOSS_DIT_OPT_TRIAL
        if(!ps5_ai_ggml_wait_backend("dit_generation_completed",1))ps5_ai_ggml_retain("dit_generation_completion_unknown");
        nlohmann::json fused_receipts=nlohmann::json::array();
        for(unsigned k=1;k<=8;++k)fused_receipts.push_back(moss_dit_opt::GPU_recordings[k].load());
        const unsigned dit_mask=PS5_AI_MOSS_DIT_OPT_MASK;
        if((dit_mask&1)&&(!moss_dit_opt::cache_hits.load()||!moss_dit_opt::cache_misses.load()))throw std::runtime_error("DiT invariant cache not used");
        if((dit_mask&2)&&!moss_dit_opt::GPU_recordings[moss_dit_opt::SITU].load())throw std::runtime_error("SiTU GPU fusion not used");
        if(dit_mask&4)for(unsigned k=2;k<=5;++k)if(!moss_dit_opt::GPU_recordings[k].load())throw std::runtime_error("Norm/residual/QK GPU fusion not used");
        if((dit_mask&8)&&(!moss_dit_opt::scale_elisions.load()||!moss_dit_opt::GPU_recordings[8].load()))throw std::runtime_error("LoRA unit-scale/packed qkv fusion not used");
        write_event({{"event","moss_dit_optimization_completed"},{"mask",dit_mask},{"cache_graph_hits",moss_dit_opt::cache_hits.load()},{"cache_graph_misses",moss_dit_opt::cache_misses.load()},{"rope_builds",moss_dit_opt::rope_builds.load()},{"scale_elisions",moss_dit_opt::scale_elisions.load()},{"GPU_recordings_kinds1_to8",fused_receipts},{"finite_DiT_outputs",precision_dit_outputs.load()},{"nonfinite",precision_nonfinite.load()}});
        moss_dit_opt::end_completed();
#endif
        // Collect already-completed sampled graphs before publishing any PNG.
#if PS5_AI_MOSS_GPU_PROFILE
        const bool gpu_profile_ok=ps5_ai_moss_gpu_profile_job_end(&gpu_profile)!=0;
        gpu_profile_open=false;
        if(!gpu_profile_ok) {
            write_event({{"event","moss_gpu_profile_rejected"},{"report",moss_gpu_profile_json(gpu_profile)}});
            throw std::runtime_error("Image rejected: bounded GPU operator timing evidence is incomplete");
        }
#endif
        if(!ps5_ai_direct_pool_log_checkpoint())
            throw std::runtime_error("Direct pool log checkpoint failed; image not published");
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
        moss_lora_verified = true;
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
#if PS5_AI_MOSS_GPU_PROFILE
    if(gpu_profile_open) {
        (void)ps5_ai_moss_gpu_profile_job_end(&gpu_profile);
        gpu_profile_open=false;
    }
#endif
    sd_set_backend_eval_callback(nullptr, nullptr);
    sd_set_progress_callback(nullptr, nullptr);
    if (images) {
        free_sd_images(images, image_count);
        images = nullptr;
    }
    phase_profile_report();
    stats->gpu_used = observation.gpu_dit_math_nodes > 0 ? 1 : 0;
    stats->generation_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    stats->peak_cpu_bytes = std::max(stats->peak_cpu_bytes, cpu_peak_bytes());
    ps5_ai_native_capacity_snapshot("after_moss_generation");
    ps5_ai_flex_stats pool = {};
    ps5_ai_flex_snapshot(&pool);
    ps5_ai_moss_quant_receipt quant_after{};ps5_ai_moss_quant_snapshot(&quant_after);
    ps5_ai_moss_q4_bk64_report candidate_after{};ps5_ai_moss_q4_bk64_snapshot(&candidate_after);
    write_event({{"event", "generation_result"}, {"success", result == 0}, {"model_family", "MOSS-Image"}, {"mode", "text2img"},
                 {"gpu_operator_profile",moss_gpu_profile_json(gpu_profile)},
                 {"width",request->width},{"height",request->height},
                 {"moss_q4_bk64_B",moss_q4_bk64_generation_json(candidate_before,candidate_after)},
                 {"moss_lora_path", engine_config.moss_lora_path}, {"moss_lora_strength", engine_config.moss_lora_strength},
                 {"moss_lora_loaded_tensors", moss_lora_loaded.load()}, {"moss_lora_matched_tensors", moss_lora_matched.load()},
                 {"moss_lora_used_tensors", moss_lora_used.load()}, {"moss_lora_skipped_tensors", moss_lora_skipped.load()},
                 {"moss_lora_application_verified", moss_lora_verified.load()},
                 {"flow_shift",request->flow_shift},{"scheduler",PS5_AI_MOSS_REQUEST_SCHEDULER},{"sampler",request->sampler},{"cfg_scale",request->cfg_scale},
                 {"moss_sampler_trial",bool(PS5_AI_MOSS_SAMPLER_TRIAL)},{"moss_sampler_policy",PS5_AI_MOSS_SAMPLER_POLICY},
                 {"moss_bounded_weight_upload_trial",PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL!=0},
                 {"moss_bounded_weight_upload_policy",PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_POLICY},
                 {"bounded_weight_upload_enabled",ps5_ai_moss_bounded_weight_upload_enabled()!=0},
                 {"moss_spectrum",moss_spectrum_json()},
                 {"moss_turbo_trial",bool(PS5_AI_MOSS_TURBO_TRIAL)},{"moss_turbo_policy",PS5_AI_MOSS_TURBO_POLICY},
                     {"moss_lora_fp32_shared_trial",bool(PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL)},
                     {"moss_lora_f16_storage_trial",bool(PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL)},
                     {"moss_lora_precision_policy",PS5_AI_MOSS_LORA_PRECISION_POLICY},
                     {"moss_turbo_bounded_upload_trial",bool(PS5_AI_MOSS_TURBO_BOUNDED_UPLOAD_TRIAL)},
                     {"moss_turbo_bounded_upload_policy",PS5_AI_MOSS_TURBO_BOUNDED_UPLOAD_POLICY},
                     {"moss_ffn_token_chunk_trial",bool(PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL)},{"moss_ffn_chunk_tokens",PS5_AI_MOSS_FFN_CHUNK_TOKENS},
                     {"moss_ffn_chunk_policy",PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL?PS5_AI_MOSS_FFN_CHUNK_POLICY:"disabled"},
                 {"effective_adapter",moss_effective_lora_identity()},
                 {"moss_lora_storage_pipeline_receipt",moss_lora_storage_generation_json(storage_before)},
                 {"custom_sigmas",PS5_AI_MOSS_TURBO_TRIAL?nlohmann::json::array({1.0,.910,.717,.024,0.0}):nlohmann::json(nullptr)},
                 {"custom_sigmas_second_flow_shift",false},
                 {"negative_cfg_skip_layers",{22}},
                 {"negative_cfg_skip_scope",PS5_AI_MOSS_TURBO_TRIAL?"no_negative_forward_CFG1":"all_negative_forwards_existing_MOSS_policy"},
                 {"negative_cfg_skip_applied",!bool(PS5_AI_MOSS_TURBO_TRIAL)},
                 {"expected_dit_forwards",PS5_AI_MOSS_TURBO_TRIAL?uint64_t(4):PS5_AI_MOSS_SPECTRUM_TRIAL?uint64_t(60):uint64_t(request->steps)*2},
                 {"expected_q4_f32acc_recordings",PS5_AI_MOSS_TURBO_TRIAL?moss_ffn_chunk_expected(observation):PS5_AI_MOSS_SPECTRUM_TRIAL?uint64_t(9960):uint64_t(request->steps)*(42+41)*4},
                 {"requested_steps",request->steps},{"actual_steps",stats->actual_steps},
                 {"mask_prepared",stats->mask_prepared!=0},{"mask_denoise_evaluations",stats->mask_denoise_evaluations},
                 {"mask_preserved_pixels",stats->mask_preserved_pixels},{"mask_changed_pixels",stats->mask_changed_pixels},
                 {"init_encoded",observation.init_encoded},{"init_encode_seconds",stats->init_encode_seconds},
                 {"ip_adapter_tokens",stats->ip_adapter_tokens},{"ip_adapter_seconds",stats->ip_adapter_seconds},
                 {"gpu_ip_attention_verified",stats->gpu_ip_attention_verified!=0},
                 {"controlnet",request->controlnet},{"gpu_controlnet_verified",stats->gpu_controlnet_verified!=0},
                 {"controlnet_evals",stats->controlnet_evals},{"controlnet_tensors",stats->controlnet_tensors},{"controlnet_seconds",stats->controlnet_seconds},
                 {"gpu_used", stats->gpu_used != 0}, {"gpu_math_nodes", observation.gpu_math_nodes},
                 {"gpu_dit_math_nodes", observation.gpu_dit_math_nodes},
                 {"gpu_vae_math_nodes", observation.gpu_vae_math_nodes},
                 {"gpu_flash_attention_nodes", observation.gpu_flash_attention_nodes},
                 {"gpu_observation", "sampled completed witnesses; not total GPU operation counts"},
                 {"dit_witness_requested", observation.requested_dit != nullptr},
                 {"vae_witness_requested", observation.requested_vae != nullptr},
                 {"flash_witness_requested", observation.requested_flash != nullptr},
                 {"owned_direct_data_reserved_bytes", pool.reserved_bytes},
                 {"owned_direct_data_used_bytes", pool.used_bytes},
                 {"owned_direct_data_peak_used_bytes", pool.peak_used_bytes},
                 {"owned_direct_data_free_bytes", pool.free_bytes},
                 {"owned_direct_data_fault_retained", pool.fault_retained != 0},
                 {"cpu_clip_shared_direct_live_bytes",cpu_clip_pool_live_bytes.load()},
                 {"cpu_clip_shared_direct_peak_bytes",cpu_clip_pool_peak_bytes.load()},
                 {"owned_pool_peak_includes_CPU_text_weights",true},
                 {"partial_gpu_taesd_decode", partial_taesd_gpu_decode},
                 {"backend", engine_config.backend},
                 {"fp16_math",precision_fp16},{"finite_checked_outputs",precision_outputs.load()},
                 {"moss_f32_accumulation",bool(PS5_AI_MOSS_F32_ACCUMULATION)},
                 {"q4_matrix_recorded_f32acc",quant_after.q4_f32acc-quant_before.q4_f32acc},
                 {"q4_matrix_recorded_f16acc",quant_after.q4_f16acc-quant_before.q4_f16acc},
                 {"q4_matrix_accumulation_invalid",quant_after.invalid-quant_before.invalid},
                 {"accumulation_observation_scope","recorded actual selected scalar matrix pipelines; generation GPU completion checked separately"},
                 {"performance_trial",bool(PS5_AI_MOSS_PERFORMANCE_TRIAL)},{"performance_variant",PS5_AI_MOSS_PERF_VARIANT},
                 {"submit_multiplier",perf_submit_multiplier.load()},{"max_nodes_per_submit",perf_max_nodes.load()},
                 {"limited_elementwise_fusion",perf_limited_fusion.load()!=0},
                 {"limited_fusion_selected_graph_nodes",perf_fused_selected.load(std::memory_order_relaxed)-fused_before},
                 {"finite_checked_values",precision_values.load()},{"nonfinite_values",precision_nonfinite.load()},
                 {"finite_checked_dit_outputs",precision_dit_outputs.load()},
                 {"finite_checked_vae_outputs",precision_vae_outputs.load()},
                 {"seconds_including_initial_model_load", stats->generation_seconds},
                 {"process_peak_rss_bytes", stats->peak_cpu_bytes ? nlohmann::json(stats->peak_cpu_bytes) : nlohmann::json(nullptr)},
                 {"peak_gpu_bytes", nullptr}, {"gpu_memory_measurement", "unavailable"}});
    if (result == 0 && progress) {
        progress(1.0, user);
    }
#if PS5_AI_MOSS_TURBO_TRIAL
    if(result==0 && request->width==256 && request->height==256) {
        if(!ps5_ai_ggml_log_healthy() || pool.fault_retained)
            throw std::runtime_error("Turbo small-result durable completion proof failed;1024 remains closed");
        moss_turbo_small_image_completed.store(true,std::memory_order_release);
    }
#endif
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
