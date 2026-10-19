#ifndef PS5_AI_ENGINE_CONFIG_H
#define PS5_AI_ENGINE_CONFIG_H
#include "json.hpp"
#include "stable-diffusion.h"
#include "memory_plan.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdint>

struct PS5AIEngineConfig {
    // MOSS uses a split DiT, Qwen3.5 text encoder, external tokenizer and FLUX.2 VAE.
    // The DiT must never be passed as an SDXL combined checkpoint.
    std::string model, llm, vae, tokenizer, taesd, backend, params_backend, max_vram;
    std::string moss_lora_path;
    double moss_lora_strength = 0.7;
    std::string manga_lora_path;
    std::string clip_vision_path, ip_adapter_path, reference_params_backend;
    std::string controlnet_canny_path, controlnet_openpose_path;
    std::string flash_attn_mode = "auto,unet=on";
    bool batched_cfg = false;
    int conditioning_cache_size = 2;
    std::string log = "/app0/ai-runtime/logs/engine.jsonl";
    int cpu_threads = 3;
    int vae_tile_size = 256; // Image pixels; the selected VAE resolves latent tile sizes.
    float vae_tile_overlap = 0.5f;
    bool vae_tiling = true, mmap = false, disable_prefetch = true;
    bool eager_load = false, auto_fit = true, tae_preview_only = false;
    uint64_t gpu_pool_bytes = UINT64_C(12) << 30;
    uint64_t working_reserve_bytes = UINT64_C(1) << 30;

    static const nlohmann::json& engine_object(const nlohmann::json& document) {
        if (!document.is_object()) throw std::runtime_error("MOSS config must be a JSON object");
        for (auto it = document.begin(); it != document.end(); ++it)
            if (it.key() != "engine" && it.key() != "logging")
                throw std::runtime_error("Unknown MOSS config top-level key; only engine and optional logging are accepted");
        if (!document.contains("engine") || !document.at("engine").is_object())
            throw std::runtime_error("MOSS config requires an explicit engine object; no default engine is selected");
        return document.at("engine");
    }

    static void validate_keys(const nlohmann::json& object) {
        if (!object.is_object()) throw std::runtime_error("engine must be a JSON object");
        static constexpr const char* allowed[] = {
            "model", "llm", "vae", "tokenizer", "taesd_path", "backend", "params_backend", "max_vram_gib",
            "moss_lora_path", "moss_lora_strength",
            "manga_lora_path", "clip_vision_path", "ip_adapter_path", "controlnet_canny_path", "controlnet_openpose_path",
            "flash_attn_mode", "batched_cfg", "conditioning_cache_size", "log", "cpu_threads", "vae_tile_size",
            "vae_tile_overlap", "vae_tiling", "mmap", "disable_prefetch", "eager_load", "auto_fit", "tae_preview_only",
            "gpu_pool_bytes", "working_reserve_bytes"
        };
        for (auto it = object.begin(); it != object.end(); ++it) {
            bool known = false;
            for (const char* key : allowed) if (it.key() == key) { known = true; break; }
            if (!known) throw std::runtime_error("Unknown engine configuration field: " + it.key());
        }
    }

    static const char* selected_moss_lora_path() {
        return "/app0/models/moss-image-nsfw-uncensored-r16-f32.safetensors";
    }

    static void validate_selected_lora(const nlohmann::json& object) {
        const std::string path = object.value("moss_lora_path", selected_moss_lora_path());
        if (path != selected_moss_lora_path())
            throw std::runtime_error("MOSS selected adapter requires its exact canonical F32 path; empty/base-only or other adapters are not selected");
        if (object.contains("moss_lora_strength") && !object.at("moss_lora_strength").is_number())
            throw std::runtime_error("engine.moss_lora_strength must be numeric");
        const double strength = object.value("moss_lora_strength", 0.7);
        if (!std::isfinite(strength) || strength <= 0.0 || strength > 1.5 ||
            !std::isfinite(static_cast<float>(strength)) || static_cast<float>(strength) <= 0.f)
            throw std::runtime_error("engine.moss_lora_strength must remain finite and positive as float, at most1.5; selected LoRA cannot be disabled");
    }

    static uint64_t memory_integer(const nlohmann::json& object, const char* key, uint64_t fallback) {
        if (!object.contains(key)) return fallback;
        const auto& value = object.at(key);
        if (!value.is_number_integer())
            throw std::runtime_error(std::string("engine.") + key + " must be an unsigned integer byte count");
        if (value.is_number_unsigned()) return value.get<uint64_t>();
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value < 0)
            throw std::runtime_error(std::string("engine.") + key + " cannot be negative");
        return static_cast<uint64_t>(signed_value);
    }

    static ps5_ai_memory_plan memory_plan(const nlohmann::json& object) {
        // Validate before explicit AGC/Direct setup as well as full engine parsing.
        validate_keys(object);
        validate_selected_lora(object);
        if (!object.contains("gpu_pool_bytes") || !object.contains("working_reserve_bytes"))
            throw std::runtime_error("MOSS requires explicit gpu_pool_bytes and working_reserve_bytes; no inherited memory fallback is used");
        ps5_ai_memory_plan result = {
            memory_integer(object, "gpu_pool_bytes", PS5_AI_MEMORY_PLAN_DEFAULT_POOL_BYTES),
            memory_integer(object, "working_reserve_bytes", PS5_AI_MEMORY_PLAN_DEFAULT_WORKING_RESERVE_BYTES)};
        // These bounds match the actual pool's page-index/address arithmetic,
        // not an arbitrary application memory ceiling. Runtime quota/backing
        // admission is performed by the pool, after this configuration read.
        if (!result.gpu_pool_bytes || result.gpu_pool_bytes % PS5_AI_MEMORY_PLAN_PAGE_BYTES ||
            result.gpu_pool_bytes / PS5_AI_MEMORY_PLAN_PAGE_BYTES > UINT32_MAX ||
            result.gpu_pool_bytes >= (UINT64_C(1) << 47) ||
            result.working_reserve_bytes < PS5_AI_MEMORY_PLAN_DEFAULT_WORKING_RESERVE_BYTES ||
            result.working_reserve_bytes % PS5_AI_MEMORY_PLAN_PAGE_BYTES ||
            result.gpu_pool_bytes > UINT64_MAX - result.working_reserve_bytes)
            throw std::runtime_error("Native Direct MOSS memory bytes must fit pool address/page arithmetic; reserve is aligned and at least1GiB outside the pool");
        return result;
    }

    static bool model_path_ok(const std::string& path) {
        return path.rfind("/app0/models/", 0) == 0 && path.size() > 13 &&
               path.find("..") == std::string::npos && path.find('\0') == std::string::npos &&
               path.find('\\') == std::string::npos;
    }

    static int bounded_integer(const nlohmann::json& object, const char* key, int fallback, int lower, int upper) {
        if (!object.contains(key)) return fallback;
        const auto& value = object.at(key);
        if (!value.is_number_integer() ||
            (value.is_number_unsigned() && value.get<uint64_t>() > static_cast<uint64_t>(upper)))
            throw std::runtime_error(std::string("engine.") + key + " must be a bounded integer");
        const auto number = value.get<int64_t>();
        if (number < lower || number > upper)
            throw std::runtime_error(std::string("engine.") + key + " is outside its supported range");
        return static_cast<int>(number);
    }

    static PS5AIEngineConfig parse(const nlohmann::json& object) {
        if (!object.is_object()) throw std::runtime_error("engine must be a JSON object");
        PS5AIEngineConfig result;
        const auto plan = memory_plan(object);
        result.gpu_pool_bytes = plan.gpu_pool_bytes;
        result.working_reserve_bytes = plan.working_reserve_bytes;
        result.model = object.value("model", "/app0/models/MOSS-Image-SFT-Q4_K.gguf");
        result.llm = object.value("llm", "/app0/models/Qwen3.5-9B-Q4_K_M.gguf");
        result.moss_lora_path = object.value("moss_lora_path", selected_moss_lora_path());
        result.moss_lora_strength = object.value("moss_lora_strength", result.moss_lora_strength);
        result.manga_lora_path = object.value("manga_lora_path", "");
        result.clip_vision_path=object.value("clip_vision_path", "");
        result.ip_adapter_path=object.value("ip_adapter_path", "");
        result.vae = object.value("vae", "/app0/models/flux2-vae_small.safetensors");
        result.tokenizer = object.value("tokenizer", "/app0/models/tokenizer_qwen3.5.json");
        result.taesd = object.value("taesd_path", "");
        result.backend = object.value("backend", "");
        result.params_backend = object.value("params_backend", "");
        result.controlnet_canny_path=object.value("controlnet_canny_path", "");
        result.controlnet_openpose_path=object.value("controlnet_openpose_path", "");
        if (!result.manga_lora_path.empty() || !result.clip_vision_path.empty() ||
            !result.ip_adapter_path.empty() || !result.controlnet_canny_path.empty() ||
            !result.controlnet_openpose_path.empty() || !result.taesd.empty())
            throw std::runtime_error("MOSS text2img does not accept WAI/SDXL LoRA, IP-Adapter, ControlNet or four-channel TAESD weights");
        result.max_vram = object.value("max_vram_gib", "");
        result.flash_attn_mode = object.value("flash_attn_mode", result.flash_attn_mode);
        result.batched_cfg = object.value("batched_cfg", result.batched_cfg);
        result.conditioning_cache_size = bounded_integer(object, "conditioning_cache_size", result.conditioning_cache_size, 0, 2);
        if (result.flash_attn_mode != "auto,unet=on" && result.flash_attn_mode != "off,unet=on" &&
            result.flash_attn_mode != "auto" && result.flash_attn_mode != "on" && result.flash_attn_mode != "off")
            throw std::runtime_error("Unsupported MOSS flash-attention policy");
        result.log = object.value("log", result.log);
        result.cpu_threads = bounded_integer(object, "cpu_threads", result.cpu_threads, 1, 4);
        result.vae_tile_size = bounded_integer(object, "vae_tile_size", result.vae_tile_size, 32, 1024);
        result.vae_tile_overlap = object.value("vae_tile_overlap", result.vae_tile_overlap);
        result.vae_tiling = object.value("vae_tiling", result.vae_tiling);
        result.mmap = object.value("mmap", result.mmap);
        if (result.mmap) throw std::runtime_error("MOSS preparation requires mmap=false; console mmap admission is unverified");
        result.disable_prefetch = object.value("disable_prefetch", result.disable_prefetch);
        result.eager_load = object.value("eager_load", result.eager_load);
        result.auto_fit = object.value("auto_fit", result.auto_fit);
        result.tae_preview_only = object.value("tae_preview_only", result.tae_preview_only);
        if (!model_path_ok(result.model) || !model_path_ok(result.llm) || !model_path_ok(result.vae) || !model_path_ok(result.tokenizer))
            throw std::runtime_error("engine model/llm/vae/tokenizer must name split model files inside /app0/models/");
        if (result.cpu_threads < 1 || result.cpu_threads > 4)
            throw std::runtime_error("engine.cpu_threads must be between 1 and 4");
        if (result.vae_tile_size < 32 || result.vae_tile_size > 1024 || result.vae_tile_size % 8 != 0)
            throw std::runtime_error("engine.vae_tile_size must be 32..1024 image pixels, divisible by 8");
        if (!(result.vae_tile_overlap >= 0.f && result.vae_tile_overlap <= 0.5f))
            throw std::runtime_error("engine.vae_tile_overlap must be finite and between 0 and 0.5");
        if (result.log != "/app0/ai-runtime/logs/engine.jsonl")
            throw std::runtime_error("engine.log must match the fixed native V9 public journal path");
        for (const auto* setting : {&result.backend, &result.params_backend, &result.max_vram}) {
            if (setting->size() > 512 || setting->find('\0') != std::string::npos)
                throw std::runtime_error("engine backend/budget setting is invalid");
        }
        return result;
    }

    void resolve_defaults(const std::string& gpu_name) {
        // SD.cpp names the diffusion backend selector 'unet' for DiT models too.
        const std::string expected = "cpu,unet=" + gpu_name + ",vae=" + gpu_name;
        if (backend.empty()) backend = expected;
        if (params_backend.empty()) params_backend = "cpu,unet=" + gpu_name + ",te=disk,vae=" + gpu_name;
        auto local_only = [&](const std::string& assignment, bool parameters) {
            size_t begin = 0;
            while (begin <= assignment.size()) {
                const auto end = assignment.find(',', begin);
                const auto item = assignment.substr(begin, end == std::string::npos ? end : end - begin);
                const auto equal = item.find('=');
                const auto value = equal == std::string::npos ? item : item.substr(equal + 1);
                if (value != "cpu" && value != "CPU" && value != gpu_name && !(parameters && value == "disk"))
                    throw std::runtime_error("Only local CPU/native Vulkan backends (or disk parameter storage) are allowed");
                if (end == std::string::npos) break;
                begin = end + 1;
            }
        };
        local_only(backend, false);
        local_only(params_backend, true);
        reference_params_backend=params_backend;
        if (backend != expected || (params_backend != "cpu" && params_backend != "disk" &&
            params_backend != "cpu,unet=disk" && params_backend != "cpu,unet=" + gpu_name &&
            params_backend != "cpu,unet=" + gpu_name + ",te=disk" &&
            params_backend != "cpu,unet=" + gpu_name + ",te=disk,vae=" + gpu_name))
            throw std::runtime_error("Native Direct MOSS requires CPU Qwen3.5 text encoding, native Vulkan DiT/VAE compute, and local parameter storage");
    }

    void validate_actual_pool(const std::string& gpu_name, uint64_t actual_bytes) const {
        if (actual_bytes != gpu_pool_bytes)
            throw std::runtime_error("Native Direct MOSS configured GPU pool does not match the actual prepared owned pool");
        const std::string prefix = gpu_name + "=";
        if (max_vram.rfind(prefix, 0) != 0 || max_vram.find(',', prefix.size()) != std::string::npos)
            throw std::runtime_error("Native Direct MOSS max_vram_gib must explicitly name only the native Vulkan device");
        const std::string number = max_vram.substr(prefix.size());
        size_t consumed = 0;
        const double gib = std::stod(number, &consumed);
        if (number.empty() || consumed != number.size() || !std::isfinite(gib) || gib <= 0.0 ||
            static_cast<long double>(gib) * (UINT64_C(1) << 30) > static_cast<long double>(actual_bytes))
            throw std::runtime_error("Native Direct MOSS max_vram_gib must be positive and no larger than the real owned pool");
    }

    bool requires_dit_flash_attention() const {
        return flash_attn_mode == "on" || flash_attn_mode == "auto,unet=on" || flash_attn_mode == "off,unet=on";
    }

    bool partial_taesd_gpu_decode(const std::string& gpu_name) const {
        return !taesd.empty() && !tae_preview_only && backend == "cpu,vae=" + gpu_name;
    }

    void apply_context(sd_ctx_params_t& params) const {
        params.model_path = nullptr;
        params.diffusion_model_path = model.c_str();
        params.llm_path = llm.c_str();
        params.tokenizer = tokenizer.c_str();
        params.vae_path = vae.c_str();
        params.taesd_path = nullptr;
        params.backend = backend.c_str();
        params.params_backend = reference_params_backend.c_str();
        params.clip_vision_path=clip_vision_path.empty()?nullptr:clip_vision_path.c_str();
        params.ip_adapter_path=ip_adapter_path.empty()?nullptr:ip_adapter_path.c_str();
        params.max_vram = max_vram.empty() ? nullptr : max_vram.c_str();
        params.n_threads = cpu_threads;
        params.enable_mmap = mmap;
        // No WAI-specific adapter is loaded; weights remain immutable.
        params.lora_apply_mode = LORA_APPLY_AT_RUNTIME;
        params.disable_prefetch = disable_prefetch;
        params.eager_load = eager_load;
        params.auto_fit = auto_fit;
        params.tae_preview_only = tae_preview_only;
        params.conditioning_cache_size = conditioning_cache_size;
        params.batched_cfg = batched_cfg;
        params.flash_attn_mode = flash_attn_mode.c_str();
    }

    void apply_image(sd_img_gen_params_t& params) const {
        params.vae_tiling_params.enabled = vae_tiling;
        params.vae_tiling_params.tile_size_w = vae_tile_size;
        params.vae_tiling_params.tile_size_h = vae_tile_size;
        params.vae_tiling_params.target_overlap = vae_tile_overlap;
    }
};
#endif
