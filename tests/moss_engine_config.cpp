// SPDX-License-Identifier: GPL-3.0-or-later
// Host-side configuration checks; this does not execute inference or contact PS5.
#include "engine_config.h"
#include "moss_runtime_admission.h"
#include "phase_profile.h"
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>

static unsigned checks = 0;
static void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
static void rejects(const std::function<void()>& action, const char* message) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}

int main(int argc, char** argv) {
    require(PS5AIEngineConfig::engine_object(nlohmann::json{{"engine", nlohmann::json::object()}}).is_object(), "explicit root engine rejected");
    for (const auto& bad : {nlohmann::json::object(), nlohmann::json{{"enginee", nlohmann::json::object()}},
            nlohmann::json{{"engine", nullptr}}, nlohmann::json{{"engine", nlohmann::json::array()}},
            nlohmann::json{{"engine", nlohmann::json::object()}, {"unknown", true}}})
        rejects([&]{ PS5AIEngineConfig::engine_object(bad); }, "root engine/schema typo/default fallback must reject");
    // Keep the requested12GiB target explicit; parsing does not prove that the
    // native runtime can allocate it with the outside-pool reserve.
    const auto empty = nlohmann::json{{"gpu_pool_bytes",12884901888ULL},{"working_reserve_bytes",1073741824ULL}};
    static_assert(PS5_AI_MEMORY_PLAN_DEFAULT_POOL_BYTES == (UINT64_C(12) << 30), "requested12GiB profile constant");
    static_assert(PS5_AI_MEMORY_PLAN_DEFAULT_WORKING_RESERVE_BYTES == (UINT64_C(1) << 30), "explicit1GiB working reserve constant");
    for (const char* key : {"gpu_pool_bytes", "working_reserve_bytes"}) {
        auto missing = empty; missing.erase(key);
        rejects([&]{ PS5AIEngineConfig::parse(missing); }, "missing memory field silently defaulted in full parsing");
        rejects([&]{ PS5AIEngineConfig::memory_plan(missing); }, "missing memory field passed pre-AGC parsing");
    }
    for (const auto& bad : {nlohmann::json::object(),nlohmann::json(nullptr),nlohmann::json::array()}) {
        rejects([&]{ PS5AIEngineConfig::parse(bad); }, "absent/invalid memory object silently defaulted");
        rejects([&]{ PS5AIEngineConfig::memory_plan(bad); }, "absent/invalid memory object passed pre-AGC parsing");
    }
    for (const char* key : {"gpu_pool_bytes", "working_reserve_bytes"}) {
        auto invalid = empty; invalid[key] = nullptr;
        rejects([&]{ PS5AIEngineConfig::parse(invalid); }, "null memory field silently defaulted");
        rejects([&]{ PS5AIEngineConfig::memory_plan(invalid); }, "null memory field passed pre-AGC parsing");
    }
    auto inherited_reserve = empty; inherited_reserve["working_reserve_bytes"] = UINT64_C(64) << 20;
    rejects([&]{ PS5AIEngineConfig::memory_plan(inherited_reserve); }, "inherited64MiB reserve passed MOSS1GiB minimum");
    auto explicit_candidate = empty; explicit_candidate["gpu_pool_bytes"] = UINT64_C(10) << 30;
    require(PS5AIEngineConfig::memory_plan(explicit_candidate).gpu_pool_bytes == (UINT64_C(10) << 30), "explicit candidate memory plan was silently restored to12GiB");
    auto config = PS5AIEngineConfig::parse(empty);
    config.resolve_defaults("Vulkan0");
    require(config.model == "/app0/models/MOSS-Image-SFT-Q4_K.gguf", "wrong DiT default");
    require(config.llm == "/app0/models/Qwen3.5-9B-Q4_K_M.gguf", "Qwen LLM missing");
    require(config.vae == "/app0/models/flux2-vae_small.safetensors", "Qwen VAE missing");
    require(config.backend == "cpu,unet=Vulkan0,vae=Vulkan0", "DiT/VAE must compute on local GPU");
    require(config.params_backend == "cpu,unet=Vulkan0,te=disk,vae=Vulkan0", "wrong parameter residency");
    require(config.moss_lora_path == PS5AIEngineConfig::selected_moss_lora_path() && config.moss_lora_strength == 0.7, "requested MOSS adapter default is missing");
    for (const char* key : {"lora_path", "tokenzier", "model_path", "gpu_pool_byte", "api_token", "engine"}) {
        auto unknown = empty; unknown[key] = "must-not-be-ignored";
        rejects([&]{ PS5AIEngineConfig::parse(unknown); }, "unknown engine field silently ignored");
        rejects([&]{ PS5AIEngineConfig::memory_plan(unknown); }, "unknown engine field passed pre-AGC memory-plan admission");
    }
    require(ps5_ai_profile_group("MOSS-Image") == 0, "actual MOSS runner is missing from DiT timings");
    require(ps5_ai_profile_group("anima") == 0 && ps5_ai_profile_group("unet") == 0, "inherited actual runner timing classification changed");
    require(ps5_ai_profile_group("control_net") == 1 && ps5_ai_profile_group("llm") == 2 && ps5_ai_profile_group("vae") == 2, "non-DiT timing classification changed");
    require(!config.batched_cfg && config.flash_attn_mode == "auto,unet=on", "wrong MOSS compute defaults");
    require(config.manga_lora_path.empty() && config.clip_vision_path.empty() && config.taesd.empty(), "SDXL-only weight default leaked");
    sd_ctx_params_t context = {};
    config.apply_context(context);
    require(context.model_path == nullptr, "DiT was incorrectly passed as combined SDXL checkpoint");
    require(context.diffusion_model_path == config.model.c_str(), "split diffusion model not connected");
    require(context.llm_path == config.llm.c_str() && context.vae_path == config.vae.c_str(), "split Qwen models not connected");
    require(context.taesd_path == nullptr && context.ip_adapter_path == nullptr && context.clip_vision_path == nullptr, "SDXL-only component connected");
    require(context.lora_apply_mode == LORA_APPLY_AT_RUNTIME, "quantized base must retain runtime LoRA application");
    require(std::string(context.backend) == config.backend && std::string(context.params_backend) == config.params_backend, "context backends differ");
    sd_img_gen_params_t image = {};
    config.apply_image(image);
    require(image.vae_tiling_params.enabled && image.vae_tiling_params.tile_size_w == 256, "VAE tiling not connected");
    for (const char* key : {"model", "llm", "vae", "tokenizer"}) {
        for (const char* bad : {"", "/data/other.safetensors", "/app0/models/../other.safetensors", "/app0/models/"}) {
            rejects([&]{ auto j = empty; j[key] = bad; PS5AIEngineConfig::parse(j); }, "unsafe or empty split model path accepted");
        }
    }
    for (const char* key : {"manga_lora_path", "clip_vision_path", "ip_adapter_path", "controlnet_canny_path", "controlnet_openpose_path", "taesd_path"}) {
        rejects([&]{ auto j = empty; j[key] = "/app0/models/old-sdxl.safetensors"; PS5AIEngineConfig::parse(j); }, "SDXL-only configured weight accepted");
    }
    for (const char* backend : {"cpu", "cpu,unet=CPU,vae=Vulkan0", "cpu,unet=Vulkan0,vae=CPU", "cpu,unet=RPC0,vae=Vulkan0", "cpu,unet=Vulkan0,vae=Vulkan0,controlnet=Vulkan0"}) {
        rejects([&]{ auto j=empty; j["backend"]=backend; auto c=PS5AIEngineConfig::parse(j); c.resolve_defaults("Vulkan0"); }, "CPU/remote or stale ControlNet compute path accepted");
    }
    for(const char* mode:{"off","on","auto","auto,unet=on","off,unet=on"}) {
        auto j=empty;j["flash_attn_mode"]=mode;auto c=PS5AIEngineConfig::parse(j);
        require(c.requires_dit_flash_attention()==(std::string(mode)!="off" && std::string(mode)!="auto"),"flash proof policy mismatch");
    }
    rejects([&]{ auto j=empty;j["flash_attn_mode"]="bogus";PS5AIEngineConfig::parse(j); }, "unknown flash policy accepted");
    rejects([&]{ auto j=empty;j["mmap"]=true;PS5AIEngineConfig::parse(j); }, "unadmitted mmap enabled");
    for (const char* path : {"", "/app0/models/other.safetensors", "/app0/models/moss-image-nsfw-uncensored-r16-bf16.safetensors"}) {
        auto j=empty;j["moss_lora_path"]=path;
        rejects([&]{ PS5AIEngineConfig::parse(j); }, "base-only or noncanonical MOSS adapter accepted");
        rejects([&]{ PS5AIEngineConfig::memory_plan(j); }, "wrong MOSS adapter passed pre-AGC admission");
    }
    for (double strength : {0.0, -0.1, 1.5001, 1e-100, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        auto j=empty;j["moss_lora_strength"]=strength;
        rejects([&]{ PS5AIEngineConfig::parse(j); }, "invalid or disabled MOSS adapter strength accepted");
        rejects([&]{ PS5AIEngineConfig::memory_plan(j); }, "invalid MOSS adapter strength passed pre-AGC admission");
    }
    for (double strength : {0.1, 0.7, 1.5}) {
        auto j=empty;j["moss_lora_strength"]=strength;
        require(PS5AIEngineConfig::parse(j).moss_lora_strength == strength, "valid MOSS adapter strength changed");
    }
    rejects([&]{ auto j=empty;j["moss_lora_strength"]="0.7";PS5AIEngineConfig::parse(j); }, "string MOSS adapter strength accepted");
    require(!config.mmap && !context.enable_mmap,"MOSS must not mmap models");
    require(context.tokenizer==config.tokenizer.c_str() && config.tokenizer=="/app0/models/tokenizer_qwen3.5.json","external Qwen3.5 tokenizer not connected");
    require(PS5_AI_MOSS_RUNTIME_ADMITTED==0,"preparation build runtime admission must stay false");
    rejects([&]{ PS5AIEngineConfig::parse(nlohmann::json::object()); }, "unspecified memory must not inherit12GiB silently");
    require(config.working_reserve_bytes==1073741824ULL,"working reserve must remain explicit outside pool");
    rejects([&]{ auto j=empty;j["gpu_pool_bytes"]=-1;PS5AIEngineConfig::parse(j); }, "negative Direct reservation accepted");
    rejects([&]{ auto j=empty;j["gpu_pool_bytes"]=12884901889ULL;PS5AIEngineConfig::parse(j); }, "unaligned Direct reservation accepted");
    config.max_vram="Vulkan0=11.5";
    config.validate_actual_pool("Vulkan0", config.gpu_pool_bytes);
    require(config.gpu_pool_bytes == 12884901888ULL, "MOSS requested12GiB reservation changed");
    config.max_vram="Vulkan0=12.001";
    rejects([&]{ config.validate_actual_pool("Vulkan0",config.gpu_pool_bytes); }, "budget exceeds actual owned reservation");
    config.max_vram="RPC0=11.5";
    rejects([&]{ config.validate_actual_pool("Vulkan0",config.gpu_pool_bytes); }, "remote budget accepted");
    for (int i = 1; i < argc; ++i) {
        std::ifstream file(argv[i]);
        require(file.good(), "configuration example could not be opened");
        nlohmann::json example; file >> example;
        auto candidate = PS5AIEngineConfig::parse(example.at("engine"));
        candidate.resolve_defaults("Vulkan0");
        candidate.validate_actual_pool("Vulkan0", 12884901888ULL);
        require(candidate.model == "/app0/models/MOSS-Image-SFT-Q4_K.gguf", "configuration example contains a different model");
        require(candidate.gpu_pool_bytes == 12884901888ULL && candidate.working_reserve_bytes == 1073741824ULL, "configuration example changed the requested memory plan");
        require(!candidate.mmap && !candidate.batched_cfg, "configuration example restored incompatible inherited defaults");
    }
    std::cout << "PASS " << checks << " MOSS split-model/backend/admission checks; no inference executed\n";
}
