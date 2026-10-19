// SPDX-License-Identifier: GPL-3.0-or-later
// Host-side configuration checks; this does not execute inference or contact PS5.
#include "engine_config.h"
#include "moss_runtime_admission.h"
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

int main() {
    // The real deployment provides this explicit owned-pool admission value;
    // retain the inherited small fail-safe default for absent configuration.
    const auto empty = nlohmann::json{{"gpu_pool_bytes",12884901888ULL}};
    auto config = PS5AIEngineConfig::parse(empty);
    config.resolve_defaults("Vulkan0");
    require(config.model == "/app0/models/MOSS-Image-SFT-Q4_K.gguf", "wrong DiT default");
    require(config.llm == "/app0/models/Qwen3.5-9B-Q4_K_M.gguf", "Qwen LLM missing");
    require(config.vae == "/app0/models/flux2-vae_small.safetensors", "Qwen VAE missing");
    require(config.backend == "cpu,unet=Vulkan0,vae=Vulkan0", "DiT/VAE must compute on local GPU");
    require(config.params_backend == "cpu,unet=Vulkan0,te=disk,vae=Vulkan0", "wrong parameter residency");
    require(!config.batched_cfg && config.flash_attn_mode == "auto,unet=on", "wrong MOSS compute defaults");
    require(config.manga_lora_path.empty() && config.clip_vision_path.empty() && config.taesd.empty(), "SDXL-only weight default leaked");
    sd_ctx_params_t context = {};
    config.apply_context(context);
    require(context.model_path == nullptr, "DiT was incorrectly passed as combined SDXL checkpoint");
    require(context.diffusion_model_path == config.model.c_str(), "split diffusion model not connected");
    require(context.llm_path == config.llm.c_str() && context.vae_path == config.vae.c_str(), "split Qwen models not connected");
    require(context.taesd_path == nullptr && context.ip_adapter_path == nullptr && context.clip_vision_path == nullptr, "SDXL-only component connected");
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
    require(!config.mmap && !context.enable_mmap,"MOSS must not mmap models");
    require(context.tokenizer==config.tokenizer.c_str() && config.tokenizer=="/app0/models/tokenizer_qwen3.5.json","external Qwen3.5 tokenizer not connected");
    require(PS5_AI_MOSS_RUNTIME_ADMITTED==0,"preparation build runtime admission must stay false");
    require(PS5AIEngineConfig::parse(nlohmann::json::object()).gpu_pool_bytes==12884901888ULL,"12 GiB target silently reduced");
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
    std::cout << "PASS " << checks << " MOSS split-model/backend/admission checks; no inference executed\n";
}
