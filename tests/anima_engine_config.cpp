// SPDX-License-Identifier: GPL-3.0-or-later
// Host-side configuration checks; this does not execute inference or contact PS5.
#include "engine_config.h"
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
    const auto empty = nlohmann::json{{"gpu_pool_bytes",9879683072ULL}};
    auto config = PS5AIEngineConfig::parse(empty);
    config.resolve_defaults("Vulkan0");
    require(config.model == "/app0/models/anima-turbo-v1.1-f16.safetensors", "wrong DiT default");
    require(config.llm == "/app0/models/qwen_3_06b_base.safetensors", "Qwen LLM missing");
    require(config.vae == "/app0/models/qwen_image_vae-f16.safetensors", "Qwen VAE missing");
    require(config.backend == "cpu,unet=Vulkan0,vae=Vulkan0", "DiT/VAE must compute on local GPU");
    require(config.params_backend == "cpu,unet=Vulkan0,te=disk,vae=Vulkan0", "wrong parameter residency");
    require(!config.batched_cfg && config.flash_attn_mode == "auto,unet=on", "wrong Turbo compute defaults");
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
    for (const char* key : {"model", "llm", "vae"}) {
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
    rejects([&]{ auto j=empty;j["flash_attn_mode"]="auto,unet=off";PS5AIEngineConfig::parse(j); }, "required DiT flash attention can be disabled");
    rejects([&]{ auto j=empty;j["gpu_pool_bytes"]=-1;PS5AIEngineConfig::parse(j); }, "negative Direct reservation accepted");
    rejects([&]{ auto j=empty;j["gpu_pool_bytes"]=9879683073ULL;PS5AIEngineConfig::parse(j); }, "unaligned Direct reservation accepted");
    config.max_vram="Vulkan0=8.75";
    config.validate_actual_pool("Vulkan0", config.gpu_pool_bytes);
    require(config.gpu_pool_bytes == 9879683072ULL, "existing 9422 MiB reservation changed");
    config.max_vram="Vulkan0=10";
    rejects([&]{ config.validate_actual_pool("Vulkan0",config.gpu_pool_bytes); }, "budget exceeds actual owned reservation");
    config.max_vram="RPC0=8.75";
    rejects([&]{ config.validate_actual_pool("Vulkan0",config.gpu_pool_bytes); }, "remote budget accepted");
    std::cout << "PASS " << checks << " Anima split-model/backend/admission checks; no inference executed\n";
}
