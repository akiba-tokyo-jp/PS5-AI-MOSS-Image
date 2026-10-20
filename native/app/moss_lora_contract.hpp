/* SPDX-License-Identifier: MIT
 * Exact selected MOSS LoRA shape/name contract, independent of GPU execution.
 * All504 F32 tensors are mandatory; q/k/v keep separate rank16 A/B pairs.
 */
#ifndef PS5_AI_MOSS_LORA_CONTRACT_HPP
#define PS5_AI_MOSS_LORA_CONTRACT_HPP
#ifndef PS5_AI_MOSS_TURBO_TRIAL
#define PS5_AI_MOSS_TURBO_TRIAL 0
#endif
#if PS5_AI_MOSS_TURBO_TRIAL != 0 && PS5_AI_MOSS_TURBO_TRIAL != 1
#error PS5_AI_MOSS_TURBO_TRIAL must be0or1
#endif
#include "moss_lora_f16_storage_trial.h"
#if PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL
#include "moss_lora_f16_storage_contract.hpp"
namespace ps5_ai_moss_lora=ps5_ai_moss_lora_f16_storage;
#elif PS5_AI_MOSS_TURBO_TRIAL
#include "moss_turbo_lora_contract.hpp"
namespace ps5_ai_moss_lora=ps5_ai_moss_turbo_lora;
#else
#include <cstdint>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>

namespace ps5_ai_moss_lora {
constexpr size_t tensor_count = 504;
struct Shape { int64_t input, output; std::string model_weight; };
inline const std::map<std::string, Shape>& expected() {
    static const auto catalog = [] {
        std::map<std::string, Shape> result;
        for (unsigned layer = 0; layer < 42; ++layer) {
            const auto base = std::string("model.diffusion_model.layers.") + std::to_string(layer) + ".proj.";
            for (const std::string module : {"q", "k", "v", "o", "ffn_up_gate", "ffn_down"}) {
                const int64_t input = module == "ffn_down" ? 8192 : 3072;
                const int64_t output = module == "ffn_up_gate" ? 16384 : 3072;
                const bool qkv = module == "q" || module == "k" || module == "v";
                const auto weight = base + (qkv ? "qkv" : module) + ".weight";
                const auto slice = module == "k" ? ".1" : module == "v" ? ".2" : "";
                const auto name = std::string("lora.") + weight + slice;
                result.emplace(name + ".lora_down", Shape{input, 16, weight});
                result.emplace(name + ".lora_up", Shape{16, output, weight});
            }
        }
        return result;
    }();
    return catalog;
}
inline void validate_tensor(const std::string& name, bool f32, int dimensions, int64_t ne0, int64_t ne1) {
    const auto found = expected().find(name);
    if (found == expected().end() || !f32 || dimensions != 2 ||
        ne0 != found->second.input || ne1 != found->second.output)
        throw std::runtime_error("MOSS selected LoRA name/F32/rank16/shape contract mismatch");
}
inline void validate_counts(size_t loaded, size_t matched, size_t used, size_t skipped, bool completed) {
    if (loaded != tensor_count || matched != tensor_count || skipped ||
        (completed && used != tensor_count) || used > tensor_count)
        throw std::runtime_error("MOSS selected LoRA requires all504 tensors matched and used; no skipped/unused tensors allowed");
}
}
#endif /* Separate Turbo rank144 core; normal rank16 body remains unchanged. */
#endif
