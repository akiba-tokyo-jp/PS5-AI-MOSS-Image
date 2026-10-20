/* SPDX-License-Identifier: MIT -- proposed optional strict contract only. */
#ifndef PS5_AI_MOSS_TURBO_LORA_CONTRACT_HPP
#define PS5_AI_MOSS_TURBO_LORA_CONTRACT_HPP
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
/* Normal rank16 validation remains in moss_lora_contract.hpp, unchanged.
 * A future explicit core-side Turbo branch must call this contract only after
 * the pinned composed adapter identity and multiplier1.0 have been admitted.
 * Native-only flags cannot change the already compiled core rank16 contract.
 */
namespace ps5_ai_moss_turbo_lora {
constexpr size_t tensor_count=504;
struct Shape { int64_t input, output; std::string model_weight; };
inline const std::map<std::string,Shape>& expected() {
    static const auto catalog=[] {
        std::map<std::string,Shape> result;
        for(unsigned layer=0;layer<42;++layer) {
            const auto base=std::string("model.diffusion_model.layers.")+std::to_string(layer)+".proj.";
            for(const std::string module:{"q","k","v","o","ffn_up_gate","ffn_down"}) {
                const int64_t input=module=="ffn_down"?8192:3072;
                const int64_t output=module=="ffn_up_gate"?16384:3072;
                const bool qkv=module=="q"||module=="k"||module=="v";
                const auto weight=base+(qkv?"qkv":module)+".weight";
                const auto slice=module=="k"?".1":module=="v"?".2":"";
                const auto name=std::string("lora.")+weight+slice;
                result.emplace(name+".lora_down",Shape{input,144,weight});
                result.emplace(name+".lora_up",Shape{144,output,weight});
            }
        }
        return result;
    }();
    return catalog;
}
inline void validate_tensor(const std::string& name,bool f32,int dimensions,int64_t ne0,int64_t ne1) {
    const auto found=expected().find(name);
    if(found==expected().end()||!f32||dimensions!=2||ne0!=found->second.input||ne1!=found->second.output)
        throw std::runtime_error("MOSS Turbo composed LoRA exact F32/rank144/name/shape contract mismatch");
}
inline void validate_counts(size_t loaded,size_t matched,size_t used,size_t skipped,bool completed) {
    if(loaded!=504||matched!=504||skipped||(completed&&used!=504)||used>504)
        throw std::runtime_error("MOSS Turbo composed adapter must load/match/use all504 matrices");
}
}
#endif
