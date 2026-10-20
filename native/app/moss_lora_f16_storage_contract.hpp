/* SPDX-License-Identifier: MIT -- separate exact F16/rank144 contract. */
#ifndef PS5_AI_MOSS_LORA_F16_STORAGE_CONTRACT_HPP
#define PS5_AI_MOSS_LORA_F16_STORAGE_CONTRACT_HPP
#include "moss_turbo_lora_contract.hpp"
namespace ps5_ai_moss_lora_f16_storage {
constexpr size_t tensor_count=ps5_ai_moss_turbo_lora::tensor_count;
inline const std::map<std::string,ps5_ai_moss_turbo_lora::Shape>& expected(){return ps5_ai_moss_turbo_lora::expected();}
inline void validate_tensor(const std::string& name,bool f16,int dimensions,int64_t ne0,int64_t ne1){
    const auto found=expected().find(name);
    if(found==expected().end()||!f16||dimensions!=2||ne0!=found->second.input||ne1!=found->second.output)
        throw std::runtime_error("MOSS lossy F16-storage experiment exact F16/rank144/name/shape mismatch");
}
inline void validate_counts(size_t loaded,size_t matched,size_t used,size_t skipped,bool completed){
    ps5_ai_moss_turbo_lora::validate_counts(loaded,matched,used,skipped,completed);
}
}
#endif
