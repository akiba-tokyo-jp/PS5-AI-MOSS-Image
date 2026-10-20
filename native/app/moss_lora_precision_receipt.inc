/* SPDX-License-Identifier: GPL-3.0-or-later
 * Actual compile/dispatch receipts only. No Vulkan operation or GPU wait here.
 */
#include "moss_lora_f16_storage_trial.h"
#if PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL
#include "moss_lora_precision_module_pins.hpp"
extern "C" {
#include "sha256.h"
}
static std::mutex moss_lora_precision_mutex;
static ps5_ai_moss_lora_storage_receipt moss_lora_storage_report{};
static void moss_lora_precision_sha(const void* data,size_t size,char out[65]){
    ai_sha256 ctx;uint8_t digest[32];ai_sha256_init(&ctx);ai_sha256_update(&ctx,data,size);ai_sha256_final(&ctx,digest);
    static const char hex[]="0123456789abcdef";
    for(unsigned i=0;i<32;++i){out[2*i]=hex[digest[i]>>4];out[2*i+1]=hex[digest[i]&15];}out[64]=0;
}
extern "C" void ps5_ai_moss_lora_storage_snapshot(ps5_ai_moss_lora_storage_receipt *out){
    if(!out)return;std::lock_guard<std::mutex> lock(moss_lora_precision_mutex);
    *out=moss_lora_storage_report;out->schema=2;out->enabled=1;
    out->storage_f16=PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL;out->shared_f32=1;
}
extern "C" void ps5_ai_vk_lora_storage_compile(unsigned slot,const void *original,size_t original_bytes,const void *actual,size_t actual_bytes,
    uint64_t pipeline,unsigned subgroup,unsigned aligned,const uint32_t *specialization,unsigned count,unsigned rte,unsigned denorm){
    std::lock_guard<std::mutex> lock(moss_lora_precision_mutex);auto& state=moss_lora_storage_report;
    const size_t expected_bytes=slot<2?13980u:14112u;
    if(slot>=4||!original||!actual||!pipeline||!specialization||count!=12||rte>1||denorm>1||
       (subgroup!=32&&subgroup!=64)||aligned!=slot%2||original_bytes!=expected_bytes||
       actual_bytes!=expected_bytes+((rte||denorm)?28u:0u)+24u*(rte+denorm)){
        ++state.invalid;return;
    }
    const uint32_t expected[11]={128,64,64,32,subgroup,32,2,4,2,1,subgroup};
    for(unsigned i=0;i<11;++i)if(specialization[i]!=expected[i]){++state.invalid;return;}
    if(specialization[11]!=aligned){++state.invalid;return;}
    auto& p=state.pipelines[slot];
    if(p.compile_count||p.pipeline){++state.invalid;return;}
    char original_sha[65],actual_sha[65];moss_lora_precision_sha(original,original_bytes,original_sha);
    moss_lora_precision_sha(actual,actual_bytes,actual_sha);
    if(std::strcmp(original_sha,slot<2?PS5_AI_MOSS_LORA_F32_MODULE_SHA256:PS5_AI_MOSS_LORA_F16_MODULE_SHA256)||
       std::strcmp(actual_sha,moss_lora_precision_actual_module_sha256[slot/2][rte+2*denorm])){
        ++state.invalid;return;
    }
    p.pipeline=pipeline;p.original_bytes=original_bytes;p.actual_bytes=actual_bytes;p.compile_count=1;
    p.required_subgroup=subgroup;p.aligned=aligned;p.rte_fp16=rte;p.denorm_preserve_fp16=denorm;
    for(unsigned i=0;i<12;++i)p.specialization[i]=specialization[i];
    std::memcpy(p.original_sha256,original_sha,65);std::memcpy(p.actual_sha256,actual_sha,65);
}
extern "C" void ps5_ai_vk_lora_storage_record(const char *name,int64_t k,int64_t m,int64_t n,
    unsigned lhs_f16,unsigned lhs_f32,unsigned rhs_f32,unsigned dst_f32,unsigned requested_f32,unsigned selected_f16acc,
    unsigned flags,uint64_t pipeline,unsigned compiled,unsigned slot){
    std::lock_guard<std::mutex> lock(moss_lora_precision_mutex);auto& s=moss_lora_storage_report;
    const bool fixture=name&&!std::strncmp(name,"moss_f16_storage_fixture.",std::strlen("moss_f16_storage_fixture."));
    const int index=ps5_ai_moss_lora_storage_index(name,k,m);
    const bool canonical=index>=0;
    const bool geometry=(m==144&&(k==3072||k==8192))||(k==144&&(m==3072||m==16384));
    if((!fixture&&!canonical)||!geometry||rhs_f32!=1||dst_f32!=1||requested_f32!=1||selected_f16acc||(flags&~8u)||
       !pipeline||!compiled||lhs_f16+lhs_f32!=1||slot>=4||unsigned(slot>=2)!=lhs_f16||
       s.pipelines[slot].pipeline!=pipeline||s.pipelines[slot].compile_count!=1||
       (!fixture&&lhs_f16!=PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL)||
       (fixture&&n!=9)||(!fixture&&n!=362&&n!=4202)){
        ++s.invalid;return;
    }
    s.last_pipeline=pipeline;s.last_K=(uint64_t)k;s.last_M=(uint64_t)m;s.last_N=(uint64_t)n;s.split_k_calls+=bool(flags&8u);
    if(fixture){if(lhs_f16)++s.fixture_f16;else ++s.fixture_f32;}
    else {if(lhs_f16)++s.runtime_f16;else ++s.runtime_f32;++s.matrix_index[index];}
}
static bool moss_lora_storage_pipelines_valid(const ps5_ai_moss_lora_storage_receipt& s){
    if(s.schema!=2||s.enabled!=1||s.shared_f32!=1||s.storage_f16!=PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL||s.invalid)return false;
    for(unsigned i=0;i<4;++i){const auto& p=s.pipelines[i];
        if(!p.pipeline||p.compile_count!=1||p.aligned!=i%2||(p.required_subgroup!=32&&p.required_subgroup!=64)||
           p.rte_fp16>1||p.denorm_preserve_fp16>1||p.original_bytes!=(i<2?13980u:14112u)||
           p.actual_bytes!=p.original_bytes+((p.rte_fp16||p.denorm_preserve_fp16)?28u:0u)+24u*(p.rte_fp16+p.denorm_preserve_fp16)||
           std::strcmp(p.actual_sha256,moss_lora_precision_actual_module_sha256[i/2][p.rte_fp16+2*p.denorm_preserve_fp16])||
           std::strcmp(p.original_sha256,i<2?PS5_AI_MOSS_LORA_F32_MODULE_SHA256:PS5_AI_MOSS_LORA_F16_MODULE_SHA256))return false;
        const uint64_t expected[12]={128,64,64,32,p.required_subgroup,32,2,4,2,1,p.required_subgroup,i%2};
        for(unsigned k=0;k<12;++k)if(p.specialization[k]!=expected[k])return false;
        for(unsigned j=0;j<i;++j)if(p.pipeline==s.pipelines[j].pipeline)return false;
    }
    return !std::strcmp(s.pipelines[0].actual_sha256,s.pipelines[1].actual_sha256)&&
           !std::strcmp(s.pipelines[2].actual_sha256,s.pipelines[3].actual_sha256);
}
static bool moss_lora_storage_generation_delta_valid(const ps5_ai_moss_lora_storage_receipt& before,const ps5_ai_moss_lora_storage_receipt& after){
    if(!moss_lora_storage_pipelines_valid(before)||!moss_lora_storage_pipelines_valid(after)||
       before.fixture_f32!=16||before.fixture_f16!=16||after.fixture_f32!=before.fixture_f32||after.fixture_f16!=before.fixture_f16||
       after.runtime_f32<before.runtime_f32||after.runtime_f16<before.runtime_f16||
       after.runtime_f32-before.runtime_f32!=(PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL?0u:2016u)||
       after.runtime_f16-before.runtime_f16!=(PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL?2016u:0u)||
       std::memcmp(before.pipelines,after.pipelines,sizeof(before.pipelines)))return false;
    for(unsigned i=0;i<504;++i)if(after.matrix_index[i]<before.matrix_index[i]||after.matrix_index[i]-before.matrix_index[i]!=4)return false;
    return true;
}
static nlohmann::json moss_lora_precision_pipelines_json(const ps5_ai_moss_lora_storage_receipt& s){
    auto result=nlohmann::json::array();
    for(unsigned i=0;i<4;++i){const auto& p=s.pipelines[i];auto spec=nlohmann::json::array();
        for(auto value:p.specialization)spec.push_back(value);
        result.push_back({{"slot",i},{"pipeline",p.pipeline},{"original_module_bytes",p.original_bytes},{"actual_module_bytes",p.actual_bytes},
            {"original_module_sha256",p.original_sha256},{"actual_module_sha256",p.actual_sha256},{"compile_count",p.compile_count},
            {"required_subgroup",p.required_subgroup},{"aligned",p.aligned},{"existing_RTE16_patch",p.rte_fp16!=0},
            {"existing_DenormPreserve16_patch",p.denorm_preserve_fp16!=0},{"specialization",spec}});
    }return result;
}
#endif
static nlohmann::json moss_lora_storage_generation_json(const ps5_ai_moss_lora_storage_receipt& before){
#if PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL
    ps5_ai_moss_lora_storage_receipt after{};ps5_ai_moss_lora_storage_snapshot(&after);
    return {{"enabled",true},{"policy",PS5_AI_MOSS_LORA_PRECISION_POLICY},{"storage_dtype",PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL?"F16":"F32"},
        {"lossy_storage",bool(PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL)},{"FP32_shared_math",true},
        {"exact_504_tensors_four_each",moss_lora_storage_generation_delta_valid(before,after)},
        {"F32_weight_private_recordings",after.runtime_f32-before.runtime_f32},{"F16_weight_private_recordings",after.runtime_f16-before.runtime_f16},
        {"invalid_delta",after.invalid-before.invalid},{"split_K_recordings",after.split_k_calls-before.split_k_calls},
        {"pipelines",moss_lora_precision_pipelines_json(after)},{"physical_memory_savings_asserted",false},{"image_quality_parity",false}};
#else
    (void)before;return {{"enabled",false},{"policy","disabled"}};
#endif
}
