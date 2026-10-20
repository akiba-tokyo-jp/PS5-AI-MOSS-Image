/* SPDX-License-Identifier: MIT -- separate lossy storage experiment, default OFF. */
#ifndef PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL_H
#define PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#ifndef PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL
#define PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL 0
#endif
#ifndef PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL
#define PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL 0
#endif
#ifndef PS5_AI_MOSS_LORA_PRECISION_TRANSPORT_ONLY
#define PS5_AI_MOSS_LORA_PRECISION_TRANSPORT_ONLY 0
#endif
#if PS5_AI_MOSS_LORA_PRECISION_TRANSPORT_ONLY != 0 && PS5_AI_MOSS_LORA_PRECISION_TRANSPORT_ONLY != 1
#error LoRA_precision_transport_marker_must_be_zero_or_one
#endif
#if PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL != 0 && PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL != 1
#error F16_storage_trial_must_be_zero_or_one
#endif
#if PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL != 0 && PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL != 1
#error LoRA_FP32_shared_trial_must_be_zero_or_one
#endif
#if PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL && !PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL
#error F16_LoRA_storage_requires_the_FP32_shared_precision_experiment
#endif
#if PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL && (!PS5_AI_MOSS_TURBO_TRIAL || (!PS5_AI_MOSS_LORA_PRECISION_TRANSPORT_ONLY && !PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL) || (PS5_AI_MOSS_LORA_PRECISION_TRANSPORT_ONLY && PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL) || !PS5_AI_MOSS_F32_ACCUMULATION || !PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL || !PS5_AI_MOSS_TRIAL_10G || !PS5_AI_MOSS_PERFORMANCE_TRIAL || PS5_AI_MOSS_PERF_VARIANT != 1 || PS5_AI_MOSS_GPU_PROFILE || PS5_AI_MOSS_Q4_BK64_RAW64 || PS5_AI_MOSS_SAMPLER_TRIAL || PS5_AI_MOSS_SPECTRUM_TRIAL || PS5_AI_MOSS_RUNTIME_ADMITTED || PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL || PS5_AI_MOSS_OPERATING_PROFILE)
#error LoRA_precision_experiment_preserves_Turbo_and_upload_with_F32acc_and_other_trials_OFF
#endif
#if PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL
#define PS5_AI_MOSS_LORA_PRECISION_POLICY "combined_Turbo_F16_LoRA_storage_FP32_shared_math_r144_v1"
#define PS5_AI_MOSS_LORA_SELECTED_SHA256 "7904da4b4437dd142b37d1d3b459f558eac1556b8f7e5b46941f6fd39fca735a"
#define PS5_AI_MOSS_LORA_SELECTED_BYTES UINT64_C(668930824)
#elif PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL
#define PS5_AI_MOSS_LORA_PRECISION_POLICY "combined_Turbo_F32_LoRA_storage_FP32_shared_math_r144_v1"
#define PS5_AI_MOSS_LORA_SELECTED_SHA256 "126fed9dd5268fe20ab0085525a4b7b7f8ff767a28aea907df2089c33360d4d3"
#define PS5_AI_MOSS_LORA_SELECTED_BYTES UINT64_C(1337791552)
#else
#define PS5_AI_MOSS_LORA_PRECISION_POLICY "disabled"
#define PS5_AI_MOSS_LORA_SELECTED_SHA256 "disabled"
#define PS5_AI_MOSS_LORA_SELECTED_BYTES UINT64_C(0)
#endif
#define PS5_AI_MOSS_LORA_F16_STORAGE_POLICY PS5_AI_MOSS_LORA_PRECISION_POLICY
#define PS5_AI_MOSS_LORA_F16_STORAGE_BASENAME "moss-nsfw07-turbo4-r144-f16.safetensors"
#define PS5_AI_MOSS_LORA_F16_STORAGE_SCHEMA "ps5-ai-moss-nsfw07-turbo1-r144-f16-storage-v1"
#define PS5_AI_MOSS_LORA_F16_STORAGE_SOURCE_SHA256 "126fed9dd5268fe20ab0085525a4b7b7f8ff767a28aea907df2089c33360d4d3"
#define PS5_AI_MOSS_LORA_F32_MODULE_SHA256 "9f875beba3c839333f3bcfdddf6c5d444e0c61f73a103f286146002bbb68d2f1"
#define PS5_AI_MOSS_LORA_F16_MODULE_SHA256 "1fdb392b957fac72c5512aa1334cbe872dd7460e922811ccf9df1086b8ed3c2c"

/* Actual private pipeline compilation and command-recording receipts.
 * F32 RHS, shared staging, arithmetic and accumulation are proved separately
 * from deliberately rounded F16 coefficient storage. No physical peak claim. */
struct ps5_ai_moss_lora_storage_pipeline {
    uint64_t pipeline,original_bytes,actual_bytes,compile_count,required_subgroup,aligned,rte_fp16,denorm_preserve_fp16;
    uint64_t specialization[12];
    char original_sha256[65],actual_sha256[65];
};
struct ps5_ai_moss_lora_storage_receipt {
    uint64_t schema,enabled,storage_f16,shared_f32,fixture_f32,fixture_f16,runtime_f32,runtime_f16,invalid,split_k_calls;
    uint64_t matrix_index[504];
    struct ps5_ai_moss_lora_storage_pipeline pipelines[4];
    uint64_t last_pipeline, last_K, last_M, last_N;
};
/* Canonical504 names and rank144 geometry only. Returns -1 for anything else. */
static inline int ps5_ai_moss_lora_storage_index(const char *name,int64_t k,int64_t m) {
    const char *prefix="lora.model.diffusion_model.layers.";
    if(!name||strncmp(name,prefix,strlen(prefix)))return -1;
    const char *p=name+strlen(prefix);unsigned layer=0,digits=0;
    if(*p=='0'&&p[1]!='.')return -1;
    while(*p>='0'&&*p<='9'){if(++digits>2)return -1;layer=layer*10+(unsigned)(*p++-'0');}
    if(!digits||layer>=42||strncmp(p,".proj.",6))return -1;
    p+=6;
    const char *tails[]={"qkv.weight","qkv.weight.1","qkv.weight.2","o.weight","ffn_up_gate.weight","ffn_down.weight"};
    for(unsigned module=0;module<6;++module){
        const size_t n=strlen(tails[module]);if(strncmp(p,tails[module],n))continue;
        const int64_t input=module==5?8192:3072,output=module==4?16384:3072;
        if(!strcmp(p+n,".lora_down")&&k==input&&m==144)return (int)(layer*12+module*2);
        if(!strcmp(p+n,".lora_up")&&k==144&&m==output)return (int)(layer*12+module*2+1);
    }
    return -1;
}
#ifdef __cplusplus
extern "C" {
#endif
#if PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL
void ps5_ai_moss_lora_storage_snapshot(struct ps5_ai_moss_lora_storage_receipt *out);
/* Existing verified Vulkan backend only. Returns0 when eligible, otherwise a
 * bounded refusal code (1backend,2physical caps,3queue,4metadata,5ownership). */
int ps5_ai_moss_lora_precision_backend_admission(void *existing_backend);
void ps5_ai_vk_lora_storage_compile(unsigned slot,const void *original,size_t original_bytes,const void *actual,size_t actual_bytes,
    uint64_t pipeline,unsigned subgroup,unsigned aligned,const uint32_t *specialization,unsigned count,unsigned rte,unsigned denorm);
/* Actual backend supplies booleans after ggml_vk_matmul command recording.
 * flags bit0=rhs reformatted,1=lhs reformatted,2=quantized RHS,3=split-K,
 * bit4=noncontiguous LHS,5=noncontiguous RHS. Only split-K may be set;
 * its existing F32 reduction is additionally covered by output readback. */
void ps5_ai_vk_lora_storage_record(const char *name,int64_t k,int64_t m,int64_t n,
    unsigned lhs_f16,unsigned lhs_f32,unsigned rhs_f32,unsigned dst_f32,
    unsigned requested_f32,unsigned selected_f16acc,unsigned flags,
    uint64_t pipeline,unsigned pipeline_compiled,unsigned private_slot);
#else
static inline void ps5_ai_moss_lora_storage_snapshot(struct ps5_ai_moss_lora_storage_receipt *out){if(out)memset(out,0,sizeof(*out));}
#endif
#ifdef __cplusplus
}
#endif
#endif
