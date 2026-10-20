// SPDX-License-Identifier: MIT
// Mask bits: FFN predecode, FA shared KV, QKV predecode/LoRA vec4, RMS fusion.
#pragma once
#ifndef PS5_AI_MOSS_DIT_KERNEL_TRIAL
#define PS5_AI_MOSS_DIT_KERNEL_TRIAL 0
#endif
#ifndef PS5_AI_MOSS_DIT_KERNEL_MASK
#define PS5_AI_MOSS_DIT_KERNEL_MASK 0
#endif
#ifndef PS5_AI_MOSS_DIT_FA_NO_OCCUPANCY_LIMIT
#define PS5_AI_MOSS_DIT_FA_NO_OCCUPANCY_LIMIT 0
#endif
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
namespace moss_kernel {
inline std::atomic<unsigned> mask{0};
inline std::atomic<uint64_t> records[5]{};
inline bool enabled(unsigned bit){return PS5_AI_MOSS_DIT_KERNEL_TRIAL && (mask.load(std::memory_order_relaxed)&bit);}
inline void reset(unsigned value){mask=value;for(auto& r:records)r=0;}
inline unsigned projection(const char* name) {
    if(!name)return 0;
    if(!std::strncmp(name,"moss_kernel_gate_ffn",20))return 1;
    if(!std::strncmp(name,"moss_kernel_gate_qkv",20))return 4;
    unsigned layer=0;int n=0;
    if(std::sscanf(name,"model.diffusion_model.layers.%u.proj.%n",&layer,&n)!=1 || n<=0 || layer>=42)return 0;
    const char* suffix=name+n;
    if(!std::strcmp(suffix,"ffn_up_gate.weight")||!std::strcmp(suffix,"ffn_down.weight"))return 1;
    if(!std::strcmp(suffix,"qkv.weight")||!std::strcmp(suffix,"o.weight"))return 4;
    return 0;
}
}
extern "C" unsigned ps5_ai_moss_dit_kernel_capabilities();
