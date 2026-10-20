/* SPDX-License-Identifier: GPL-3.0-or-later
 * Optional complete-FFN token slicing. No attention/QKV, parameter, precision,
 * LoRA-scale or sampler change. The original full path remains default.
 */
#pragma once
#include "ggml.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
#ifndef PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL
#define PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL 0
#endif
#ifndef PS5_AI_MOSS_FFN_CHUNK_TOKENS
#define PS5_AI_MOSS_FFN_CHUNK_TOKENS 0
#endif
#if PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL != 0 && PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL != 1
#error PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL must be0or1
#endif
#if PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL && (!PS5_AI_MOSS_F32_ACCUMULATION || !PS5_AI_MOSS_TURBO_TRIAL)
#error FFN trial requires the actual Turbo F32-accumulation core
#endif
#if PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL && PS5_AI_MOSS_FFN_CHUNK_TOKENS != 512 && PS5_AI_MOSS_FFN_CHUNK_TOKENS != 1024 && PS5_AI_MOSS_FFN_CHUNK_TOKENS != 2048
#error FFN enabled size must be512,1024or2048
#endif
#if !PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL && PS5_AI_MOSS_FFN_CHUNK_TOKENS != 0
#error Disabled FFN trial must have zero configured chunk size
#endif

namespace moss_ffn_token_chunk {
constexpr int64_t configured_tokens=PS5_AI_MOSS_FFN_CHUNK_TOKENS;
inline bool size_allowed(int64_t n) {return n==0||n==512||n==1024||n==2048;}
inline int64_t pieces(int64_t tokens,int64_t chunk) {
    if(tokens<=0||tokens>4202||!size_allowed(chunk))
        throw std::runtime_error("MOSS FFN tokens/chunk outside the explicit experiment");
    return chunk==0?1:(tokens+chunk-1)/chunk;
}
template<class FullFfn>
ggml_tensor* apply(ggml_context* c,ggml_tensor* input,int64_t chunk,FullFfn&& full_ffn) {
    if(!c||!input||input->type!=GGML_TYPE_F32||input->ne[0]!=3072||
       input->ne[1]<=0||input->ne[1]>4202||input->ne[2]!=1||input->ne[3]!=1||
       input->nb[0]!=sizeof(float)||input->nb[1]!=3072*sizeof(float)||!size_allowed(chunk))
        throw std::runtime_error("MOSS FFN slicing requires contiguous F32[3072,N,1,1] and admitted size");
    auto checked=[&](ggml_tensor* part) {
        auto* out=full_ffn(part);
        if(!out||out->type!=GGML_TYPE_F32||out->ne[0]!=3072||out->ne[1]!=part->ne[1]||
           out->ne[2]!=1||out->ne[3]!=1)
            throw std::runtime_error("MOSS complete FFN callback output geometry changed");
        return out;
    };
    if(chunk==0||input->ne[1]<=chunk)return checked(input);
    std::vector<ggml_tensor*> completed;
    completed.reserve(size_t(pieces(input->ne[1],chunk)));
    for(int64_t first=0;first<input->ne[1];first+=chunk) {
        const int64_t n=std::min(chunk,input->ne[1]-first);
        auto* part=ggml_view_2d(c,input,3072,n,input->nb[1],size_t(first)*input->nb[1]);
        completed.push_back(checked(part));
    }
    // Balanced assembly of only3072-wide final FFN results. Never construct
    // the full16384xN intermediate before slicing. Generic SET would scan
    // the entire destination for every piece in the current Vulkan backend.
    for(size_t stride=1;stride<completed.size();stride*=2)
        for(size_t i=0;i+stride<completed.size();i+=stride*2)
            completed[i]=ggml_concat(c,completed[i],completed[i+stride],1);
    return completed[0];
}
template<class FullFfn>
ggml_tensor* forward(ggml_context* c,ggml_tensor* input,FullFfn&& full_ffn) {
#if PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL
    return apply(c,input,configured_tokens,full_ffn);
#else
    (void)c;
    return full_ffn(input);
#endif
}
} // namespace moss_ffn_token_chunk
