// SPDX-License-Identifier: MIT
// MOSS-only exact forward fusion. vLLM-Omni's Apache-2.0 implementation
// informed the scope; these F32 GGML/GLSL paths preserve this port's arithmetic.
#pragma once
#ifndef PS5_AI_MOSS_DIT_OPT_TRIAL
#define PS5_AI_MOSS_DIT_OPT_TRIAL 0
#endif
#if PS5_AI_MOSS_DIT_OPT_TRIAL != 0 && PS5_AI_MOSS_DIT_OPT_TRIAL != 1
#error Invalid MOSS DiT trial selector
#endif
#include "ggml.h"
#include "ggml-backend.h"
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include "ggml-impl.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
extern "C" unsigned ps5_ai_moss_dit_capabilities();
namespace moss_dit_opt {
inline std::atomic<unsigned> active_mask{0};
inline std::atomic<uint64_t> generation{0}, cache_misses{0},cache_hits{0},rope_builds{0},scale_elisions{0};
inline std::atomic<uint64_t> GPU_recordings[9]{};
inline std::atomic<void*> cache_owner{nullptr};
using clear_fn=void(*)(void*);
inline std::atomic<clear_fn> clear_cache{nullptr};
inline bool enabled(unsigned bit) {return PS5_AI_MOSS_DIT_OPT_TRIAL && (active_mask.load(std::memory_order_relaxed)&bit);}
inline void register_cache(void* owner,clear_fn fn) {
    void* expected=nullptr;
    if(!cache_owner.compare_exchange_strong(expected,owner) && expected!=owner)throw std::runtime_error("Only one MOSS cache owner is admitted");
    clear_cache.store(fn);
}
inline void unregister_cache(void* owner) {if(cache_owner.load()==owner){clear_cache.store(nullptr);cache_owner.store(nullptr);}}
inline void reset_cache() {auto fn=clear_cache.load();auto ptr=cache_owner.load();if(fn&&ptr)fn(ptr);}
inline void begin(unsigned mask) {
    if(mask>15)throw std::runtime_error("Invalid DiT optimization mask");
    reset_cache();generation.fetch_add(1);active_mask.store(mask);
    cache_misses=0;cache_hits=0;rope_builds=0;scale_elisions=0;
    for(auto& x:GPU_recordings)x=0;
}
inline void end_completed() {reset_cache();active_mask=0;}
// Vulkan tensor->data is an offset from a shared sentinel, not a unique
// storage identity. Check the resolved VkBuffer and actual byte ranges.
inline bool storage_ranges_disjoint(bool same_buffer,uint64_t input_offset,uint64_t input_bytes,uint64_t output_offset,uint64_t output_bytes) {
    if(!same_buffer)return true;
    if(input_bytes>UINT64_MAX-input_offset||output_bytes>UINT64_MAX-output_offset)return false;
    return input_offset+input_bytes<=output_offset||output_offset+output_bytes<=input_offset;
}
enum Kind:unsigned {SITU=1,NORM_MOD0=2,NORM_MOD3=3,GATED_RESIDUAL=4,QK_WEIGHT_ROPE=5,LORA_ADD1=6,LORA_PACK2=7,LORA_PACK3=8};
inline float load(const ggml_tensor* t,uint64_t i0,uint64_t i1=0,uint64_t i2=0,uint64_t i3=0) {
    float x;std::memcpy(&x,(const unsigned char*)t->data+i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3],4);return x;
}
inline float mul(float a,float b){volatile float x=a*b;return x;}
inline float add(float a,float b){volatile float x=a+b;return x;}
inline float tanh_reference(float x){return 1.f-2.f/(std::exp(2.f*x)+1.f);}
inline float scalar(unsigned kind,const ggml_tensor* a,const ggml_tensor* b,const ggml_tensor* c,const ggml_tensor* d,uint64_t i) {
    uint64_t j=i%a->ne[0],r=i/a->ne[0],y=r%a->ne[1],z=(r/a->ne[1])%a->ne[2],w=r/(a->ne[1]*a->ne[2]);
    if(kind==QK_WEIGHT_ROPE){
        const uint64_t H=a->ne[0],T=a->ne[2],heads=a->ne[1],half=H/2;
        j=i%H;r=i/H;const uint64_t token=r%T,head=(r/T)%heads,batch=r/(T*heads),p=j%half;
        float av=mul(load(a,p,head,token,batch),load(b,p));
        float bv=mul(load(a,p+half,head,token,batch),load(b,p+half));
        float co=load(c,0,0,p,token),si=load(c,0,1,p,token);
        return j<half?add(mul(av,co),-mul(bv,si)):add(mul(av,si),mul(bv,co));
    }
    const float av=load(a,j,y,z,w);
    if(kind==SITU){float g=load(b,j,y,z,w);float ag=mul(tanh_reference(mul(g,.25f)),4.f);ag=mul(ag,1.f/(1.f+std::exp(-g)));return mul(ag,mul(tanh_reference(mul(av,1.f/25.f)),25.f));}
    if(kind==NORM_MOD0||kind==NORM_MOD3){unsigned off=kind==NORM_MOD0?0:3;float v=mul(av,load(b,j));return add(add(v,mul(v,load(c,j+(off+1)*a->ne[0]))),load(c,j+off*a->ne[0]));}
    if(kind==GATED_RESIDUAL)return add(av,mul(load(b,j,y,z,w),load(c,j)));
    const ggml_tensor* part=b;
    if(kind==LORA_PACK2||kind==LORA_PACK3){if(j>=uint64_t(b->ne[0])){j-=b->ne[0];part=c;if(j>=uint64_t(c->ne[0])){j-=c->ne[0];part=d;}}}
    return add(av,load(part,j,y,z,w));
}
inline void cpu(ggml_tensor* dst,const ggml_tensor* a,const ggml_tensor* b,const ggml_tensor* c,int ith,int nth,void* userdata) {
    unsigned kind=unsigned(uintptr_t(userdata));GGML_ASSERT(kind>=1&&kind<=8);
    for(uint64_t i=ith;i<uint64_t(ggml_nelements(dst));i+=nth){float x=scalar(kind,a,b,c,dst->src[3],i);std::memcpy((unsigned char*)dst->data+i*4,&x,4);}
}
inline unsigned kind(const ggml_tensor* t) {
#if PS5_AI_MOSS_DIT_OPT_TRIAL
    if(!t||t->op!=GGML_OP_MAP_CUSTOM3)return 0;
    ggml_map_custom3_op_params p{};std::memcpy(&p,t->op_params,sizeof(p));
    if(p.fun!=cpu||p.n_tasks!=1)return 0;auto k=uintptr_t(p.userdata);return k>=1&&k<=8?unsigned(k):0;
#else
    (void)t;return 0;
#endif
}
inline bool addressable(const ggml_tensor* t) {
    if(!t||t->type!=GGML_TYPE_F32||t->nb[0]!=4||ggml_nelements(t)<=0||uint64_t(ggml_nelements(t))>UINT32_MAX)return false;
    uint64_t span=4;
    for(int i=0;i<4;++i){if(t->ne[i]<=0||uint64_t(t->ne[i])>UINT32_MAX||t->nb[i]%4||t->nb[i]>UINT32_MAX)return false;uint64_t n=(t->ne[i]-1)*t->nb[i];if(n>UINT32_MAX-span)return false;span+=n;}
    return true;
}
inline bool valid(const ggml_tensor* t) {
    unsigned k=kind(t);if(!k||!addressable(t)||!ggml_is_contiguous(t)||t->view_src)return false;
    auto a=t->src[0],b=t->src[1],c=t->src[2],d=t->src[3];
    if(!addressable(a)||!addressable(b)||!addressable(c))return false;
    if(k==QK_WEIGHT_ROPE)return a->ne[0]==128&&a->ne[3]==1&&b->ne[0]==128&&ggml_nelements(b)==128&&c->ne[0]==2&&c->ne[1]==2&&c->ne[2]==64&&c->ne[3]==a->ne[2]&&t->ne[0]==128&&t->ne[1]==a->ne[2]&&t->ne[2]==a->ne[1]&&t->ne[3]==1;
    if(!ggml_are_same_shape(t,a))return false;
    if(k==SITU)return ggml_are_same_shape(a,b);
    if(k==NORM_MOD0||k==NORM_MOD3)return ggml_nelements(b)==a->ne[0]&&c->ne[0]==6*a->ne[0]&&ggml_nelements(c)==c->ne[0];
    if(k==GATED_RESIDUAL)return ggml_are_same_shape(a,b)&&ggml_nelements(c)==a->ne[0];
    if(k==LORA_ADD1)return ggml_are_same_shape(a,b);
    if(a->ne[2]!=1||a->ne[3]!=1||b->ne[1]!=a->ne[1]||c->ne[1]!=a->ne[1]||b->ne[2]!=1||b->ne[3]!=1||c->ne[2]!=1||c->ne[3]!=1)return false;
    if(k==LORA_PACK2)return a->ne[0]==b->ne[0]+c->ne[0];
    return addressable(d)&&d->ne[1]==a->ne[1]&&d->ne[2]==1&&d->ne[3]==1&&a->ne[0]==b->ne[0]+c->ne[0]+d->ne[0];
}
inline ggml_tensor* make(ggml_context* ctx,unsigned k,ggml_tensor* a,ggml_tensor* b,ggml_tensor* c,ggml_tensor* d=nullptr,ggml_backend_t backend=nullptr) {
    auto out=ggml_map_custom3(ctx,a,b,c,cpu,1,reinterpret_cast<void*>(uintptr_t(k)));out->src[3]=d;
    if(k==QK_WEIGHT_ROPE){out->ne[1]=a->ne[2];out->ne[2]=a->ne[1]*a->ne[3];out->ne[3]=1;for(int i=1;i<4;++i)out->nb[i]=out->nb[i-1]*out->ne[i-1];}
    if(!valid(out))throw std::runtime_error("Invalid MOSS fused shape/stride; no fallback");
    if(backend&&!ggml_backend_supports_op(backend,out))throw std::runtime_error("MOSS fused Vulkan operator unavailable; CPU fallback forbidden");
    ggml_format_name(out,"moss_dit_fused_%u",k);return out;
}
} // namespace moss_dit_opt
