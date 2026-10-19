// SPDX-License-Identifier: GPL-3.0-or-later
// Actual pinned GGML tensor constructors; metadata only, no graph execution.
#include <cstdio>
#define LOG_INFO(...) std::fprintf(stderr,__VA_ARGS__)
#include "runner_admission.hpp"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
using namespace native_scratch_bounds;
static unsigned checks;
static Profile profile{true,true,true,true,36};
extern "C" void ggml_backend_tensor_memset(ggml_tensor*,uint8_t,size_t,size_t){std::abort();}
extern "C" void ggml_backend_tensor_set(ggml_tensor*,const void*,size_t,size_t){std::abort();}
extern "C" ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t){std::abort();}
static void check(bool value,const char* message){++checks;if(!value){std::fprintf(stderr,"FAIL %s\n",message);std::exit(1);}}
static native_graph_admission::ScratchDecision inspect_one(ggml_tensor* tensor,uint64_t maximum){
    ggml_tensor* nodes[]={tensor};ggml_cgraph graph={};graph.n_nodes=1;graph.nodes=nodes;
    return native_graph_admission::inspect_scratch(&graph,profile,maximum);
}
int main(){
 auto* c=ggml_init({1<<20,nullptr,true});check(c!=nullptr,"metadata context");
 constexpr uint64_t limit=UINT64_C(1)<<30;
 for(auto type:{GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_Q8_0}){
    auto* a=ggml_new_tensor_2d(c,type,3072,16384);auto* b=ggml_new_tensor_2d(c,GGML_TYPE_F32,3072,5120);
    auto* d=ggml_mul_mat(c,a,b);auto out=inspect_one(d,limit);
    check(out.per_claim && out.upper.known && out.upper.x==UINT64_C(201326592) && out.upper.y==UINT64_C(201326592),"actual MOSS quantized FFN staging admitted conservatively");
    auto old=a->nb[1];a->nb[1]+=16;check(!inspect_one(d,limit).per_claim,"strided quantized input refused");a->nb[1]=old;
 }
 auto* a=ggml_new_tensor_2d(c,GGML_TYPE_Q4_K,16384,16384);auto* b=ggml_new_tensor_2d(c,GGML_TYPE_F32,16384,128);auto* d=ggml_mul_mat(c,a,b);
 check(inspect_one(d,limit).per_claim,"exact 1GiB conservative scratch bound admitted");
 check(!inspect_one(d,limit-16384).per_claim,"same claim over actual region refused");
 auto* unknown=ggml_new_tensor_2d(c,GGML_TYPE_IQ4_XS,3072,16384);auto* unknown_d=ggml_mul_mat(c,unknown,ggml_new_tensor_2d(c,GGML_TYPE_F32,3072,128));
 check(!inspect_one(unknown_d,limit).per_claim,"unaudited IQ4 input refused");
 auto* q=ggml_new_tensor_4d(c,GGML_TYPE_F32,128,24,5120,1);auto* pe=ggml_new_tensor_4d(c,GGML_TYPE_F32,2,2,64,5120);
 auto* rms=ggml_rms_norm(c,q,1e-6f);check(inspect_one(rms,limit).per_claim,"actual MOSS24head RMS admitted");
 auto* rope=ggml_rope_apply(c,rms,pe,false);check(rope->ne[0]==128 && rope->ne[1]==5120 && rope->ne[2]==24 && rope->ne[3]==1,"actual MOSS RoPE reshapes heads/tokens");check(inspect_one(rope,limit).per_claim,"actual MOSS RoPE admitted");
 auto* fq=ggml_new_tensor_4d(c,GGML_TYPE_F32,128,5120,24,1);auto* fk=ggml_new_tensor_4d(c,GGML_TYPE_F16,128,5120,24,1);auto* fv=ggml_new_tensor_4d(c,GGML_TYPE_F16,128,5120,24,1);
 auto* fa=ggml_flash_attn_ext(c,fq,fk,fv,nullptr,1.f,0.f,0.f);check(inspect_one(fa,limit).per_claim,"actual MOSS24head flash scratch bounded");
 auto* h=ggml_new_tensor_2d(c,GGML_TYPE_F32,8192,5120);auto* tanh=ggml_tanh(c,h);auto* sig=ggml_sigmoid(c,h);
 check(inspect_one(tanh,limit).per_claim && inspect_one(sig,limit).per_claim,"MOSS SiTU TANH/SIGMOID have no four-buffer prealloc growth");
 ggml_free(c);std::printf("PASS %u MOSS quantized scratch/constructor checks; model inference=0, GPU calls=0\n",checks);
}
