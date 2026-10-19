// SPDX-License-Identifier: GPL-3.0-or-later
// Actual admission helpers, synthetic tensor metadata only; no GGML backend,
// GPU allocation, shader execution, model payload or PS5 connection.
[[maybe_unused]] static void ignored_log(const char*,...) {}
#define LOG_INFO(...) ignored_log(__VA_ARGS__)
#include "runner_admission.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace native_scratch_bounds;
static unsigned checks;
static Profile profile{true,true,true,true,36};
static void check(bool condition,const char* message) {
    ++checks;if(!condition){std::fprintf(stderr,"FAIL %s\n",message);std::exit(1);}
}
static ggml_tensor tensor(ggml_type type,int64_t a,int64_t b=1,int64_t c=1,int64_t d=1) {
    ggml_tensor t={};t.type=type;t.ne[0]=a;t.ne[1]=b;t.ne[2]=c;t.ne[3]=d;
    size_t stride=type==GGML_TYPE_F16 || type==GGML_TYPE_BF16?2:4;
    for(unsigned i=0;i<4;++i){t.nb[i]=stride;stride*=size_t(t.ne[i]);}return t;
}
static Bounds inspect_one(ggml_tensor& t,const Profile& p=profile) {
    ggml_tensor* nodes[]={&t};ggml_cgraph graph={};graph.n_nodes=1;graph.nodes=nodes;
    return inspect(&graph,p);
}
static void accepted(ggml_tensor& t,const char* message) {
    auto result=inspect_one(t);check(result.known && !result.x && !result.y && !result.split_k &&
        result.add_rms_baseline==16384,message);
}
static void rejected(ggml_tensor& t,const char* message) {check(!inspect_one(t).known,message);}
static void convolution_params(ggml_tensor& t,int32_t ic,int32_t oc=0,int32_t batch=1) {
    const int32_t p[]={1,1,1,1,1,0,1,1,1,ic,batch,oc};std::memcpy(t.op_params,p,sizeof(p));
}
static void param(ggml_tensor& t,unsigned index,int32_t value) {
    std::memcpy(reinterpret_cast<char*>(t.op_params)+index*sizeof(value),&value,sizeof(value));
}
int main() {
    (void)native_graph_admission::current_profile();
    auto x=tensor(GGML_TYPE_F32,128,16,1024),r=x;r.op=GGML_OP_RMS_NORM;r.src[0]=&x;
    accepted(r,"Anima attention RMS has no extra four-buffer scratch");
    auto vae_x=tensor(GGML_TYPE_F32,384,32,32),vae_r=vae_x;vae_r.op=GGML_OP_RMS_NORM;vae_r.src[0]=&vae_x;
    accepted(vae_r,"Qwen VAE RMS supported");
    auto bad=r;bad.ne[0]=64;rejected(bad,"RMS shape mismatch refused");
    bad=r;bad.type=GGML_TYPE_F16;rejected(bad,"unaudited F16 RMS output refused");
    auto huge=tensor(GGML_TYPE_F32,INT64_C(1)<<30);bad=huge;bad.op=GGML_OP_RMS_NORM;bad.src[0]=&huge;
    rejected(bad,"RMS 32-bit byte-count overflow refused");
    auto rows=tensor(GGML_TYPE_F32,128,65536);bad=rows;bad.op=GGML_OP_RMS_NORM;bad.src[0]=&rows;
    rejected(bad,"RMS over guaranteed dispatch row count refused");
    auto strided=x;strided.nb[2]=uint64_t(UINT32_MAX)+1;bad=r;bad.src[0]=&strided;
    rejected(bad,"RMS oversized source stride refused");

    auto pe=tensor(GGML_TYPE_F32,2,2,64,1024),rope=tensor(GGML_TYPE_F32,128,1024,16);rope.op=GGML_OP_ROPE_APPLY;rope.src[0]=&x;rope.src[1]=&pe;
    accepted(rope,"Anima 128-dimension image RoPE supported");
    auto ax=tensor(GGML_TYPE_F32,64,16,512),ape=tensor(GGML_TYPE_F32,2,2,32,512),ar=tensor(GGML_TYPE_F32,64,512,16);
    ar.op=GGML_OP_ROPE_APPLY;ar.src[0]=&ax;ar.src[1]=&ape;accepted(ar,"Anima 64-dimension adapter RoPE supported");
    auto odd=tensor(GGML_TYPE_F32,127,16,1024);bad=odd;bad.op=GGML_OP_ROPE_APPLY;bad.src[0]=&odd;bad.src[1]=&pe;
    rejected(bad,"odd RoPE head refused");
    auto wrong_pe=pe;wrong_pe.ne[3]=1023;bad=rope;bad.src[1]=&wrong_pe;rejected(bad,"RoPE position length mismatch refused");
    wrong_pe=pe;wrong_pe.type=GGML_TYPE_F16;bad=rope;bad.src[1]=&wrong_pe;rejected(bad,"unaudited RoPE positional type refused");
    bad=rope;bad.nb[1]+=4;rejected(bad,"noncontiguous RoPE output refused");
    bad=x;bad.op=GGML_OP_ROPE_APPLY;bad.src[0]=&x;bad.src[1]=&pe;rejected(bad,"old incorrect same-shape RoPE fixture refused");
    auto huge_stride=x;huge_stride.nb[2]=UINT32_MAX;bad=rope;bad.src[0]=&huge_stride;
    rejected(bad,"RoPE stride overflow/unaligned addressing refused");

    auto kernel=tensor(GGML_TYPE_F16,3,3,3,16*384),image=tensor(GGML_TYPE_F32,32,32,3,16);
    auto conv=tensor(GGML_TYPE_F32,32,32,1,384);conv.op=GGML_OP_CONV_3D;conv.src[0]=&kernel;conv.src[1]=&image;
    convolution_params(conv,16,384);accepted(conv,"Qwen VAE 16-to384 causal 3D convolution supported");
    auto k32=kernel;k32.type=GGML_TYPE_F32;
    size_t step=4;for(unsigned i=0;i<4;++i){k32.nb[i]=step;step*=size_t(k32.ne[i]);}
    bad=conv;bad.src[0]=&k32;accepted(bad,"F32 convolution kernel remains supported");
    bad=conv;param(bad,9,15);rejected(bad,"3D convolution channel mismatch refused");
    bad=conv;param(bad,10,2);rejected(bad,"3D convolution packed batch mismatch refused");
    bad=conv;param(bad,0,0);rejected(bad,"zero convolution stride refused");
    bad=conv;param(bad,3,-1);rejected(bad,"negative convolution padding refused");
    bad=conv;param(bad,6,0);rejected(bad,"zero convolution dilation refused");
    bad=conv;bad.ne[2]=2;rejected(bad,"incorrect convolution output extent refused");
    auto noncontig=image;noncontig.nb[1]+=4;bad=conv;bad.src[1]=&noncontig;rejected(bad,"unsupported noncontiguous convolution input refused");
    auto bf=kernel;bf.type=GGML_TYPE_BF16;bad=conv;bad.src[0]=&bf;rejected(bad,"unsupported GPU BF16 convolution kernel refused");
    auto too_large=tensor(GGML_TYPE_F32,65535,65535,3,16);bad=conv;bad.src[1]=&too_large;
    rejected(bad,"3D convolution shader input indexing overflow refused");

    auto col=tensor(GGML_TYPE_F16,16*27,32,32);col.op=GGML_OP_IM2COL_3D;col.src[0]=&kernel;col.src[1]=&image;
    convolution_params(col,16);accepted(col,"Qwen VAE 3D im2col fallback bounded without extra scratch");
    auto col32=tensor(GGML_TYPE_F32,16*27,32,32);col32.op=col.op;col32.src[0]=&kernel;col32.src[1]=&image;
    convolution_params(col32,16);accepted(col32,"F32 3D im2col output supported");
    bad=col;bad.ne[0]--;rejected(bad,"3D im2col encoded kernel product mismatch refused");
    bad=col;param(bad,9,15);rejected(bad,"3D im2col nonintegral packed batch refused");
    bad=col;bad.ne[1]=31;rejected(bad,"3D im2col wrong output geometry refused");
    auto half_image=image;half_image.type=GGML_TYPE_F16;bad=col;bad.src[1]=&half_image;
    rejected(bad,"unsupported half input to 3D im2col refused");
    bad=col;bad.type=GGML_TYPE_BF16;rejected(bad,"unsupported BF16 im2col output refused");
    bad=col;param(bad,8,INT32_MAX);rejected(bad,"im2col invalid expanded kernel extent refused");

    auto invalid_profile=profile;invalid_profile.no_fusion=false;
    check(!inspect_one(r,invalid_profile).known,"RMS fusion profile remains refused");
    invalid_profile=profile;invalid_profile.no_async=false;check(!inspect_one(conv,invalid_profile).known,"async profile remains refused");
    invalid_profile=profile;invalid_profile.ps5_scalar_no_coopmat=false;check(!inspect_one(rope,invalid_profile).known,"unaudited cooperative profile remains refused");
    invalid_profile=profile;invalid_profile.shader_cores_upper=40;check(!inspect_one(col,invalid_profile).known,"different device profile remains refused");
    bad=r;bad.op=GGML_OP_MEAN;rejected(bad,"unused VAE encoder mean not blindly admitted");
    bad=r;bad.op=GGML_OP_ROPE;rejected(bad,"different RoPE opcode not blindly admitted");
    auto a=tensor(GGML_TYPE_F16,2048,8192),b=tensor(GGML_TYPE_F32,2048,1024),mm=tensor(GGML_TYPE_F32,8192,1024);
    mm.op=GGML_OP_MUL_MAT;mm.src[0]=&a;mm.src[1]=&b;
    ggml_tensor* nodes[]={&mm};ggml_cgraph graph={};graph.n_nodes=1;graph.nodes=nodes;
    check(native_graph_admission::inspect_scratch(&graph,profile,UINT64_C(512)<<20).per_claim,"Anima initial 512 image MLP scratch within actual claim bound");
    a=tensor(GGML_TYPE_F16,16384,16384);b=tensor(GGML_TYPE_F32,16384,1024);mm=tensor(GGML_TYPE_F32,16384,1024);mm.op=GGML_OP_MUL_MAT;mm.src[0]=&a;mm.src[1]=&b;
    check(!native_graph_admission::inspect_scratch(&graph,profile,UINT64_C(512)<<20).per_claim,"oversized matrix scratch still refused");
    std::printf("PASS %u Anima/Qwen VAE operator and scratch bound checks; metadata only, GPU calls=0\n",checks);
}
