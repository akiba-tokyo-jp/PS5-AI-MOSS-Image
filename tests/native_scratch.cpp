#define LOG_INFO(...) ((void)0)
#include "runner_admission.hpp"
#include <cstdio>
#include <cstdlib>
using namespace native_scratch_bounds;
static unsigned cases;
static void check(bool ok,const char *name){if(!ok){std::fprintf(stderr,"FAIL %s\n",name);std::exit(1);}++cases;}
static ggml_tensor tensor(ggml_type type,int64_t a,int64_t b=1,int64_t c=1,int64_t d=1){
    ggml_tensor t={};t.type=type;t.ne[0]=a;t.ne[1]=b;t.ne[2]=c;t.ne[3]=d;return t;
}
static Profile profile{true,true,true,true,36};
static Bounds mm(uint32_t k,uint32_t m,uint32_t n){
    auto a=tensor(GGML_TYPE_F16,k,m),b=tensor(GGML_TYPE_F32,k,n),d=tensor(GGML_TYPE_F32,m,n);
    d.src[0]=&a;d.src[1]=&b;Bounds out;check(matmul(&d,profile,out),"matmul metadata known");return out;
}
int main(){
    auto proj=mm(640,5120,4096);
    check(proj.x==13107200&&proj.y==13107200&&!proj.split_k,"1024 GEGLU projection geometry");
    auto down=mm(2560,640,4096);
    check(down.x==41943040&&down.y==41943040&&!down.split_k,"1024 GEGLU down geometry");
    auto deep=mm(5120,1280,1024);
    check(deep.x==26214400&&deep.y==26214400&&!deep.split_k,"1024 deep feedforward geometry");
    auto small=mm(5120,1280,256);
    check(small.split_k==3932160,"source finite tile split3 bound");
    auto q=tensor(GGML_TYPE_F32,64,4096,10),k=tensor(GGML_TYPE_F16,64,4096,10),v=k;
    auto d=tensor(GGML_TYPE_F32,64,10,4096);d.src[0]=&q;d.src[1]=&k;d.src[2]=&v;
    Bounds fa;check(dense_flash(&d,profile,fa)&&!fa.split_k,"1024 dense self attention no split");
    q=tensor(GGML_TYPE_F32,64,1);k=tensor(GGML_TYPE_F16,64,4096);v=k;d=tensor(GGML_TYPE_F32,64,1,1);
    d.src[0]=&q;d.src[1]=&k;d.src[2]=&v;Bounds low;
    check(dense_flash(&d,profile,low)&&low.split_k==32768,"FLASH bound allows64 splits not8");
    d.op=GGML_OP_FLASH_ATTN_EXT;ggml_tensor *nodes[]={&d};ggml_cgraph graph={};graph.n_nodes=1;graph.nodes=nodes;
    auto mask=tensor(GGML_TYPE_F16,4096);d.src[3]=&mask;
    auto unknown=inspect(&graph,profile);check(!unknown.known&&unknown.unknown_nodes==1,"masked FLASH unknown");
    d.src[3]=nullptr;auto no_profile=profile;no_profile.no_fusion=false;
    check(!inspect(&graph,no_profile).known,"unverified profile unknown");
    d.op=GGML_OP_MUL_MAT_ID;check(!inspect(&graph,profile).known,"MoE unknown");
    auto huge_a=tensor(GGML_TYPE_F16,1,UINT32_MAX),huge_b=tensor(GGML_TYPE_F32,1,UINT32_MAX);
    auto huge_d=tensor(GGML_TYPE_F32,UINT32_MAX,UINT32_MAX);huge_d.src[0]=&huge_a;huge_d.src[1]=&huge_b;Bounds huge;
    check(!matmul(&huge_d,profile,huge),"source32-bit tile/stride overflow unknown");
    q=tensor(GGML_TYPE_F32,64,2000000000,3);k=tensor(GGML_TYPE_F16,64,4096,3);v=k;
    d=tensor(GGML_TYPE_F32,64,3,2000000000);d.src[0]=&q;d.src[1]=&k;d.src[2]=&v;
    check(!dense_flash(&d,profile,huge),"source32-bit FLASH workgroup product overflow unknown");
    d.op=GGML_OP_TIMESTEP_EMBEDDING;d.src[0]=nullptr;d.src[1]=nullptr;d.src[2]=nullptr;
    auto timestep=inspect(&graph,profile);check(timestep.known,"SDXL timestep source-zero prealloc");
    d.op=GGML_OP_SOFT_MAX;check(!inspect(&graph,profile).known,"manual softmax not silently admitted");
    auto large_a=tensor(GGML_TYPE_F16,8192,8192),large_b=tensor(GGML_TYPE_F32,8192,64);
    auto large_d=tensor(GGML_TYPE_F32,8192,64);large_d.op=GGML_OP_MUL_MAT;large_d.src[0]=&large_a;large_d.src[1]=&large_b;
    nodes[0]=&large_d;
    auto exact_limit=native_graph_admission::inspect_scratch(&graph,profile,UINT64_C(256)<<20);
    check(exact_limit.per_claim && exact_limit.upper.x==(UINT64_C(256)<<20),"scratch actual256MiB boundary admitted");
    large_a.ne[0]=8193;large_b.ne[0]=8193;
    auto over_limit=native_graph_admission::inspect_scratch(&graph,profile,UINT64_C(256)<<20);
    check(over_limit.upper.known&&!over_limit.per_claim,"oversize scratch refused before GPU allocation");
    auto unavailable=profile;unavailable.no_async=false;
    check(!native_graph_admission::inspect_scratch(&graph,unavailable,UINT64_C(256)<<20).per_claim,"unknown actual profile refused");
    // Real blocked1024 convolution: F16[17280,4096] im2col staging.
    auto ca=tensor(GGML_TYPE_F16,17280,640),cb=tensor(GGML_TYPE_F16,17280,4096);
    auto cd=tensor(GGML_TYPE_F32,640,4096);cd.src[0]=&ca;cd.src[1]=&cb;cd.op=GGML_OP_MUL_MAT;
    ggml_tensor *cnodes[]={&cd};ggml_cgraph cg={};cg.n_nodes=1;cg.nodes=cnodes;
    Profile half_profile{true,false,true,true,36};
    auto half=native_graph_admission::inspect_scratch(&cg,half_profile,UINT64_C(256)<<20);
    check(half.per_claim&&half.upper.x==141557760&&half.upper.y==141557760,"actual F16 convolution uses two-byte staging upper");
    auto old_profile=half_profile;old_profile.f16_math_disabled=true;
    check(!native_graph_admission::inspect_scratch(&cg,old_profile,UINT64_C(256)<<20).per_claim,"F32 profile retains conservative refusal");
    cb.type=GGML_TYPE_F32;
    check(!native_graph_admission::inspect_scratch(&cg,half_profile,UINT64_C(256)<<20).per_claim,"mixed F16 F32 stays four-byte conservative");
    cb.type=GGML_TYPE_F16;ca.ne[0]=cb.ne[0]=32769;
    check(!native_graph_admission::inspect_scratch(&cg,half_profile,UINT64_C(256)<<20).per_claim,"F16 staging over actual256MiB still refused");
    std::printf("{\"checks_passed\":%u,\"GPU_calls\":0,\"backend_allocations\":0,\"model_inference\":false,\"full_1024_graph_built\":false}\n",cases);
}
