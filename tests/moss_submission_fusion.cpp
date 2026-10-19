// SPDX-License-Identifier: GPL-3.0-or-later
// Actual GGML constructors and graph dependencies; metadata only, no backend.
#include "ggml-vulkan-ps5-profile.h"
#include "ggml-impl.h"
#include "vulkan_scratch_bounds.hpp"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
#define LOG_INFO(...) std::fprintf(stderr,__VA_ARGS__)
#include "runner_admission.hpp"
static unsigned checks;
static void check(bool v,const char * name){++checks;if(!v){std::fprintf(stderr,"FAIL %s\n",name);std::exit(1);}}
extern "C" ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t){std::abort();}
extern "C" void ggml_backend_tensor_memset(ggml_tensor*,uint8_t,size_t,size_t){std::abort();}
extern "C" void ggml_backend_tensor_set(ggml_tensor*,const void*,size_t,size_t){std::abort();}
using namespace ggml_vk_ps5_profile;
static bool consumer_safe(ggml_cgraph * g,ggml_tensor * u){
    for(int i=0;i+1<g->n_nodes;++i)if(g->nodes[i]==u){
        const ggml_op ops[]={GGML_OP_UNARY,GGML_OP_MUL};const int out[]={i+1};
        return ggml_can_fuse_subgraph(g,i,2,ops,out,1);
    }
    return false;
}
int main(){
    Options out;const char * error=nullptr;Environment env;
    check(parse(env,out,error)&&out.submit_multiplier==1&&!out.elementwise_fusion&&!error,"unset unchanged");
    // Legacy values are intentionally ignored when neither new switch is present.
    env.no_fusion="0";env.generic_max_nodes="175";
    check(parse(env,out,error)&&out.submit_multiplier==1,"legacy default parsing unchanged");
    env={"2","1","1","1","1","1","1","1",nullptr};
    check(parse(env,out,error)&&out.submit_multiplier==2&&out.elementwise_fusion,"exact optimized profile");
    for(const char * bad:{"","0","1","02","2x"," 2","4294967295"}){
        auto e=env;e.submit_batch=bad;check(!parse(e,out,error)&&error,"invalid batch rejected");
    }
    for(const char * bad:{"","0","2","01","true"}){
        auto e=env;e.elementwise_fusion=bad;check(!parse(e,out,error),"invalid fusion rejected");
    }
    for(unsigned which=0;which<6;++which){
        auto e=env;const char ** fields[]={&e.serialize,&e.no_async,&e.no_fusion,&e.no_f16,&e.no_coopmat,&e.no_coopmat2};
        *fields[which]=nullptr;check(!parse(e,out,error),"missing safety profile rejected");
        *fields[which]="0";check(!parse(e,out,error),"presence zero safety profile rejected");
    }
    auto conflict=env;conflict.generic_max_nodes="100";check(!parse(conflict,out,error),"generic node cap override rejected");
    env.elementwise_fusion=nullptr;check(parse(env,out,error)&&!out.elementwise_fusion,"batch-only valid");
    env.elementwise_fusion="1";env.submit_batch=nullptr;check(parse(env,out,error)&&out.submit_multiplier==1,"fusion-only valid");
    for(uint64_t cap:{UINT64_C(500000000),UINT64_C(18000000000),UINT64_C(200000000000)}){
        for(uint64_t last:{UINT64_C(0),UINT64_C(39),UINT64_C(40),UINT64_C(4000000000000),UINT64_MAX}){
            const auto before=std::min(cap,last/40u);check(initial_flops(cap,last,1)==before,"default threshold exact");
            check(initial_flops(cap,last,2)==before*2 && initial_flops(cap,last,2)*8<=UINT64_C(3200000000000),"bounded doubled threshold and three ramps");
        }
    }
    auto * c=ggml_init({8<<20,nullptr,true});check(c!=nullptr,"metadata context");
    check(_putenv_s("GGML_VK_PS5_ELEMENTWISE_FUSION","")==0 &&
          native_graph_admission::current_profile().ps5_elementwise_fusion_mode==0,"runtime profile unset is original mode0");
    check(_putenv_s("GGML_VK_PS5_ELEMENTWISE_FUSION","1")==0 &&
          native_graph_admission::current_profile().ps5_elementwise_fusion_mode==1,"runtime profile exact opt is known mode1");
    check(_putenv_s("GGML_VK_PS5_ELEMENTWISE_FUSION","0")==0 &&
          native_graph_admission::current_profile().ps5_elementwise_fusion_mode==2,"runtime profile presence-zero is invalid mode2");
    check(_putenv_s("GGML_VK_PS5_ELEMENTWISE_FUSION","")==0,"restore host process environment");
    auto * base=ggml_new_tensor_2d(c,GGML_TYPE_F32,16384,304);
    auto * left=ggml_view_2d(c,base,8192,304,base->nb[1],0);
    auto * right=ggml_view_2d(c,base,8192,304,base->nb[1],8192*sizeof(float));
    check(left->nb[1]==65536&&!ggml_is_contiguous(left)&&ggml_is_contiguous_1(left),"actual FFN strided halfview");
    for(bool silu:{false,true})for(bool reversed:{false,true}){
        auto * u=silu?ggml_silu(c,left):ggml_sigmoid(c,left);
        auto * m=reversed?ggml_mul(c,right,u):ggml_mul(c,u,right);
        auto * g=ggml_new_graph_custom(c,32,false);ggml_build_forward_expand(g,right);ggml_build_forward_expand(g,m);
        // Scheduler's compute marking, without executing any scheduler/backend.
        for(int i=0;i<g->n_nodes;++i)g->nodes[i]->flags|=GGML_TENSOR_FLAG_COMPUTE;
        check(limited_unary_mul_metadata(u,m),"F32 same-shape halfview limited candidate");
        check(consumer_safe(g,u),"existing graph dependency accepts sole consumer");
        u->flags|=GGML_TENSOR_FLAG_OUTPUT;check(!consumer_safe(g,u),"externally requested intermediate rejected");u->flags&=~GGML_TENSOR_FLAG_OUTPUT;
        auto * extra=ggml_scale(c,u,2);ggml_build_forward_expand(g,extra);
        check(!consumer_safe(g,u),"second external consumer rejected");
        auto saved=m->type;m->type=GGML_TYPE_F16;check(!limited_unary_mul_metadata(u,m),"F16 output rejected");m->type=saved;
        saved=right->type;right->type=GGML_TYPE_F16;check(!limited_unary_mul_metadata(u,m),"F16 other rejected");right->type=saved;
        auto old=right->ne[1];right->ne[1]=1;check(!limited_unary_mul_metadata(u,m),"broadcast rejected");right->ne[1]=old;
        auto stride=right->nb[1];right->nb[1]=UINT64_C(1)<<32;check(!limited_unary_mul_metadata(u,m),"32-bit shader address overflow rejected");right->nb[1]=stride;
        auto * gelu=ggml_gelu(c,left);auto * gm=ggml_mul(c,gelu,right);check(!limited_unary_mul_metadata(gelu,gm),"general GELU fusion stays excluded");
        // Existing per-node bounds stay identical; mode1 adds no scratch request.
        native_scratch_bounds::Profile original{true,true,true,true,36};auto limited=original;limited.ps5_elementwise_fusion_mode=1;
        auto before=native_scratch_bounds::inspect(g,original),after=native_scratch_bounds::inspect(g,limited);
        check(before.known&&after.known&&before.x==after.x&&before.y==after.y&&before.split_k==after.split_k&&before.add_rms_baseline==after.add_rms_baseline,"limited mode retains exact four-prealloc upper bound");
        limited.no_fusion=false;check(!native_scratch_bounds::inspect(g,limited).known,"general fusion remains refused");
        limited=original;limited.ps5_elementwise_fusion_mode=2;check(!native_scratch_bounds::inspect(g,limited).known,"unknown fusion mode refused");
        limited=original;limited.ps5_elementwise_fusion_mode=1;limited.no_async=false;check(!native_scratch_bounds::inspect(g,limited).known,"async remains refused");
        limited=original;limited.ps5_elementwise_fusion_mode=1;limited.f16_math_disabled=false;check(!native_scratch_bounds::inspect(g,limited).known,"unaudited FP16 fusion combination refused");
    }
    ggml_free(c);
    std::printf("PASS %u submission/fusion configuration and real GGML metadata checks; GPU=0 model_inference=0\n",checks);
}
