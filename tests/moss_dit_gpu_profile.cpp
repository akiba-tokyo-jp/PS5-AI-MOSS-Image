// SPDX-License-Identifier: MIT
// Metadata tests only; never execute a tensor or initialize a GPU.
#include "ggml.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "ggml-impl.h"
#pragma clang diagnostic pop
#include "moss_dit_ops.hpp"
#include "moss_gpu_profile.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
extern "C" unsigned ps5_ai_test_actual_node_class(const ggml_tensor*);
extern "C" ps5_ai_moss_gpu_coverage ps5_ai_test_actual_graph_coverage(const ggml_cgraph*);
static unsigned checks;
static void need(bool x){++checks;if(!x)std::abort();}
int main(){
 static_assert(GGML_MAX_NAME==160,"Actual tensor name ABI");
 static_assert(PS5_AI_MOSS_GPU_PROFILE_CLASSES==25,"Diagnostic ABI");
 const char* tails[]={"qkv.weight","qkv.weight.1","qkv.weight.2","o.weight","ffn_up_gate.weight","ffn_down.weight"};
 std::vector<ggml_tensor> weights(672),rhs(672),nodes(672);
 std::vector<ggml_tensor*> pointers;unsigned n=0;
 auto mm=[&](const std::string& name,ggml_type type,int64_t k,int64_t m,unsigned cls){
  auto& a=weights[n];a.type=type;a.ne[0]=k;a.ne[1]=m;a.ne[2]=a.ne[3]=1;ggml_set_name(&a,name.c_str());
  auto& b=rhs[n];b.type=GGML_TYPE_F32;b.ne[0]=k;b.ne[1]=4202;b.ne[2]=b.ne[3]=1;
  auto& o=nodes[n];o.type=GGML_TYPE_F32;o.op=GGML_OP_MUL_MAT;o.src[0]=&a;o.src[1]=&b;o.ne[0]=m;o.ne[1]=4202;o.ne[2]=o.ne[3]=1;
  need(ps5_ai_test_actual_node_class(&o)==cls);
  auto actual=a.ne[0];a.ne[0]++;need(ps5_ai_test_actual_node_class(&o)!=cls);a.ne[0]=actual;
  pointers.push_back(&o);++n;
 };
 for(unsigned l=0;l<42;l++){
  const auto prefix="model.diffusion_model.layers."+std::to_string(l)+".proj.";
  mm(prefix+tails[0],GGML_TYPE_Q4_K,3072,9216,PS5_AI_GPU_Q4_QKV);
  mm(prefix+tails[3],GGML_TYPE_Q4_K,3072,3072,PS5_AI_GPU_Q4_O);
  mm(prefix+tails[4],GGML_TYPE_Q4_K,3072,16384,PS5_AI_GPU_Q4_FFN_UP);
  mm(prefix+tails[5],GGML_TYPE_Q4_K,8192,3072,PS5_AI_GPU_Q4_FFN_DOWN);
  for(unsigned t=0;t<6;t++){
   mm("lora."+prefix+tails[t]+".lora_down",GGML_TYPE_F32,t==5?8192:3072,144,PS5_AI_GPU_LORA_DOWN);
   mm("lora."+prefix+tails[t]+".lora_up",GGML_TYPE_F32,144,t==4?16384:3072,PS5_AI_GPU_LORA_UP);
   need(ps5_ai_moss_gpu_lora_class(("lora."+prefix+tails[t]+".lora_up").c_str(),16,t==4?16384:3072)==PS5_AI_GPU_OTHER_MM);
  }
 }
 need(n==672);ggml_cgraph graph{};graph.n_nodes=n;graph.nodes=pointers.data();
 auto c=ps5_ai_test_actual_graph_coverage(&graph);need(ps5_ai_moss_gpu_coverage_complete(&c));need(c.lora_down_nodes==252&&c.lora_up_nodes==252);
 c.lora_up_nodes--;need(!ps5_ai_moss_gpu_coverage_complete(&c));c.lora_up_nodes++;
 for(unsigned i=0;i<4;i++){c.masks[i]^=UINT64_C(1)<<22;c.q4_main_nodes[i]--;}
 need(!ps5_ai_moss_gpu_coverage_complete(&c));
 ggml_tensor a{},out{};a.type=GGML_TYPE_F32;out.type=GGML_TYPE_F16;out.op=GGML_OP_CPY;out.src[0]=&a;
 need(ps5_ai_test_actual_node_class(&out)==PS5_AI_GPU_CAST);out.type=GGML_TYPE_F32;
 need(ps5_ai_test_actual_node_class(&out)==PS5_AI_GPU_COPY);out.op=GGML_OP_CONT;
 need(ps5_ai_test_actual_node_class(&out)==PS5_AI_GPU_CONT);out.op=GGML_OP_RMS_NORM;
 need(ps5_ai_test_actual_node_class(&out)==PS5_AI_GPU_RMS);
 const unsigned fused[]={PS5_AI_GPU_OTHER,PS5_AI_GPU_SITU,PS5_AI_GPU_NORM_MOD,PS5_AI_GPU_NORM_MOD,PS5_AI_GPU_RESIDUAL,PS5_AI_GPU_QK_ROPE,PS5_AI_GPU_LORA_ADD,PS5_AI_GPU_LORA_PACK,PS5_AI_GPU_LORA_PACK};
 for(unsigned k=1;k<=8;k++){
  out.op=GGML_OP_MAP_CUSTOM3;ggml_map_custom3_op_params p{moss_dit_opt::cpu,1,reinterpret_cast<void*>(uintptr_t(k))};
  std::memcpy(out.op_params,&p,sizeof(p));need(ps5_ai_test_actual_node_class(&out)==fused[k]);
  p.n_tasks=2;std::memcpy(out.op_params,&p,sizeof(p));need(ps5_ai_test_actual_node_class(&out)==PS5_AI_GPU_OTHER);
 }
 const char* pipelines[]={"matmul_q4_k_f16","flash_attn_f32_f16","copy_f32_f16","split_k_reduce","rms_norm_f32","moss_dit_fused_f32","scale_f32","unknown"};
 for(unsigned i=0;i<8;i++)need(ps5_ai_moss_gpu_shader_class(pipelines[i])==i);
 need(ps5_ai_moss_gpu_shader_class("fa_split_k_reduce")==3);
 need(ps5_ai_moss_gpu_shader_class(nullptr)==7);
 ps5_ai_moss_gpu_class_stat stat{};need(ps5_ai_moss_gpu_accumulate(&stat,0));need(ps5_ai_moss_gpu_accumulate(&stat,10));need(stat.ticks==10&&stat.minimum_ticks==0&&stat.maximum_ticks==10);
 stat.ticks=UINT64_MAX;need(!ps5_ai_moss_gpu_accumulate(&stat,1));
 ps5_ai_moss_gpu_query_value x{100,1},y{200,1};uint64_t ticks=0;
 need(ps5_ai_moss_gpu_pair_ticks(&x,&y,&ticks)&&ticks==100);y.available=0;need(!ps5_ai_moss_gpu_pair_ticks(&x,&y,&ticks));y.available=1;y.ticks=99;need(!ps5_ai_moss_gpu_pair_ticks(&x,&y,&ticks));
 std::printf("PASS %u actual metadata checks; GPU calls0; inference0\n",checks);
}
