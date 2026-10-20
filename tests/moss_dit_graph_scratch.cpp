// SPDX-License-Identifier: GPL-3.0-or-later
// Actual MOSS Model::forward + selected GGUF tensor metadata, without weights,
// graph execution, allocation of tensor data, a GPU backend or PS5 access.
#include "json.hpp"
#include "model/diffusion/moss_image.hpp"
#define native_scratch_bounds baseline_native_scratch_bounds
#include "baseline/vulkan_scratch_bounds.hpp"
#undef native_scratch_bounds
#include "runner_admission.hpp"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

static unsigned checks;
static void check(bool ok,const char* message) {
    ++checks;if(!ok){std::fprintf(stderr,"FAIL %s\n",message);std::exit(1);}
}
// Only capability queries select the ROPE/FLASH graph variants observed on
// the console. No actual backend exists, and every tensor data call traps.
extern "C" bool ggml_backend_supports_op(ggml_backend_t backend,const ggml_tensor* t) {
    check(backend==reinterpret_cast<ggml_backend_t>(uintptr_t(1)),"metadata capability sentinel");
    return t->op==GGML_OP_ROPE_APPLY || t->op==GGML_OP_FLASH_ATTN_EXT || moss_dit_opt::valid(t);
}
bool sd_backend_is_cpu(ggml_backend_t) { return false; }
bool sd_backend_supports_cuda_mma(ggml_backend_t) { return false; }
bool sd_backend_is(ggml_backend_t,const std::string&) { return false; }
extern "C" void ggml_backend_tensor_memset(ggml_tensor*,uint8_t,size_t,size_t) { std::abort(); }
extern "C" void ggml_backend_tensor_set(ggml_tensor*,const void*,size_t,size_t) { std::abort(); }
extern "C" ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t) { std::abort(); }
void log_printf(sd_log_level_t,const char*,int,const char*,...) {}

int main(int argc,char** argv){
 check(argc==2,"header-only fixture argument");std::ifstream stream(argv[1]);nlohmann::json metadata;stream>>metadata;
 check(metadata.at("payload_bytes_read")==0&&metadata.at("tensors").size()==379,"actual379 header shapes, no model values");
 String2TensorStorage weights;const std::string prefix="model.diffusion_model";
 for(const auto& row:metadata.at("tensors")){TensorStorage s;s.name=prefix+"."+row.at("name").get<std::string>();s.type=static_cast<ggml_type>(row.at("type_id").get<int>());s.n_dims=row.at("shape").size();for(int i=0;i<s.n_dims;++i)s.ne[i]=row.at("shape").at(i).get<int64_t>();weights.emplace(s.name,s);}
 const auto config=MOSSImage::MOSSImageConfig::detect_from_weights(weights,prefix);check(config.depth==42&&config.dim==3072&&config.head_dim==128,"actual model dimensions");
 const native_scratch_bounds::Profile profile{true,false,true,true,36};nlohmann::json cases=nlohmann::json::array();
 for(int size:{256,1024})for(unsigned mask:{0u,1u,3u,7u,15u})for(bool refined:{false,true}){
  if(refined&&!(mask&1))continue;moss_dit_opt::begin(mask);
  auto*params=ggml_init({32<<20,nullptr,true});auto*c=ggml_init({128<<20,nullptr,true});check(params&&c,"metadata only contexts");
  {
   MOSSImage::Model model(config);model.init(params,weights,prefix);GGMLRunnerContext runner;runner.backend=reinterpret_cast<ggml_backend_t>(uintptr_t(1));runner.ggml_ctx=c;runner.flash_attn_enabled=true;
   auto*x=ggml_new_tensor_4d(c,GGML_TYPE_F32,size/16,size/16,128,1);auto*time=ggml_new_tensor_3d(c,GGML_TYPE_F32,256,1,1);auto*text=ggml_new_tensor_3d(c,GGML_TYPE_F32,refined?3072:4096,106,1);auto*pe=ggml_new_tensor_4d(c,GGML_TYPE_F32,2,2,64,(size/16)*(size/16)+106);auto*g=ggml_new_graph_custom(c,65536,false);
   ggml_build_forward_expand(g,model.forward(&runner,x,time,text,pe,{},refined));auto bound=native_graph_admission::inspect_scratch(g,profile,UINT64_C(1)<<30);
   check(bound.per_claim&&bound.upper.known&&!bound.upper.unknown_nodes,"real MOSS graph admits only known bounded operators");check(bound.upper.x==226492416&&bound.upper.y==226492416,"original conservative BF16 modulation bound retained");
   unsigned fused=0,flash=0;ggml_tensor*sample=nullptr;
   for(int i=0;i<g->n_nodes;++i){auto*t=g->nodes[i];flash+=t->op==GGML_OP_FLASH_ATTN_EXT;if(t->op==GGML_OP_MAP_CUSTOM3){++fused;sample=t;check(moss_dit_opt::valid(t),"every fused node exact recognized metadata");}}
   check(flash==(refined?42u:44u),"refined text skips only the two invariant text blocks");if(mask&6)check(fused>0,"requested actual model fusion nodes present");else check(fused==0,"cache/control-only model retains original pointwise ops");
   if(sample){auto saved=sample->op_params[0];sample->op_params[0]^=1;check(!native_graph_admission::inspect_scratch(g,profile,UINT64_C(1)<<30).per_claim,"unrecognized custom callback rejected");sample->op_params[0]=saved;}
   auto bad=profile;bad.no_fusion=false;check(!native_graph_admission::inspect_scratch(g,bad,UINT64_C(1)<<30).per_claim,"general fusion still refused");
   check(!native_graph_admission::inspect_scratch(g,profile,bound.upper.x-16384).per_claim,"original region limit still enforced");
   cases.push_back({{"size",size},{"mask",mask},{"refined_text",refined},{"nodes",g->n_nodes},{"fused_nodes",fused},{"flash_nodes",flash},{"x_upper",bound.upper.x},{"y_upper",bound.upper.y}});
   // Canonical rank144 independent q/k/v graph, metadata only. No LoRA values.
   const int N=(size/16)*(size/16)+106;auto*in=ggml_new_tensor_2d(c,GGML_TYPE_F32,3072,N);auto*w=ggml_new_tensor_2d(c,GGML_TYPE_Q4_K,3072,9216);auto*base=ggml_mul_mat(c,w,in);
   auto diff=[&](){return ggml_mul_mat(c,ggml_new_tensor_2d(c,GGML_TYPE_F32,144,3072),ggml_mul_mat(c,ggml_new_tensor_2d(c,GGML_TYPE_F32,3072,144),in));};auto*q=diff(),*k=diff(),*v=diff();auto*out=moss_dit_opt::make(c,moss_dit_opt::LORA_PACK3,base,q,k,v);auto*lg=ggml_new_graph_custom(c,64,false);ggml_build_forward_expand(lg,out);
   check(native_graph_admission::inspect_scratch(lg,profile,UINT64_C(1)<<30).per_claim,"rank144 packed qkv and fourth input bound included");auto old=v->ne[0];v->ne[0]++;check(!native_graph_admission::inspect_scratch(lg,profile,UINT64_C(1)<<30).per_claim,"invalid fourth input rejected");v->ne[0]=old;
  }
  ggml_free(c);ggml_free(params);moss_dit_opt::end_completed();
 }
 std::cout<<nlohmann::json({{"checks",checks},{"cases",cases},{"actual_model_forward",true},{"tensor_data_allocated",false},{"graph_executed",false},{"backend_initialized",false},{"PS5_access",false}}).dump(2)<<"\n";
}
