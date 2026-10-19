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
    return t->op==GGML_OP_ROPE_APPLY || t->op==GGML_OP_FLASH_ATTN_EXT;
}
bool sd_backend_is_cpu(ggml_backend_t) { return false; }
bool sd_backend_supports_cuda_mma(ggml_backend_t) { return false; }
bool sd_backend_is(ggml_backend_t,const std::string&) { return false; }
extern "C" void ggml_backend_tensor_memset(ggml_tensor*,uint8_t,size_t,size_t) { std::abort(); }
extern "C" void ggml_backend_tensor_set(ggml_tensor*,const void*,size_t,size_t) { std::abort(); }
extern "C" ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t) { std::abort(); }
void log_printf(sd_log_level_t,const char*,int,const char*,...) {}

int main(int argc,char** argv) {
    check(argc==2,"selected header metadata argument");
    std::ifstream stream(argv[1]);nlohmann::json metadata;stream>>metadata;
    check(metadata.at("selected_filename")=="MOSS-Image-SFT-Q4_K.gguf" &&
        metadata.at("payload_bytes_read")==0 && metadata.at("tensors").size()==379,"actual selected header-only379 tensors");
    String2TensorStorage weights;
    const std::string prefix="model.diffusion_model";
    for(const auto& row:metadata.at("tensors")) {
        TensorStorage s;s.name=prefix+"."+row.at("name").get<std::string>();
        s.type=static_cast<ggml_type>(row.at("type_id").get<int>());
        s.n_dims=static_cast<int>(row.at("shape").size());
        for(int i=0;i<s.n_dims;++i)s.ne[i]=row.at("shape").at(i).get<int64_t>();
        weights.emplace(s.name,s);
    }
    const auto config=MOSSImage::MOSSImageConfig::detect_from_weights(weights,prefix);
    check(config.depth==42 && config.dim==3072 && config.head_dim==128 && config.ffn_dim==8192 && config.channels==128,"actual selected MOSS shape detection");
    const native_scratch_bounds::Profile profile{true,true,true,true,36};
    nlohmann::json cases=nlohmann::json::array();
    for(int tokens:{48,45}) {
        auto* params=ggml_init({32<<20,nullptr,true});
        auto* context=ggml_init({128<<20,nullptr,true});
        check(params && context,"no-alloc metadata contexts");
        {
            MOSSImage::Model model(config);model.init(params,weights,prefix);
            std::map<std::string,ggml_tensor*> actual_params;model.get_param_tensors(actual_params,prefix);
            check(actual_params.size()==378,"actual model378 parameters; source rope.bands is generated separately");
            for(const auto& entry:weights)
                check(actual_params.count(entry.first)==1 || entry.first==prefix+".rope.bands","all selected model tensors accounted for");
            GGMLRunnerContext runner;runner.backend=reinterpret_cast<ggml_backend_t>(uintptr_t(1));
            runner.ggml_ctx=context;runner.flash_attn_enabled=true;
            auto* image=ggml_new_tensor_4d(context,GGML_TYPE_F32,16,16,128,1);
            auto* time=ggml_new_tensor_3d(context,GGML_TYPE_F32,256,1,1);
            auto* text=ggml_new_tensor_3d(context,GGML_TYPE_F32,4096,tokens,1);
            auto* pe=ggml_new_tensor_4d(context,GGML_TYPE_F32,2,2,64,256+tokens);
            auto* graph=ggml_new_graph_custom(context,65536,false);
            ggml_build_forward_expand(graph,model.forward(&runner,image,time,text,pe,{}));
            auto before=baseline_native_scratch_bounds::inspect(graph,{true,true,true,true,36});
            auto after=native_graph_admission::inspect_scratch(graph,profile,UINT64_C(1)<<30);
            unsigned bf16_matmuls=0,flash=0;ggml_tensor* first=nullptr;
            nlohmann::json rejected=nlohmann::json::array();
            for(int i=0;i<graph->n_nodes;++i) {
                auto* t=graph->nodes[i];
                if(t->op==GGML_OP_FLASH_ATTN_EXT)++flash;
                ggml_tensor* one[]={t};ggml_cgraph isolated={};isolated.n_nodes=1;isolated.nodes=one;
                if(!baseline_native_scratch_bounds::inspect(&isolated,{true,true,true,true,36}).known) {
                    check(t->op==GGML_OP_MUL_MAT && t->src[0] && t->src[0]->type==GGML_TYPE_BF16,"all previous unknowns are BF16-left MUL_MAT");
                    rejected.push_back({{"op",ggml_op_name(t->op)},{"left_name",t->src[0]->name},
                        {"left_shape",{t->src[0]->ne[0],t->src[0]->ne[1],t->src[0]->ne[2],t->src[0]->ne[3]}},
                        {"right_shape",{t->src[1]->ne[0],t->src[1]->ne[1],t->src[1]->ne[2],t->src[1]->ne[3]}}});
                }
                if(t->op==GGML_OP_MUL_MAT && t->src[0]->type==GGML_TYPE_BF16) {++bf16_matmuls;if(!first)first=t;}
            }
            check(before.unknown_nodes==17 && before.first_unknown==GGML_OP_MUL_MAT && bf16_matmuls==17,"reproduces console unknown17 op29 exactly");
            check(flash==44,"actual observed44 flash nodes");
            check(after.per_claim && after.upper.known && after.upper.unknown_nodes==0,"all actual model graph scratch bounds known within actual1GiB region");
            check(after.upper.x==226492416 && after.upper.y==226492416,"shared modulation BF16 keeps conservative four-byte staging upper");
            auto* a=first->src[0];const auto old_stride=a->nb[1];a->nb[1]+=2;
            check(!native_graph_admission::inspect_scratch(graph,profile,UINT64_C(1)<<30).per_claim,"actual graph strided BF16 weight still rejected");a->nb[1]=old_stride;
            check(!native_graph_admission::inspect_scratch(graph,profile,after.upper.x-16384).per_claim,"actual graph scratch above claimed region still refused");
            auto old_type=first->src[1]->type;first->src[1]->type=GGML_TYPE_BF16;
            check(!native_graph_admission::inspect_scratch(graph,profile,UINT64_C(1)<<30).per_claim,"BF16 activations remain unaudited and rejected");first->src[1]->type=old_type;
            auto unsupported=profile;unsupported.no_fusion=false;
            check(!native_graph_admission::inspect_scratch(graph,unsupported,UINT64_C(1)<<30).per_claim,"unverified fusion profile remains refused");
            cases.push_back({{"tokens",tokens},{"nodes",graph->n_nodes},{"previous_unknown_nodes",before.unknown_nodes},
                {"previous_first_unknown_op",int(before.first_unknown)},{"bf16_matmuls",bf16_matmuls},{"flash_nodes",flash},
                {"x_upper",after.upper.x},{"y_upper",after.upper.y},{"split_k_upper",after.upper.split_k},
                {"all_prior_unknowns",rejected},{"selected_LoRA_graph_included",false}});
        }
        ggml_free(context);ggml_free(params);
    }
    std::cout<<nlohmann::json({{"checks",checks},{"cases",cases},{"actual_model_forward",true},
        {"tensor_data_allocated",false},{"graph_executed",false},{"backend_initialized",false},{"PS5_access",false}}).dump(2)<<"\n";
}
