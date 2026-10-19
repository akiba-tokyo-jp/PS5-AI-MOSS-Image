// SPDX-License-Identifier: GPL-3.0-or-later
// Link actual pinned GGML tensor constructors. Metadata only, no model data,
// graph execution, backend initialization, GPU call or PS5 connection.
#include "vulkan_scratch_bounds.hpp"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
using namespace native_scratch_bounds;
static unsigned checks;
static Profile profile{true,true,true,true,36};
// GGML's base object also contains graph/backend helpers. Trap rather than
// link any backend: tensor construction must never reach these entry points.
extern "C" void ggml_backend_tensor_memset(ggml_tensor*,uint8_t,size_t,size_t) { std::abort(); }
extern "C" void ggml_backend_tensor_set(ggml_tensor*,const void*,size_t,size_t) { std::abort(); }
extern "C" ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t) { std::abort(); }
static void check(bool value,const char* message) {
    ++checks;if(!value){std::fprintf(stderr,"FAIL %s\n",message);std::exit(1);}
}
static void admitted(ggml_tensor* tensor,const char* message) {
    ggml_tensor* nodes[]={tensor};ggml_cgraph graph={};graph.n_nodes=1;graph.nodes=nodes;
    auto bounds=inspect(&graph,profile);check(bounds.known && !bounds.x && !bounds.y && !bounds.split_k,message);
}
int main() {
    auto* context=ggml_init({1<<20,nullptr,true});check(context!=nullptr,"metadata context");
    auto* q=ggml_new_tensor_4d(context,GGML_TYPE_F32,128,16,1024,1);
    auto* pe=ggml_new_tensor_4d(context,GGML_TYPE_F32,2,2,64,1024);
    auto* rms=ggml_rms_norm(context,q,1e-6f);admitted(rms,"actual Anima RMS constructor admitted");
    auto* rope=ggml_rope_apply(context,rms,pe,false);
    check(rope->ne[0]==128 && rope->ne[1]==1024 && rope->ne[2]==16 && rope->ne[3]==1,"actual RoPE layout D,L,H*N,1");
    admitted(rope,"actual Anima image RoPE constructor admitted");
    auto* batched_q=ggml_new_tensor_4d(context,GGML_TYPE_F32,64,16,512,2);
    auto* adapter_pe=ggml_new_tensor_4d(context,GGML_TYPE_F32,2,2,32,512);
    auto* batched_rope=ggml_rope_apply(context,batched_q,adapter_pe,false);
    check(batched_rope->ne[0]==64 && batched_rope->ne[1]==512 && batched_rope->ne[2]==32 && batched_rope->ne[3]==1,"actual adapter RoPE packed batch layout");
    admitted(batched_rope,"actual adapter RoPE packed heads/batch admitted");
    auto* kernel=ggml_new_tensor_4d(context,GGML_TYPE_F16,3,3,3,16*384);
    auto* image=ggml_new_tensor_4d(context,GGML_TYPE_F32,32,32,3,16);
    auto* conv=ggml_conv_3d_direct(context,kernel,image,1,1,1,1,1,0,1,1,1,16,1,384);
    admitted(conv,"actual Qwen VAE direct conv3d constructor admitted");
    auto* col=ggml_im2col_3d(context,kernel,image,16,1,1,1,1,1,0,1,1,1,GGML_TYPE_F16);
    admitted(col,"actual Qwen VAE im2col3d fallback constructor admitted");
    ggml_free(context);
    std::printf("PASS %u actual pinned GGML constructor checks; metadata only, model inference=0, GPU calls=0\n",checks);
}
