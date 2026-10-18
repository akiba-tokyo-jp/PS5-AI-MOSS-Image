/* SPDX-License-Identifier: GPL-3.0-or-later
 * Verify direct convolution against a CPU double oracle, not an FP16 matmul
 * reference. Uses the upstream auto-probe's two actual2D shapes and inputs.
 * No CPU image/model inference and no new VkDevice/AGC initialization.
 */
static void precision_conv_readback_gate() {
    struct OwnedConv {
        ggml_backend_t backend=nullptr;ggml_context *ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;
        void clear(){ggml_backend_synchronize(backend);
            if(!ps5_ai_ggml_wait_backend("conv_precision_cleanup",0))ps5_ai_ggml_retain("conv_precision_cleanup_unknown");
            if(buffer){ggml_backend_buffer_free(buffer);buffer=nullptr;}
            if(ctx){ggml_free(ctx);ctx=nullptr;}}
        ~OwnedConv(){if(backend){clear();ggml_backend_free(backend);}}
    } r;
    r.backend=ggml_backend_vk_init(0);
    if(!r.backend||ggml_backend_get_device(r.backend)!=gpu_device)throw std::runtime_error("Convolution gate requires shared Vulkan device");
    for(unsigned shape=0;shape<2;++shape){
        const int s=shape?16:32,c=shape?128:64;
        const size_t xn=size_t(s)*s*c,wn=size_t(9)*c*c,on=xn;
        std::vector<float> input(xn),weight(wn),actual(on);
        std::vector<ggml_fp16_t> half(wn);
        uint32_t state=42;
        for(auto &v:input){state=state*1664525u+1013904223u;v=ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(state>>8)/16777216.f-0.5f));}
        const float scale=1.f/std::sqrt(float(c*9));
        for(size_t i=0;i<wn;++i){state=state*1664525u+1013904223u;half[i]=ggml_fp32_to_fp16((float(state>>8)/16777216.f-0.5f)*scale);weight[i]=ggml_fp16_to_fp32(half[i]);}
        std::vector<double> expected(on);
        for(int oc=0;oc<c;++oc)for(int y=0;y<s;++y)for(int x=0;x<s;++x){
            double sum=0;
            for(int ic=0;ic<c;++ic)for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){
                const int iy=y+ky-1,ix=x+kx-1;
                if(iy>=0&&iy<s&&ix>=0&&ix<s)sum+=double(input[(size_t(ic)*s+iy)*s+ix])*weight[(size_t(oc)*c+ic)*9+ky*3+kx];
            }
            expected[(size_t(oc)*s+y)*s+x]=sum;
        }
        for(unsigned direct=0;direct<2;++direct){
            r.ctx=ggml_init({1<<20,nullptr,true});if(!r.ctx)throw std::runtime_error("Convolution metadata unavailable");
            auto *x=ggml_new_tensor_4d(r.ctx,GGML_TYPE_F32,s,s,c,1);
            auto *w=ggml_new_tensor_4d(r.ctx,GGML_TYPE_F16,3,3,c,c);
            auto *out=direct?ggml_conv_2d_direct(r.ctx,w,x,1,1,1,1,1,1):ggml_conv_2d(r.ctx,w,x,1,1,1,1,1,1);
            ggml_set_input(x);ggml_set_input(w);ggml_set_output(out);
            auto *graph=ggml_new_graph_custom(r.ctx,32,false);ggml_build_forward_expand(graph,out);
            for(int i=0;i<ggml_graph_n_nodes(graph);++i)if(!ggml_backend_supports_op(r.backend,ggml_graph_node(graph,i)))throw std::runtime_error("Convolution gate unsupported; no CPU fallback");
            r.buffer=ggml_backend_alloc_ctx_tensors(r.ctx,r.backend);if(!r.buffer)throw std::runtime_error("Convolution buffer unavailable");
            ggml_backend_tensor_set(x,input.data(),0,xn*sizeof(float));ggml_backend_tensor_set(w,half.data(),0,wn*sizeof(ggml_fp16_t));ggml_backend_synchronize(r.backend);
            ps5_ai_ggml_dispatch_stats before{},after{};ps5_ai_ggml_dispatch_snapshot(&before);
            const auto status=ggml_backend_graph_compute(r.backend,graph);ggml_backend_synchronize(r.backend);
            if(status!=GGML_STATUS_SUCCESS||!ps5_ai_ggml_wait_backend("conv_precision_result",1))ps5_ai_ggml_retain("conv_precision_completion_unknown");
            ps5_ai_ggml_dispatch_snapshot(&after);ggml_backend_tensor_get(out,actual.data(),0,on*sizeof(float));
            double maximum=0;uint64_t invalid=0,mismatches=0;
            for(size_t i=0;i<on;++i){const double error=std::fabs(double(actual[i])-expected[i]);maximum=std::max(maximum,error);invalid+=!std::isfinite(actual[i]);mismatches+=!std::isfinite(actual[i])||error>0.001+0.02*std::fabs(expected[i]);}
            write_event({{"event","native_conv_precision_readback"},{"shape",{s,s,c}},{"path",direct?"direct":"im2col"},{"fp16_math",precision_fp16},{"CPU_oracle","double convolution of actual rounded inputs"},{"compared_values",on},{"nonfinite",invalid},{"mismatches",mismatches},{"max_absolute_error",maximum},{"absolute_tolerance",0.001},{"relative_tolerance",0.02},{"dispatch_delta",after.dispatches-before.dispatches},{"fence_delta",after.fenced_completions-before.fenced_completions},{"CPU_fallback",false}});
            if(invalid||after.dispatches<=before.dispatches||after.fenced_completions<=before.fenced_completions||(direct&&mismatches))throw std::runtime_error("Direct convolution failed CPU oracle; refusing model initialization");
            r.clear();
        }
    }
}
