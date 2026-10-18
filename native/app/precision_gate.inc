/* SPDX-License-Identifier: GPL-3.0-or-later
 * Tiny actual Vulkan F16-weight matmul and dense Flash readbacks. CPU oracle
 * uses only fixed test data, never a model or image inference on Windows.
 */
static void precision_readback_gate() {
    struct Owned {
        ggml_backend_t backend=nullptr;ggml_context *ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;
        void clear() {
            ggml_backend_synchronize(backend);
            if(!ps5_ai_ggml_wait_backend("precision_gate_cleanup",0))ps5_ai_ggml_retain("precision_gate_cleanup_unknown");
            if(buffer){ggml_backend_buffer_free(buffer);buffer=nullptr;}
            if(ctx){ggml_free(ctx);ctx=nullptr;}
        }
        ~Owned(){if(backend){clear();ggml_backend_free(backend);}}
    } r;
    r.backend=ggml_backend_vk_init(0);
    if(!r.backend || ggml_backend_get_device(r.backend)!=gpu_device)throw std::runtime_error("Precision gate requires existing shared Vulkan device");
    for(unsigned test=0;test<2;++test) {
        r.ctx=ggml_init({1<<20,nullptr,true});if(!r.ctx)throw std::runtime_error("Precision gate metadata unavailable");
        ggml_tensor *a=nullptr,*b=nullptr,*v=nullptr,*out=nullptr;
        if(test==0){a=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F16,64,8);b=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,64,7);out=ggml_mul_mat(r.ctx,a,b);}
        else {a=ggml_new_tensor_3d(r.ctx,GGML_TYPE_F32,64,16,2);b=ggml_new_tensor_3d(r.ctx,GGML_TYPE_F16,64,16,2);v=ggml_new_tensor_3d(r.ctx,GGML_TYPE_F16,64,16,2);out=ggml_flash_attn_ext(r.ctx,a,b,v,nullptr,0.125f,0.f,0.f);}
        ggml_set_input(a);ggml_set_input(b);if(v)ggml_set_input(v);ggml_set_output(out);
        if(!ggml_backend_supports_op(r.backend,out))throw std::runtime_error("Precision gate operation unsupported; no CPU fallback");
        auto *graph=ggml_new_graph_custom(r.ctx,16,false);ggml_build_forward_expand(graph,out);
        r.buffer=ggml_backend_alloc_ctx_tensors(r.ctx,r.backend);if(!r.buffer)throw std::runtime_error("Precision gate buffer allocation failed");
        const size_t an=ggml_nelements(a),bn=ggml_nelements(b),vn=v?ggml_nelements(v):0,on=ggml_nelements(out);
        std::vector<float> af(an),bf(bn),vf(vn),actual(on);std::vector<ggml_fp16_t> half;
        for(size_t i=0;i<an;++i)af[i]=float(int(i%13)-6)/16.f;
        for(size_t i=0;i<bn;++i)bf[i]=float(int(i%11)-5)/16.f;
        for(size_t i=0;i<vn;++i)vf[i]=float(int(i%17)-8)/32.f;
        auto upload=[&](ggml_tensor *t,const std::vector<float> &data){
            if(t->type==GGML_TYPE_F16){half.resize(data.size());for(size_t i=0;i<data.size();++i)half[i]=ggml_fp32_to_fp16(data[i]);ggml_backend_tensor_set(t,half.data(),0,half.size()*sizeof(ggml_fp16_t));}
            else ggml_backend_tensor_set(t,data.data(),0,data.size()*sizeof(float));
        };
        upload(a,af);upload(b,bf);if(v)upload(v,vf);ggml_backend_synchronize(r.backend);
        ps5_ai_ggml_dispatch_stats before{},after{};ps5_ai_ggml_dispatch_snapshot(&before);
        ps5_ai_ggml_report("BEFORE precision_readback case=%u fp16=%d CPU_fallback=0",test,precision_fp16);
        auto status=ggml_backend_graph_compute(r.backend,graph);ggml_backend_synchronize(r.backend);
        if(status!=GGML_STATUS_SUCCESS || !ps5_ai_ggml_wait_backend("precision_gate_result",1))ps5_ai_ggml_retain("precision_gate_completion_unknown");
        ps5_ai_ggml_dispatch_snapshot(&after);
        ggml_backend_tensor_get(out,actual.data(),0,on*sizeof(float));
        double maximum=0;uint64_t invalid=0,mismatches=0;
        for(size_t i=0;i<on;++i) {
            double expected=0;
            if(test==0){size_t m=i%8,n=i/8;for(size_t k=0;k<64;++k)expected+=double(af[m*64+k])*bf[n*64+k];}
            else {
                const size_t d=i%64,head=(i/64)%2,row=(i/128)%16;
                double scores[16],denom=0;
                for(size_t col=0;col<16;++col){double dot=0;for(size_t k=0;k<64;++k)dot+=double(af[head*1024+row*64+k])*bf[head*1024+col*64+k];scores[col]=std::exp(dot*0.125);denom+=scores[col];}
                for(size_t col=0;col<16;++col)expected+=scores[col]/denom*vf[head*1024+col*64+d];
            }
            const double error=std::fabs(double(actual[i])-expected);invalid+=!std::isfinite(actual[i]);mismatches+=!std::isfinite(actual[i])||error>0.002+0.02*std::fabs(expected);maximum=std::max(maximum,error);
        }
        write_event({{"event","native_precision_readback"},{"case",test==0?"F16_weight_matmul":"F32_Q_F16_KV_flash"},{"fp16_math",precision_fp16},
            {"compared_values",on},{"nonfinite",invalid},{"mismatches",mismatches},{"max_absolute_error",maximum},
            {"dispatch_delta",after.dispatches-before.dispatches},{"fence_delta",after.fenced_completions-before.fenced_completions},{"CPU_fallback",false}});
        if(invalid || mismatches || after.dispatches<=before.dispatches || after.fenced_completions<=before.fenced_completions)throw std::runtime_error("Precision readback mismatch or missing GPU execution proof");
        r.clear();
    }
}
