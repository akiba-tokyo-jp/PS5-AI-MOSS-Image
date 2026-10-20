// SPDX-License-Identifier: MIT
// Synthetic fixed data only; CPU quantization creates input, no learned host inference.
static void moss_kernel_matrix_attention_gate(){
    for(unsigned test=0;test<8;++test){
        const unsigned test_bit=test==2||test==3?4:test>=6?2:1;
        if(!(PS5_AI_MOSS_DIT_KERNEL_MASK&test_bit))continue;
        ps5_ai_flex_stats before_pool{};ps5_ai_flex_snapshot(&before_pool);
        struct Owned{
            ggml_backend_t backend=nullptr;ggml_context* ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;
            ~Owned(){if(backend){ggml_backend_synchronize(backend);if(!ps5_ai_ggml_wait_backend("kernel_gate_cleanup",0))ps5_ai_ggml_retain("kernel_gate_unknown");if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);ggml_backend_free(backend);}}
        } r;
        r.backend=ggml_backend_vk_init(0);r.ctx=ggml_init({4<<20,nullptr,true});
        if(!r.backend||!r.ctx||ggml_backend_get_device(r.backend)!=gpu_device)throw std::runtime_error("Kernel gate Vulkan device unavailable");
        std::vector<float> input,query;std::vector<block_q4_K> quant;std::vector<ggml_fp16_t> key,value;
        ggml_tensor*out=nullptr;unsigned bit=0;
        if(test<6){
            const int K=test==1||test==5?8192:3072;
            const int M=test==0?16384:test==2?9216:test<4?3072:32;
            const int N=test<4?17:4202;bit=test==2||test==3?4:1;
            std::vector<float> row(K);quant.resize(size_t(K/256)*M);input.resize(size_t(K)*N);
            for(int m=0;m<M;++m){for(int k=0;k<K;++k)row[k]=float(int((size_t(k)*37+size_t(m)*17)%127)-63)/128.f;quantize_row_q4_K_ref(row.data(),quant.data()+size_t(m)*(K/256),K);}
            for(size_t i=0;i<input.size();++i)input[i]=float(int((i*23+i/K*7)%101)-50)/256.f;
            auto*a=ggml_new_tensor_2d(r.ctx,GGML_TYPE_Q4_K,K,M);auto*b=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,K,N);
            ggml_set_name(a,bit==1?"moss_kernel_gate_ffn":"moss_kernel_gate_qkv");ggml_set_input(a);ggml_set_input(b);
            out=ggml_mul_mat(r.ctx,a,b);if(!ggml_prec_set_acc(out,GGML_PREC_F32))throw std::runtime_error("Kernel F32 precision rejected");
        }else{
            bit=2;const int N=test==6?362:4202;const size_t count=size_t(128)*N*2;
            query.resize(count);key.resize(count);value.resize(count);
            for(size_t i=0;i<count;++i){query[i]=float(int((i*17+i/128*3)%41)-20)/32.f;key[i]=ggml_fp32_to_fp16(float(int((i*11+i/128)%37)-18)/32.f);value[i]=ggml_fp32_to_fp16(float(int((i*7+i/128*5)%43)-21)/32.f);}
            auto*q=ggml_new_tensor_3d(r.ctx,GGML_TYPE_F32,128,N,2);auto*k=ggml_new_tensor_3d(r.ctx,GGML_TYPE_F16,128,N,2);auto*v=ggml_new_tensor_3d(r.ctx,GGML_TYPE_F16,128,N,2);
            ggml_set_input(q);ggml_set_input(k);ggml_set_input(v);out=ggml_flash_attn_ext(r.ctx,q,k,v,nullptr,1.f/std::sqrt(128.f),0.f,0.f);if(!ggml_prec_set_acc(out,GGML_PREC_F32))throw std::runtime_error("Kernel FA F32 precision rejected");
        }
        ggml_set_output(out);auto*g=ggml_new_graph_custom(r.ctx,32,false);ggml_build_forward_expand(g,out);
        if(!ggml_backend_supports_op(r.backend,out))throw std::runtime_error("Kernel gate unsupported GPU operator");
        r.buffer=ggml_backend_alloc_ctx_tensors(r.ctx,r.backend);if(!r.buffer||ggml_backend_buffer_is_host(r.buffer))throw std::runtime_error("Kernel gate GPU allocation failed");
        if(test<6){ggml_backend_tensor_set(out->src[0],quant.data(),0,quant.size()*sizeof(block_q4_K));ggml_backend_tensor_set(out->src[1],input.data(),0,input.size()*4);}
        else{ggml_backend_tensor_set(out->src[0],query.data(),0,query.size()*4);ggml_backend_tensor_set(out->src[1],key.data(),0,key.size()*2);ggml_backend_tensor_set(out->src[2],value.data(),0,value.size()*2);}
        ggml_backend_synchronize(r.backend);if(!ps5_ai_ggml_wait_backend("kernel_gate_upload",0))ps5_ai_ggml_retain("kernel_gate_upload_unknown");
        std::vector<float> baseline(ggml_nelements(out)),actual(baseline.size());double seconds[2]{};
        for(unsigned run=0;run<2;++run){
            moss_kernel::reset(run?bit:0);
            ps5_ai_ggml_dispatch_stats before{},after{};ps5_ai_ggml_dispatch_snapshot(&before);
            auto start=std::chrono::steady_clock::now();const auto status=ggml_backend_graph_compute(r.backend,g);ggml_backend_synchronize(r.backend);
            if(status!=GGML_STATUS_SUCCESS||!ps5_ai_ggml_wait_backend("kernel_gate_compute",1))ps5_ai_ggml_retain("kernel_gate_compute_unknown");
            seconds[run]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            ps5_ai_ggml_dispatch_snapshot(&after);if(after.work_pending||after.dispatches<=before.dispatches||after.fenced_completions<=before.fenced_completions)throw std::runtime_error("Kernel gate dispatch/fence unavailable");
            if(run&&!moss_kernel::records[bit==1?0:bit==2?1:2].load())throw std::runtime_error("Kernel gate candidate not selected");
            auto& target=run?actual:baseline;ggml_backend_tensor_get(out,target.data(),0,target.size()*4);ggml_backend_synchronize(r.backend);
            if(!ps5_ai_ggml_wait_backend("kernel_gate_readback",1))ps5_ai_ggml_retain("kernel_gate_readback_unknown");
        }
        uint64_t bad=0,nonfinite=0;double maximum=0;const double abs=test<6?.001:2e-5,rel=test<6?.01:2e-5;
        for(size_t i=0;i<actual.size();++i){nonfinite+=!std::isfinite(actual[i])||!std::isfinite(baseline[i]);double error=std::fabs(double(actual[i])-baseline[i]);maximum=std::max(maximum,error);bad+=error>abs+rel*std::fabs(baseline[i]);}
        ggml_backend_buffer_free(r.buffer);r.buffer=nullptr;ggml_free(r.ctx);r.ctx=nullptr;ggml_backend_free(r.backend);r.backend=nullptr;
        ps5_ai_flex_stats after_pool{};ps5_ai_flex_snapshot(&after_pool);
        const bool released=before_pool.used_bytes==after_pool.used_bytes&&before_pool.allocations==after_pool.allocations&&before_pool.active_imports==after_pool.active_imports&&!after_pool.fault_retained&&after_pool.log_healthy;
        write_event({{"event","native_moss_kernel_matrix_readback"},{"test",test},{"bit",bit},{"compared",actual.size()},{"nonfinite",nonfinite},{"mismatches",bad},{"max_absolute_error",maximum},{"absolute_tolerance",abs},{"relative_tolerance",rel},{"baseline_gate_seconds",seconds[0]},{"candidate_gate_seconds",seconds[1]},{"warm_benchmark",false},{"owned_buffers_released",released},{"CPU_inference",false},{"success",!bad&&!nonfinite&&released}});
        if(bad||nonfinite||!released)throw std::runtime_error("Kernel matrix/attention readback rejected");
    }
}
