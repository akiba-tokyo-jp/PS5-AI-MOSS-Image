/* SPDX-License-Identifier: GPL-3.0-or-later
 * Fixed F32 elementwise fixture on the existing Vulkan device. No model data.
 * Uses independent double references and retains every completion/owner guard.
 */
static void moss_performance_readback_gate() {
#if PS5_AI_MOSS_PERFORMANCE_TRIAL
    struct Owned {
        ggml_backend_t backend=nullptr;ggml_context* context=nullptr;ggml_backend_buffer_t buffer=nullptr;
        void clear() noexcept {
            if(!backend)return;
            ggml_backend_synchronize(backend);
            ps5_ai_ggml_dispatch_stats state{};ps5_ai_ggml_dispatch_snapshot(&state);
            if(state.work_pending || !ps5_ai_ggml_wait_backend("moss_performance_gate_cleanup",0))
                ps5_ai_ggml_retain("moss_performance_gate_cleanup_completion_unknown");
            if(buffer){ggml_backend_buffer_free(buffer);buffer=nullptr;}
            if(context){ggml_free(context);context=nullptr;}
        }
        ~Owned() noexcept {if(backend){clear();ggml_backend_free(backend);}}
    } r;
    r.backend=ggml_backend_vk_init(0);
    if(!r.backend || !ggml_backend_is_vk(r.backend) || ggml_backend_get_device(r.backend)!=gpu_device)
        throw std::runtime_error("MOSS performance gate requires existing shared Vulkan hardware");
    constexpr int columns=8192,rows=304;
    constexpr size_t values=size_t(columns)*rows;
    std::vector<float> input(values*2);
    for(size_t i=0;i<input.size();++i)input[i]=float(int((i*17+i/columns)%257)-128)/32.f;
    for(unsigned test=0;test<4;++test) {
        const bool sigmoid=test>=2,reverse=(test&1)!=0;
        r.context=ggml_init({1<<20,nullptr,true});
        if(!r.context)throw std::runtime_error("MOSS performance gate metadata unavailable");
        auto* source=ggml_new_tensor_2d(r.context,GGML_TYPE_F32,columns*2,rows);
        auto* a=ggml_view_2d(r.context,source,columns,rows,size_t(columns)*2*sizeof(float),0);
        auto* b=ggml_view_2d(r.context,source,columns,rows,size_t(columns)*2*sizeof(float),size_t(columns)*sizeof(float));
        auto* unary=sigmoid?ggml_sigmoid(r.context,a):ggml_silu(r.context,a);
        auto* out=reverse?ggml_mul(r.context,b,unary):ggml_mul(r.context,unary,b);
        ggml_set_input(source);ggml_set_output(out);
        auto* graph=ggml_new_graph_custom(r.context,32,false);
        /* Materialize the other half-view before UNARY so both MUL operand
         * orders expose the same adjacent pair to direct backend execution. */
        ggml_build_forward_expand(graph,b);ggml_build_forward_expand(graph,out);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)
            if(!ggml_backend_supports_op(r.backend,ggml_graph_node(graph,i)))
                throw std::runtime_error("MOSS performance gate operation unsupported; CPU fallback forbidden");
        r.buffer=ggml_backend_alloc_ctx_tensors(r.context,r.backend);
        if(!r.buffer)throw std::runtime_error("MOSS performance gate buffer unavailable");
        ggml_backend_tensor_set(source,input.data(),0,input.size()*sizeof(float));
        ggml_backend_synchronize(r.backend);
        if(!ps5_ai_ggml_wait_backend("moss_performance_gate_upload",0))
            ps5_ai_ggml_retain("moss_performance_gate_upload_completion_unknown");
        ps5_ai_ggml_dispatch_stats before{},after{};ps5_ai_ggml_dispatch_snapshot(&before);
        if(before.work_pending)ps5_ai_ggml_retain("moss_performance_gate_before_pending");
        const uint64_t selected_before=perf_fused_selected.load(std::memory_order_relaxed);
        ps5_ai_ggml_report("BEFORE moss_performance_gate case=%u CPU_fallback=0 model_loaded=0",test);
        const auto start=std::chrono::steady_clock::now();
        const ggml_status status=ggml_backend_graph_compute(r.backend,graph);
        ggml_backend_synchronize(r.backend);
        if(status!=GGML_STATUS_SUCCESS || !ps5_ai_ggml_wait_backend("moss_performance_gate_result",1))
            ps5_ai_ggml_retain("moss_performance_gate_compute_completion_unknown");
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        ps5_ai_ggml_dispatch_snapshot(&after);
        if(after.work_pending || !after.hardware_verified || after.dispatches<=before.dispatches ||
           after.submits<=before.submits || after.fenced_completions<=before.fenced_completions)
            throw std::runtime_error("MOSS performance gate lacks hardware dispatch/submit/fence proof");
        std::vector<float> actual(values);ggml_backend_tensor_get(out,actual.data(),0,actual.size()*sizeof(float));
        ggml_backend_synchronize(r.backend);
        if(!ps5_ai_ggml_wait_backend("moss_performance_gate_readback",1))
            ps5_ai_ggml_retain("moss_performance_gate_readback_completion_unknown");
        uint64_t invalid=0,mismatches=0;double maximum=0;
        for(size_t i=0;i<values;++i) {
            const size_t row=i/columns,col=i%columns;
            const double a_value=input[row*columns*2+col],b_value=input[row*columns*2+columns+col];
            const double activation=1.0/(1.0+std::exp(-a_value));
            const double expected=(sigmoid?activation:a_value*activation)*b_value;
            const double difference=std::fabs(double(actual[i])-expected);
            const bool finite=std::isfinite(actual[i]) && std::isfinite(expected);
            invalid+=!finite;mismatches+=!finite || difference>0.00002+0.00002*std::fabs(expected);
            if(finite)maximum=std::max(maximum,difference);
        }
        const uint64_t selected=perf_fused_selected.load(std::memory_order_relaxed)-selected_before;
        const bool expected_fusion=PS5_AI_MOSS_PERF_VARIANT==2;
        write_event({{"event","native_moss_performance_gate"},{"case",test},{"variant",PS5_AI_MOSS_PERF_VARIANT},
            {"unary",sigmoid?"SIGMOID":"SILU"},{"reverse_mul",reverse},{"shape",{columns,rows}},
            {"row_stride_bytes",columns*2*sizeof(float)},{"source_halves_share_backing",true},
            {"compared_values",values},{"nonfinite",invalid},{"mismatches",mismatches},{"max_absolute_error",maximum},
            {"absolute_tolerance",0.00002},{"relative_tolerance",0.00002},
            {"limited_fusion_selected_graph_nodes",selected},{"expected_limited_fusion",expected_fusion},
            {"graph_and_wait_seconds",seconds},{"CPU_wall_not_GPU_timestamp",true},
            {"dispatch_delta",after.dispatches-before.dispatches},{"submit_delta",after.submits-before.submits},
            {"fence_delta",after.fenced_completions-before.fenced_completions},{"CPU_fallback",false},{"model_inference",false}});
        if(invalid || mismatches || (expected_fusion?selected!=1:selected!=0) || !ps5_ai_ggml_log_healthy())
            throw std::runtime_error("MOSS performance gate numerical/fusion-selection/durable-receipt mismatch");
        r.clear();
    }
#endif
}
