/* SPDX-License-Identifier: GPL-3.0-or-later
 * Additional fixed Q4_K FFN-up admission, not model inference.
 * The original fourteen gates and tolerances remain unchanged.
 */
#include "moss_q4_candidate.h"

struct MossQ4CandidateGateResult {
    bool numerical_pass=false, microtiming_faster=false, candidate_enabled=false;
    double baseline_seconds=0, candidate_seconds=0;
};
[[maybe_unused]] static bool moss_q4_candidate_admission_passed=false;

[[maybe_unused]] static nlohmann::json moss_q4_candidate_json(const ps5_ai_moss_q4_candidate_report &r) {
    nlohmann::json pipelines=nlohmann::json::array();
    for(unsigned i=0;i<4;i++) {
        const auto &p=r.pipelines[i];nlohmann::json statistics=nlohmann::json::array();
        for(unsigned j=0;j<p.stats_recorded && j<16;j++) {
            const auto &s=p.stats[j];
            statistics.push_back({{"name",std::string(s.name,strnlen(s.name,sizeof(s.name)))},
                {"format",s.format},{"raw_value_bits",s.value_bits}});
        }
        pipelines.push_back({{"slot",i},{"role",i<2?"candidate_A":"baseline_medium64"},
            {"pipeline_handle",p.pipeline},{"original_module_bytes",p.original_module_bytes},
            {"actual_module_bytes",p.actual_module_bytes},{"compile_count",p.compile_count},
            {"specialization",std::vector<uint32_t>(p.specialization,p.specialization+13)},
            {"required_subgroup",p.required_subgroup},{"aligned",bool(p.aligned)},
            {"stats_available",bool(p.stats_available)},{"stats_recorded",p.stats_recorded},
            {"stats_total",p.stats_total},{"statistics",statistics}});
    }
    return {{"schema",r.schema},{"enabled",bool(r.enabled)},{"active",bool(r.active)},
        {"recording_scope","post-dispatch receipts; actual hardware completion checked separately"},
        {"selected_calls",r.selected_calls},{"baseline_calls",r.baseline_calls},
        {"invalid_receipts",r.invalid_receipts},{"device_generation",r.device_generation},
        {"layer_calls",std::vector<uint64_t>(r.layer_calls,r.layer_calls+42)},
        {"baseline_layer_calls",std::vector<uint64_t>(r.baseline_layer_calls,r.baseline_layer_calls+42)},
        {"token_order",{270,362,4110,4202}},
        {"token_calls",std::vector<uint64_t>(r.token_calls,r.token_calls+4)},
        {"baseline_token_calls",std::vector<uint64_t>(r.baseline_token_calls,r.baseline_token_calls+4)},
        {"pipelines",pipelines}};
}

static nlohmann::json moss_q4_candidate_generation_json(
        const ps5_ai_moss_q4_candidate_report &before,const ps5_ai_moss_q4_candidate_report &after) {
#if PS5_AI_MOSS_Q4_CANDIDATE_A
    std::vector<uint64_t> layers(42),baseline_layers(42),tokens(4),baseline_tokens(4);
    for(unsigned i=0;i<42;i++) {
        layers[i]=after.layer_calls[i]-before.layer_calls[i];
        baseline_layers[i]=after.baseline_layer_calls[i]-before.baseline_layer_calls[i];
    }
    for(unsigned i=0;i<4;i++) {
        tokens[i]=after.token_calls[i]-before.token_calls[i];
        baseline_tokens[i]=after.baseline_token_calls[i]-before.baseline_token_calls[i];
    }
    return {{"admission_passed",moss_q4_candidate_admission_passed},
        {"selected_calls_before",before.selected_calls},{"baseline_calls_before",before.baseline_calls},
        {"token_calls_before",std::vector<uint64_t>(before.token_calls,before.token_calls+4)},
        {"baseline_token_calls_before",std::vector<uint64_t>(before.baseline_token_calls,before.baseline_token_calls+4)},
        {"selected_recording_delta",after.selected_calls-before.selected_calls},
        {"baseline_recording_delta",after.baseline_calls-before.baseline_calls},
        {"invalid_receipts_delta",after.invalid_receipts-before.invalid_receipts},
        {"layer_recording_deltas",layers},{"baseline_layer_recording_deltas",baseline_layers},
        {"token_order",{270,362,4110,4202}},
        {"token_recording_deltas",tokens},{"baseline_token_recording_deltas",baseline_tokens},
        {"actual_cumulative_after",moss_q4_candidate_json(after)},
        {"scope","recorded candidate FFN-up graph nodes; excludes startup warm/timed gates through before/after deltas; completion checked before PNG publication"}};
#else
    (void)before;(void)after;
    return nullptr;
#endif
}

[[maybe_unused]] static MossQ4CandidateGateResult moss_q4_candidate_readback_gate() {
    MossQ4CandidateGateResult result;
#if PS5_AI_MOSS_Q4_CANDIDATE_A
    moss_q4_candidate_admission_passed=false;
    nlohmann::json shape_results=nlohmann::json::array();
    uint64_t gate_device_generation=0;
    for(unsigned shape=0;shape<2;shape++) {
    const int N=shape==0?270:362;
    double shape_baseline_seconds=0,shape_candidate_seconds=0;
    struct Owned {
        ggml_backend_t backend=nullptr;ggml_context *context=nullptr;ggml_backend_buffer_t buffer=nullptr;
        void clear() noexcept {
            if(!backend)return;
            ggml_backend_synchronize(backend);
            ps5_ai_ggml_dispatch_stats state{};ps5_ai_ggml_dispatch_snapshot(&state);
            if(state.work_pending || !ps5_ai_ggml_wait_backend("moss_q4_candidate_gate_cleanup",0))
                ps5_ai_ggml_retain("moss_q4_candidate_gate_cleanup_completion_unknown");
            if(!ps5_ai_moss_q4_candidate_set_enabled(0))
                ps5_ai_ggml_retain("moss_q4_candidate_gate_cleanup_control_not_idle");
            if(buffer)ggml_backend_buffer_free(buffer);
            if(context)ggml_free(context);
            ggml_backend_free(backend);
            backend=nullptr;context=nullptr;buffer=nullptr;
        }
        ~Owned() noexcept {clear();}
    } r;
    r.backend=ggml_backend_vk_init(0);
    if(!r.backend || !ggml_backend_is_vk(r.backend) || ggml_backend_get_device(r.backend)!=gpu_device)
        throw std::runtime_error("MOSS Q4 candidate gate requires existing shared Vulkan hardware");
    if(!ps5_ai_moss_q4_candidate_set_enabled(0))
        throw std::runtime_error("MOSS Q4 candidate gate cannot select idle baseline");
    ps5_ai_moss_q4_candidate_report admission{};ps5_ai_moss_q4_candidate_snapshot(&admission);
    write_event({{"event","native_moss_q4_candidate_admission"},{"shape_index",shape},{"N",N},
        {"enabled",bool(admission.enabled)},{"active",bool(admission.active)},
        {"device_generation",admission.device_generation},{"selected_calls",admission.selected_calls},
        {"baseline_calls",admission.baseline_calls},{"invalid_receipts",admission.invalid_receipts}});
    if(admission.enabled!=1 || admission.active!=0 || !admission.device_generation)
        throw std::runtime_error("MOSS Q4 candidate NAVI21/subgroup capability admission unavailable; no fixed-data graph submitted");
    if(gate_device_generation && gate_device_generation!=admission.device_generation)
        throw std::runtime_error("MOSS Q4 candidate gate device changed between exact real token shapes");
    gate_device_generation=admission.device_generation;
    constexpr int K=3072,M=16384;
    const size_t output_values=size_t(M)*N;
    static_assert(K%QK_K==0,"whole Q4_K blocks");
    moss_precision::Fixture fixture;
    fixture.left_rows=M;fixture.right_cols=N;
    fixture.a.resize(K);fixture.b.resize(size_t(K)*N);
    fixture.quantized.resize(size_t(K/QK_K)*M);
    uint32_t state=0x6d6f7373u;
    auto next=[&]() {state=state*1664525u+1013904223u;return float(int((state>>16)&1023u)-511)/1024.f;};
    for(int row=0;row<M;row++) {
        for(float &x:fixture.a)x=next();
        quantize_row_q4_K_ref(fixture.a.data(),fixture.quantized.data()+size_t(row)*(K/QK_K),K);
    }
    for(float &x:fixture.b)x=next()*0.5f;
    std::vector<float> decoded(K);
    for(int row:{0,1,15,16,31,32,127,128,8191,8192,16383}) {
        dequantize_row_q4_K(fixture.quantized.data()+size_t(row)*(K/QK_K),decoded.data(),K);
        for(int col:{0,1,7,8,31,32,127,128,N-1}) {
            double sum=0;
            for(int d=0;d<K;d++)sum+=double(decoded[d])*fixture.b[size_t(col)*K+d];
            fixture.sampled_output_indices.push_back(size_t(col)*M+row);fixture.expected.push_back(sum);
        }
    }
    if(fixture.expected.size()!=99 || fixture.sampled_output_indices.size()!=99)
        throw std::runtime_error("MOSS Q4 candidate CPU oracle must contain exactly99 stored-Q4 samples");
    r.context=ggml_init({1<<20,nullptr,true});
    if(!r.context)throw std::runtime_error("MOSS Q4 candidate metadata unavailable");
    auto *a=ggml_new_tensor_2d(r.context,GGML_TYPE_Q4_K,K,M);
    auto *b=ggml_new_tensor_2d(r.context,GGML_TYPE_F32,K,N);
    ggml_set_name(a,"model.diffusion_model.layers.0.proj.ffn_up_gate.weight");
    auto *out=ggml_mul_mat(r.context,a,b);
    if(!ggml_prec_set_acc(out,GGML_PREC_F32))
        throw std::runtime_error("MOSS Q4 candidate F32 accumulation policy unavailable");
    ggml_set_input(a);ggml_set_input(b);ggml_set_output(out);
    auto *graph=ggml_new_graph_custom(r.context,16,false);ggml_build_forward_expand(graph,out);
    for(int i=0;i<ggml_graph_n_nodes(graph);i++)
        if(!ggml_backend_supports_op(r.backend,ggml_graph_node(graph,i)))
            throw std::runtime_error("MOSS Q4 candidate unsupported graph; CPU fallback forbidden");
    if(size_t(ggml_nelements(out))!=output_values)
        throw std::runtime_error("MOSS Q4 candidate output layout mismatch");
    r.buffer=ggml_backend_alloc_ctx_tensors(r.context,r.backend);
    if(!r.buffer)throw std::runtime_error("MOSS Q4 candidate GPU allocation unavailable");
    ggml_backend_tensor_set(a,fixture.quantized.data(),0,fixture.quantized.size()*sizeof(block_q4_K));
    ggml_backend_tensor_set(b,fixture.b.data(),0,fixture.b.size()*sizeof(float));
    ggml_backend_synchronize(r.backend);
    if(!ps5_ai_ggml_wait_backend("moss_q4_candidate_gate_upload",0))
        ps5_ai_ggml_retain("moss_q4_candidate_gate_upload_completion_unknown");
    std::vector<float> actual(output_values);
    for(unsigned run=0;run<4;run++) {
        const unsigned test=shape*4+run;
        const unsigned active=run%2;const bool warmup=run<2;
        if(!ps5_ai_moss_q4_candidate_set_enabled(active))
            throw std::runtime_error("MOSS Q4 candidate switch requires idle completed hardware");
        ps5_ai_ggml_dispatch_stats before{},after{};ps5_ai_ggml_dispatch_snapshot(&before);
        ps5_ai_moss_quant_receipt quant_before{},quant_after{};ps5_ai_moss_quant_snapshot(&quant_before);
        ps5_ai_moss_q4_candidate_report selected_before{},selected_after{};
        ps5_ai_moss_q4_candidate_snapshot(&selected_before);
        if(before.work_pending || !before.hardware_verified)
            ps5_ai_ggml_retain("moss_q4_candidate_gate_before_not_idle_hardware");
        ps5_ai_ggml_report("BEFORE moss_q4_candidate_gate case=%u N=%u active=%u warmup=%u CPU_fallback=0 model_loaded=0",test,unsigned(N),active,unsigned(warmup));
        const auto start=std::chrono::steady_clock::now();
        const auto status=ggml_backend_graph_compute(r.backend,graph);
        ggml_backend_synchronize(r.backend);
        if(status!=GGML_STATUS_SUCCESS || !ps5_ai_ggml_wait_backend("moss_q4_candidate_gate_result",1))
            ps5_ai_ggml_retain("moss_q4_candidate_gate_compute_completion_unknown");
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        ps5_ai_ggml_dispatch_snapshot(&after);ps5_ai_moss_quant_snapshot(&quant_after);
        ps5_ai_moss_q4_candidate_snapshot(&selected_after);
        if(after.work_pending || !after.hardware_verified || after.dispatches<=before.dispatches ||
           after.submits<=before.submits || after.fenced_completions<=before.fenced_completions)
            throw std::runtime_error("MOSS Q4 candidate lacks actual completed hardware dispatch/submit/fence proof");
        if(!std::isfinite(seconds) || seconds<=0 || quant_after.q4_f32acc!=quant_before.q4_f32acc+1 ||
           quant_after.q4_f16acc!=quant_before.q4_f16acc || quant_after.invalid!=quant_before.invalid)
            throw std::runtime_error("MOSS Q4 candidate lacks one actual F32 accumulation selection");
        const uint64_t chosen=selected_after.selected_calls-selected_before.selected_calls;
        const uint64_t baseline=selected_after.baseline_calls-selected_before.baseline_calls;
        if(selected_after.schema!=1 || selected_after.enabled!=1 || selected_after.active!=active ||
           !selected_before.device_generation || selected_after.device_generation!=selected_before.device_generation ||
           selected_after.invalid_receipts!=selected_before.invalid_receipts || chosen!=active || baseline!=(1-active))
            throw std::runtime_error("MOSS Q4 candidate actual selected/baseline post-dispatch receipt mismatch");
        const auto &pipeline=selected_after.pipelines[active?1:3];
        const uint32_t expected_A[13]={256,64,64,32,16,32,2,2,2,1,32,1,unsigned(GGML_TYPE_Q4_K)};
        const uint32_t expected_baseline[13]={128,64,64,32,32,32,2,4,2,1,32,1,unsigned(GGML_TYPE_Q4_K)};
        const auto *expected_spec=active?expected_A:expected_baseline;
        const uint32_t baseline_warp=pipeline.specialization[10];
        if(!active && ((baseline_warp!=32 && baseline_warp!=64) || pipeline.specialization[4]!=baseline_warp))
            throw std::runtime_error("MOSS Q4 candidate baseline WM/WARP receipt differs from actual32/64 subgroup family");
        if(!pipeline.pipeline || pipeline.compile_count!=1 || pipeline.aligned!=1 ||
           !pipeline.original_module_bytes || !pipeline.actual_module_bytes ||
           pipeline.original_module_bytes%4 || pipeline.actual_module_bytes%4 ||
           (active && pipeline.required_subgroup!=32))
            throw std::runtime_error("MOSS Q4 candidate actual compiled aligned pipeline receipt incomplete");
        for(unsigned i=0;i<13;i++)if(pipeline.specialization[i]!=(!active && (i==4 || i==10)?baseline_warp:expected_spec[i]))
            throw std::runtime_error("MOSS Q4 candidate actual13 specialization constants differ");
        if(active && pipeline.original_module_bytes!=selected_after.pipelines[3].original_module_bytes)
            throw std::runtime_error("MOSS Q4 candidate must use the same F32 accumulation SPIR-V module as baseline");
        for(unsigned layer=0;layer<42;layer++) {
            const unsigned expected=layer==0;
            if(selected_after.layer_calls[layer]-selected_before.layer_calls[layer]!=expected*active ||
               selected_after.baseline_layer_calls[layer]-selected_before.baseline_layer_calls[layer]!=expected*(1-active))
                throw std::runtime_error("MOSS Q4 candidate canonical layer receipt mismatch");
        }
        for(unsigned token=0;token<4;token++) {
            const unsigned expected=token==shape;
            if(selected_after.token_calls[token]-selected_before.token_calls[token]!=expected*active ||
               selected_after.baseline_token_calls[token]-selected_before.baseline_token_calls[token]!=expected*(1-active))
                throw std::runtime_error("MOSS Q4 candidate actual input-token recording receipt mismatch");
        }
        ggml_backend_tensor_get(out,actual.data(),0,actual.size()*sizeof(float));
        ggml_backend_synchronize(r.backend);
        if(!ps5_ai_ggml_wait_backend("moss_q4_candidate_gate_readback",1))
            ps5_ai_ggml_retain("moss_q4_candidate_gate_readback_completion_unknown");
        const auto comparison=moss_precision::compare_fixture(actual,fixture,0.001,0.01);
        write_event({{"event","native_moss_q4_candidate_gate"},{"schema",1},{"case",test},
            {"shape_index",shape},{"shape_case",run},
            {"candidate_active",bool(active)},{"warmup",warmup},{"K",K},{"M",M},{"N",N},
            {"canonical_weight","model.diffusion_model.layers.0.proj.ffn_up_gate.weight"},
            {"same_graph_buffers_and_uploaded_inputs",true},{"input_upload_count",1},
            {"CPU_oracle","double dot of actual stored reference-quantized Q4_K weights and unchanged F32 RHS"},
            {"compared_values",fixture.expected.size()},{"sampled_output_indices",fixture.sampled_output_indices},
            {"finite_checked_values",actual.size()},{"nonfinite",comparison.invalid},
            {"mismatches",comparison.mismatches},{"max_absolute_error",comparison.maximum},
            {"absolute_tolerance",0.001},{"relative_tolerance",0.01},
            {"q4_matrix_recorded_f32acc",quant_after.q4_f32acc-quant_before.q4_f32acc},
            {"q4_matrix_recorded_f16acc",quant_after.q4_f16acc-quant_before.q4_f16acc},
            {"q4_matrix_accumulation_invalid",quant_after.invalid-quant_before.invalid},
            {"candidate_recording_delta",chosen},{"baseline_recording_delta",baseline},
            {"actual_pipeline_receipts",moss_q4_candidate_json(selected_after)},
            {"graph_and_wait_seconds",seconds},{"CPU_wall_not_GPU_timestamp",true},
            {"timing_scope","graph_compute+synchronize+existing checked wait; upload/readback/oracle/logging excluded"},
            {"dispatch_delta",after.dispatches-before.dispatches},{"submit_delta",after.submits-before.submits},
            {"fence_delta",after.fenced_completions-before.fenced_completions},
            {"CPU_fallback",false},{"model_inference",false}});
        if(comparison.invalid || comparison.mismatches || !ps5_ai_ggml_log_healthy())
            throw std::runtime_error("MOSS Q4 candidate numerical mismatch/nonfinite/durable receipt failure");
        if(!warmup) {
            if(active)shape_candidate_seconds=seconds;else shape_baseline_seconds=seconds;
        }
    }
    // Release only our known completed fixed-data buffers before the final
    // activation choice. The shared model/backend Vulkan device stays alive.
    r.clear();
    result.baseline_seconds+=shape_baseline_seconds;result.candidate_seconds+=shape_candidate_seconds;
    shape_results.push_back({{"N",N},{"baseline_loaded_graph_and_wait_seconds",shape_baseline_seconds},
        {"candidate_loaded_graph_and_wait_seconds",shape_candidate_seconds},
        {"candidate_to_baseline_time_ratio",shape_candidate_seconds/shape_baseline_seconds},
        {"finite_checked_values_per_run",output_values},{"compared_values_per_run",99}});
    }
    result.numerical_pass=true;
    result.microtiming_faster=result.candidate_seconds<result.baseline_seconds;
    try {
        if(!ps5_ai_moss_q4_candidate_set_enabled(result.microtiming_faster?1u:0u))
            throw std::runtime_error("MOSS Q4 candidate final activation requires completed idle hardware");
        result.candidate_enabled=result.microtiming_faster;
        ps5_ai_moss_q4_candidate_report final_receipt{};ps5_ai_moss_q4_candidate_snapshot(&final_receipt);
        if(final_receipt.active!=unsigned(result.candidate_enabled))
            throw std::runtime_error("MOSS Q4 candidate final activation receipt mismatch");
        write_event({{"event","native_moss_q4_candidate_gate_summary"},{"numerical_pass",true},
        {"shape_results",shape_results},{"gate_cases",8},{"token_shapes",{270,362}},
        {"baseline_loaded_graph_and_wait_seconds",result.baseline_seconds},
        {"candidate_loaded_graph_and_wait_seconds",result.candidate_seconds},
        {"candidate_to_baseline_time_ratio",result.candidate_seconds/result.baseline_seconds},
        {"microtiming_faster",result.microtiming_faster},{"candidate_enabled_after_gate",result.candidate_enabled},
        {"microtiming_reduction_percent",100.0*(1.0-result.candidate_seconds/result.baseline_seconds)},
        {"CPU_wall_not_GPU_timestamp",true},{"whole_generation_speedup_prediction",nullptr},
        {"performance_policy","enable only after all eight strict numerical/completion/receipt gates and lower summed warmed CPU-wall microtime across real token shapes270/362; root reviews before any image POST"}});
        if(!ps5_ai_ggml_log_healthy())throw std::runtime_error("MOSS Q4 candidate summary not durable");
        moss_q4_candidate_admission_passed=result.numerical_pass && result.microtiming_faster;
    } catch(...) {
        if(!ps5_ai_moss_q4_candidate_set_enabled(0))
            ps5_ai_ggml_retain("moss_q4_candidate_gate_final_failure_control_not_idle");
        throw;
    }
#endif
    return result;
}
