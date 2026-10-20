/* SPDX-License-Identifier: GPL-3.0-or-later
 * Extra F16 storage experiment rank144 startup gate. Existing14 gates are retained separately.
 * Fixed synthetic matrices only; this does not validate actual LoRA file bytes.
 */
#include "moss_lora_f16_storage_trial.h"
#if PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL
#include "moss_lora_f16_storage_fixture.hpp"
#include "moss_lora_subnormal_fixture.hpp"

namespace moss_lora_f16_storage_gate_state {
static std::atomic<bool> admitted{false};
static bool attempted=false;
}
static bool moss_lora_f16_storage_gate_admitted() noexcept {
    return moss_lora_f16_storage_gate_state::admitted.load(std::memory_order_acquire);
}

static void moss_lora_f16_storage_readback_gate() {
    using namespace moss_lora_f16_storage_fixture;
    if(moss_lora_f16_storage_gate_state::attempted||moss_lora_f16_storage_gate_admitted()||
       model_context||!hardware_verified||!gpu_device||!current_status.available)
        throw std::runtime_error("F16 storage experiment rank144 gate requires one verified deferred-model context");
    moss_lora_f16_storage_gate_state::attempted=true;
    struct Owned {
        ggml_backend_t backend=nullptr;
        ggml_context* context=nullptr;
        ggml_backend_buffer_t buffer=nullptr;
        void release() noexcept {
            if(!backend)return;
            try {ggml_backend_synchronize(backend);}
            catch(...) {ps5_ai_ggml_retain("moss_lora_f16_storage_cleanup_sync_unknown");}
            ps5_ai_ggml_dispatch_stats state{};ps5_ai_ggml_dispatch_snapshot(&state);
            if(state.work_pending||!ps5_ai_ggml_wait_backend("moss_lora_f16_storage_cleanup",0))
                ps5_ai_ggml_retain("moss_lora_f16_storage_cleanup_completion_unknown");
            if(buffer){ggml_backend_buffer_free(buffer);buffer=nullptr;}
            if(context){ggml_free(context);context=nullptr;}
            ggml_backend_free(backend);backend=nullptr;
        }
        ~Owned() noexcept {if(backend)release();}
    };
    for(unsigned case_index=0;case_index<4;++case_index) {
        // Own all CPU source/reference vectors; keep each fixture below16MiB
        // F32 payload. No CPU matrix evaluator is registered as a backend.
        const Fixture fixture=case_index==3?moss_lora_subnormal_fixture::make():make(case_index==0?Kind::qkv:case_index==1?Kind::ffn_up:Kind::ffn_down);
        std::vector<float> baseline_readbacks;
        for(const bool half_storage:{false,true}) {
        ps5_ai_ggml_dispatch_stats entry{};ps5_ai_ggml_dispatch_snapshot(&entry);
        if(entry.work_pending||!ps5_ai_ggml_wait_backend("moss_lora_f16_storage_entry",0))
            ps5_ai_ggml_retain("moss_lora_f16_storage_entry_not_quiescent");
        ps5_ai_flex_stats pool_before{};ps5_ai_flex_snapshot(&pool_before);
        if(pool_before.fault_retained||!pool_before.log_healthy||!pool_before.mapping_owned||!pool_before.gpu_access_granted)
            throw std::runtime_error("F16 storage experiment rank144 gate requires a healthy owned GPU pool");
        Owned r;r.backend=ggml_backend_vk_init(0);
        if(!r.backend||!ggml_backend_is_vk(r.backend)||ggml_backend_get_device(r.backend)!=gpu_device)
            throw std::runtime_error("F16 storage experiment rank144 requires existing shared Vulkan hardware; CPU fallback forbidden");
        const int admission=ps5_ai_moss_lora_precision_backend_admission(r.backend);
        if((case_index==0&&!half_storage)||admission) {
            write_event({{"event","native_moss_lora_precision_backend_admission"},{"success",admission==0},
                {"refusal_code",admission},{"policy",PS5_AI_MOSS_LORA_PRECISION_POLICY},
                {"physical_caps_and_serial_queue_verified",admission==0},{"four_private_metadata_slots_initialized",admission==0},
                {"fixture_GPU_buffers_allocated",false},{"model_weights_loaded",false},{"CPU_fallback",false}});
            if(!ps5_ai_ggml_log_healthy())throw std::runtime_error("LoRA precision backend admission receipt is not durable");
        }
        if(admission)throw std::runtime_error("Private LoRA precision backend refused before GPU fixture allocation; see bounded refusal code");
        r.context=ggml_init({1<<20,nullptr,true});
        if(!r.context)throw std::runtime_error("F16 storage experiment rank144 metadata arena unavailable");
        auto* input=ggml_new_tensor_2d(r.context,GGML_TYPE_F32,fixture.k,tokens);
        ggml_set_name(input,"turbo_rank144_gate_F32_input");ggml_set_input(input);
        struct Tensors {ggml_tensor* down_weight;ggml_tensor* up_weight;ggml_tensor* down;ggml_tensor* up;};
        std::vector<Tensors> chain;chain.reserve(fixture.branches);
        for(int i=0;i<fixture.branches;++i) {
            auto* a=ggml_new_tensor_2d(r.context,half_storage?GGML_TYPE_F16:GGML_TYPE_F32,fixture.k,rank);
            auto* b=ggml_new_tensor_2d(r.context,half_storage?GGML_TYPE_F16:GGML_TYPE_F32,rank,fixture.m);
            auto* down=ggml_mul_mat(r.context,a,input);
            auto* up=ggml_mul_mat(r.context,b,down);
            if(!ggml_prec_set_acc(down,GGML_PREC_F32)||!ggml_prec_set_acc(up,GGML_PREC_F32))
                throw std::runtime_error("F16 storage experiment rank144 F32 matrix accumulation request failed");
            ggml_set_input(a);ggml_set_input(b);ggml_set_output(down);ggml_set_output(up);
            char name[96];std::snprintf(name,sizeof(name),"moss_f16_storage_fixture.branch%d.down",i);ggml_set_name(a,name);
            std::snprintf(name,sizeof(name),"moss_f16_storage_fixture.branch%d.up",i);ggml_set_name(b,name);
            chain.push_back({a,b,down,up});
        }
        ggml_tensor* output=chain[0].up;
        if(fixture.branches==3)output=ggml_concat(r.context,ggml_concat(r.context,chain[0].up,chain[1].up,0),chain[2].up,0);
        ggml_set_output(output);
        auto* graph=ggml_new_graph_custom(r.context,32,false);ggml_build_forward_expand(graph,output);
        unsigned forced_f32_nodes=0,concat_nodes=0;
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            const auto* node=ggml_graph_node(graph,i);
            if(!ggml_backend_supports_op(r.backend,node))
                throw std::runtime_error("F16 storage experiment rank144 graph unsupported; CPU fallback forbidden");
            if(node->op==GGML_OP_MUL_MAT) {
                if(node->type!=GGML_TYPE_F32||!node->src[0]||!node->src[1]||
                   node->src[0]->type!=(half_storage?GGML_TYPE_F16:GGML_TYPE_F32)||node->src[1]->type!=GGML_TYPE_F32||
                   (ggml_prec)node->op_params[0]!=GGML_PREC_F32)
                    throw std::runtime_error("F16 storage experiment rank144 graph lost exact F32 matrix policy");
                ++forced_f32_nodes;
            } else if(node->op==GGML_OP_CONCAT)++concat_nodes;
            else throw std::runtime_error("F16 storage experiment rank144 fixture has an unexpected graph operation");
        }
        if(forced_f32_nodes!=unsigned(2*fixture.branches)||concat_nodes!=unsigned(fixture.branches==3?2:0)||
           output->ne[0]!=int64_t(fixture.m)*fixture.branches||output->ne[1]!=tokens||
           ggml_nelements(output)!=int64_t(fixture.m)*fixture.branches*tokens)
            throw std::runtime_error("F16 storage experiment rank144 graph topology or output shape mismatch");
        r.buffer=ggml_backend_alloc_ctx_tensors(r.context,r.backend);
        if(!r.buffer||ggml_backend_buffer_is_host(r.buffer)||
           ggml_backend_buft_get_device(ggml_backend_buffer_get_type(r.buffer))!=gpu_device)
            throw std::runtime_error("F16 storage experiment rank144 requires owned non-host GPU tensors");
        const uint64_t buffer_bytes=ggml_backend_buffer_get_size(r.buffer);
        ggml_backend_tensor_set(input,fixture.input.data(),0,fixture.input.size()*sizeof(float));
        for(int i=0;i<fixture.branches;++i) {
            const auto& source=fixture.chain[i];const auto& tensor=chain[i];
            const size_t element_bytes=half_storage?sizeof(uint16_t):sizeof(float);
            if(ggml_nbytes(tensor.down_weight)!=source.down.size()*element_bytes||
               ggml_nbytes(tensor.up_weight)!=source.up.size()*element_bytes)
                throw std::runtime_error("F16 storage weight byte span mismatch");
            ggml_backend_tensor_set(tensor.down_weight,half_storage?static_cast<const void*>(source.down_half.data()):source.down.data(),0,source.down.size()*element_bytes);
            ggml_backend_tensor_set(tensor.up_weight,half_storage?static_cast<const void*>(source.up_half.data()):source.up.data(),0,source.up.size()*element_bytes);
        }
        ggml_backend_synchronize(r.backend);
        if(!ps5_ai_ggml_wait_backend("moss_lora_f16_storage_uploaded",0))
            ps5_ai_ggml_retain("moss_lora_f16_storage_upload_completion_unknown");
        ps5_ai_ggml_dispatch_stats before{},after{};ps5_ai_ggml_dispatch_snapshot(&before);
        if(before.work_pending)ps5_ai_ggml_retain("moss_lora_f16_storage_before_graph_pending");
        ps5_ai_moss_lora_storage_receipt receipt_before{},receipt_after{};
        ps5_ai_moss_lora_storage_snapshot(&receipt_before);
        const auto started=std::chrono::steady_clock::now();
        const ggml_status status=ggml_backend_graph_compute(r.backend,graph);
        ggml_backend_synchronize(r.backend);
        if(status!=GGML_STATUS_SUCCESS||!ps5_ai_ggml_wait_backend("moss_lora_f16_storage_computed",1))
            ps5_ai_ggml_retain("moss_lora_f16_storage_compute_completion_unknown");
        const double graph_wait_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        ps5_ai_ggml_dispatch_snapshot(&after);
        if(after.work_pending)ps5_ai_ggml_retain("moss_lora_f16_storage_after_graph_pending");
        if(!after.hardware_verified||after.dispatches<=before.dispatches||after.submits<=before.submits||
           after.fenced_completions<=before.fenced_completions)
            throw std::runtime_error("F16 storage experiment rank144 lacks positive hardware dispatch/submit/fence proof");
        ps5_ai_moss_lora_storage_snapshot(&receipt_after);
        const uint64_t expected_mm=uint64_t(2*fixture.branches);
        if(receipt_after.schema!=2||receipt_after.shared_f32!=1||receipt_after.storage_f16!=PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL||receipt_after.enabled!=1||receipt_after.invalid!=receipt_before.invalid||
           receipt_after.fixture_f32-receipt_before.fixture_f32!=(half_storage?0:expected_mm)||
           receipt_after.fixture_f16-receipt_before.fixture_f16!=(half_storage?expected_mm:0)||
           receipt_after.runtime_f16!=receipt_before.runtime_f16||receipt_after.runtime_f32!=receipt_before.runtime_f32||!receipt_after.last_pipeline)
            throw std::runtime_error("F16 storage actual selected pipeline/precision/geometry receipt mismatch");
        Comparison comparison,versus_baseline;uint64_t readback_calls=0;size_t baseline_offset=0;
        auto checked_readback=[&](ggml_tensor* tensor,const std::vector<double>& expected) {
            if(ggml_nelements(tensor)!=int64_t(expected.size())||ggml_nbytes(tensor)!=expected.size()*sizeof(float))
                throw std::runtime_error("F16 storage experiment rank144 readback shape or byte mismatch");
            std::vector<float> actual(expected.size());
            ggml_backend_tensor_get(tensor,actual.data(),0,actual.size()*sizeof(float));
            ggml_backend_synchronize(r.backend);
            if(!ps5_ai_ggml_wait_backend("moss_lora_f16_storage_readback",1))
                ps5_ai_ggml_retain("moss_lora_f16_storage_readback_completion_unknown");
            ++readback_calls;add(comparison,compare(actual,expected));
            if(!half_storage)baseline_readbacks.insert(baseline_readbacks.end(),actual.begin(),actual.end());
            else {
                if(actual.size()>baseline_readbacks.size()-baseline_offset)
                    throw std::runtime_error("F16 storage F32 baseline readback boundary mismatch");
                std::vector<double> previous(actual.size());
                for(size_t i=0;i<actual.size();++i)previous[i]=baseline_readbacks[baseline_offset+i];
                add(versus_baseline,compare(actual,previous));baseline_offset+=actual.size();
            }
        };
        for(int i=0;i<fixture.branches;++i) {
            checked_readback(chain[i].down,(half_storage?fixture.chain[i].rounded_down:fixture.chain[i].baseline_down));
            checked_readback(chain[i].up,(half_storage?fixture.chain[i].rounded_up:fixture.chain[i].baseline_up));
        }
        if(fixture.branches==3)checked_readback(output,(half_storage?fixture.rounded_concat:fixture.baseline_concat));
        const bool numerical_pass=!(comparison.nonfinite||comparison.mismatches||versus_baseline.nonfinite||versus_baseline.mismatches||
           (half_storage&&baseline_offset!=baseline_readbacks.size()));
        r.release();
        ps5_ai_flex_stats pool_after{};ps5_ai_flex_snapshot(&pool_after);
        if(pool_after.fault_retained||!pool_after.log_healthy||pool_after.used_bytes!=pool_before.used_bytes||
           pool_after.allocations!=pool_before.allocations||pool_after.active_imports!=pool_before.active_imports)
            throw std::runtime_error("F16 storage experiment rank144 owned buffers were not safely released");
        write_event({{"event","native_moss_lora_precision_readback"},{"success",numerical_pass},{"case",fixture.name},
            {"variant",half_storage?"F16_storage_FP32_shared":"F32_storage_FP32_shared"},{"weight_dtype",half_storage?"F16":"F32"},
            {"synthetic_fixture",true},{"model_weights_loaded",false},{"actual_LoRA_file_bytes_verified",false},
            {"K",fixture.k},{"M_per_branch",fixture.m},{"N",tokens},{"branches",fixture.branches},{"rank",rank},
            {"non_dyadic_RHS",case_index!=3},{"separate_subnormal_probe",case_index==3},{"original_non_dyadic_fixture_unchanged",true},{"CPU_oracle",half_storage?"FP64 products of actual RNE-rounded F16 weights with original non-dyadic F32 RHS":"FP64 products of original F32 synthetic weights and non-dyadic F32 RHS"},
            {"RHS_storage","F32"},{"accumulation_request","F32"},{"selected_F16_accumulator",false},
            {"shader_RHS_load_precision","existing FLOAT_TYPE=F32 private module; no F16 shared RHS conversion"},{"private_pipeline_policies_checked",true},
            {"actual_matrix_pipeline_recordings",expected_mm},{"last_pipeline_handle",receipt_after.last_pipeline},
            {"receipt_invalid_delta",receipt_after.invalid-receipt_before.invalid},
            {"compared_values",comparison.compared},{"finite_checked_values",comparison.compared},
            {"nonfinite",comparison.nonfinite},{"mismatches",comparison.mismatches},{"max_absolute_error",comparison.max_absolute_error},
            {"versus_F32_baseline_values",versus_baseline.compared},{"versus_F32_baseline_nonfinite",versus_baseline.nonfinite},
            {"versus_F32_baseline_mismatches",versus_baseline.mismatches},{"versus_F32_baseline_max_absolute_error",versus_baseline.max_absolute_error},
            {"absolute_tolerance",absolute_tolerance},{"relative_tolerance",relative_tolerance},
            {"readback_calls",readback_calls},{"fixture_host_payload_bytes",fixture.host_payload_bytes()},{"GPU_buffer_bytes",buffer_bytes},
            {"payload_scope","software tensor payload; not physical VRAM or whole-pool savings"},
            {"graph_and_wait_CPU_wall_seconds",graph_wait_seconds},{"timing_scope","compute plus existing synchronization; not GPU timestamp"},
            {"dispatch_delta",after.dispatches-before.dispatches},{"submit_delta",after.submits-before.submits},{"fence_delta",after.fenced_completions-before.fenced_completions},
            {"pool_used_before",pool_before.used_bytes},{"pool_used_after",pool_after.used_bytes},
            {"pool_allocations_before",pool_before.allocations},{"pool_allocations_after",pool_after.allocations},
            {"active_imports_before",pool_before.active_imports},{"active_imports_after",pool_after.active_imports},
            {"CPU_fallback",false},{"automatic_retries",0},{"image_quality_parity",false}});
        if(!ps5_ai_ggml_log_healthy())throw std::runtime_error("F16 storage experiment rank144 admission receipt is not durable");
        if(!numerical_pass)throw std::runtime_error("F16 storage rejected by unchanged numerical tolerance; no model loading permitted");
    }
    } // baseline and candidate
    ps5_ai_moss_lora_storage_receipt complete{};ps5_ai_moss_lora_storage_snapshot(&complete);
    if(!moss_lora_storage_pipelines_valid(complete)||complete.fixture_f32!=16||complete.fixture_f16!=16||
       complete.runtime_f32||complete.runtime_f16)
        throw std::runtime_error("Private LoRA precision experiment requires all8 fixed readbacks and four compiled modules");
    write_event({{"event","native_moss_lora_precision_gates_complete"},{"success",true},
        {"policy",PS5_AI_MOSS_LORA_PRECISION_POLICY},{"readback_events",8},
        {"F32_weight_private_matrix_recordings",complete.fixture_f32},{"F16_weight_private_matrix_recordings",complete.fixture_f16},
        {"fixed_non_dyadic_cases",3},{"separate_subnormal_cases",1},{"absolute_tolerance",absolute_tolerance},
        {"relative_tolerance",relative_tolerance},{"invalid",complete.invalid},
        {"pipelines",moss_lora_precision_pipelines_json(complete)},{"actual_LoRA_file_bytes_verified",false},
        {"physical_memory_savings_asserted",false},{"image_quality_parity",false}});
    if(!ps5_ai_ggml_log_healthy())throw std::runtime_error("Private LoRA precision final admission receipt is not durable");
    moss_lora_f16_storage_gate_state::admitted.store(true,std::memory_order_release);
}
#endif
