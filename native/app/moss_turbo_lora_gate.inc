/* SPDX-License-Identifier: GPL-3.0-or-later
 * Extra Turbo rank144 startup gate. Existing14 gates are retained separately.
 * Fixed synthetic matrices only; this does not validate actual LoRA file bytes.
 */
#include "moss_turbo_trial.h"
#if PS5_AI_MOSS_TURBO_TRIAL
#include "moss_turbo_lora_fixture.hpp"

namespace moss_turbo_lora_gate_state {
static std::atomic<bool> admitted{false};
static bool attempted=false;
}
static bool moss_turbo_lora_gate_admitted() noexcept {
    return moss_turbo_lora_gate_state::admitted.load(std::memory_order_acquire);
}

static void moss_turbo_lora_readback_gate() {
    using namespace moss_turbo_lora_fixture;
    if(moss_turbo_lora_gate_state::attempted||moss_turbo_lora_gate_admitted()||
       model_context||!hardware_verified||!gpu_device||!current_status.available)
        throw std::runtime_error("Turbo rank144 gate requires one verified deferred-model context");
    moss_turbo_lora_gate_state::attempted=true;
    struct Owned {
        ggml_backend_t backend=nullptr;
        ggml_context* context=nullptr;
        ggml_backend_buffer_t buffer=nullptr;
        void release() noexcept {
            if(!backend)return;
            try {ggml_backend_synchronize(backend);}
            catch(...) {ps5_ai_ggml_retain("moss_turbo_lora_cleanup_sync_unknown");}
            ps5_ai_ggml_dispatch_stats state{};ps5_ai_ggml_dispatch_snapshot(&state);
            if(state.work_pending||!ps5_ai_ggml_wait_backend("moss_turbo_lora_cleanup",0))
                ps5_ai_ggml_retain("moss_turbo_lora_cleanup_completion_unknown");
            if(buffer){ggml_backend_buffer_free(buffer);buffer=nullptr;}
            if(context){ggml_free(context);context=nullptr;}
            ggml_backend_free(backend);backend=nullptr;
        }
        ~Owned() noexcept {if(backend)release();}
    };
    for(const Kind kind:{Kind::qkv,Kind::ffn_up,Kind::ffn_down}) {
        // Own all CPU source/reference vectors; keep each fixture below16MiB
        // F32 payload. No CPU matrix evaluator is registered as a backend.
        const Fixture fixture=make(kind);
        ps5_ai_ggml_dispatch_stats entry{};ps5_ai_ggml_dispatch_snapshot(&entry);
        if(entry.work_pending||!ps5_ai_ggml_wait_backend("moss_turbo_lora_entry",0))
            ps5_ai_ggml_retain("moss_turbo_lora_entry_not_quiescent");
        ps5_ai_flex_stats pool_before{};ps5_ai_flex_snapshot(&pool_before);
        if(pool_before.fault_retained||!pool_before.log_healthy||!pool_before.mapping_owned||!pool_before.gpu_access_granted)
            throw std::runtime_error("Turbo rank144 gate requires a healthy owned GPU pool");
        Owned r;r.backend=ggml_backend_vk_init(0);
        if(!r.backend||!ggml_backend_is_vk(r.backend)||ggml_backend_get_device(r.backend)!=gpu_device)
            throw std::runtime_error("Turbo rank144 requires existing shared Vulkan hardware; CPU fallback forbidden");
        r.context=ggml_init({1<<20,nullptr,true});
        if(!r.context)throw std::runtime_error("Turbo rank144 metadata arena unavailable");
        auto* input=ggml_new_tensor_2d(r.context,GGML_TYPE_F32,fixture.k,tokens);
        ggml_set_name(input,"turbo_rank144_gate_F32_input");ggml_set_input(input);
        struct Tensors {ggml_tensor* down_weight;ggml_tensor* up_weight;ggml_tensor* down;ggml_tensor* up;};
        std::vector<Tensors> chain;chain.reserve(fixture.branches);
        for(int i=0;i<fixture.branches;++i) {
            auto* a=ggml_new_tensor_2d(r.context,GGML_TYPE_F32,fixture.k,rank);
            auto* b=ggml_new_tensor_2d(r.context,GGML_TYPE_F32,rank,fixture.m);
            auto* down=ggml_mul_mat(r.context,a,input);
            auto* up=ggml_mul_mat(r.context,b,down);
            if(!ggml_prec_set_acc(down,GGML_PREC_F32)||!ggml_prec_set_acc(up,GGML_PREC_F32))
                throw std::runtime_error("Turbo rank144 F32 matrix accumulation request failed");
            ggml_set_input(a);ggml_set_input(b);ggml_set_output(down);ggml_set_output(up);
            char name[96];std::snprintf(name,sizeof(name),"turbo_rank144_gate_branch%d_down_weight",i);ggml_set_name(a,name);
            std::snprintf(name,sizeof(name),"turbo_rank144_gate_branch%d_up_weight",i);ggml_set_name(b,name);
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
                throw std::runtime_error("Turbo rank144 graph unsupported; CPU fallback forbidden");
            if(node->op==GGML_OP_MUL_MAT) {
                if(node->type!=GGML_TYPE_F32||!node->src[0]||!node->src[1]||
                   node->src[0]->type!=GGML_TYPE_F32||node->src[1]->type!=GGML_TYPE_F32||
                   (ggml_prec)node->op_params[0]!=GGML_PREC_F32)
                    throw std::runtime_error("Turbo rank144 graph lost exact F32 matrix policy");
                ++forced_f32_nodes;
            } else if(node->op==GGML_OP_CONCAT)++concat_nodes;
            else throw std::runtime_error("Turbo rank144 fixture has an unexpected graph operation");
        }
        if(forced_f32_nodes!=unsigned(2*fixture.branches)||concat_nodes!=unsigned(fixture.branches==3?2:0)||
           output->ne[0]!=int64_t(fixture.m)*fixture.branches||output->ne[1]!=tokens||
           ggml_nelements(output)!=int64_t(fixture.m)*fixture.branches*tokens)
            throw std::runtime_error("Turbo rank144 graph topology or output shape mismatch");
        r.buffer=ggml_backend_alloc_ctx_tensors(r.context,r.backend);
        if(!r.buffer||ggml_backend_buffer_is_host(r.buffer)||
           ggml_backend_buft_get_device(ggml_backend_buffer_get_type(r.buffer))!=gpu_device)
            throw std::runtime_error("Turbo rank144 requires owned non-host GPU tensors");
        const uint64_t buffer_bytes=ggml_backend_buffer_get_size(r.buffer);
        ggml_backend_tensor_set(input,fixture.input.data(),0,fixture.input.size()*sizeof(float));
        for(int i=0;i<fixture.branches;++i) {
            const auto& source=fixture.chain[i];const auto& tensor=chain[i];
            if(ggml_nbytes(tensor.down_weight)!=source.down.size()*sizeof(float)||
               ggml_nbytes(tensor.up_weight)!=source.up.size()*sizeof(float))
                throw std::runtime_error("Turbo rank144 composed weight byte span mismatch");
            ggml_backend_tensor_set(tensor.down_weight,source.down.data(),0,source.down.size()*sizeof(float));
            ggml_backend_tensor_set(tensor.up_weight,source.up.data(),0,source.up.size()*sizeof(float));
        }
        ggml_backend_synchronize(r.backend);
        if(!ps5_ai_ggml_wait_backend("moss_turbo_lora_uploaded",0))
            ps5_ai_ggml_retain("moss_turbo_lora_upload_completion_unknown");
        ps5_ai_ggml_dispatch_stats before{},after{};ps5_ai_ggml_dispatch_snapshot(&before);
        if(before.work_pending)ps5_ai_ggml_retain("moss_turbo_lora_before_graph_pending");
        const auto started=std::chrono::steady_clock::now();
        const ggml_status status=ggml_backend_graph_compute(r.backend,graph);
        ggml_backend_synchronize(r.backend);
        if(status!=GGML_STATUS_SUCCESS||!ps5_ai_ggml_wait_backend("moss_turbo_lora_computed",1))
            ps5_ai_ggml_retain("moss_turbo_lora_compute_completion_unknown");
        const double graph_wait_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        ps5_ai_ggml_dispatch_snapshot(&after);
        if(after.work_pending)ps5_ai_ggml_retain("moss_turbo_lora_after_graph_pending");
        if(!after.hardware_verified||after.dispatches<=before.dispatches||after.submits<=before.submits||
           after.fenced_completions<=before.fenced_completions)
            throw std::runtime_error("Turbo rank144 lacks positive hardware dispatch/submit/fence proof");
        Comparison comparison;uint64_t readback_calls=0;
        auto checked_readback=[&](ggml_tensor* tensor,const std::vector<double>& expected) {
            if(ggml_nelements(tensor)!=int64_t(expected.size())||ggml_nbytes(tensor)!=expected.size()*sizeof(float))
                throw std::runtime_error("Turbo rank144 readback shape or byte mismatch");
            std::vector<float> actual(expected.size());
            ggml_backend_tensor_get(tensor,actual.data(),0,actual.size()*sizeof(float));
            ggml_backend_synchronize(r.backend);
            if(!ps5_ai_ggml_wait_backend("moss_turbo_lora_readback",1))
                ps5_ai_ggml_retain("moss_turbo_lora_readback_completion_unknown");
            ++readback_calls;add(comparison,compare(actual,expected));
        };
        for(int i=0;i<fixture.branches;++i) {
            checked_readback(chain[i].down,fixture.chain[i].expected_down);
            checked_readback(chain[i].up,fixture.chain[i].expected_up);
        }
        if(fixture.branches==3)checked_readback(output,fixture.expected_concat);
        if(comparison.nonfinite||comparison.mismatches)
            throw std::runtime_error("Turbo rank144 fixed numerical mismatch/nonfinite");
        r.release();
        ps5_ai_flex_stats pool_after{};ps5_ai_flex_snapshot(&pool_after);
        if(pool_after.fault_retained||!pool_after.log_healthy||pool_after.used_bytes!=pool_before.used_bytes||
           pool_after.allocations!=pool_before.allocations||pool_after.active_imports!=pool_before.active_imports)
            throw std::runtime_error("Turbo rank144 owned buffers were not safely released");
        write_event({{"event","native_moss_turbo_lora_readback"},{"success",true},{"case",fixture.name},
            {"synthetic_fixture",true},{"model_weights_loaded",false},{"actual_LoRA_file_bytes_verified",false},
            {"K",fixture.k},{"M_per_branch",fixture.m},{"N",tokens},{"branches",fixture.branches},
            {"combined_rank",rank},{"component_order",{"NSFW","Turbo"}},{"component_ranks",{nsfw_rank,turbo_rank}},
            {"synthetic_baked_up_scales",{nsfw_baked_scale,turbo_baked_scale}},{"runtime_global_scale",1.0},
            {"CPU_oracle","independent rank128 and rank16 FP64 products of exact stored F32 inputs; sum components; separate Q/K/V plus concatenation"},
            {"forced_F32_matmul_nodes",forced_f32_nodes},{"concat_nodes",concat_nodes},
            {"F32_pipeline_receipt_scope","actual supported F32 graph inputs/output and explicit F32 accumulation requests plus completed hardware math; no per-shader pipeline selection counter"},
            {"compared_values",comparison.compared},{"finite_checked_values",comparison.compared},
            {"nonfinite",comparison.nonfinite},{"mismatches",comparison.mismatches},{"max_absolute_error",comparison.max_absolute_error},
            {"absolute_tolerance",absolute_tolerance},{"relative_tolerance",relative_tolerance},
            {"readback_calls",readback_calls},{"fixture_owned_F32_payload_bytes",fixture.f32_payload_bytes()},
            {"CPU_FP64_oracle_payload_bytes",fixture.oracle_payload_bytes()},{"GPU_buffer_bytes",buffer_bytes},
            {"payload_scope","software buffer payload; not physical VRAM or whole pool admission"},
            {"graph_and_wait_CPU_wall_seconds",graph_wait_seconds},
            {"timing_scope","compute+synchronize+checked completion; excludes CPU oracle, upload and readback; not GPU timestamp"},
            {"dispatch_delta",after.dispatches-before.dispatches},{"submit_delta",after.submits-before.submits},
            {"fence_delta",after.fenced_completions-before.fenced_completions},
            {"pool_used_before",pool_before.used_bytes},{"pool_used_after",pool_after.used_bytes},
            {"pool_allocations_before",pool_before.allocations},{"pool_allocations_after",pool_after.allocations},
            {"active_imports_before",pool_before.active_imports},{"active_imports_after",pool_after.active_imports},
            {"CPU_fallback",false},{"automatic_retries",0},{"whole_pool_GPU_verified",false}});
        if(!ps5_ai_ggml_log_healthy())throw std::runtime_error("Turbo rank144 admission receipt is not durable");
    }
    moss_turbo_lora_gate_state::admitted.store(true,std::memory_order_release);
}
#endif
