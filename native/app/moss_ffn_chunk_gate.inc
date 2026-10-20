/* SPDX-License-Identifier: GPL-3.0-or-later
 * Additional complete-FFN numerical gate. Does not replace old14+Turbo3.
 * No model parameters are read. Unknown GPU completion retains ownership.
 */
#include "moss_ffn_chunk_trial.h"
#if PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL
#include "moss_ffn_chunk_fixture.hpp"
#include "core/ggml_extend.h"
namespace moss_ffn_chunk_gate_state {static std::atomic<bool> admitted{false};static bool attempted=false;}
static bool moss_ffn_chunk_gate_admitted(){return moss_ffn_chunk_gate_state::admitted.load(std::memory_order_acquire);}
static void moss_ffn_chunk_readback_gate() {
    using namespace moss_ffn_chunk_fixture;
    if(moss_ffn_chunk_gate_state::attempted||model_context||!hardware_verified||!gpu_device||!moss_turbo_lora_gate_admitted())
        throw std::runtime_error("FFN gate requires completed old14+Turbo3, deferred model, one attempt");
    moss_ffn_chunk_gate_state::attempted=true;
    constexpr int tokens=2057; // 2048+9 tail; actual3072/8192/16384 and rank144.
    const Fixture f(tokens);
    std::vector<float> baseline;
    struct Owned {
        ggml_backend_t backend=nullptr;ggml_context* ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;
        void release()noexcept {
            if(!backend)return;
            try{ggml_backend_synchronize(backend);}catch(...){ps5_ai_ggml_retain("FFN_cleanup_sync_unknown");}
            ps5_ai_ggml_dispatch_stats s{};ps5_ai_ggml_dispatch_snapshot(&s);
            if(s.work_pending||!ps5_ai_ggml_wait_backend("FFN_cleanup",0))ps5_ai_ggml_retain("FFN_cleanup_pending");
            if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);
            ggml_backend_free(backend);backend=nullptr;buffer=nullptr;ctx=nullptr;
        }
        ~Owned()noexcept{if(backend)release();}
    };
    for(const int size:{0,int(moss_ffn_token_chunk::configured_tokens)}) {
        ps5_ai_ggml_dispatch_stats entry{};ps5_ai_ggml_dispatch_snapshot(&entry);
        if(entry.work_pending||!ps5_ai_ggml_wait_backend("FFN_entry",0))ps5_ai_ggml_retain("FFN_entry_pending");
        ps5_ai_flex_stats pool_before{};ps5_ai_flex_snapshot(&pool_before);
        if(pool_before.fault_retained||!pool_before.log_healthy||!pool_before.mapping_owned||!pool_before.gpu_access_granted)
            throw std::runtime_error("FFN fixture GPU pool unhealthy");
        Owned r;r.backend=ggml_backend_vk_init(0);
        if(!r.backend||!ggml_backend_is_vk(r.backend)||ggml_backend_get_device(r.backend)!=gpu_device)
            throw std::runtime_error("FFN fixture requires the same actual Vulkan GPU");
        r.ctx=ggml_init({4<<20,nullptr,true});if(!r.ctx)throw std::runtime_error("FFN graph metadata allocation");
        auto tensor=[&](ggml_type type,int64_t k,int64_t m,const char* name){auto*t=ggml_new_tensor_2d(r.ctx,type,k,m);ggml_set_name(t,name);ggml_set_input(t);return t;};
        auto*x=tensor(GGML_TYPE_F32,K,tokens,"FFN_fixture_input");
        auto*wu=tensor(GGML_TYPE_Q4_K,K,U,"model.diffusion_model.layers.0.proj.ffn_up_gate.weight");
        auto*wd=tensor(GGML_TYPE_Q4_K,F,K,"model.diffusion_model.layers.0.proj.ffn_down.weight");
        auto*au=tensor(GGML_TYPE_F32,K,R,"lora.model.diffusion_model.layers.0.proj.ffn_up_gate.weight.lora_down");
        auto*bu=tensor(GGML_TYPE_F32,R,U,"lora.model.diffusion_model.layers.0.proj.ffn_up_gate.weight.lora_up");
        auto*ad=tensor(GGML_TYPE_F32,F,R,"lora.model.diffusion_model.layers.0.proj.ffn_down.weight.lora_down");
        auto*bd=tensor(GGML_TYPE_F32,R,K,"lora.model.diffusion_model.layers.0.proj.ffn_down.weight.lora_up");
        auto mm=[&](ggml_tensor*a,ggml_tensor*b){auto*t=ggml_mul_mat(r.ctx,a,b);if(!ggml_prec_set_acc(t,GGML_PREC_F32))throw std::runtime_error("FFN fixture F32 accumulation refused");return t;};
        auto linear=[&](ggml_tensor*w,ggml_tensor*a,ggml_tensor*b,ggml_tensor*in){return ggml_add(r.ctx,mm(w,in),mm(b,mm(a,in)));};
        auto full=[&](ggml_tensor*in){
            auto*ug=linear(wu,au,bu,in);
            auto*hidden=ggml_ext_slice(r.ctx,ug,0,0,F);
            auto*gate=ggml_ext_slice(r.ctx,ug,0,F,2*F);
            // Same operations/constants as MOSSImage::situ, including both
            // saturations, not generic SiLU or an approximated activation.
            auto*a=ggml_scale(r.ctx,ggml_tanh(r.ctx,ggml_scale(r.ctx,gate,.25f)),4.f);
            a=ggml_mul(r.ctx,a,ggml_sigmoid(r.ctx,gate));
            auto*b=ggml_scale(r.ctx,ggml_tanh(r.ctx,ggml_scale(r.ctx,hidden,1.f/25.f)),25.f);
            return linear(wd,ad,bd,ggml_mul(r.ctx,a,b));
        };
        auto*out=moss_ffn_token_chunk::apply(r.ctx,x,size,full);ggml_set_output(out);
        auto*g=ggml_new_graph_custom(r.ctx,256,false);ggml_build_forward_expand(g,out);
        const uint64_t pieces=moss_ffn_token_chunk::pieces(tokens,size);
        uint64_t mm_nodes=0,q4_nodes=0;
        for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);
            if(!ggml_backend_supports_op(r.backend,t))throw std::runtime_error("CompleteFFN op unsupported; CPU fallback forbidden");
            if(t->op==GGML_OP_MUL_MAT){++mm_nodes;q4_nodes+=t->src[0]->type==GGML_TYPE_Q4_K;
                if(t->type!=GGML_TYPE_F32||t->src[1]->type!=GGML_TYPE_F32||(ggml_prec)t->op_params[0]!=GGML_PREC_F32)
                    throw std::runtime_error("CompleteFFN graph precision changed");}
        }
        if(mm_nodes!=6*pieces||q4_nodes!=2*pieces||out->ne[0]!=K||out->ne[1]!=tokens)
            throw std::runtime_error("CompleteFFN graph topology/capacity differs");
        r.buffer=ggml_backend_alloc_ctx_tensors(r.ctx,r.backend);
        if(!r.buffer||ggml_backend_buffer_is_host(r.buffer)||ggml_backend_buft_get_device(ggml_backend_buffer_get_type(r.buffer))!=gpu_device)
            throw std::runtime_error("CompleteFFN nonhost GPU buffer allocation failed");
        auto upload=[&](ggml_tensor*t,const auto&v){if(ggml_nbytes(t)!=v.size()*sizeof(v[0]))throw std::runtime_error("FFN upload exact span");ggml_backend_tensor_set(t,v.data(),0,ggml_nbytes(t));};
        upload(x,f.input);upload(wu,f.q_up);upload(wd,f.q_down);upload(au,f.a_up);upload(bu,f.b_up);upload(ad,f.a_down);upload(bd,f.b_down);
        ggml_backend_synchronize(r.backend);if(!ps5_ai_ggml_wait_backend("FFN_uploaded",0))ps5_ai_ggml_retain("FFN_upload_unknown");
        ps5_ai_ggml_dispatch_stats before{},after{};ps5_ai_ggml_dispatch_snapshot(&before);
        ps5_ai_moss_quant_receipt qb{},qa{};ps5_ai_moss_quant_snapshot(&qb);
        const auto start=std::chrono::steady_clock::now();const auto status=ggml_backend_graph_compute(r.backend,g);
        ggml_backend_synchronize(r.backend);
        if(status!=GGML_STATUS_SUCCESS||!ps5_ai_ggml_wait_backend("FFN_completed",1))ps5_ai_ggml_retain("FFN_compute_unknown");
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        ps5_ai_ggml_dispatch_snapshot(&after);ps5_ai_moss_quant_snapshot(&qa);
        if(after.work_pending)ps5_ai_ggml_retain("FFN_post_compute_pending");
        if(!after.hardware_verified||after.dispatches<=before.dispatches||after.submits<=before.submits||after.fenced_completions<=before.fenced_completions||
           qa.q4_f32acc-qb.q4_f32acc!=2*pieces||qa.q4_f16acc!=qb.q4_f16acc||qa.invalid!=qb.invalid) {
            write_event({{"event","native_moss_FFN_chunk_readback"},{"success",false},{"failure_stage","GPU_counter_or_hardware_receipt"},
                {"synthetic_fixture",true},{"model_weights_loaded",false},{"N",tokens},{"chunk_tokens",size},{"pieces",pieces},
                {"readback_performed",false},{"hardware_verified",after.hardware_verified},{"work_pending",bool(after.work_pending)},
                {"expected_Q4_F32_recordings",2*pieces},{"Q4_F32_before",qb.q4_f32acc},{"Q4_F32_after",qa.q4_f32acc},
                {"Q4_F16_before",qb.q4_f16acc},{"Q4_F16_after",qa.q4_f16acc},{"invalid_before",qb.invalid},{"invalid_after",qa.invalid},
                {"dispatch_before",before.dispatches},{"dispatch_after",after.dispatches},{"submit_before",before.submits},{"submit_after",after.submits},
                {"fence_before",before.fenced_completions},{"fence_after",after.fenced_completions},{"automatic_retry",false}});
            throw std::runtime_error("CompleteFFN hardware/F32 counter receipt failed");
        }
        std::vector<float> actual(size_t(K)*tokens);
        ggml_backend_tensor_get(out,actual.data(),0,actual.size()*sizeof(float));ggml_backend_synchronize(r.backend);
        if(!ps5_ai_ggml_wait_backend("FFN_readback",1))ps5_ai_ggml_retain("FFN_readback_unknown");
        const auto oracle=compare(actual,[&](size_t i){return f.oracle(i);});
        Comparison pair{};
        if(size)pair=compare(actual,[&](size_t i){return double(baseline[i]);});
        else baseline=actual;
        if(oracle.nonfinite||oracle.mismatches||pair.nonfinite||pair.mismatches) {
            write_event({{"event","native_moss_FFN_chunk_readback"},{"success",false},{"failure_stage","completed_numeric_readback"},
                {"synthetic_fixture",true},{"model_weights_loaded",false},{"N",tokens},{"K",K},{"FFN_hidden",F},{"rank",R},
                {"chunk_tokens",size},{"pieces",pieces},{"oracle_compared_values",oracle.compared},{"oracle_nonfinite",oracle.nonfinite},
                {"oracle_mismatches",oracle.mismatches},{"max_oracle_absolute_error",oracle.max_absolute_error},
                {"pair_compared_values",pair.compared},{"pair_nonfinite",pair.nonfinite},{"pair_mismatches",pair.mismatches},
                {"max_pair_absolute_error",pair.max_absolute_error},{"max_errors_scope","finite value pairs only; nonfinite counted separately"},
                {"absolute_tolerance",absolute_tolerance},{"relative_tolerance",relative_tolerance},{"readback_performed",true},
                {"Q4_F32_recordings",qa.q4_f32acc-qb.q4_f32acc},{"Q4_F16_recordings",qa.q4_f16acc-qb.q4_f16acc},
                {"invalid",qa.invalid-qb.invalid},{"automatic_retry",false}});
            throw std::runtime_error("CompleteFFN unchunked/oracle or chunked/oracle/pair numeric failure");
        }
        const uint64_t buffer_bytes=ggml_backend_buffer_get_size(r.buffer);r.release();
        ps5_ai_flex_stats pool_after{};ps5_ai_flex_snapshot(&pool_after);
        if(pool_after.fault_retained||!pool_after.log_healthy||pool_before.used_bytes!=pool_after.used_bytes||pool_before.allocations!=pool_after.allocations||pool_before.active_imports!=pool_after.active_imports)
            throw std::runtime_error("CompleteFFN GPU ownership not restored");
        write_event({{"event","native_moss_FFN_chunk_readback"},{"success",true},{"synthetic_fixture",true},{"model_weights_loaded",false},
            {"N",tokens},{"K",K},{"FFN_hidden",F},{"rank",R},{"runtime_LoRA_scale",1.0},{"NSFW_baked_scale",.7},{"Turbo_baked_scale",1.0},
            {"chunk_tokens",size},{"pieces",pieces},{"matrix_nodes",mm_nodes},{"Q4_F32_recordings",qa.q4_f32acc-qb.q4_f32acc},{"F16_recordings",0},{"invalid",0},
            {"CPU_oracle","FP64 dots of actual Q4K reference-decoded base and exact F32 component coefficients, exact MOSS situ; repeated-row synthetic classes only"},
            {"oracle_compared_values",oracle.compared},{"oracle_mismatches",oracle.mismatches},{"pair_compared_values",pair.compared},{"pair_mismatches",pair.mismatches},
            {"nonfinite",oracle.nonfinite+pair.nonfinite},{"max_oracle_absolute_error",oracle.max_absolute_error},{"max_pair_absolute_error",pair.max_absolute_error},
            {"absolute_tolerance",absolute_tolerance},{"relative_tolerance",relative_tolerance},{"graph_wait_CPU_seconds",seconds},{"pure_GPU_timestamp",false},
            {"GPU_buffer_bytes",buffer_bytes},{"buffer_scope","gate allocation payload, not whole-graph runtime workspace or physical VRAM"},
            {"dispatch_delta",after.dispatches-before.dispatches},{"submit_delta",after.submits-before.submits},{"fence_delta",after.fenced_completions-before.fenced_completions},
            {"CPU_fallback",false},{"automatic_retry",false},{"pool_restored",true}});
        if(!ps5_ai_ggml_log_healthy())throw std::runtime_error("CompleteFFN gate receipt not durable");
    }
    moss_ffn_chunk_gate_state::admitted.store(true,std::memory_order_release);
}
#endif
