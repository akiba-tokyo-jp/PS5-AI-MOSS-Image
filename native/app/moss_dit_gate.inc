/* SPDX-License-Identifier: MIT
 * Synthetic F32 original-graph vs fused Vulkan readback. No learned host inference.
 * Old14+Turbo3/upload gates remain mandatory. Unknown completion retains GPU ownership.
 */
#include "model/diffusion/moss_dit_ops.hpp"
#ifndef PS5_AI_MOSS_DIT_OPT_MASK
#define PS5_AI_MOSS_DIT_OPT_MASK 0
#endif
#if PS5_AI_MOSS_DIT_OPT_TRIAL
extern "C" unsigned ps5_ai_moss_dit_capabilities();
static bool moss_dit_gate_passed=false;
static void moss_dit_readback_gate(){
    if(moss_dit_gate_passed||model_context||!hardware_verified||!moss_turbo_lora_gate_admitted()||!ps5_ai_moss_bounded_weight_upload_enabled()||ps5_ai_moss_dit_capabilities()!=15)
        throw std::runtime_error("DiT gate requires deferred model and existing completed baseline gates");
    struct Owned{
        ggml_backend_t backend=nullptr;ggml_context* ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;std::vector<ggml_backend_buffer_t> external_buffers;
        void release()noexcept{
            if(!backend)return;
            try{ggml_backend_synchronize(backend);}catch(...){ps5_ai_ggml_retain("dit_gate_cleanup_unknown");}
            if(!ps5_ai_ggml_wait_backend("dit_gate_cleanup",0))ps5_ai_ggml_retain("dit_gate_cleanup_pending");
            if(buffer)ggml_backend_buffer_free(buffer);for(auto b:external_buffers)ggml_backend_buffer_free(b);external_buffers.clear();if(ctx)ggml_free(ctx);ggml_backend_free(backend);backend=nullptr;
        }
        ~Owned()noexcept{if(backend)release();}
    };
    for(unsigned kind=1;kind<=8;++kind)for(unsigned strided=0;strided<=1;++strided){
        if(!ps5_ai_ggml_wait_backend("dit_gate_entry",0))ps5_ai_ggml_retain("dit_gate_entry_pending");
        ps5_ai_flex_stats pool_before{};ps5_ai_flex_snapshot(&pool_before);
        Owned r;r.backend=ggml_backend_vk_init(0);r.ctx=ggml_init({4<<20,nullptr,true});
        if(!r.backend||!r.ctx||!ggml_backend_is_vk(r.backend)||ggml_backend_get_device(r.backend)!=gpu_device)throw std::runtime_error("DiT gate existing non-CPU Vulkan backend required");
        struct Host{ggml_tensor* root;std::vector<float> data;};std::vector<Host> host;
        auto input=[&](int64_t d,int64_t n,int64_t z=1,int64_t w=1,bool view=true){
            const int pad=strided&&view?4:0;
            auto* base=ggml_new_tensor_4d(r.ctx,GGML_TYPE_F32,d+pad,n,z,w);ggml_set_input(base);
            Host h{base,std::vector<float>(ggml_nelements(base))};const unsigned salt=host.size()+1;
            for(size_t i=0;i<h.data.size();++i)h.data[i]=float(int((i*37+salt*17)%127)-63)/19.f;
            host.push_back(std::move(h));
            return pad?ggml_view_4d(r.ctx,base,d,n,z,w,base->nb[1],base->nb[2],base->nb[3],4):base;
        };
        const int d=kind==7?12:kind==8?18:kind==1?16384:128;
        auto*a=kind==5?input(128,2,17):input(d,kind==1?3:17);
        ggml_tensor*b=nullptr,*c=nullptr,*e=nullptr;
        if(kind==1||kind==6){b=input(d,a->ne[1]);c=b;}
        else if(kind==2||kind==3){b=input(d,1);c=input(6*d,1);}
        else if(kind==4){b=input(d,a->ne[1]);c=input(d,1);}
        else if(kind==5){b=input(128,1);c=input(2,2,64,17,false);}
        else{b=input(5,a->ne[1]);c=input(7,a->ne[1]);if(kind==8)e=input(6,a->ne[1]);}
        auto*out=moss_dit_opt::make(r.ctx,kind,a,b,c,e,r.backend);ggml_set_output(out);
        // The unfused SCALE/unary backend requires contiguous inputs. Keep
        // the fused path on the original strided/misaligned views; materialize
        // only the independent reference, exactly as the original model's
        // ggml_ext_slice/scale boundary does. This is a GPU CONT, never CPU math.
        a=ggml_cont(r.ctx,a);b=ggml_cont(r.ctx,b);c=ggml_cont(r.ctx,c);if(e)e=ggml_cont(r.ctx,e);
        ggml_tensor*ref=nullptr;
        if(kind==1){
            auto* x=ggml_scale(r.ctx,ggml_tanh(r.ctx,ggml_scale(r.ctx,b,.25f)),4.f);
            x=ggml_mul(r.ctx,x,ggml_sigmoid(r.ctx,b));
            ref=ggml_mul(r.ctx,x,ggml_scale(r.ctx,ggml_tanh(r.ctx,ggml_scale(r.ctx,a,1.f/25.f)),25.f));
        }else if(kind==2||kind==3){
            unsigned off=kind==2?0:3;auto part=[&](unsigned j){return ggml_view_1d(r.ctx,c,d,j*d*4);};
            auto* y=ggml_mul(r.ctx,a,b);ref=ggml_add(r.ctx,ggml_add(r.ctx,y,ggml_mul(r.ctx,y,part(off+1))),part(off));
        }else if(kind==4)ref=ggml_add(r.ctx,a,ggml_mul(r.ctx,b,c));
        else if(kind==5)ref=ggml_rope_apply(r.ctx,ggml_mul(r.ctx,a,b),c,false);
        else if(kind==6)ref=ggml_add(r.ctx,a,ggml_scale(r.ctx,b,1.f));
        else{auto*parts=ggml_concat(r.ctx,b,c,0);if(e)parts=ggml_concat(r.ctx,parts,e,0);ref=ggml_add(r.ctx,a,ggml_scale(r.ctx,parts,1.f));}
        ggml_set_output(ref);auto*g=ggml_new_graph_custom(r.ctx,128,false);ggml_build_forward_expand(g,ref);ggml_build_forward_expand(g,out);
        for(int i=0;i<ggml_graph_n_nodes(g);++i){
            const auto* t=ggml_graph_node(g,i);
            if(!ggml_backend_supports_op(r.backend,t)){
                write_event({{"event","native_moss_dit_gate_unsupported"},{"kind",kind},{"strided",strided!=0},{"op",ggml_op_name(t->op)},{"shape",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},{"CPU_fallback",false}});
                throw std::runtime_error("DiT gate operator unsupported; CPU fallback forbidden");
            }
        }
        // Cached-input fixture: distinct VkBuffers share the GGML virtual base.
        const bool separate_buffers=kind==5&&strided==0;
        if(separate_buffers)for(auto&h:host){
            auto* buf=ggml_backend_alloc_buffer(r.backend,ggml_nbytes(h.root));
            if(!buf)throw std::runtime_error("DiT separate-buffer gate allocation failed");
            r.external_buffers.push_back(buf);
            if(ggml_backend_tensor_alloc(buf,h.root,ggml_backend_buffer_get_base(buf))!=GGML_STATUS_SUCCESS)throw std::runtime_error("DiT separate-buffer gate bind failed");
        }
        r.buffer=ggml_backend_alloc_ctx_tensors(r.ctx,r.backend);
        if(!r.buffer||ggml_backend_buffer_is_host(r.buffer))throw std::runtime_error("DiT gate non-host allocation required");
        if(separate_buffers)for(auto&h:host)if(h.root->buffer==out->buffer||h.root->data!=out->data)throw std::runtime_error("DiT separate-buffer equal-sentinel fixture was not exercised");
        for(auto&h:host)ggml_backend_tensor_set(h.root,h.data.data(),0,h.data.size()*4);
        ggml_backend_synchronize(r.backend);if(!ps5_ai_ggml_wait_backend("dit_gate_upload",0))ps5_ai_ggml_retain("dit_gate_upload_unknown");
        ps5_ai_ggml_dispatch_stats before{},after{};ps5_ai_ggml_dispatch_snapshot(&before);
        auto records=moss_dit_opt::GPU_recordings[kind].load();
        const auto status=ggml_backend_graph_compute(r.backend,g);ggml_backend_synchronize(r.backend);
        if(status!=GGML_STATUS_SUCCESS||!ps5_ai_ggml_wait_backend("dit_gate_compute",1))ps5_ai_ggml_retain("dit_gate_compute_unknown");
        ps5_ai_ggml_dispatch_snapshot(&after);
        if(after.work_pending||!after.hardware_verified||after.dispatches<=before.dispatches||after.submits<=before.submits||after.fenced_completions<=before.fenced_completions||moss_dit_opt::GPU_recordings[kind].load()!=records+1)throw std::runtime_error("DiT gate actual dispatch/fence receipt missing");
        std::vector<float> actual(ggml_nelements(out)),original(actual.size());
        ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);ggml_backend_tensor_get(ref,original.data(),0,original.size()*4);ggml_backend_synchronize(r.backend);
        if(!ps5_ai_ggml_wait_backend("dit_gate_readback",1))ps5_ai_ggml_retain("dit_gate_readback_unknown");
        uint64_t nonfinite=0,mismatches=0;double max_error=0;
        for(size_t i=0;i<actual.size();++i){
            nonfinite+=!std::isfinite(actual[i])||!std::isfinite(original[i]);
            double error=std::abs(double(actual[i])-original[i]);max_error=std::max(max_error,error);
            mismatches+=error>2e-5+2e-5*std::abs(double(original[i]));
        }
        r.release();ps5_ai_flex_stats pool_after{};ps5_ai_flex_snapshot(&pool_after);
        bool released=pool_before.used_bytes==pool_after.used_bytes&&pool_before.allocations==pool_after.allocations&&pool_before.active_imports==pool_after.active_imports&&!pool_after.fault_retained&&pool_after.log_healthy;
        write_event({{"event","native_moss_dit_fused_readback"},{"kind",kind},{"strided_misaligned",strided!=0},{"distinct_buffers_equal_virtual_base",separate_buffers},{"compared",actual.size()},{"nonfinite",nonfinite},{"mismatches",mismatches},{"max_absolute_error",max_error},{"absolute_tolerance",2e-5},{"relative_tolerance",2e-5},{"dispatch_delta",after.dispatches-before.dispatches},{"fence_delta",after.fenced_completions-before.fenced_completions},{"owned_buffers_released",released},{"model_loaded",false},{"CPU_inference",false},{"success",!nonfinite&&!mismatches&&released}});
        if(nonfinite||mismatches||!released)throw std::runtime_error("DiT fused original-graph readback mismatch or unsafe release");
    }
    moss_dit_gate_passed=true;
}
#endif
