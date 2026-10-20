/* SPDX-License-Identifier: MIT -- extra startup gate; existing gates stay intact. */
#include "moss_bounded_weight_upload.h"
#if PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL
#include "model_io/bounded_weight_upload.hpp"
#include "moss_bounded_upload_fixture.hpp"

namespace moss_bounded_upload_gate_state {
static std::atomic<bool> admitted{false};
static bool attempted=false;
}
#endif

extern "C" int ps5_ai_moss_bounded_weight_upload_enabled(void) {
#if PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL
    return moss_bounded_upload_gate_state::admitted.load(std::memory_order_acquire) ? 1 : 0;
#else
    return 0;
#endif
}

#if PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL
static void moss_bounded_weight_upload_readback_gate() {
    using namespace moss_bounded_upload_fixture;
    if (moss_bounded_upload_gate_state::attempted ||
        moss_bounded_upload_gate_state::admitted.load(std::memory_order_acquire) ||
        model_context || !hardware_verified || !gpu_device || !current_status.available)
        throw std::runtime_error("Bounded upload gate requires one deferred, verified MOSS context");
    moss_bounded_upload_gate_state::attempted=true;
    struct Owned {
        ggml_backend_t backend=nullptr;
        ggml_context* context=nullptr;
        ggml_backend_buffer_t buffer=nullptr;
        void release() noexcept {
            if (!backend) return;
            try { ggml_backend_synchronize(backend); }
            catch (...) { ps5_ai_ggml_retain("bounded_upload_gate_cleanup_sync_unknown"); }
            ps5_ai_ggml_dispatch_stats state{};ps5_ai_ggml_dispatch_snapshot(&state);
            if (state.work_pending || !ps5_ai_ggml_wait_backend("bounded_upload_gate_cleanup",0))
                ps5_ai_ggml_retain("bounded_upload_gate_cleanup_not_quiescent");
            if (buffer) { ggml_backend_buffer_free(buffer);buffer=nullptr; }
            if (context) { ggml_free(context);context=nullptr; }
            ggml_backend_free(backend);backend=nullptr;
        }
        ~Owned() noexcept { if (backend) release(); }
    } r;
    ps5_ai_ggml_dispatch_stats entry{};ps5_ai_ggml_dispatch_snapshot(&entry);
    if (entry.work_pending || !ps5_ai_ggml_wait_backend("bounded_upload_gate_entry",0))
        ps5_ai_ggml_retain("bounded_upload_gate_entry_not_quiescent");
    ps5_ai_flex_stats pool_before{};ps5_ai_flex_snapshot(&pool_before);
    if (pool_before.fault_retained || !pool_before.log_healthy)
        throw std::runtime_error("Bounded upload gate requires healthy owned Direct pool");
    r.backend=ggml_backend_vk_init(0);
    if (!r.backend || !ggml_backend_is_vk(r.backend) || ggml_backend_get_device(r.backend)!=gpu_device)
        throw std::runtime_error("Bounded upload gate requires the existing shared Vulkan device");
    r.context=ggml_init({1<<20,nullptr,true});
    if (!r.context) throw std::runtime_error("Bounded upload gate metadata allocation failed");
    ggml_tensor* input_tensor=ggml_new_tensor_1d(r.context,GGML_TYPE_F32,elements);
    ggml_tensor* output_tensor=ggml_scale(r.context,input_tensor,2.0f);
    ggml_set_name(input_tensor,"bounded_upload_9MiB_plus68_input");
    ggml_set_name(output_tensor,"bounded_upload_SCALE2_exact_output");
    ggml_set_input(input_tensor);ggml_set_output(output_tensor);
    if (!ggml_backend_supports_op(r.backend,output_tensor))
        throw std::runtime_error("Bounded upload SCALE unsupported; CPU fallback forbidden");
    ggml_cgraph* graph=ggml_new_graph_custom(r.context,8,false);
    ggml_build_forward_expand(graph,output_tensor);
    r.buffer=ggml_backend_alloc_ctx_tensors(r.context,r.backend);
    if (!r.buffer || ggml_backend_buffer_is_host(r.buffer) ||
        ggml_backend_buffer_get_size(r.buffer)<tensor_bytes*2 ||
        ggml_nbytes(input_tensor)!=tensor_bytes || ggml_nbytes(output_tensor)!=tensor_bytes)
        throw std::runtime_error("Bounded upload gate did not allocate full non-host input/output");

    uint64_t read_calls=0,upload_calls=0,next_source=source_offset,next_upload=0;
    ps5_ai_weight_io::Result transfer;
    {
        // No full input vector: fill the loader's one8MiB Scratch on demand.
        // Synchronous backend_set consumes each host fragment before reuse/free.
        ps5_ai_weight_io::Scratch scratch;
        transfer=ps5_ai_weight_io::transfer_exact(source_offset,tensor_bytes,scratch,
            [&](uint64_t absolute,unsigned char* memory,size_t n) {
                ++read_calls;
                if (absolute!=next_source || !read_exact(absolute,memory,n)) return false;
                next_source+=n;return true;
            },
            [&](size_t relative,const unsigned char* memory,size_t n) {
                ++upload_calls;
                if (!memory || relative!=next_upload || relative>tensor_bytes ||
                    !n || n>tensor_bytes-relative || relative%sizeof(float) || n%sizeof(float)) return false;
                ggml_backend_tensor_set(input_tensor,memory,relative,n);
                next_upload+=n;return true;
            });
    } // Release8MiB owned CPU Scratch before graph/readback-buffer allocation.
    if (!transfer.success || transfer.error!=ps5_ai_weight_io::Error::none ||
        transfer.bytes!=tensor_bytes || transfer.chunks!=2 || transfer.staging_payload!=ps5_ai_weight_io::staging_bytes ||
        read_calls!=2 || upload_calls!=2 || next_source!=source_offset+tensor_bytes || next_upload!=tensor_bytes)
        throw std::runtime_error("Bounded fragment upload failed; no fallback or retry admitted");
    ggml_backend_synchronize(r.backend);
    if (!ps5_ai_ggml_wait_backend("bounded_upload_gate_before_graph",0))
        ps5_ai_ggml_retain("bounded_upload_gate_upload_completion_unknown");
    ps5_ai_ggml_dispatch_stats before{},after{};ps5_ai_ggml_dispatch_snapshot(&before);
    if (before.work_pending) ps5_ai_ggml_retain("bounded_upload_gate_pre_graph_pending");
    const ggml_status status=ggml_backend_graph_compute(r.backend,graph);
    ggml_backend_synchronize(r.backend);
    if (status!=GGML_STATUS_SUCCESS || !ps5_ai_ggml_wait_backend("bounded_upload_gate_after_graph",1))
        ps5_ai_ggml_retain("bounded_upload_gate_compute_completion_unknown");
    ps5_ai_ggml_dispatch_snapshot(&after);
    if (after.work_pending || !after.hardware_verified || after.dispatches<=before.dispatches ||
        after.submits<=before.submits || after.fenced_completions<=before.fenced_completions)
        throw std::runtime_error("Bounded upload gate lacks actual hardware dispatch/submit/fence proof");

    std::vector<float> chunk(readback_elements);
    uint64_t compared=0,nonfinite=0,mismatches=0,checksum=fnv_offset,expected_checksum=fnv_offset,readback_calls=0;
    for(size_t first=0;first<elements;first+=readback_elements) {
        const size_t count=std::min(readback_elements,elements-first);
        ggml_backend_tensor_get(output_tensor,chunk.data(),first*sizeof(float),count*sizeof(float));
        ++readback_calls;
        for(size_t i=0;i<count;++i) {
            const float expected=moss_bounded_upload_fixture::output(first+i);
            const uint32_t actual_bits=bits(chunk[i]),expected_bits=bits(expected);
            nonfinite+=!std::isfinite(chunk[i]);
            mismatches+=actual_bits!=expected_bits;
            checksum=(checksum^actual_bits)*fnv_prime;
            expected_checksum=(expected_checksum^expected_bits)*fnv_prime;
            ++compared;
        }
    }
    ggml_backend_synchronize(r.backend);
    if (!ps5_ai_ggml_wait_backend("bounded_upload_gate_after_readback",1))
        ps5_ai_ggml_retain("bounded_upload_gate_readback_completion_unknown");
    ps5_ai_ggml_dispatch_stats final{};ps5_ai_ggml_dispatch_snapshot(&final);
    if (final.work_pending || !final.hardware_verified || compared!=elements || nonfinite || mismatches || checksum!=expected_checksum)
        throw std::runtime_error("Bounded upload full exact GPU readback mismatch");
    r.release();
    ps5_ai_flex_stats pool_after{};ps5_ai_flex_snapshot(&pool_after);
    if (pool_after.fault_retained || !pool_after.log_healthy ||
        pool_after.used_bytes!=pool_before.used_bytes || pool_after.allocations!=pool_before.allocations ||
        pool_after.active_imports!=pool_before.active_imports || !ps5_ai_ggml_log_healthy())
        throw std::runtime_error("Bounded upload owned buffers were not safely released");
    write_event({{"event","native_moss_bounded_weight_upload_gate"},{"success",true},
        {"policy",PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_POLICY},{"tensor_bytes",tensor_bytes},
        {"source_offset",source_offset},{"upload_chunk_bytes",ps5_ai_weight_io::staging_bytes},
        {"upload_chunks",transfer.chunks},{"upload_tail_bytes",tensor_bytes-ps5_ai_weight_io::staging_bytes},
        {"read_calls",read_calls},{"upload_calls",upload_calls},{"uploaded_bytes",transfer.bytes},
        {"compared_values",compared},{"nonfinite",nonfinite},{"mismatches",mismatches},
        {"checksum",checksum},{"expected_checksum",expected_checksum},{"readback_calls",readback_calls},
        {"readback_payload_bytes",readback_elements*sizeof(float)},{"metadata_arena_bytes",1<<20},
        {"new_host_scratch_payload_bytes",ps5_ai_weight_io::staging_bytes},
        {"new_host_scratch_scope","per active worker; released before readback; existing backend staging excluded"},
        {"dispatch_delta",after.dispatches-before.dispatches},{"submit_delta",after.submits-before.submits},
        {"fence_delta",after.fenced_completions-before.fenced_completions},
        {"pool_used_before",pool_before.used_bytes},{"pool_used_after",pool_after.used_bytes},
        {"pool_allocations_before",pool_before.allocations},{"pool_allocations_after",pool_after.allocations},
        {"active_imports_before",pool_before.active_imports},{"active_imports_after",pool_after.active_imports},
        {"model_loaded",false},{"CPU_inference",false},{"whole_pool_GPU_verified",false},
        {"enable_hook_only_after_this_durable_gate",true},{"automatic_retries",0}});
    if (!ps5_ai_ggml_log_healthy())
        throw std::runtime_error("Bounded upload admission log was not durable");
    moss_bounded_upload_gate_state::admitted.store(true,std::memory_order_release);
}
#endif
