#pragma once
#include "ggml-plan-admission.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <cstring>
#include "pool.h"
#include "vulkan_scratch_bounds.hpp"

extern "C" uint64_t ps5_ai_direct_pool_max_claim_bytes(void);
static_assert(sizeof(ps5_ai_flex_stats)==72,"Preserve deployed Direct pool snapshot ABI");

namespace native_graph_admission {

/* Only pointer tags are temporarily changed, exactly as the existing graph
 * measurement does. Save before changing; restore on success, failure and throw. */
struct SavedBinding {
    ggml_tensor * tensor;
    ggml_backend_buffer_t buffer;
    void * data;
    void * extra;
};
struct ExternalBindings {
    std::vector<SavedBinding> saved;
    void mark(ggml_tensor * tensor) {
        for (auto * current = tensor; current != nullptr; current = current->view_src) {
            bool known = false;
            for (const auto & entry : saved) {
                if (entry.tensor == current) { known = true; break; }
            }
            if (known) continue;
            saved.push_back({current, current->buffer, current->data, current->extra});
            current->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
        }
    }
    ~ExternalBindings() {
        for (auto it = saved.rbegin(); it != saved.rend(); ++it) {
            it->tensor->buffer = it->buffer;
            it->tensor->data = it->data;
            it->tensor->extra = it->extra;
        }
    }
};
struct AllocatorOwner {
    ggml_gallocr_t value = nullptr;
    ~AllocatorOwner() { if (value != nullptr) ggml_gallocr_free(value); }
};

struct Result {
    bool accepted = false;
    const char * reason = "unmeasured";
    uint64_t claim_limit = 0;
    uint64_t max_tensor_bytes = 0;
    ggml_tensor * max_tensor = nullptr;
    uint32_t tensor_references_scanned = 0;
    uint32_t im2col_nodes = 0;
    uint32_t flash_nodes = 0;
    ggml_gallocr_plan_snapshot plan = {};
    /* Intentionally separate: no whole-graph backend scratch planning API. */
    bool scratch_measured = false;
};

template<class IsParameter>
static Result inspect(ggml_backend_buffer_type_t buft, ggml_cgraph * graph,
                      uint64_t claim_limit, IsParameter is_parameter) {
    Result result;
    result.claim_limit = claim_limit;
    /* Same graph bound as the actual runner; this metadata pass has no GPU data. */
    constexpr int max_graph_references = 2 * 327680;
    if (buft == nullptr || graph == nullptr || claim_limit == 0 || graph->n_nodes < 0 || graph->n_leafs < 0 ||
        graph->n_nodes > max_graph_references || graph->n_leafs > max_graph_references - graph->n_nodes) {
        result.reason = "invalid_or_unavailable_owned_plan";
        return result;
    }
    auto scan = [&](ggml_tensor * tensor) {
        if (tensor == nullptr) return false;
        uint64_t bytes = static_cast<uint64_t>(ggml_backend_buft_get_alloc_size(buft, tensor));
        ++result.tensor_references_scanned;
        if (bytes > result.max_tensor_bytes) {
            result.max_tensor_bytes = bytes;
            result.max_tensor = tensor;
        }
        if (tensor->op == GGML_OP_IM2COL) ++result.im2col_nodes;
        if (tensor->op == GGML_OP_FLASH_ATTN_EXT) ++result.flash_nodes;
        return bytes <= claim_limit;
    };
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (!scan(graph->leafs[i])) { result.reason = "tensor_exceeds_actual_region"; return result; }
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (!scan(graph->nodes[i])) { result.reason = "tensor_exceeds_actual_region"; return result; }
    }
    ExternalBindings bindings;
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (is_parameter(graph->leafs[i])) bindings.mark(graph->leafs[i]);
    }
    AllocatorOwner allocator{ggml_gallocr_new(buft)};
    if (allocator.value == nullptr) { result.reason = "metadata_allocator_unavailable"; return result; }
    size_t measured[1] = {0};
    ggml_gallocr_reserve_n_size(allocator.value, graph, nullptr, nullptr, measured);
    if (!ggml_gallocr_get_plan_snapshot(allocator.value, 0, &result.plan) ||
        result.plan.buffer_count != 1 || result.plan.total_bytes != static_cast<uint64_t>(measured[0])) {
        result.reason = "metadata_plan_snapshot_invalid";
        return result;
    }
    if (result.plan.max_chunk_bytes > claim_limit) {
        result.reason = "chunk_exceeds_actual_region";
        return result;
    }
    result.accepted = true;
    result.reason = "data_shape_and_chunks_within_actual_region";
    return result;
}

/* All five switches are set before backend creation by the pinned native startup.
 * Do not reinterpret a different backend/profile using these scalar bounds. */
static bool exact_profile_switch(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}
static native_scratch_bounds::Profile current_profile() {
    return {exact_profile_switch("GGML_VK_DISABLE_COOPMAT") && exact_profile_switch("GGML_VK_DISABLE_COOPMAT2"),
            exact_profile_switch("GGML_VK_DISABLE_F16"), exact_profile_switch("GGML_VK_DISABLE_ASYNC"),
            exact_profile_switch("GGML_VK_DISABLE_FUSION"), 36};
}
struct ScratchDecision {
    native_scratch_bounds::Bounds upper;
    bool per_claim = false;
};
static ScratchDecision inspect_scratch(ggml_cgraph * graph, const native_scratch_bounds::Profile & profile,
                                      uint64_t claim_limit) {
    ScratchDecision result{native_scratch_bounds::inspect(graph,profile),false};
    const auto & bound=result.upper;
    result.per_claim=claim_limit && bound.known && bound.x<=claim_limit && bound.y<=claim_limit &&
        bound.split_k<=claim_limit && bound.add_rms_baseline<=claim_limit;
    return result;
}

template<class IsParameter>
static bool admit(ggml_backend_t backend, ggml_cgraph * graph, IsParameter is_parameter, const char * description, uint64_t reusable_workspace) {
    auto result = inspect(ggml_backend_get_default_buffer_type(backend), graph,
                          ps5_ai_direct_pool_max_claim_bytes(), is_parameter);
    const auto * tensor = result.max_tensor;
    LOG_INFO("native graph data admission %s: accepted=%d reason=%s actual_max_claim=%llu tensor_refs=%u max_tensor=%llu type=%s ne=[%lld,%lld,%lld,%lld] chunks=%u max_chunk=%llu total_chunks=%llu im2col_nodes=%u flash_nodes=%u scratch_measured=0 full_1024_readiness_proven=0 GPU_allocation_attempted=0 scope=metadata_inspection_only preexisting_backend_resources_not_counted=1",
        description, result.accepted ? 1 : 0, result.reason,
        (unsigned long long)result.claim_limit, result.tensor_references_scanned,
        (unsigned long long)result.max_tensor_bytes, tensor ? ggml_type_name(tensor->type) : "none",
        tensor ? (long long)tensor->ne[0] : 0LL, tensor ? (long long)tensor->ne[1] : 0LL,
        tensor ? (long long)tensor->ne[2] : 0LL, tensor ? (long long)tensor->ne[3] : 0LL,
        result.plan.chunk_count, (unsigned long long)result.plan.max_chunk_bytes,
        (unsigned long long)result.plan.total_bytes, result.im2col_nodes, result.flash_nodes);
    if (!result.accepted) return false;
    const auto scratch = inspect_scratch(graph,current_profile(),result.claim_limit);
    const auto & upper=scratch.upper;
    ps5_ai_flex_stats pool={};
    ps5_ai_flex_snapshot(&pool);
    const bool pool_healthy=pool.mapping_owned && pool.gpu_access_granted && !pool.fault_retained &&
        pool.log_healthy && pool.reserved_bytes && pool.free_bytes<=pool.reserved_bytes;
    const uint64_t new_workspace=result.plan.total_bytes>reusable_workspace ? result.plan.total_bytes-reusable_workspace : 0;
    /* Requested maxima are conservative and may count scratch already held by
     * this backend. This diagnostic never pretends to simulate fragmentation,
     * preceding context highwaters, parameter staging or the physical driver. */
    uint64_t diagnostic_total=new_workspace;
    bool sum_known=true;
    for (const auto bytes : {upper.x,upper.y,upper.split_k,upper.add_rms_baseline}) {
        if (bytes>UINT64_MAX-diagnostic_total) { sum_known=false;break; }
        diagnostic_total+=bytes;
    }
    LOG_INFO("native graph scratch metadata %s: requested_four_prealloc_upper_known=%d reason=%s x_bytes=%llu y_bytes=%llu split_k_bytes=%llu add_rms_baseline_bytes=%llu unknown_nodes=%u first_unknown_op=%d per_claim_bound_pass=%d owned_pool_healthy=%d owned_pool_free=%llu workspace_reusable=%llu new_workspace=%llu conservative_requested_plus_new_workspace=%llu aggregate_diagnostic_sum_known=%d diagnostic_sum_fits_current_free=%d aggregate_capacity_proven=0 fragmentation_proven=0 previous_context_highwater_measured=0 driver_headroom_proven=0 full_1024_readiness_proven=0 GPU_allocation_attempted=0 scope=metadata_inspection_only preexisting_backend_resources_not_counted=1",
        description,upper.known?1:0,upper.reason,(unsigned long long)upper.x,(unsigned long long)upper.y,
        (unsigned long long)upper.split_k,(unsigned long long)upper.add_rms_baseline,
        upper.unknown_nodes,(int)upper.first_unknown,scratch.per_claim?1:0,pool_healthy?1:0,
        (unsigned long long)pool.free_bytes,(unsigned long long)reusable_workspace,(unsigned long long)new_workspace,
        (unsigned long long)(sum_known?diagnostic_total:0),sum_known?1:0,
        sum_known && diagnostic_total<=pool.free_bytes?1:0);
    return scratch.per_claim && pool_healthy;
}
} // namespace native_graph_admission
