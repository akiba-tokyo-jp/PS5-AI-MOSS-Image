// SPDX-License-Identifier: MIT
#pragma once
/* Metadata/configuration only. This does not submit work or allocate GPU memory.
 * The explicit PS5 profile changes batching and one existing zero-scratch shader,
 * never the serialized completion/ownership protocol or general fusion switch. */
#include "ggml.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace ggml_vk_ps5_profile {
struct Options {
    unsigned submit_multiplier = 1;
    bool elementwise_fusion = false;
};
struct Environment {
    const char * submit_batch = nullptr;
    const char * elementwise_fusion = nullptr;
    const char * serialize = nullptr;
    const char * no_async = nullptr;
    const char * no_fusion = nullptr;
    const char * no_f16 = nullptr;
    const char * no_coopmat = nullptr;
    const char * no_coopmat2 = nullptr;
    const char * generic_max_nodes = nullptr;
};
inline bool exact(const char * value, const char * expected) {
    return value && std::strcmp(value, expected) == 0;
}
inline bool parse(const Environment & env, Options & out, const char * & error) {
    out = {}; error = nullptr;
    if (env.submit_batch && !exact(env.submit_batch, "2")) {
        error = "GGML_VK_PS5_SUBMIT_BATCH must be unset or exactly 2"; return false;
    }
    if (env.elementwise_fusion && !exact(env.elementwise_fusion, "1")) {
        error = "GGML_VK_PS5_ELEMENTWISE_FUSION must be unset or exactly 1"; return false;
    }
    if (!env.submit_batch && !env.elementwise_fusion) return true;
    if (!exact(env.serialize, "1") || !exact(env.no_async, "1") || !exact(env.no_fusion, "1") ||
        !exact(env.no_f16, "1") || !exact(env.no_coopmat, "1") || !exact(env.no_coopmat2, "1")) {
        error = "PS5 optimization requires exact serialized scalar FP32/no-async/no-general-fusion profile";
        return false;
    }
    if (env.generic_max_nodes) {
        error = "PS5 optimization conflicts with GGML_VK_MAX_NODES_PER_SUBMIT"; return false;
    }
    out.submit_multiplier = env.submit_batch ? 2 : 1;
    out.elementwise_fusion = env.elementwise_fusion != nullptr;
    return true;
}
inline uint64_t initial_flops(uint64_t base_cap, uint64_t last_total, unsigned multiplier) {
    // Callers supply the frozen <=200GFLOP base and the parsed multiplier 1 or 2.
    return std::min(base_cap, last_total / 40u) * multiplier;
}
inline bool f32_addressable(const ggml_tensor * t) {
    if (!t || t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || !ggml_is_contiguous_1(t)) return false;
    uint64_t count = 1, span = sizeof(float);
    for (unsigned d = 0; d < GGML_MAX_DIMS; ++d) {
        if (t->ne[d] <= 0 || uint64_t(t->ne[d]) > UINT32_MAX ||
            t->nb[d] > UINT32_MAX || t->nb[d] % sizeof(float) ||
            count > UINT32_MAX / uint64_t(t->ne[d])) return false;
        count *= uint64_t(t->ne[d]);
        const uint64_t additional = (uint64_t(t->ne[d])-1) * t->nb[d];
        if (additional > UINT32_MAX - span) return false;
        span += additional;
    }
    return true;
}
inline bool limited_unary_mul_metadata(const ggml_tensor * unary, const ggml_tensor * product) {
    if (!unary || !product || unary->op != GGML_OP_UNARY || product->op != GGML_OP_MUL ||
        (ggml_get_unary_op(unary) != GGML_UNARY_OP_SIGMOID && ggml_get_unary_op(unary) != GGML_UNARY_OP_SILU) ||
        !f32_addressable(unary) || !f32_addressable(product) || !f32_addressable(unary->src[0])) return false;
    if (product->src[0] != unary && product->src[1] != unary) return false;
    const auto * other = product->src[0] == unary ? product->src[1] : product->src[0];
    return f32_addressable(other) && ggml_are_same_shape(unary, product) &&
           ggml_are_same_shape(unary, unary->src[0]) && ggml_are_same_shape(unary, other);
}
} // namespace ggml_vk_ps5_profile
