/* SPDX-License-Identifier: MIT
 * CPU-only CLIP weights borrow owned Direct data claims. No Vulkan import,
 * physical allocation, driver modification or transfer of pool ownership.
 */
#pragma once
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "pool.h"
#include <cstddef>
#include <cstdint>

extern "C" void ps5_ai_cpu_clip_pool_event(const char*, uint64_t, int) noexcept;

static void ps5_ai_cpu_clip_pool_buffer_free(ggml_backend_buffer_t buffer) {
    // CPU_Mapped keeps the original aligned pointer as its context. This
    // claim was never imported by Vulkan. Pool free checks completed GPU work
    // before making its pages available to another claimant.
    const int released = ps5_ai_flex_free(buffer->context, buffer->size) == 0;
    ps5_ai_cpu_clip_pool_event("free", buffer->size, released);
}

static ggml_backend_buffer_t ps5_ai_cpu_clip_pool_buffer(size_t size) {
    if (!size || size > (static_cast<size_t>(256) << 20)) return nullptr;
    void* pointer = ps5_ai_flex_alloc(size, 64);
    if (!pointer) {
        ps5_ai_cpu_clip_pool_event("allocation_denied", size, 0);
        return nullptr; // No hidden second physical allocation attempt.
    }
    ggml_backend_buffer_t buffer = nullptr;
    try { buffer = ggml_backend_cpu_buffer_from_ptr(pointer, size); }
    catch (...) {
        const int released = ps5_ai_flex_free(pointer, size) == 0;
        ps5_ai_cpu_clip_pool_event("descriptor_failure_cleanup", size, released);
        throw;
    }
    if (!buffer) {
        const int released = ps5_ai_flex_free(pointer, size) == 0;
        ps5_ai_cpu_clip_pool_event("descriptor_failure_cleanup", size, released);
        return nullptr;
    }
    // All tensor access remains the upstream CPU_Mapped implementation.
    // Its default free callback is null; only this exact claim is returned.
    buffer->iface.free_buffer = ps5_ai_cpu_clip_pool_buffer_free;
    ps5_ai_cpu_clip_pool_event("allocated", size, 1);
    return buffer;
}
