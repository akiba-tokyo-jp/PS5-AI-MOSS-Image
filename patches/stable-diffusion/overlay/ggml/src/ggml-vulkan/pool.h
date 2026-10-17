/* SPDX-License-Identifier: MIT
 * Application-owned host storage for Vulkan imports. No credential, budget,
 * foreign-process, kernel-memory or protected-filesystem operations.
 */
#ifndef PS5_AI_FLEXIBLE_GPU_POOL_H
#define PS5_AI_FLEXIBLE_GPU_POOL_H
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define PS5_AI_FLEX_POOL_BYTES (UINT64_C(1) << 30)
#define PS5_AI_FLEX_PAGE_BYTES UINT64_C(16384)
struct ps5_ai_flex_stats {
    uint64_t reserved_bytes, used_bytes, peak_used_bytes, free_bytes;
    uint64_t allocations, active_imports, denied_allocations;
    int mapping_owned, gpu_access_granted, fault_retained, log_healthy;
};
/* Must succeed before physical-device enumeration exposes the separate heap. */
int ps5_ai_flex_prepare(void);
void *ps5_ai_flex_alloc(uint64_t bytes, uint64_t alignment);
/* Import callback accepts exactly one current owned allocation. */
bool ps5_ai_flex_grant_gpu(void *address, uint64_t bytes);
int ps5_ai_flex_import_committed(void *address, uint64_t bytes);
/* Only call after destroying the imported VkBuffer/VkDeviceMemory. */
int ps5_ai_flex_free(void *address, uint64_t bytes);
void ps5_ai_flex_snapshot(struct ps5_ai_flex_stats *out);
uint64_t ps5_ai_flex_pool_bytes(void);
void ps5_ai_flex_retain_fault(const char *reason);
#ifdef __cplusplus
}
#endif
#endif
