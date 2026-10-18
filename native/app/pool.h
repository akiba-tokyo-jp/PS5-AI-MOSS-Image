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
#define PS5_AI_FLEX_POOL_DEFAULT_BYTES (UINT64_C(1) << 30)
#define PS5_AI_FLEX_PAGE_BYTES UINT64_C(16384)
struct ps5_ai_flex_stats {
    uint64_t reserved_bytes, used_bytes, peak_used_bytes, free_bytes;
    uint64_t allocations, active_imports, denied_allocations;
    int mapping_owned, gpu_access_granted, fault_retained, log_healthy;
};
/* Must succeed before physical-device enumeration exposes the separate heap. */
int ps5_ai_flex_prepare(void);
/* One explicit request per process, no retry/automatic size ladder.
 * Reserve>=64MiB is an accounting admission floor, not physical-capacity proof.
 * Return0 ready, -2 CPU-only cancellation with checked rollback, -1 failure.
 * This function never exits the process or installs signal handlers. */
int ps5_ai_flex_prepare_sized(uint64_t requested_bytes, uint64_t working_reserve_bytes);
enum ps5_ai_flex_prepare_phase {
    PS5_AI_FLEX_NEW, PS5_AI_FLEX_CHECKING, PS5_AI_FLEX_CPU_MAPPING,
    PS5_AI_FLEX_CPU_WRITING, PS5_AI_FLEX_CPU_VERIFYING,
    PS5_AI_FLEX_GRANT_PENDING, PS5_AI_FLEX_GPU_READY,
    PS5_AI_FLEX_CPU_ABORTED, PS5_AI_FLEX_RETAINED
};
struct ps5_ai_flex_prepare_stats {
    uint64_t requested_bytes, working_reserve_bytes;
    uint64_t configured_bytes, available_bytes, metadata_bytes;
    uint64_t pages, claim_capacity, written_pages, verified_pages;
    int phase, cancel_requested, gpu_permission_attempts, ownership_uncertain;
};
/* Own-process metadata only:0 request accepted before permission transition;
 * 1 not cancellable. Checked at block boundaries; blocked page faults have no
 * user-space hard timeout. No signal or foreign-process action. */
int ps5_ai_flex_cancel_cpu_prefault(void);
void ps5_ai_flex_prepare_snapshot(struct ps5_ai_flex_prepare_stats *out);
void *ps5_ai_flex_alloc(uint64_t bytes, uint64_t alignment);
/* Import callback accepts exactly one current owned allocation. */
bool ps5_ai_flex_grant_gpu(void *address, uint64_t bytes);
int ps5_ai_flex_import_committed(void *address, uint64_t bytes);
/* Only call after destroying the imported VkBuffer/VkDeviceMemory. */
int ps5_ai_flex_free(void *address, uint64_t bytes);
/* Checked metadata writes; required durable checkpoint before publishing PNG. */
int ps5_ai_direct_pool_log_checkpoint(void);
void ps5_ai_flex_snapshot(struct ps5_ai_flex_stats *out);
uint64_t ps5_ai_flex_pool_bytes(void);
void ps5_ai_flex_retain_fault(const char *reason);
#ifdef __cplusplus
}
#endif
#endif
