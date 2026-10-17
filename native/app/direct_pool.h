/* SPDX-License-Identifier: MIT */
#ifndef PS5_AI_NATIVE_DIRECT_POOL_H
#define PS5_AI_NATIVE_DIRECT_POOL_H
#include "pool.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Startup-thread-only one-shot configuration/prepare, before GGML uses pool.
 * Snapshot/claim APIs are synchronized. No implicit large default.
 * Every claim stays inside one separately owned, nonfixed Direct mapping. */
int ps5_ai_direct_pool_configure(uint64_t target_bytes, uint64_t region_max_bytes,
                                 uint64_t cpu_direct_reserve_bytes);
struct ps5_ai_direct_pool_stats {
    uint64_t target_bytes, region_max_bytes, cpu_direct_reserve_bytes;
    uint64_t physical_owned_bytes, mapped_bytes, largest_region_bytes;
    uint64_t region_count, claim_capacity;
    int configured, attempted, ready, ownership_uncertain;
};
void ps5_ai_direct_pool_snapshot(struct ps5_ai_direct_pool_stats *out);
uint64_t ps5_ai_direct_pool_max_claim_bytes(void);
/* Known, unused data regions only; no implicit process exit or driver32 free.
 * Caller must have closed submit producers and destroyed all imported Vk
 * objects; zero claims alone does not prove no future external submission. */
int ps5_ai_direct_pool_release_if_idle(void);
#ifdef __cplusplus
}
#endif
#endif
