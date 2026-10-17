/* SPDX-License-Identifier: MIT */
#ifndef PS5_AI_BOUNDED_ARENA_H
#define PS5_AI_BOUNDED_ARENA_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Legacy statistics ABI, with dynamic per-BO ownership; no fixed byte cap. */
struct ps5_ai_arena_stats {
    uint64_t reserved_bytes;
    uint64_t used_bytes;
    uint64_t peak_used_bytes;
    uint64_t live_allocations;
    uint64_t allocation_attempts;
    uint64_t denied_allocations;
    int physical_owned;
    int mapping_owned;
    int mapping_uncertain;
    int gpu_fault_retained;
    int global_capacity_measured; /* always zero: only explicitly owned BO bytes */
};
struct ps5_ai_arena_kick_stats {
    uint64_t requested_epoch;
    uint64_t completed_epoch;
    int initialized;
    int thread_started;
    int pending;
    int active;
    int submit_inflight;
    int fault_retained;
    int log_healthy;
};
/* Prepare owner metadata/logging only; no Direct allocation is performed. */
int ps5_ai_arena_prepare(void);
int ps5_ai_native_arena_open_log(int sandbox);
const char *ps5_ai_native_arena_log_path(void);
void ps5_ai_arena_snapshot(struct ps5_ai_arena_stats *out);
/* Irreversible in this process; never release the arena after uncertain GPU work. */
void ps5_ai_arena_retain_gpu_fault(void);
/* Metadata-only: 0 no ownership; 1 retained nodes/fault. Never aggregate-free. */
int ps5_ai_arena_release_if_idle(void);
__attribute__((visibility("hidden"))) int ps5_ai_driver_owner_ready(void);
/* Only the sole frontend producer, after existing positive quiet, may call.
 * No wait is performed here or in RADV allocation/free callbacks. */
__attribute__((visibility("hidden"))) int ps5_ai_driver_retire_quiet(uint64_t expected_epoch);
/* Read-only metadata: 0 snapshot, 1 mutex busy (retry), -1 error.
 * Does not initialize the worker, wait, submit GPU work or perform filesystem I/O.
 */
int ps5_ai_arena_kick_snapshot(struct ps5_ai_arena_kick_stats *out);
#ifdef __cplusplus
}
#endif
#endif
