/* SPDX-License-Identifier: MIT */
#ifndef PS5_AI_NATIVE_ARENA_JOURNAL_METRICS_H
#define PS5_AI_NATIVE_ARENA_JOURNAL_METRICS_H
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Metadata only. A snapshot performs no GPU call or filesystem operation.
 * Totals cover this arena journal only. Write/fsync times exclude lock wait;
 * lock_wait_ns measures it separately. Formatting, stderr and emergency
 * failure diagnostics are excluded. Call outside the logger and subtract. */
struct ps5_ai_native_arena_journal_stats {
    uint64_t records, checked_write_ns, fsync_calls, fsync_ns, lock_wait_ns;
    uint32_t log_healthy, abi_bytes;
};
__attribute__((visibility("hidden"))) void ps5_ai_native_arena_journal_snapshot(struct ps5_ai_native_arena_journal_stats *out);
/* Checked durable arena-journal boundary after an existing positive GPU quiet
 * gate (initial data gate, generation, or shutdown). This performs filesystem
 * I/O, unlike the metadata snapshot above. */
__attribute__((visibility("hidden"))) bool ps5_ai_native_arena_journal_checkpoint(void);
#ifdef __cplusplus
}
#endif
#endif
