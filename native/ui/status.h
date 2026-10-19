/* SPDX-License-Identifier: GPL-3.0-or-later
 * Optional native status display integrated into the native WAI app.
 * Publishers copy measured POD data only; no renderer/GPU calls under engine locks.
 */
#ifndef PS5_AI_STATUS_DISPLAY_V1_H
#define PS5_AI_STATUS_DISPLAY_V1_H
#include <stddef.h>
#include <stdint.h>

#define PS5_AI_STATUS_ABI 1u
#define PS5_AI_STATUS_APP_NAME "PS5 AI Anima"
#define PS5_AI_STATUS_FRAME_BYTES UINT64_C(16777216)
#define PS5_AI_STATUS_SCANOUT_BYTES UINT64_C(33554432)
#define PS5_AI_STATUS_MAX_ROWS 24
#define PS5_AI_STATUS_ROW_BYTES 128
#define PS5_AI_STATUS_MAX_SAMPLE_AGE_NS UINT64_C(3000000000)

enum ps5_ai_status_valid {
    PS5_AI_STATUS_CLOCK=1u, PS5_AI_STATUS_CPU_RSS=2u,
    PS5_AI_STATUS_CPU_BUSY=4u, PS5_AI_STATUS_GPU_BUSY=8u,
    PS5_AI_STATUS_TEMPERATURE=16u, PS5_AI_STATUS_DIRECT_POOL=32u,
    PS5_AI_STATUS_DRIVER_ARENA=64u, PS5_AI_STATUS_COUNTERS=128u,
    PS5_AI_STATUS_PROGRESS=256u, PS5_AI_STATUS_LOGS=512u,
    PS5_AI_STATUS_MODEL=1024u, PS5_AI_STATUS_PHASE=2048u,
    PS5_AI_STATUS_JOB=4096u
};
enum ps5_ai_status_rss_source {
    PS5_AI_STATUS_RSS_CURRENT=1u, PS5_AI_STATUS_RSS_PEAK=2u
};
enum ps5_ai_status_busy_source {
    PS5_AI_STATUS_CPU_CAPACITY_TIME_WINDOW=1u,
    PS5_AI_STATUS_GPU_HARDWARE_BUSY_TIME_WINDOW=2u
};
enum ps5_ai_status_phase {
    PS5_AI_STATUS_IDLE, PS5_AI_STATUS_LOADING, PS5_AI_STATUS_GENERATING,
    PS5_AI_STATUS_COMPLETED, PS5_AI_STATUS_FAILED, PS5_AI_STATUS_STOPPING
};
struct ps5_ai_status_snapshot_v1 {
    uint32_t abi_bytes, abi_version, valid, phase;
    uint64_t sequence;
    int64_t clock_unix_seconds;
    uint64_t monotonic_ns, cpu_rss_bytes;
    uint64_t direct_reserved_bytes, direct_used_bytes, direct_peak_bytes;
    uint64_t driver_reserved_bytes, driver_used_bytes;
    uint64_t dispatches, submits, fences;
    uint32_t process_id, cpu_busy_source, gpu_busy_source, progress_basis_points;
    int32_t temperature_millic;
    uint32_t cpu_rss_source;
    uint64_t cpu_busy_ns, cpu_capacity_ns, gpu_busy_ns, gpu_elapsed_ns;
    char model[64], job_id[33], padding[7], logs[4][96];
};
struct ps5_ai_status_gate_v1 {
    uint32_t abi_bytes, abi_version, process_id, completed_gate_passed;
    uint64_t completed_gate_bytes;
    uint32_t explicit_AGC_attempts, reserved;
};
struct ps5_ai_status_display_state_v1 {
    uint32_t abi_bytes, abi_version, thread_started, ready;
    uint32_t physical_owned, mapping_owned, buffers_registered, ownership_uncertain;
    uint32_t display_fault, stopped, video_opened, reserved;
    int32_t error_stage, error_code;
    uint64_t completed_flip_argument, heartbeat;
};
#ifdef __cplusplus
#define PS5_AI_STATUS_ASSERT static_assert
extern "C" {
#else
#define PS5_AI_STATUS_ASSERT _Static_assert
#endif
PS5_AI_STATUS_ASSERT(sizeof(struct ps5_ai_status_snapshot_v1)==656, "Status POD ABI size");
PS5_AI_STATUS_ASSERT(offsetof(struct ps5_ai_status_snapshot_v1,logs)==272, "Status log packet offset");
PS5_AI_STATUS_ASSERT(sizeof(struct ps5_ai_status_gate_v1)==32, "Gate POD ABI size");
PS5_AI_STATUS_ASSERT(sizeof(struct ps5_ai_status_display_state_v1)==72, "Display state ABI size");

/* Every publish replaces all valid bits; they are never merged with old bits.
 * monotonic_ns is the acquisition time, not a fresh label for cached samples.
 * Missing valid bits omit the entire row, including its label. A percentage is
 * accepted only with its measured time-window source and numerator/denominator;
 * pending work and fence counts must never be published as GPU busy percent.
 * No CPU/GPU busy measurement provider exists in this optional module, so an
 * integration without one MUST leave those bits clear. CPU capacity is measured
 * process CPU-time capacity for the same window; GPU busy is a hardware counter.
 * ru_maxrss is PEAK only, never CURRENT. No current RSS getter is called here.
 * Phase, job and all metrics need a fresh acquisition time (max age 3 seconds).
 * Display heartbeat proves the renderer loop only, not inference progress.
 * Direct data pool percentages describe this app's reserved pool, not VRAM.
 */
int ps5_ai_status_snapshot_valid(const struct ps5_ai_status_snapshot_v1 *);
int ps5_ai_status_live_view(struct ps5_ai_status_snapshot_v1 *out,
                           const struct ps5_ai_status_snapshot_v1 *source,
                           uint64_t now_ns,int refresh_ok);
size_t ps5_ai_status_format(const struct ps5_ai_status_snapshot_v1 *,
                           char rows[PS5_AI_STATUS_MAX_ROWS][PS5_AI_STATUS_ROW_BYTES]);

/* Hardware adapter is separate. Start is one-shot and only after the existing
 * same-process 64 MiB GPU gate and sole explicit Init8 have passed. It returns
 * after pthread creation; readiness is obtained separately, without blocking
 * inference. The display owner alone performs VideoOut/scanout operations.
 * Every owned mapping survives display failure/unknown flip until OS teardown;
 * no resource-free API exists. No AGC/Vulkan/GPU math calls are permitted.
 * request_stop/stopped describe CPU drawing only, not join or scanout release.
 */
int ps5_ai_status_start_after_gate(const struct ps5_ai_status_gate_v1 *);
int ps5_ai_status_publish(const struct ps5_ai_status_snapshot_v1 *);
int ps5_ai_status_display_state(struct ps5_ai_status_display_state_v1 *);
void ps5_ai_status_request_stop(void);
#ifdef __cplusplus
}
#endif
#undef PS5_AI_STATUS_ASSERT
#endif
