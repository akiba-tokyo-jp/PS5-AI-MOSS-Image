/* SPDX-License-Identifier: MIT
 * Observational native lifetime state. This never authorizes an OS close,
 * assumes constructor side effects absent, or releases a GPU resource.
 */
#ifndef PS5_AI_ANIMA_LIFECYCLE_H
#define PS5_AI_ANIMA_LIFECYCLE_H
#include <stddef.h>
#include <stdint.h>
enum ps5_ai_lifecycle_phase {
    PS5_AI_LIFE_ENTRY, PS5_AI_LIFE_BOOTSTRAP, PS5_AI_LIFE_SERVER,
    PS5_AI_LIFE_CONFIG, PS5_AI_LIFE_AGC, PS5_AI_LIFE_DRIVER,
    PS5_AI_LIFE_DIRECT, PS5_AI_LIFE_BACKEND, PS5_AI_LIFE_GPU_GATE,
    PS5_AI_LIFE_READY, PS5_AI_LIFE_FAILED, PS5_AI_LIFE_RETAINED,
    PS5_AI_LIFE_PHASE_COUNT
};
#define PS5_AI_LIFE_READY_FLAG UINT32_C(0x10000)
#define PS5_AI_LIFE_INIT_FAILED UINT32_C(0x20000)
#define PS5_AI_LIFE_SERVER_FAILED UINT32_C(0x40000)
#define PS5_AI_LIFE_RETAINED_FLAG UINT32_C(0x80000)
#define PS5_AI_LIFE_BOOTSTRAP_FAILED UINT32_C(0x100000)
#define PS5_AI_LIFE_FAILURE_MASK UINT32_C(0x1e0000)
struct ps5_ai_native_lifecycle {
    uint32_t abi_bytes, abi_version, phase, failed_phase;
    uint32_t engine_ready, failure_observed, fault_retention_entered;
    uint32_t constructor_effects_unknown;
};
static inline const char *ps5_ai_lifecycle_phase_name(uint32_t phase) {
    static const char *const names[] = {"entry", "bootstrap", "server_startup",
        "configuration", "explicit_agc", "driver_prepare", "direct_prepare",
        "backend_prepare", "gpu_readback_gate", "ready", "failed", "retained"};
    return phase < PS5_AI_LIFE_PHASE_COUNT ? names[phase] : "unknown";
}
/* Ready means the existing checked init/readback path completed. It does not
 * replace the normal worker/listener/quiescence/context-cleanup exit proof.
 * All other states require a separately verified normal OS title-close path.
 * In particular, a zero AGC request counter is never an exit permission.
 */
static inline int ps5_ai_lifecycle_checked_shutdown_allowed(const struct ps5_ai_native_lifecycle *s) {
    return s && s->abi_bytes == sizeof(*s) && s->abi_version == 1 &&
        s->phase == PS5_AI_LIFE_READY && s->failed_phase < PS5_AI_LIFE_PHASE_COUNT &&
        s->engine_ready == 1 && s->failure_observed == 0 && s->fault_retention_entered == 0 &&
        s->constructor_effects_unknown == 1;
}
#ifdef __cplusplus
extern "C" {
#endif
void ps5_ai_native_lifecycle_snapshot(struct ps5_ai_native_lifecycle *out);
#ifdef __cplusplus
}
#endif
#endif
