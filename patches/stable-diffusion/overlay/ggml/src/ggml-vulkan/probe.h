/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef PS5_AI_GGML_PROBE_H
#define PS5_AI_GGML_PROBE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void ps5_ai_ggml_report(const char *format, ...);
int ps5_ai_ggml_log_healthy(void);
int ps5_ai_ggml_wait_backend(const char *phase, int expect_work);
__attribute__((noreturn)) void ps5_ai_ggml_retain(const char *reason);
struct ps5_ai_ggml_dispatch_stats {
    uint64_t submits, fenced_completions, dispatches;
    int work_pending, hardware_verified;
};
void ps5_ai_ggml_dispatch_snapshot(struct ps5_ai_ggml_dispatch_stats *out);
void ps5_ai_ggml_graph_phase(int active);
struct ps5_ai_ggml_result {
    unsigned completed_rounds, compared_values, mismatches;
    uint64_t graph_submits, graph_dispatches, fence_completions;
    double max_absolute_error;
    int backend_is_vulkan, hardware_verified;
};
int ps5_ai_ggml_matrix(struct ps5_ai_ggml_result *out);
#ifdef __cplusplus
}
#endif
#endif
