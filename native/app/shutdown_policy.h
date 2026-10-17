/* SPDX-License-Identifier: MIT */
#ifndef PS5_AI_V3_SHUTDOWN_POLICY_H
#define PS5_AI_V3_SHUTDOWN_POLICY_H
/* These are observed receipts, not requests to free or terminate anything. */
struct ps5_ai_v3_exit_state {
    int authenticated_idle_request, listener_closed, own_worker_joined;
    int model_cleanup_checked, pre_cleanup_quiescent, post_cleanup_quiescent;
    int generation_inactive, dispatch_pending, backend_quiescent;
    int arena_fault, mapping_uncertain, retained_fault, persistent_log_healthy;
    int server_return_code;
};
static inline int ps5_ai_v3_exit_allowed(const struct ps5_ai_v3_exit_state *s) {
    return s && s->authenticated_idle_request == 1 && s->listener_closed == 1 &&
        s->own_worker_joined == 1 && s->model_cleanup_checked == 1 &&
        s->pre_cleanup_quiescent == 1 && s->post_cleanup_quiescent == 1 &&
        s->generation_inactive == 1 && s->dispatch_pending == 0 &&
        s->backend_quiescent == 1 && s->arena_fault == 0 && s->mapping_uncertain == 0 &&
        s->retained_fault == 0 && s->persistent_log_healthy == 1 && s->server_return_code == 0;
}
#endif
