/* SPDX-License-Identifier: MIT */
#ifndef PS5_AI_V3_SHUTDOWN_H
#define PS5_AI_V3_SHUTDOWN_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Called only by the authenticated, empty-body, idle admin route/main thread. */
void ps5_ai_v3_shutdown_request_accepted(int running, int queued);
/* Called by that same main thread after listener close and own worker join. */
void ps5_ai_v3_worker_join_receipt(int join_rc, int running, int queued, int listener_closed);
/* Returns0 only after model/context destruction and status reset completed. */
int ps5_ai_engine_shutdown_checked(char *error, size_t error_size);
#ifdef __cplusplus
}
#endif
#endif
