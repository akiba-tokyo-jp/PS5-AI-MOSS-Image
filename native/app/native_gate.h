/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef PS5_AI_NATIVE_DIRECT_GPU_GATE_H
#define PS5_AI_NATIVE_DIRECT_GPU_GATE_H
/* Fixed own log locations only. Call and check before any explicit AGC init. */
int ps5_ai_native_arena_open_log(int sandbox);
const char *ps5_ai_native_arena_log_path(void);
#endif
