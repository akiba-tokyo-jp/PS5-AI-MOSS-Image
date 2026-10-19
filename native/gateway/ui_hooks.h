/* SPDX-License-Identifier: GPL-3.0-or-later
 * Optional TV state copies only. Never calls an engine, driver or renderer
 * from server/SD callbacks. Publication locks are try-only and bounded. */
#ifndef PS5_AI_NATIVE_ANIMA_UI_HOOKS_H
#define PS5_AI_NATIVE_ANIMA_UI_HOOKS_H
#include "status.h"
#ifdef __cplusplus
extern "C" {
#endif
void ps5_ai_native_ui_phase(unsigned phase);
void ps5_ai_native_ui_job_begin(const char *job_id);
void ps5_ai_native_ui_sampler_progress(unsigned basis_points);
void ps5_ai_native_ui_log(const char *nonsecret_line);
#ifdef __cplusplus
}
#endif
#endif
