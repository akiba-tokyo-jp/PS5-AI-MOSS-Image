/* SPDX-License-Identifier: MIT -- CPU-only gateway; no native display/GPU owner. */
#include "v3_shutdown.h"
void ps5_ai_native_ui_phase(unsigned p){(void)p;}
void ps5_ai_native_ui_job_begin(const char *s){(void)s;}
void ps5_ai_native_ui_sampler_progress(unsigned p){(void)p;}
void ps5_ai_native_ui_log(const char *s){(void)s;}
void ps5_ai_v3_shutdown_request_accepted(int r,int q){(void)r;(void)q;}
void ps5_ai_v3_worker_join_receipt(int j,int r,int q,int c){(void)j;(void)r;(void)q;(void)c;}
