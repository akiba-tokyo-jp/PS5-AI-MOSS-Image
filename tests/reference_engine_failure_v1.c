/* SPDX-License-Identifier: MIT -- HOST-ONLY; no model inference or PNG creation. */
#include "engine.h"
#include "v3_shutdown.h"
#include <stdio.h>
#include <string.h>
#include <windows.h>
int ps5_ai_engine_init(const char *p,char *e,size_t n){(void)p;(void)e;(void)n;return 0;}
void ps5_ai_engine_get_status(ps5_ai_engine_status *s){memset(s,0,sizeof(*s));s->available=s->gpu_available=1;strcpy(s->backend,"HOST FAILURE FIXTURE - NO INFERENCE");strcpy(s->model,"/app0/models/waiIllustriousSDXL_v170.safetensors");}
int ps5_ai_engine_generate(const ps5_ai_request *r,const char *p,ps5_ai_stats *s,ps5_ai_progress_cb cb,void *u,char *e,size_t n){
    (void)r;(void)p;memset(s,0,sizeof(*s));if(cb)cb(.1,u);Sleep(1000);snprintf(e,n,"Expected host-only failure; no inference performed");return -1;
}
void ps5_ai_engine_shutdown(void){}
int ps5_ai_engine_shutdown_checked(char *e,size_t n){(void)e;(void)n;return 0;}
void ps5_ai_native_ui_phase(unsigned p){(void)p;}
void ps5_ai_native_ui_job_begin(const char *p){(void)p;}
void ps5_ai_native_ui_log(const char *p){(void)p;}
void ps5_ai_v3_shutdown_request_accepted(int r,int q){(void)r;(void)q;}
void ps5_ai_v3_worker_join_receipt(int j,int r,int q,int c){(void)j;(void)r;(void)q;(void)c;}
