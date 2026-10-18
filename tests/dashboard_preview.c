/* SPDX-License-Identifier: GPL-3.0-or-later. Sample display data; no inference. */

#include "dashboard.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>
int main(void){
 size_t cap=(size_t)PS5_AI_STATUS_FRAME_BYTES;unsigned char *guard=malloc(cap+128);assert(guard);memset(guard,0xa5,cap+128);unsigned char *frame=guard+64;
 struct ps5_ai_status_snapshot_v1 s={0};s.abi_bytes=sizeof(s);s.abi_version=1;s.process_id=1234;s.sequence=128;
 s.valid=PS5_AI_STATUS_MODEL|PS5_AI_STATUS_CLOCK|PS5_AI_STATUS_PHASE|PS5_AI_STATUS_DIRECT_POOL|PS5_AI_STATUS_DRIVER_ARENA|PS5_AI_STATUS_CPU_RSS|PS5_AI_STATUS_COUNTERS|PS5_AI_STATUS_LOGS;
 s.clock_unix_seconds=1792301633;s.phase=PS5_AI_STATUS_IDLE;s.cpu_rss_source=PS5_AI_STATUS_RSS_PEAK;s.cpu_rss_bytes=9098674176ULL;
 s.direct_reserved_bytes=6442450944ULL;s.direct_used_bytes=5141463040ULL;s.direct_peak_bytes=5372903424ULL;
 s.driver_reserved_bytes=33554432;s.driver_used_bytes=17219584;s.dispatches=137630;s.submits=29143;s.fences=29143;
 strcpy(s.model,"WAI-Illustrious-SDXL v17.0");
 strcpy(s.logs[0],"Generation completed / Vulkan UNet and TAESD");strcpy(s.logs[1],"Output PNG saved / ready for the next request");
 strcpy(s.logs[2],"Native API healthy / model remains loaded");strcpy(s.logs[3],"Authorization: Bearer SECRET_DO_NOT_DRAW");
 assert(ps5_ai_dashboard_draw(frame,&s,1,1));
 FILE *f=fopen("dashboard-idle.ppm","wb");assert(f);fprintf(f,"P6\n1920 1080\n255\n");
 for(unsigned y=0;y<1080;y++)for(unsigned x=0;x<1920;x++){uint32_t c;memcpy(&c,frame+((size_t)y*1920+x)*4,4);unsigned char rgb[]={(unsigned char)(c>>16),(unsigned char)(c>>8),(unsigned char)c};assert(fwrite(rgb,1,3,f)==3);}assert(!fclose(f));
 s.valid|=PS5_AI_STATUS_PROGRESS|PS5_AI_STATUS_JOB;s.phase=PS5_AI_STATUS_GENERATING;s.progress_basis_points=6250;strcpy(s.job_id,"preview-job-sample");
 strcpy(s.logs[0],"Sampling / processing diffusion steps");strcpy(s.logs[1],"GPU compute / waiting for completed fences");
 assert(ps5_ai_dashboard_draw(frame,&s,1,1));
 f=fopen("dashboard-generating.ppm","wb");assert(f);fprintf(f,"P6\n1920 1080\n255\n");
 for(unsigned y=0;y<1080;y++)for(unsigned x=0;x<1920;x++){uint32_t c;memcpy(&c,frame+((size_t)y*1920+x)*4,4);unsigned char rgb[]={(unsigned char)(c>>16),(unsigned char)(c>>8),(unsigned char)c};assert(fwrite(rgb,1,3,f)==3);}assert(!fclose(f));
 clock_t begin=clock();for(unsigned i=0;i<10;i++)assert(ps5_ai_dashboard_draw(frame,&s,0,0));double ms=(double)(clock()-begin)*1000.0/CLOCKS_PER_SEC/10;
 for(unsigned i=0;i<64;i++)assert(guard[i]==0xa5&&guard[cap+64+i]==0xa5);
 /* Missing telemetry must remain omitted by the same native renderer path. */
 s.valid=PS5_AI_STATUS_MODEL;assert(ps5_ai_dashboard_draw(frame,&s,0,0));
 for(unsigned i=0;i<64;i++)assert(guard[i]==0xa5&&guard[cap+64+i]==0xa5);
 printf("{\"CPU_tiled_render_ms_per_frame_host_only\":%.3f,\"frame_guard\":true,\"PS5_calls\":0,\"PS5_timing_measured\":false}\n",ms);free(guard);return 0;
}
