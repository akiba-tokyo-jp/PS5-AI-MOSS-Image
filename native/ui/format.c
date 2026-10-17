/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "status.h"
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

int ps5_ai_status_snapshot_valid(const struct ps5_ai_status_snapshot_v1 *s) {
    if(!s || s->abi_bytes!=sizeof(*s) || s->abi_version!=PS5_AI_STATUS_ABI ||
       s->valid&~8191u || s->phase>PS5_AI_STATUS_STOPPING || !s->process_id ||
       !memchr(s->model,0,sizeof(s->model)) || !memchr(s->job_id,0,sizeof(s->job_id)))return 0;
    for(unsigned i=0;i<4;i++)if(!memchr(s->logs[i],0,sizeof(s->logs[i])))return 0;
    if((s->valid&PS5_AI_STATUS_PROGRESS)&&s->progress_basis_points>10000)return 0;
    if((s->valid&PS5_AI_STATUS_DIRECT_POOL)&&(!s->direct_reserved_bytes ||
       s->direct_used_bytes>s->direct_peak_bytes || s->direct_peak_bytes>s->direct_reserved_bytes))return 0;
    if((s->valid&PS5_AI_STATUS_DRIVER_ARENA)&&s->driver_used_bytes>s->driver_reserved_bytes)return 0;
    return 1;
}
int ps5_ai_status_live_view(struct ps5_ai_status_snapshot_v1 *out,
                          const struct ps5_ai_status_snapshot_v1 *source,
                          uint64_t now_ns,int refresh_ok) {
    if(!out||!source||out==source)return 0;
    memset(out,0,sizeof(*out));if(!ps5_ai_status_snapshot_valid(source))return 0;
    *out=*source;
    if(!refresh_ok||!source->monotonic_ns||now_ns<source->monotonic_ns||
       now_ns-source->monotonic_ns>PS5_AI_STATUS_MAX_SAMPLE_AGE_NS) {
        out->valid&=PS5_AI_STATUS_MODEL;out->job_id[0]=0;
    }
    /* The owner obtains a new clock value each frame; never reuse an old one. */
    out->valid&=~PS5_AI_STATUS_CLOCK;return 1;
}
static void clean(char *out,size_t capacity,const char *in,size_t limit) {
    size_t n=0;
    while(n+1<capacity && n<limit && in[n]) {
        unsigned char c=(unsigned char)in[n];out[n]=(c>=32&&c<127)?(char)c:' ';n++;
    }
    out[n]=0;
}
static int private_line(const char *line) {
    char upper[96];clean(upper,sizeof(upper),line,95);
    for(char *p=upper;*p;p++)*p=(char)toupper((unsigned char)*p);
    return strstr(upper,"TOKEN") || strstr(upper,"BEARER") ||
           strstr(upper,"AUTHORIZATION") || strstr(upper,"SECRET");
}
static void row(char rows[][128],size_t *count,const char *format,...) {
    if(*count>=PS5_AI_STATUS_MAX_ROWS)return;
    va_list a;va_start(a,format);int n=vsnprintf(rows[*count],128,format,a);va_end(a);
    if(n>=0 && n<128)(*count)++;else rows[*count][0]=0;
}
size_t ps5_ai_status_format(const struct ps5_ai_status_snapshot_v1 *s,char rows[24][128]) {
    if(!rows)return 0;
    memset(rows,0,PS5_AI_STATUS_MAX_ROWS*128);
    if(!ps5_ai_status_snapshot_valid(s))return 0;
    size_t count=0;char text[128];
    static const char *phase[]={"IDLE","LOADING","GENERATING","COMPLETED","FAILED","STOPPING"};
    row(rows,&count,PS5_AI_STATUS_APP_NAME);
    if((s->valid&PS5_AI_STATUS_MODEL)&&s->model[0]) {
        clean(text,sizeof(text),s->model,sizeof(s->model)-1);row(rows,&count,"MODEL: %s",text);
    }
    if(s->valid&PS5_AI_STATUS_CLOCK) {
        time_t seconds=(time_t)s->clock_unix_seconds;struct tm date;
        if(s->clock_unix_seconds>0 && (int64_t)seconds==s->clock_unix_seconds && gmtime_r(&seconds,&date) &&
           strftime(text,sizeof(text),"PS5 UTC %Y-%m-%d %H:%M:%S",&date))row(rows,&count,"%s",text);
    }
    row(rows,&count,"PID: %u  DISPLAY HEARTBEAT: %llu",s->process_id,(unsigned long long)s->sequence);
    if(s->valid&PS5_AI_STATUS_PHASE)row(rows,&count,"STATE: %s",phase[s->phase]);
    if((s->valid&PS5_AI_STATUS_JOB)&&s->job_id[0]){clean(text,sizeof(text),s->job_id,32);row(rows,&count,"JOB: %s",text);}
    if(s->valid&PS5_AI_STATUS_PROGRESS)row(rows,&count,"PROGRESS: %u.%02u%%",s->progress_basis_points/100,s->progress_basis_points%100);
    if((s->valid&PS5_AI_STATUS_CPU_RSS)&&s->cpu_rss_source==PS5_AI_STATUS_RSS_CURRENT)
        row(rows,&count,"PROCESS CURRENT RSS: %llu B",(unsigned long long)s->cpu_rss_bytes);
    else if((s->valid&PS5_AI_STATUS_CPU_RSS)&&s->cpu_rss_source==PS5_AI_STATUS_RSS_PEAK)
        row(rows,&count,"PROCESS PEAK RSS: %llu B",(unsigned long long)s->cpu_rss_bytes);
    if((s->valid&PS5_AI_STATUS_CPU_BUSY)&&s->cpu_busy_source==PS5_AI_STATUS_CPU_CAPACITY_TIME_WINDOW&&
       s->cpu_capacity_ns&&s->cpu_busy_ns<=s->cpu_capacity_ns)
        row(rows,&count,"PROCESS CPU: %.2f%%",100.0*(double)s->cpu_busy_ns/(double)s->cpu_capacity_ns);
    if((s->valid&PS5_AI_STATUS_GPU_BUSY)&&s->gpu_busy_source==PS5_AI_STATUS_GPU_HARDWARE_BUSY_TIME_WINDOW&&
       s->gpu_elapsed_ns&&s->gpu_busy_ns<=s->gpu_elapsed_ns)
        row(rows,&count,"GPU BUSY: %.2f%%",100.0*(double)s->gpu_busy_ns/(double)s->gpu_elapsed_ns);
    if((s->valid&PS5_AI_STATUS_TEMPERATURE)&&s->temperature_millic>=-40000&&s->temperature_millic<=200000)
        row(rows,&count,"TEMPERATURE: %.1f C",s->temperature_millic/1000.0);
    if(s->valid&PS5_AI_STATUS_DIRECT_POOL) {
        row(rows,&count,"DIRECT DATA USED: %llu / %llu B (%.1f%% OF APP POOL)",
            (unsigned long long)s->direct_used_bytes,(unsigned long long)s->direct_reserved_bytes,
            100.0*(double)s->direct_used_bytes/(double)s->direct_reserved_bytes);
        row(rows,&count,"DIRECT DATA PEAK: %llu B",(unsigned long long)s->direct_peak_bytes);
    }
    if(s->valid&PS5_AI_STATUS_DRIVER_ARENA)row(rows,&count,"DRIVER ARENA USED: %llu / %llu B",(unsigned long long)s->driver_used_bytes,(unsigned long long)s->driver_reserved_bytes);
    if(s->valid&PS5_AI_STATUS_COUNTERS)row(rows,&count,"DISPATCH: %llu  SUBMIT: %llu  FENCE: %llu",(unsigned long long)s->dispatches,(unsigned long long)s->submits,(unsigned long long)s->fences);
    if(s->valid&PS5_AI_STATUS_LOGS)for(unsigned i=0;i<4;i++)if(s->logs[i][0]&&!private_line(s->logs[i])) {
        clean(text,sizeof(text),s->logs[i],95);row(rows,&count,"%s",text);
    }
    return count;
}
