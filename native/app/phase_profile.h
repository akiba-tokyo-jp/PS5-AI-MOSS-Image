/* SPDX-License-Identifier: MIT -- CPU wall timing only; no GPU calls. */
#ifndef PS5_AI_PHASE_PROFILE_H
#define PS5_AI_PHASE_PROFILE_H
#include <stdint.h>
#include <time.h>
#include <errno.h>
#ifdef __cplusplus
extern "C" {
#endif
int ps5_ai_profile_enabled(void);
void ps5_ai_profile_add(unsigned metric,uint64_t ns,uint64_t bytes,int valid);
unsigned ps5_ai_profile_set_runner(unsigned group);
unsigned ps5_ai_profile_get_runner(void);
/* Actual selected matrix map after fallback; command recording, not completion. */
void ps5_ai_vk_quant_accumulation_selected(unsigned lhs_type,unsigned f16acc);
#ifdef __cplusplus
}
#endif
#define PS5_AI_PROFILE_METRICS 42u
static inline uint64_t ps5_ai_profile_clock(void) {
    int saved=errno;struct timespec t;uint64_t value=0;
    if(ps5_ai_profile_enabled() && !clock_gettime(CLOCK_MONOTONIC,&t) && t.tv_sec>=0 && t.tv_nsec>=0 && t.tv_nsec<1000000000 && (uint64_t)t.tv_sec<UINT64_MAX/1000000000u)
        value=(uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;
    errno=saved;return value;
}
static inline void ps5_ai_profile_finish(unsigned id,uint64_t begin,uint64_t bytes) {
    if(!begin)return;uint64_t end=ps5_ai_profile_clock();ps5_ai_profile_add(id,end>=begin?end-begin:0,bytes,end>=begin);
}
#ifdef __cplusplus
#include <cstring>
struct PS5AIProfileScope {
    unsigned id,previous_runner;uint64_t begin,bytes;bool runner_scope;
    PS5AIProfileScope(unsigned i,uint64_t b=0):id(i),previous_runner(2),begin(ps5_ai_profile_clock()),bytes(b),runner_scope(i<24 && i%8==4){
        if(runner_scope)previous_runner=ps5_ai_profile_set_runner(i/8);
    }
    ~PS5AIProfileScope(){ps5_ai_profile_finish(id,begin,bytes);if(runner_scope)ps5_ai_profile_set_runner(previous_runner);}
    PS5AIProfileScope(const PS5AIProfileScope&)=delete;
};
/* The caller's runner graph-call scope assigns nested Vulkan CPU timings.
 * Slot4 is the backend graph call; slots0–3 measure individual driver calls.
 * All include only CPU wall time and may overlap their parent graph metric. */
static inline unsigned ps5_ai_profile_vulkan_metric(unsigned slot){return (ps5_ai_profile_get_runner()==0?32u:37u)+slot;}
template<class F> auto ps5_ai_profile_call(unsigned id,uint64_t bytes,F&& f){PS5AIProfileScope scope(id,bytes);return f();}
static inline unsigned ps5_ai_profile_group(const char* name){return (!std::strncmp(name,"unet",4)||!std::strncmp(name,"anima",5)||!std::strcmp(name,"MOSS-Image"))?0:!std::strncmp(name,"control_net",11)?1:2;}
#endif
#endif
