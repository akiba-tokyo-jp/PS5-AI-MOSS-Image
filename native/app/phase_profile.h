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
#ifdef __cplusplus
}
#endif
#define PS5_AI_PROFILE_METRICS 32u
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
    unsigned id;uint64_t begin,bytes;
    PS5AIProfileScope(unsigned i,uint64_t b=0):id(i),begin(ps5_ai_profile_clock()),bytes(b){}
    ~PS5AIProfileScope(){ps5_ai_profile_finish(id,begin,bytes);}
    PS5AIProfileScope(const PS5AIProfileScope&)=delete;
};
template<class F> auto ps5_ai_profile_call(unsigned id,uint64_t bytes,F&& f){PS5AIProfileScope scope(id,bytes);return f();}
static inline unsigned ps5_ai_profile_group(const char* name){return !std::strncmp(name,"unet",4)?0:!std::strncmp(name,"control_net",11)?1:2;}
#endif
#endif
