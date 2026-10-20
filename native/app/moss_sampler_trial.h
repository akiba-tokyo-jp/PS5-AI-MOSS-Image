/* SPDX-License-Identifier: MIT -- explicit native/gateway sampler trial only. */
#ifndef PS5_AI_MOSS_SAMPLER_TRIAL_H
#define PS5_AI_MOSS_SAMPLER_TRIAL_H
#include "moss_turbo_trial.h"
#include <string.h>
#include <stdint.h>
#ifndef PS5_AI_MOSS_SAMPLER_TRIAL
#define PS5_AI_MOSS_SAMPLER_TRIAL 0
#endif
#if PS5_AI_MOSS_SAMPLER_TRIAL != 0 && PS5_AI_MOSS_SAMPLER_TRIAL != 1
#error PS5_AI_MOSS_SAMPLER_TRIAL must be 0 or 1
#endif
#if PS5_AI_MOSS_SAMPLER_TRIAL && (!PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL || !PS5_AI_MOSS_TRIAL_10G || !PS5_AI_MOSS_PERFORMANCE_TRIAL || PS5_AI_MOSS_PERF_VARIANT != 1 || !PS5_AI_MOSS_F32_ACCUMULATION || PS5_AI_MOSS_GPU_PROFILE || PS5_AI_MOSS_Q4_BK64_RAW64 || PS5_AI_MOSS_RUNTIME_ADMITTED || PS5_AI_MOSS_TURBO_TRIAL || PS5_AI_MOSS_SPECTRUM_TRIAL || PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL)
#error Sampler trial requires the separate FP16/F32 small10GiB performance trial with production, GPU profiling and Q4 candidates disabled
#endif
#if PS5_AI_MOSS_TURBO_TRIAL
#define PS5_AI_MOSS_SAMPLER_POLICY "disabled"
#define PS5_AI_MOSS_DEFAULT_REQUEST_STEPS 4
#define PS5_AI_MOSS_DEFAULT_REQUEST_CFG 1.0
#define PS5_AI_MOSS_DEFAULT_REQUEST_SAMPLER "euler"
#define PS5_AI_MOSS_REQUEST_SCHEDULER "custom_sigmas_author_training_grid"
#define PS5_AI_MOSS_DEFAULT_REQUEST_SEED INT64_C(20261020)
#elif PS5_AI_MOSS_SAMPLER_TRIAL
#define PS5_AI_MOSS_SAMPLER_POLICY "res_multistep_simple_30_cfg4_v1"
#define PS5_AI_MOSS_DEFAULT_REQUEST_STEPS 30
#define PS5_AI_MOSS_DEFAULT_REQUEST_CFG 4.0
#define PS5_AI_MOSS_DEFAULT_REQUEST_SAMPLER "res_multistep"
#define PS5_AI_MOSS_REQUEST_SCHEDULER "simple"
#define PS5_AI_MOSS_DEFAULT_REQUEST_SEED INT64_C(20261020)
#else
#define PS5_AI_MOSS_SAMPLER_POLICY "disabled"
#define PS5_AI_MOSS_DEFAULT_REQUEST_STEPS 50
#define PS5_AI_MOSS_DEFAULT_REQUEST_CFG 3.0
#define PS5_AI_MOSS_DEFAULT_REQUEST_SAMPLER "euler"
#define PS5_AI_MOSS_REQUEST_SCHEDULER "discrete"
#define PS5_AI_MOSS_DEFAULT_REQUEST_SEED INT64_C(-1)
#endif
static inline int ps5_ai_moss_sampler_name_allowed(const char *name) {
    if(!name)return 0;
#if PS5_AI_MOSS_TURBO_TRIAL
    return !strcmp(name,"euler");
#elif PS5_AI_MOSS_SAMPLER_TRIAL
    return !strcmp(name,"res_multistep");
#else
    return !strcmp(name,"euler") || !strcmp(name,"euler_a") || !strcmp(name,"dpm++2m");
#endif
}
static inline int ps5_ai_moss_sampler_request_allowed(int width,int height,int steps,double cfg,
                                                     double flow,const char *sampler,int64_t seed) {
#if PS5_AI_MOSS_TURBO_TRIAL
    return ps5_ai_moss_turbo_request_allowed(width,height,steps,cfg,flow,sampler,seed);
#elif PS5_AI_MOSS_SAMPLER_TRIAL
    return ((width==256&&height==256)||(width==1024&&height==1024)) && steps==30 &&
           cfg==4.0 && flow==5.0 && seed==INT64_C(20261020) && ps5_ai_moss_sampler_name_allowed(sampler);
#else
    (void)width;(void)height;(void)steps;(void)cfg;(void)flow;(void)sampler;(void)seed;return 1;
#endif
}
#endif
