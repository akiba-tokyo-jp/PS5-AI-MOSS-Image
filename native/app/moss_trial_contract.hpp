/* SPDX-License-Identifier: MIT -- service policy and legacy diagnostic claims. */
#pragma once
#include "moss_runtime_admission.h"
#include <atomic>
#include <cstdint>
#include <cstring>
#include "moss_sampler_trial.h"

#ifndef PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL
#define PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL 0
#endif
#if PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL != 0 && PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL != 1
#error PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL must be 0 or 1
#endif
#if PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL && !PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL
#error The quality comparison requires the explicit experimental small-trial build
#endif
#ifndef PS5_AI_MOSS_1024_COLD_WARM_TRIAL
#define PS5_AI_MOSS_1024_COLD_WARM_TRIAL 0
#endif
#if PS5_AI_MOSS_1024_COLD_WARM_TRIAL != 0 && PS5_AI_MOSS_1024_COLD_WARM_TRIAL != 1
#error PS5_AI_MOSS_1024_COLD_WARM_TRIAL must be 0 or 1
#endif
#if PS5_AI_MOSS_1024_COLD_WARM_TRIAL && (!PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL || PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL)
#error The1024 cold/warm trial is distinct from production and the256 quality trial
#endif

#ifndef PS5_AI_MOSS_PERFORMANCE_TRIAL
#define PS5_AI_MOSS_PERFORMANCE_TRIAL 0
#endif
#ifndef PS5_AI_MOSS_PERF_VARIANT
#define PS5_AI_MOSS_PERF_VARIANT 0
#endif
#if PS5_AI_MOSS_PERFORMANCE_TRIAL != 0 && PS5_AI_MOSS_PERFORMANCE_TRIAL != 1
#error PS5_AI_MOSS_PERFORMANCE_TRIAL must be 0 or 1
#endif
#if PS5_AI_MOSS_PERF_VARIANT < 0 || PS5_AI_MOSS_PERF_VARIANT > 2
#error PS5_AI_MOSS_PERF_VARIANT must be 0 baseline, 1 fp16, or 2 submit/fusion
#endif
#if PS5_AI_MOSS_PERFORMANCE_TRIAL && (!PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL || PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL || PS5_AI_MOSS_1024_COLD_WARM_TRIAL)
#error The performance trial requires its distinct experimental build
#endif
#if !PS5_AI_MOSS_PERFORMANCE_TRIAL && PS5_AI_MOSS_PERF_VARIANT != 0
#error A performance variant requires the explicit performance-trial build
#endif

static inline bool ps5_ai_moss_trial_request_valid(int width,int height,int steps,
                                                  double cfg,double flow_shift,
                                                  const char* sampler,int64_t seed=20261019) noexcept {
#if PS5_AI_MOSS_SERVICE_MODE
    return ps5_ai_moss_turbo_request_allowed(width,height,steps,cfg,flow_shift,sampler,seed);
#elif PS5_AI_MOSS_PERFORMANCE_TRIAL
    return ((width==256 && height==256) || (width==1024 && height==1024)) &&
           steps==PS5_AI_MOSS_DEFAULT_REQUEST_STEPS && cfg==PS5_AI_MOSS_DEFAULT_REQUEST_CFG && flow_shift==5.0 && seed==20261020 &&
           sampler && !std::strcmp(sampler,PS5_AI_MOSS_DEFAULT_REQUEST_SAMPLER);
#elif PS5_AI_MOSS_1024_COLD_WARM_TRIAL
    return width==1024 && height==1024 && steps==50 && cfg==3.0 && flow_shift==5.0 &&
           seed==20261020 && sampler && !std::strcmp(sampler,"euler");
#elif PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL
    (void)seed;
    return width==256 && height==256 && (steps==4 || (PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL && steps==50)) && cfg==3.0 && flow_shift==5.0 &&
#if PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL
           seed==20261019 &&
#endif
           sampler && !std::strcmp(sampler,"euler");
#elif PS5_AI_MOSS_RUNTIME_ADMITTED
    (void)width;(void)height;(void)steps;(void)cfg;(void)flow_shift;(void)sampler;(void)seed;return true;
#else
    (void)width;(void)height;(void)steps;(void)cfg;(void)flow_shift;(void)sampler;(void)seed;return false;
#endif
}

/* Separate helper preserves the original256 single-claim contract unchanged.
 * Reservation is consumed before inference; neither failure nor completion
 * resets it. The caller serializes jobs; CAS also bounds concurrent callers. */
static inline bool ps5_ai_moss_benchmark_trial_claim(std::atomic<unsigned>& attempts,int width,int height) noexcept {
#if PS5_AI_MOSS_1024_COLD_WARM_TRIAL
    if(width!=1024 || height!=1024)return false;
    unsigned observed=attempts.load(std::memory_order_acquire);
    while(observed<2) {
        if(attempts.compare_exchange_weak(observed,observed+1,std::memory_order_acq_rel,std::memory_order_acquire))return true;
    }
    return false;
#else
    (void)attempts;(void)width;(void)height;return false;
#endif
}

static inline bool ps5_ai_moss_trial_claim(std::atomic<bool>& claimed,int width,int height) noexcept {
#if PS5_AI_MOSS_1024_COLD_WARM_TRIAL || PS5_AI_MOSS_PERFORMANCE_TRIAL
    (void)claimed;(void)width;(void)height;return false;
#elif PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL
    if (width!=256 || height!=256) return false;
    bool unused=false;
    return claimed.compare_exchange_strong(unused,true,std::memory_order_acq_rel);
#elif PS5_AI_MOSS_RUNTIME_ADMITTED
    (void)claimed;(void)width;(void)height;return true;
#else
    (void)claimed;(void)width;(void)height;return false;
#endif
}

/* Performance-only bounded attempts. Invalid requests never consume a claim;
 * valid claims are consumed before inference and are never refunded/reset.
 * The serial client additionally requires a successful, visually checked256
 * result and idle queues before it may submit1024. This helper makes no claim
 * that a prior image, GPU admission or numerical readback has passed. */
static inline bool ps5_ai_moss_performance_trial_claim(std::atomic<unsigned>& attempts,
    int width,int height,int steps,double cfg,double flow_shift,
    const char* sampler,int64_t seed) noexcept {
#if PS5_AI_MOSS_SERVICE_MODE
    (void)attempts;return ps5_ai_moss_trial_request_valid(width,height,steps,cfg,flow_shift,sampler,seed);
#elif PS5_AI_MOSS_PERFORMANCE_TRIAL
    if(!ps5_ai_moss_trial_request_valid(width,height,steps,cfg,flow_shift,sampler,seed))return false;
    unsigned observed=attempts.load(std::memory_order_acquire);
    while(observed<4) {
        if(attempts.compare_exchange_weak(observed,observed+1,std::memory_order_acq_rel,std::memory_order_acquire))return true;
    }
    return false;
#else
    (void)attempts;(void)width;(void)height;(void)steps;(void)cfg;(void)flow_shift;(void)sampler;(void)seed;return false;
#endif
}
