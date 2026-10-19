/* SPDX-License-Identifier: MIT -- one explicit verification attempt per process. */
#pragma once
#include "moss_runtime_admission.h"
#include <atomic>
#include <cstdint>
#include <cstring>

#ifndef PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL
#define PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL 0
#endif
#if PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL != 0 && PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL != 1
#error PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL must be 0 or 1
#endif
#if PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL && !PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL
#error The quality comparison requires the explicit experimental small-trial build
#endif

static inline bool ps5_ai_moss_trial_request_valid(int width,int height,int steps,
                                                  double cfg,double flow_shift,
                                                  const char* sampler,int64_t seed=20261019) noexcept {
#if PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL
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

static inline bool ps5_ai_moss_trial_claim(std::atomic<bool>& claimed,int width,int height) noexcept {
#if PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL
    if (width!=256 || height!=256) return false;
    bool unused=false;
    return claimed.compare_exchange_strong(unused,true,std::memory_order_acq_rel);
#elif PS5_AI_MOSS_RUNTIME_ADMITTED
    (void)claimed;(void)width;(void)height;return true;
#else
    (void)claimed;(void)width;(void)height;return false;
#endif
}
