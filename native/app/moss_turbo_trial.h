/* SPDX-License-Identifier: MIT -- proposed, not enabled in a running build. */
#ifndef PS5_AI_MOSS_TURBO_TRIAL_H
#define PS5_AI_MOSS_TURBO_TRIAL_H
#include <stdint.h>
#include <string.h>
#ifndef PS5_AI_MOSS_TURBO_TRIAL
#define PS5_AI_MOSS_TURBO_TRIAL 0
#endif
#if PS5_AI_MOSS_TURBO_TRIAL != 0 && PS5_AI_MOSS_TURBO_TRIAL != 1
#error PS5_AI_MOSS_TURBO_TRIAL must be0or1
#endif
#if PS5_AI_MOSS_TURBO_TRIAL && (!PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL || !PS5_AI_MOSS_TRIAL_10G || !PS5_AI_MOSS_PERFORMANCE_TRIAL || PS5_AI_MOSS_PERF_VARIANT != 1 || !PS5_AI_MOSS_F32_ACCUMULATION || PS5_AI_MOSS_GPU_PROFILE || PS5_AI_MOSS_Q4_BK64_RAW64 || PS5_AI_MOSS_SAMPLER_TRIAL || PS5_AI_MOSS_SPECTRUM_TRIAL || PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL || PS5_AI_MOSS_RUNTIME_ADMITTED)
#error Turbo requires a separate F32acc small10GiB trial with sampler/B/profiling/production disabled
#endif
#if PS5_AI_MOSS_TURBO_TRIAL
#define PS5_AI_MOSS_TURBO_POLICY "euler_custom1000_910_717_24_cfg1_nsfw07_turbo1_rank144_v1"
#else
#define PS5_AI_MOSS_TURBO_POLICY "disabled"
#endif
#define PS5_AI_MOSS_TURBO_ADAPTER_BASENAME "moss-nsfw07-turbo4-r144-f32.safetensors"
#define PS5_AI_MOSS_TURBO_ADAPTER_SCHEMA "ps5-ai-moss-nsfw07-turbo1-r144-f32-v1"
#define PS5_AI_MOSS_TURBO_NSFW_SHA256 "907c1e0220c8aa301595b4a369f7b09447001306ffebb53da1fc58f24542cdf9"
#define PS5_AI_MOSS_TURBO_SOURCE_SHA256 "154c7f1455c3bf9925945c2f1ea2e4f09d0d0c67256081384a094b2b585f913f"
enum { PS5_AI_MOSS_TURBO_STEPS=4, PS5_AI_MOSS_TURBO_SIGMA_COUNT=5,
       PS5_AI_MOSS_TURBO_DIT_FORWARDS=4, PS5_AI_MOSS_TURBO_Q4_F32_RECORDINGS=672,
       PS5_AI_MOSS_TURBO_LORA_RANK=144, PS5_AI_MOSS_TURBO_LORA_TENSORS=504 };
/* These are already the author's shifted-and-training-grid-snapped sigmas.
 * Assign directly to sd_sample_params_t.custom_sigmas and custom_sigmas_count;
 * never pass them through SimpleScheduler/FlowMatch shift a second time.
 * Caller-owned storage lives through the synchronous generate_image call.
 */
static inline void ps5_ai_moss_turbo_sigmas(float out[PS5_AI_MOSS_TURBO_SIGMA_COUNT]) {
    out[0]=1.0f; out[1]=0.910f; out[2]=0.717f; out[3]=0.024f; out[4]=0.0f;
}
static inline int ps5_ai_moss_turbo_request_allowed(int width,int height,int steps,
        double cfg,double flow,const char *sampler,int64_t seed) {
#if PS5_AI_MOSS_TURBO_TRIAL
    return ((width==256&&height==256)||(width==1024&&height==1024)) && steps==4 &&
        cfg==1.0 && flow==5.0 && seed==INT64_C(20261020) && sampler && !strcmp(sampler,"euler");
#else
    (void)width;(void)height;(void)steps;(void)cfg;(void)flow;(void)sampler;(void)seed;
    return 0;
#endif
}
#endif
