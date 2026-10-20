/* SPDX-License-Identifier: MIT -- isolated, default-off weight-upload trial. */
#ifndef PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_H
#define PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_H
#ifndef PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL
#define PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL 0
#endif
#if PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL != 0 && PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL != 1
#error PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL must be 0 or 1
#endif
#if PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL && (!PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL || !PS5_AI_MOSS_TRIAL_10G || !PS5_AI_MOSS_PERFORMANCE_TRIAL || PS5_AI_MOSS_PERF_VARIANT != 1 || !PS5_AI_ENABLE_FP16 || !PS5_AI_MOSS_F32_ACCUMULATION || PS5_AI_MOSS_RUNTIME_ADMITTED || PS5_AI_MOSS_GPU_PROFILE || PS5_AI_MOSS_Q4_BK64_RAW64 || PS5_AI_MOSS_SAMPLER_TRIAL || PS5_AI_MOSS_SPECTRUM_TRIAL || PS5_AI_MOSS_TURBO_TRIAL)
#error Bounded upload requires isolated FP16/F32 small10GiB performance trial with production, sampler, Spectrum, GPU profiling and Q4 candidates disabled
#endif
#if PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL
#define PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_POLICY "raw_same_format_GPU_weights_8MiB_per_worker_v1"
#else
#define PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_POLICY "disabled"
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* Always defined by the native app; true only after the fixed GPU gate passes. */
int ps5_ai_moss_bounded_weight_upload_enabled(void);
#ifdef __cplusplus
}
#endif
#endif
