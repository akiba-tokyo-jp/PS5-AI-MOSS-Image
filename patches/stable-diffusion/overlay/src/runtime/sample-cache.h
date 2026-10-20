#ifndef __SD_RUNTIME_SAMPLE_CACHE_H__
#define __SD_RUNTIME_SAMPLE_CACHE_H__

#include <vector>

#include "core/tensor.hpp"
#include "core/util.h"
#include "model.h"
#include "runtime/cache_dit.hpp"
#include "runtime/denoiser.hpp"
#include "runtime/easycache.hpp"
#include "runtime/spectrum.hpp"
#ifndef PS5_AI_MOSS_SPECTRUM_TRIAL
#define PS5_AI_MOSS_SPECTRUM_TRIAL 0
#endif
#if PS5_AI_MOSS_SPECTRUM_TRIAL != 0 && PS5_AI_MOSS_SPECTRUM_TRIAL != 1
#error PS5_AI_MOSS_SPECTRUM_TRIAL must be 0 or 1
#endif
#ifndef PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL
#define PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL 0
#endif
#if PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL != 0 && PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL != 1
#error PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL must be 0 or 1
#endif
#if PS5_AI_MOSS_SPECTRUM_TRIAL && PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL
#error Spectrum and bounded weight upload require isolated trials
#endif
#if PS5_AI_MOSS_SPECTRUM_TRIAL
#if !PS5_AI_MOSS_F32_ACCUMULATION
#error MOSS Spectrum trial must retain explicit F32 matrix accumulation
#endif
#include "runtime/moss_spectrum.hpp"
#endif
#include "runtime/ucache.hpp"

namespace sd_sample {

    enum class SampleCacheMode {
        NONE,
        EASYCACHE,
        UCACHE,
        CACHEDIT,
    };

    struct SampleCacheRuntime {
        SampleCacheMode mode = SampleCacheMode::NONE;

        EasyCacheState easycache;
        UCacheState ucache;
        CacheDitConditionState cachedit;
        SpectrumState spectrum;

        bool spectrum_enabled = false;
#if PS5_AI_MOSS_SPECTRUM_TRIAL
        sd_moss_spectrum::State moss_spectrum;
        bool moss_spectrum_enabled = false;
#endif

        bool easycache_enabled() const;
        bool ucache_enabled() const;
        bool cachedit_enabled() const;
    };

    struct SampleStepCacheDispatcher {
        SampleCacheRuntime& runtime;
        int step;
        float sigma;
        int step_index;

        SampleStepCacheDispatcher(SampleCacheRuntime& runtime, int step, float sigma);

        bool before_condition(const void* condition, const sd::Tensor<float>& input, sd::Tensor<float>* output);
        void after_condition(const void* condition, const sd::Tensor<float>& input, const sd::Tensor<float>& output);
        void invalidate_condition(const void* condition);
        bool is_step_skipped() const;
    };

    SampleCacheRuntime init_sample_cache_runtime(SDVersion version,
                                                 const sd_cache_params_t* cache_params,
                                                 Denoiser* denoiser,
                                                 const std::vector<float>& sigmas);

    void log_sample_cache_summary(const SampleCacheRuntime& runtime, size_t total_steps);

}  // namespace sd_sample

#endif  // __SD_RUNTIME_SAMPLE_CACHE_H__
