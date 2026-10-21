/* SPDX-License-Identifier: MIT -- ordinary serial API operation. */
#ifndef PS5_AI_MOSS_SERVICE_MODE_H
#define PS5_AI_MOSS_SERVICE_MODE_H
#include <stdint.h>
#ifndef PS5_AI_MOSS_SERVICE_MODE
#define PS5_AI_MOSS_SERVICE_MODE 0
#endif
#if PS5_AI_MOSS_SERVICE_MODE != 0 && PS5_AI_MOSS_SERVICE_MODE != 1
#error MOSS service mode must be zero or one
#endif
#if PS5_AI_MOSS_SERVICE_MODE
#define PS5_AI_MOSS_ATTEMPT_LIMIT_JSON "null"
#elif PS5_AI_MOSS_PERFORMANCE_TRIAL
#define PS5_AI_MOSS_ATTEMPT_LIMIT_JSON "4"
#elif PS5_AI_MOSS_1024_COLD_WARM_TRIAL
#define PS5_AI_MOSS_ATTEMPT_LIMIT_JSON "2"
#elif PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL
#define PS5_AI_MOSS_ATTEMPT_LIMIT_JSON "1"
#else
#define PS5_AI_MOSS_ATTEMPT_LIMIT_JSON "null"
#endif
/* The Turbo adapter's authored four-step sampling recipe is a model profile,
 * separate from obsolete verification-only shape/seed/attempt restrictions. */
static inline int ps5_ai_moss_service_geometry(int w,int h) {
    return w>=64 && h>=64 && w<=1024 && h<=1024 && !(w%16) && !(h%16);
}
/* Image seeds are public reproduction parameters, not security credentials.
 * Mix a monotonic timestamp, process identity and serial counter for seed=-1.
 * Bearer tokens and signed URLs retain their independent existing controls. */
static inline int64_t ps5_ai_moss_new_image_seed(uint64_t tick,uint64_t process,uint64_t serial) {
    uint64_t v=tick^(process<<32)^(serial*UINT64_C(0x9e3779b97f4a7c15));
    v=(v^(v>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    v=(v^(v>>27))*UINT64_C(0x94d049bb133111eb);
    return (int64_t)((v^(v>>31))&UINT64_C(0x7fffffffffffffff));
}
#endif
