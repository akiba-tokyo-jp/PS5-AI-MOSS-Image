/* SPDX-License-Identifier: MIT -- explicit native/gateway image-input trial. */
#ifndef PS5_AI_MOSS_IMG2IMG_TRIAL_H
#define PS5_AI_MOSS_IMG2IMG_TRIAL_H
#include <math.h>
#ifndef PS5_AI_MOSS_IMG2IMG_TRIAL
#define PS5_AI_MOSS_IMG2IMG_TRIAL 0
#endif
#if PS5_AI_MOSS_IMG2IMG_TRIAL != 0 && PS5_AI_MOSS_IMG2IMG_TRIAL != 1
#error MOSS img2img trial must be zero or one
#endif
#if PS5_AI_MOSS_IMG2IMG_TRIAL && !PS5_AI_MOSS_TURBO_TRIAL
#error MOSS img2img requires the explicit Turbo transport/inference parent
#endif
/* Match existing SD.cpp image.cpp: floor(4*float(strength))+1 forwards,
 * retaining a suffix of [1,.910,.717,.024,0]. A suffix starting at sigma1
 * discards the reference in flow noise_scaling, so it is not admitted here.
 * This is step-fraction strength, not a continuous noise-level parameter.
 */
static inline int ps5_ai_moss_img2img_steps(double strength) {
    const float s=(float)strength;
    if(!__builtin_isfinite(strength)||!__builtin_isfinite(s)||strength<0.01||s<0.01f||s>=0.75f)return 0;
    return (int)(4.0f*s)+1;
}
static inline float ps5_ai_moss_img2img_sigma(double strength) {
    const int steps=ps5_ai_moss_img2img_steps(strength);
    return steps==3?0.910f:steps==2?0.717f:steps==1?0.024f:0.0f;
}
#endif
