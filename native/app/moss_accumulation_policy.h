/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <stdint.h>
#ifndef PS5_AI_MOSS_F32_ACCUMULATION
#define PS5_AI_MOSS_F32_ACCUMULATION 0
#endif
#if PS5_AI_MOSS_F32_ACCUMULATION != 0 && PS5_AI_MOSS_F32_ACCUMULATION != 1
#error PS5_AI_MOSS_F32_ACCUMULATION must be 0 or 1
#endif
#if PS5_AI_MOSS_F32_ACCUMULATION && (!PS5_AI_MOSS_PERFORMANCE_TRIAL || PS5_AI_MOSS_PERF_VARIANT != 1 || !PS5_AI_ENABLE_FP16)
#error Mixed MOSS accumulation requires the explicit FP16 performance candidate
#endif
struct ps5_ai_moss_quant_receipt { uint64_t q4_f32acc, q4_f16acc, invalid; };
#ifdef __cplusplus
extern "C" {
#endif
void ps5_ai_vk_quant_accumulation_selected(unsigned lhs_type,unsigned f16acc);
void ps5_ai_moss_quant_snapshot(struct ps5_ai_moss_quant_receipt *out);
#ifdef __cplusplus
}
#endif
