/* SPDX-License-Identifier: MIT */
#pragma once
#include "moss_turbo_trial.h"
#include "model/diffusion/moss_ffn_token_chunk.hpp"
#if PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL && (!PS5_AI_MOSS_TURBO_TRIAL || !PS5_AI_MOSS_F32_ACCUMULATION || !PS5_AI_ENABLE_FP16 || PS5_AI_MOSS_SPECTRUM_TRIAL || PS5_AI_MOSS_SAMPLER_TRIAL || PS5_AI_MOSS_GPU_PROFILE || PS5_AI_MOSS_Q4_BK64_RAW64 || PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL || PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL || PS5_AI_MOSS_RUNTIME_ADMITTED)
#error FFN chunking requires a separate admitted Turbo FP16/F32acc trial, with other compute changes off
#endif
// Bounded raw upload is independent of computation. Its combination remains
// subject to the parent's explicit reviewed Turbo+upload build identity.
#define PS5_AI_MOSS_FFN_CHUNK_POLICY "complete_FFN_tokens_balanced_concat_F32_rank144_v1"
