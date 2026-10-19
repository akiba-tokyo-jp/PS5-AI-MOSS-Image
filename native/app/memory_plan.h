/* SPDX-License-Identifier: MIT
 * Configuration-only pre-AGC interface. No GPU, model or driver operation.
 * working_reserve_bytes is outside the pool's accounting admission; it is not
 * the model manager's GPU workspace budget.
 */
#ifndef PS5_AI_MEMORY_PLAN_H
#define PS5_AI_MEMORY_PLAN_H
#include <stddef.h>
#include <stdint.h>
#define PS5_AI_MEMORY_PLAN_DEFAULT_POOL_BYTES (UINT64_C(12) << 30)
#define PS5_AI_MEMORY_PLAN_DEFAULT_WORKING_RESERVE_BYTES (UINT64_C(1) << 30)
#define PS5_AI_MEMORY_PLAN_PAGE_BYTES UINT64_C(16384)
#define PS5_AI_MEMORY_PLAN_MAX_CONFIG_BYTES 65536u
struct ps5_ai_memory_plan {
    uint64_t gpu_pool_bytes;
    uint64_t working_reserve_bytes;
};
#ifdef __cplusplus
extern "C" {
#define PS5_AI_MEMORY_PLAN_NOEXCEPT noexcept
#else
#define PS5_AI_MEMORY_PLAN_NOEXCEPT
#endif
int ps5_ai_v12_read_memory_plan(const char *config_path, struct ps5_ai_memory_plan *out,
                              char *error, size_t error_size) PS5_AI_MEMORY_PLAN_NOEXCEPT;
int ps5_ai_native_gpu_data_gate(char *error, size_t error_size) PS5_AI_MEMORY_PLAN_NOEXCEPT;
#ifdef __cplusplus
}
static_assert(sizeof(ps5_ai_memory_plan) == 16, "Fixed C memory-plan ABI");
#endif
#undef PS5_AI_MEMORY_PLAN_NOEXCEPT
#endif
