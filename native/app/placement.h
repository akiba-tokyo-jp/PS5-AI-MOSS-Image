/* SPDX-License-Identifier: MIT
 * Partition the existing reference full-address VA range, not physical quota.
 * Production keeps the requested12GiB profile. An explicit experimental build
 * may select10GiB; this changes VA partitioning, never native physical quota.
 */
#ifndef PS5_AI_PERBO_DATA_PLACEMENT_H
#define PS5_AI_PERBO_DATA_PLACEMENT_H
#include <stdint.h>
#define PS5_AI_REFERENCE_VA_BASE UINT64_C(0x4000000000)
#define PS5_AI_REFERENCE_VA_BYTES (UINT64_C(256)<<30)
#ifndef PS5_AI_MOSS_TRIAL_10G
#define PS5_AI_MOSS_TRIAL_10G 0
#endif
#if PS5_AI_MOSS_TRIAL_10G != 0 && PS5_AI_MOSS_TRIAL_10G != 1
#error PS5_AI_MOSS_TRIAL_10G must be0 or1
#endif
#if PS5_AI_MOSS_TRIAL_10G
#define PS5_AI_DATA_VA_BYTES (UINT64_C(10)<<30)
#else
#define PS5_AI_DATA_VA_BYTES (UINT64_C(12)<<30)
#endif
#define PS5_AI_DATA_REGION_BYTES (UINT64_C(1)<<30)
#define PS5_AI_DATA_VA_END (PS5_AI_REFERENCE_VA_BASE+PS5_AI_REFERENCE_VA_BYTES)
#define PS5_AI_DATA_VA_BASE (PS5_AI_DATA_VA_END-PS5_AI_DATA_VA_BYTES)
#define PS5_AI_DRIVER_VA_BYTES (PS5_AI_REFERENCE_VA_BYTES-PS5_AI_DATA_VA_BYTES)
#if PS5_AI_MOSS_TRIAL_10G
_Static_assert(PS5_AI_DATA_VA_BASE==UINT64_C(0x7d80000000),"Explicit10GiB trial data hint interval");
#else
_Static_assert(PS5_AI_DATA_VA_BASE==UINT64_C(0x7d00000000),"Production12GiB data hint interval");
#endif
_Static_assert(PS5_AI_REFERENCE_VA_BASE+PS5_AI_DRIVER_VA_BYTES==PS5_AI_DATA_VA_BASE,"Driver/data VA intervals remain disjoint");
_Static_assert(PS5_AI_DRIVER_VA_BYTES+PS5_AI_DATA_VA_BYTES==PS5_AI_REFERENCE_VA_BYTES,"Exact reference VA partition");
_Static_assert(PS5_AI_DATA_REGION_BYTES==(UINT64_C(1)<<30),"OneGiB regions retained for Qwen embedding claims");
_Static_assert(PS5_AI_DATA_VA_END==UINT64_C(0x8000000000),"Within reference512GiB boundary");
_Static_assert(PS5_AI_DATA_VA_BASE>UINT64_C(0x300000000),"Disjoint32-bit window");
_Static_assert(PS5_AI_DATA_VA_END<(UINT64_C(1)<<47),"GPU full-address bound");
_Static_assert(!(PS5_AI_DATA_VA_BASE%UINT64_C(2097152))&&!(PS5_AI_DRIVER_VA_BYTES%UINT64_C(2097152)),"2MiB metadata granules");
#endif
