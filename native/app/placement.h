/* SPDX-License-Identifier: MIT
 * Partition the existing reference full-address VA range, not physical quota.
 * Candidate 9GiB data profile; driver and data VA intervals stay disjoint.
 */
#ifndef PS5_AI_PERBO_DATA_PLACEMENT_H
#define PS5_AI_PERBO_DATA_PLACEMENT_H
#include <stdint.h>
#define PS5_AI_REFERENCE_VA_BASE UINT64_C(0x4000000000)
#define PS5_AI_REFERENCE_VA_BYTES (UINT64_C(256)<<30)
#define PS5_AI_DATA_VA_BYTES (UINT64_C(9)<<30)
#define PS5_AI_DATA_VA_END (PS5_AI_REFERENCE_VA_BASE+PS5_AI_REFERENCE_VA_BYTES)
#define PS5_AI_DATA_VA_BASE (PS5_AI_DATA_VA_END-PS5_AI_DATA_VA_BYTES)
#define PS5_AI_DRIVER_VA_BYTES (PS5_AI_REFERENCE_VA_BYTES-PS5_AI_DATA_VA_BYTES)
_Static_assert(PS5_AI_DATA_VA_BASE==UINT64_C(0x7dc0000000),"9GiB data hint interval");
_Static_assert(PS5_AI_DATA_VA_END==UINT64_C(0x8000000000),"Within reference512GiB boundary");
_Static_assert(PS5_AI_DATA_VA_BASE>UINT64_C(0x300000000),"Disjoint32-bit window");
_Static_assert(PS5_AI_DATA_VA_END<(UINT64_C(1)<<47),"GPU full-address bound");
_Static_assert(!(PS5_AI_DATA_VA_BASE%UINT64_C(2097152))&&!(PS5_AI_DRIVER_VA_BYTES%UINT64_C(2097152)),"2MiB metadata granules");
#endif
