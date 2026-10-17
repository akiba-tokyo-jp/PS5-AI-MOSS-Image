/* SPDX-License-Identifier: MIT
 * Vulkan bookkeeping only. No allocation, permission or kernel operations.
 */
#ifndef PS5_AI_RADV_FLEX_HEAP_H
#define PS5_AI_RADV_FLEX_HEAP_H
#include <stdint.h>
#define PS5_AI_RADV_NO_FLEX_INDEX UINT32_MAX
/* pdev is RADV's internal physical-device identity, not an arbitrary address
 * to dereference. A single PS5 physical device is registered in this variant.
 */
uint32_t ps5_ai_radv_flex_type_index(const void *pdev);
uint32_t ps5_ai_radv_flex_heap_index(const void *pdev);
#endif
