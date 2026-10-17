/* SPDX-License-Identifier: GPL-3.0-or-later
 * CPU tiling formula from pinned psllama / PS5_Vulkan; no console calls. */
#ifndef PS5_AI_STATUS_RASTER_V1_H
#define PS5_AI_STATUS_RASTER_V1_H
#include <stddef.h>
#include <stdint.h>
#define PS5_AI_STATUS_WIDTH 1920u
#define PS5_AI_STATUS_HEIGHT 1080u
static inline size_t ps5_ai_status_pixel_offset(unsigned x,unsigned y) {
    uint32_t offset=((y<<4)&0x70u)^((y<<5)&0xf00u)^((y<<9)&0x1000u)^((y<<8)&0x4000u)^
                    ((x<<2)&0xcu)^((x<<5)&0x380u)^((x<<4)&0x400u)^((x<<6)&0x800u)^((x<<9)&0xa000u);
    return ((size_t)((y>>7)*((PS5_AI_STATUS_WIDTH+127u)>>7)+(x>>7))<<16)+offset;
}
#endif
