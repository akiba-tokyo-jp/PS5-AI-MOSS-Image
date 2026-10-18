/* SPDX-License-Identifier: MIT */
#ifndef PS5_AI_REFERENCE_IMAGE_H
#define PS5_AI_REFERENCE_IMAGE_H
#include <stddef.h>
#include <stdint.h>
#define PS5_AI_INPUT_MAX_BYTES (8U * 1024U * 1024U)
#define PS5_AI_INPUT_MAX_PIXELS (2048U * 2048U)
#define PS5_AI_INPUT_MAX_DIMENSION 2048U
#ifdef __cplusplus
extern "C" {
#endif
/* A successful decode returns owned RGB; caller frees it with free(). */
int ps5_ai_input_decode(const unsigned char *, size_t, unsigned char **, uint32_t *, uint32_t *);
int ps5_ai_input_read(const char *, unsigned char **, uint32_t *, uint32_t *);
int ps5_ai_input_write(const char *, const unsigned char *, uint32_t, uint32_t);
int ps5_ai_input_info(const char *, uint32_t *, uint32_t *, uint64_t *, char[65]);
#ifdef __cplusplus
}
#endif
#endif
