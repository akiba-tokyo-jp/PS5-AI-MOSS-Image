/* SPDX-License-Identifier: MIT */
#ifndef PS5_AI_MODEL_SHA256_H
#define PS5_AI_MODEL_SHA256_H
#include <stddef.h>
#include <stdint.h>
struct ai_sha256 { uint32_t state[8]; uint64_t bytes; uint8_t block[64]; size_t used; };
void ai_sha256_init(struct ai_sha256 *ctx);
void ai_sha256_update(struct ai_sha256 *ctx, const void *data, size_t size);
void ai_sha256_final(struct ai_sha256 *ctx, uint8_t digest[32]);
#endif
