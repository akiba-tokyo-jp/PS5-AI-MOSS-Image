/* SPDX-License-Identifier: MIT */
#ifndef PS5_AI_MOSS_LORA_RECEIPT_H
#define PS5_AI_MOSS_LORA_RECEIPT_H
#include <stdint.h>
struct ps5_ai_moss_lora_receipt {
    uint32_t loaded, matched, used, skipped;
    float strength;
    int configured, application_verified;
};
#ifdef __cplusplus
extern "C" {
#endif
int ps5_ai_moss_lora_snapshot(struct ps5_ai_moss_lora_receipt *);
#ifdef __cplusplus
}
#endif
#endif
