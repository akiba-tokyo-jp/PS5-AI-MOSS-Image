/* SPDX-License-Identifier: MIT
 * Preparation build stays closed until actual native memory/operator/readback
 * admission has been proved. This is a build-time control, never a JSON option.
 */
#ifndef PS5_AI_MOSS_RUNTIME_ADMISSION_H
#define PS5_AI_MOSS_RUNTIME_ADMISSION_H
#ifndef PS5_AI_MOSS_RUNTIME_ADMITTED
#define PS5_AI_MOSS_RUNTIME_ADMITTED 0
#endif
#if PS5_AI_MOSS_RUNTIME_ADMITTED != 0 && PS5_AI_MOSS_RUNTIME_ADMITTED != 1
#error PS5_AI_MOSS_RUNTIME_ADMITTED must be 0 or 1
#endif
#define PS5_AI_MOSS_UNADMITTED_REASON "MOSS runtime not admitted: 12GiB Direct capacity, Qwen3.5 tensor residency and native GPU readback are unverified; no explicit AGC/data-pool initialization attempted"
#endif
