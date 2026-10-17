#pragma once
#include "ggml-alloc.h"
#include <stdint.h>

/* Read only the metadata produced by reserve_n_size(no_alloc=true).
 * This is an internal static-link API, not a native provider or GPU operation. */
#define GGML_PLAN_ADMISSION_MAX_CHUNKS 16
struct ggml_gallocr_plan_snapshot {
    uint32_t buffer_count;
    uint32_t chunk_count;
    uint64_t total_bytes;
    uint64_t max_chunk_bytes;
    uint64_t chunk_bytes[GGML_PLAN_ADMISSION_MAX_CHUNKS];
};
#if defined(__GNUC__)
#define GGML_PLAN_ADMISSION_LOCAL __attribute__((visibility("hidden")))
#else
#define GGML_PLAN_ADMISSION_LOCAL
#endif
#ifdef __cplusplus
extern "C" {
#endif
GGML_PLAN_ADMISSION_LOCAL bool ggml_gallocr_get_plan_snapshot(
    ggml_gallocr_t allocator, int buffer_index, struct ggml_gallocr_plan_snapshot * output);
#ifdef __cplusplus
}
#endif
