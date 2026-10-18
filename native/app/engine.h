#ifndef PS5_AI_ENGINE_H
#define PS5_AI_ENGINE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Every entry point runs in the same process. No shell or child process.
 * get_status must be thread safe and must not initialize/load large resources.
 * generate is called by one worker only. Return 0 only on a complete PNG.
 * gpu_used must reflect observed inference dispatch, not device enumeration.
 */
typedef struct ps5_ai_request {
    char prompt[4097];
    char negative_prompt[4097];
    int width, height, steps;
    double cfg_scale;
    int64_t seed;
    char sampler[32];
    double manga_strength; /* Default 1 for existing requests; native override 0..1.5. */
} ps5_ai_request;

typedef struct ps5_ai_engine_status {
    int available;
    int gpu_available;
    int model_loaded;
    int gpu_verified;
    char backend[64];
    char device[256];
    char driver[256];
    char model[512];
    char reason[512];
} ps5_ai_engine_status;

typedef struct ps5_ai_stats {
    int gpu_used;
    double generation_seconds;
    uint64_t peak_cpu_bytes;
    uint64_t peak_gpu_bytes;
} ps5_ai_stats;

typedef void (*ps5_ai_progress_cb)(double progress, void *user);

/* Return 0 on configured engine; nonzero describes unavailability via status.
 * init is called once before worker creation; config_path may not exist.
 */
int ps5_ai_engine_init(const char *config_path, char *error, size_t error_size);
void ps5_ai_engine_get_status(ps5_ai_engine_status *status);
int ps5_ai_engine_generate(const ps5_ai_request *request,
    const char *output_path, ps5_ai_stats *stats,
    ps5_ai_progress_cb progress, void *user, char *error, size_t error_size);
void ps5_ai_engine_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif
