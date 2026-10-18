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
    double manga_strength; /* Default 1, native/OpenAI extension 0..1.5. */
    int manga_monochrome; /* Default false; opt in to automatic prompt prefix. */
    char init_image_id[33], reference_image_id[33];
    double denoise_strength, ip_adapter_strength;
    /* Resolved by authenticated native admission, never accepted as client paths. */
    char init_image_path[1024], reference_image_path[1024];
    char init_image_sha256[65], reference_image_sha256[65];
    int transient_inputs; /* Internal gateway ownership; not a JSON request field. */
    char controlnet[16], control_image_id[33];
    char control_image_path[1024], control_image_sha256[65];
    double control_strength;
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
    int actual_steps, ip_adapter_tokens, gpu_ip_attention_verified;
    double init_encode_seconds, ip_adapter_seconds;
    int gpu_controlnet_verified, controlnet_evals, controlnet_tensors;
    double controlnet_seconds;
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
