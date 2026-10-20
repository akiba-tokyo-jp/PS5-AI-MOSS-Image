/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#include "engine.h"
#include "moss_sampler_trial.h"
#include "reference_image.h"
#include "v3_shutdown.h"
#include "lifecycle.h"
#include "moss_lora_receipt.h"
#ifndef _WIN32
extern int ps5_ai_moss_lora_snapshot(struct ps5_ai_moss_lora_receipt *) __attribute__((weak));
#if PS5_AI_MOSS_TURBO_TRIAL
extern int ps5_ai_moss_turbo_identity_json(char*,size_t) __attribute__((weak));
#endif
#endif
#include "ui_hooks.h"
#include <ctype.h>
#include <stdatomic.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <direct.h>
#include <io.h>
#define O_NOFOLLOW 0
#define O_DIRECTORY 0
typedef SOCKET socket_t;
#define BAD_SOCKET INVALID_SOCKET
#define close_socket closesocket
#define make_dir(p) _mkdir(p)
#define get_pid() GetCurrentProcessId()
static CRITICAL_SECTION state_mutex, log_mutex;
static void lock_state(void) { EnterCriticalSection(&state_mutex); }
static void unlock_state(void) { LeaveCriticalSection(&state_mutex); }
static void lock_log(void) { EnterCriticalSection(&log_mutex); }
static void unlock_log(void) { LeaveCriticalSection(&log_mutex); }
static void pause_ms(unsigned ms) { Sleep(ms); }
#else
#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
typedef int socket_t;
#define BAD_SOCKET (-1)
#define close_socket close
#define make_dir(p) mkdir(p, 0700)
#define get_pid() getpid()
static pthread_mutex_t state_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
static void lock_state(void) { pthread_mutex_lock(&state_mutex); }
static void unlock_state(void) { pthread_mutex_unlock(&state_mutex); }
static void lock_log(void) { pthread_mutex_lock(&log_mutex); }
static void unlock_log(void) { pthread_mutex_unlock(&log_mutex); }
static void pause_ms(unsigned ms) {
    struct timespec t = {(time_t)(ms / 1000), (long)(ms % 1000) * 1000000};
    nanosleep(&t, NULL);
}
#endif

#define MAX_PATH_SIZE 1024
#define MAX_HEADER 8192
#define MAX_BODY 24576
#define MAX_JOBS 32
#define MAX_FIELDS 32 /* External request field limit remains unchanged. */
#define MAX_JOB_FIELDS 64 /* Bounded internal saved-job metadata capacity. */
#define MAX_JSON_STRING 8192
#define JOB_JSON_SIZE 8192
#define MOSS_IMAGE_INPUTS_SUPPORTED 0 /* No MOSS Image-compatible image-conditioning backend yet. */

typedef struct { char key[64]; char value[MAX_JSON_STRING + 1]; int type; } field;
/* type: 1 string, 2 JSON number, 3 true, 4 false, 5 null */
typedef struct { field values[MAX_JOB_FIELDS]; size_t count; } object;
typedef struct {
    char id[33], status[16], error[768];
    ps5_ai_request request;
    ps5_ai_stats stats;
    double progress;
    int64_t created, updated;
    int occupied;
} job;

static char root_path[MAX_PATH_SIZE] = "/app0/ai-runtime";
static char bind_ip[64] = "192.168.68.67";
static int listen_port = 8192;
static char api_token[257];
static size_t token_len;
static FILE *log_file;
#ifdef __PROSPERO__
extern void ps5_ai_native_server_stage(const char*,int,int);
#define STARTUP_STAGE(name,code) ps5_ai_native_server_stage(name,code,errno)
#else
#define STARTUP_STAGE(name,code) ((void)0)
#endif
static char bounded_api_log_path[MAX_PATH_SIZE];
#include "bounded_stdio_log.h"
static job jobs[MAX_JOBS];
static volatile sig_atomic_t stopping;
static volatile sig_atomic_t listener_reload;
static uint32_t sequence;
static int input_pin_request(ps5_ai_request *);
static void input_unpin_request(const ps5_ai_request *);
static struct in_addr bind_address;
static uint32_t listener_generation;
static int64_t server_started_at;
#include "retained_fault.inc"

static int finite_double(double value) { return value >= -DBL_MAX && value <= DBL_MAX; }
#if defined(__PROSPERO__) || defined(PS5_AI_LIFECYCLE_HOST_TEST)
static void lifecycle_json(char *out,size_t cap) {
    struct ps5_ai_native_lifecycle s;
    ps5_ai_native_lifecycle_snapshot(&s);
    snprintf(out,cap,"{\"initialization_stage\":\"%s\",\"failed_stage\":\"%s\",\"engine_ready\":%s,\"fault_retention_entered\":%s,\"in_app_shutdown\":\"%s\",\"constructor_effects\":\"unknown\",\"title_id\":\"PPSA99190\",\"process_id\":%" PRId64 ",\"external_fresh_owner_verification_required\":true,\"os_close_authorized\":false,\"manual_gpu_release\":false}",
        ps5_ai_lifecycle_phase_name(s.phase),s.failure_observed?ps5_ai_lifecycle_phase_name(s.failed_phase):"none",
        s.engine_ready?"true":"false",s.fault_retention_entered?"true":"false",
        ps5_ai_lifecycle_checked_shutdown_allowed(&s)?"checked_gpu_cleanup":"trusted_os_close_required",(int64_t)get_pid());
}
#endif
static void wipe_secret(void *data, size_t size) {
    volatile unsigned char *p = data;
    while (size--) *p++ = 0;
}

static void log_line(const char *format, ...) {
    char line[2048];va_list ap;
    lock_log();
    int prefix=snprintf(line,sizeof(line),"[%" PRId64 "] ",(int64_t)time(NULL));
    va_start(ap, format);
    int body=prefix>0 && (size_t)prefix<sizeof(line)-1?
        vsnprintf(line+prefix,sizeof(line)-(size_t)prefix-1,format,ap):-1;
    va_end(ap);
    if(body>=0 && prefix>0 && (size_t)body<sizeof(line)-(size_t)prefix-1) {
        size_t bytes=(size_t)prefix+(size_t)body;line[bytes++]='\n';
        if(!ps5_ai_bounded_stdio_append(&log_file,bounded_api_log_path,line,bytes,UINT64_C(2097152)))
            fputs("API journal unavailable; bounded append skipped\n",stderr);
    } else fputs("API journal oversized record skipped\n",stderr);
    unlock_log();
}

static int path_for(char *out, size_t size, const char *folder, const char *name) {
    int n = snprintf(out, size, "%s/%s/%s", root_path, folder, name);
    return n >= 0 && (size_t)n < size;
}

static void skip_space(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') ++*p;
}
static int hex_digit(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static int unicode_escape(const char **p, unsigned *value) {
    unsigned n = 0;
    for (int i = 0; i < 4; ++i) {
        int h = hex_digit((unsigned char)**p);
        if (h < 0) return 0;
        n = n * 16 + (unsigned)h; ++*p;
    }
    *value = n; return 1;
}
static int append_utf8(char *out, size_t cap, size_t *used, unsigned value) {
    unsigned char bytes[4]; size_t n;
    if (!value || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return 0;
    if (value < 128) { bytes[0] = (unsigned char)value; n = 1; }
    else if (value < 2048) {
        bytes[0] = (unsigned char)(0xc0 | (value >> 6));
        bytes[1] = (unsigned char)(0x80 | (value & 63)); n = 2;
    } else if (value < 65536) {
        bytes[0] = (unsigned char)(0xe0 | (value >> 12));
        bytes[1] = (unsigned char)(0x80 | ((value >> 6) & 63));
        bytes[2] = (unsigned char)(0x80 | (value & 63)); n = 3;
    } else {
        bytes[0] = (unsigned char)(0xf0 | (value >> 18));
        bytes[1] = (unsigned char)(0x80 | ((value >> 12) & 63));
        bytes[2] = (unsigned char)(0x80 | ((value >> 6) & 63));
        bytes[3] = (unsigned char)(0x80 | (value & 63)); n = 4;
    }
    if (*used + n >= cap) return 0;
    memcpy(out + *used, bytes, n); *used += n; return 1;
}
static int valid_utf8(const unsigned char *s) {
    while (*s) {
        unsigned v; int n;
        if (*s < 128) { ++s; continue; }
        if (*s >= 0xc2 && *s <= 0xdf) { v = *s & 31; n = 1; }
        else if (*s >= 0xe0 && *s <= 0xef) { v = *s & 15; n = 2; }
        else if (*s >= 0xf0 && *s <= 0xf4) { v = *s & 7; n = 3; }
        else return 0;
        ++s;
        for (int i = 0; i < n; ++i) {
            if ((*s & 0xc0) != 0x80) return 0;
            v = (v << 6) | (*s & 63); ++s;
        }
        if ((n == 1 && v < 128) || (n == 2 && v < 2048) ||
            (n == 3 && v < 65536) || v > 0x10ffff ||
            (v >= 0xd800 && v <= 0xdfff)) return 0;
    }
    return 1;
}
static int parse_string(const char **p, char *out, size_t cap) {
    size_t used = 0;
    if (*(*p)++ != '"') return 0;
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c < 32) return 0;
        if (c == '\\') {
            c = (unsigned char)*(*p)++;
            switch (c) {
            case '"': case '\\': case '/': break;
            case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case 'n': c = '\n'; break;
            case 'r': c = '\r'; break;
            case 't': c = '\t'; break;
            case 'u': {
                unsigned v;
                if (!unicode_escape(p, &v)) return 0;
                if (v >= 0xd800 && v <= 0xdbff) {
                    unsigned low;
                    if ((*p)[0] != '\\' || (*p)[1] != 'u') return 0;
                    *p += 2;
                    if (!unicode_escape(p, &low) || low < 0xdc00 || low > 0xdfff) return 0;
                    v = 0x10000 + ((v - 0xd800) << 10) + low - 0xdc00;
                }
                if (!append_utf8(out, cap, &used, v)) return 0;
                continue;
            }
            default: return 0;
            }
        }
        if (used + 1 >= cap) return 0;
        out[used++] = (char)c;
    }
    if (**p != '"') return 0;
    ++*p; out[used] = 0;
    return valid_utf8((const unsigned char *)out);
}
static int parse_number(const char **p, char *out, size_t cap) {
    const char *start = *p;
    if (**p == '-') ++*p;
    if (**p == '0') ++*p;
    else {
        if (**p < '1' || **p > '9') return 0;
        do { ++*p; } while (**p >= '0' && **p <= '9');
    }
    if (**p == '.') {
        ++*p; if (!isdigit((unsigned char)**p)) return 0;
        do { ++*p; } while (isdigit((unsigned char)**p));
    }
    if (**p == 'e' || **p == 'E') {
        ++*p; if (**p == '+' || **p == '-') ++*p;
        if (!isdigit((unsigned char)**p)) return 0;
        do { ++*p; } while (isdigit((unsigned char)**p));
    }
    size_t n = (size_t)(*p - start);
    if (!n || n >= cap) return 0;
    memcpy(out, start, n); out[n] = 0; return 1;
}
/* Strict flat JSON objects only. No dependency parser, nesting, or coercion. */
static int parse_object_limit(const char *p, object *obj, size_t limit) {
    if(limit>MAX_JOB_FIELDS)return 0;
    memset(obj, 0, sizeof(*obj)); skip_space(&p);
    if (*p++ != '{') return 0;
    skip_space(&p);
    if (*p == '}') { ++p; skip_space(&p); return *p == 0; }
    for (;;) {
        if (obj->count == limit || *p != '"') return 0;
        field *f = &obj->values[obj->count];
        if (!parse_string(&p, f->key, sizeof(f->key))) return 0;
        for (size_t i = 0; i < obj->count; ++i)
            if (!strcmp(f->key, obj->values[i].key)) return 0;
        skip_space(&p); if (*p++ != ':') return 0; skip_space(&p);
        if (*p == '"') {
            f->type = 1;
            if (!parse_string(&p, f->value, sizeof(f->value))) return 0;
        } else if (*p == '-' || isdigit((unsigned char)*p)) {
            f->type = 2;
            if (!parse_number(&p, f->value, sizeof(f->value))) return 0;
        } else if (!strncmp(p, "true", 4)) { f->type = 3; p += 4; }
        else if (!strncmp(p, "false", 5)) { f->type = 4; p += 5; }
        else if (!strncmp(p, "null", 4)) { f->type = 5; p += 4; }
        else return 0;
        ++obj->count; skip_space(&p);
        if (*p == '}') { ++p; skip_space(&p); return *p == 0; }
        if (*p++ != ',') return 0;
        skip_space(&p);
    }
}
static int parse_object(const char *p, object *obj) {return parse_object_limit(p,obj,MAX_FIELDS);}
static int parse_job_object(const char *p, object *obj) {return parse_object_limit(p,obj,MAX_JOB_FIELDS);}
static field *find_field(object *obj, const char *name) {
    for (size_t i = 0; i < obj->count; ++i)
        if (!strcmp(name, obj->values[i].key)) return &obj->values[i];
    return NULL;
}
static int get_int(field *f, int64_t min, int64_t max, int64_t *out) {
    char *end; long long v;
    if (!f || f->type != 2 || strchr(f->value, '.') ||
        strchr(f->value, 'e') || strchr(f->value, 'E')) return 0;
    errno = 0; v = strtoll(f->value, &end, 10);
    if (errno || *end || v < min || v > max) return 0;
    *out = (int64_t)v; return 1;
}
static int get_double(field *f, double min, double max, double *out) {
    char *end; double v;
    if (!f || f->type != 2) return 0;
    errno = 0; v = strtod(f->value, &end);
    if (errno || *end || !finite_double(v) || v < min || v > max) return 0;
    *out = v; return 1;
}
static int valid_id(const char *id);
static int validate_request(const char *body, ps5_ai_request *r, char *error, size_t cap) {
    object *obj = malloc(sizeof(*obj));
    if (!obj) { snprintf(error, cap, "Out of memory parsing request"); return 0; }
    int ok = 0; int64_t v;
    memset(r, 0, sizeof(*r));
    r->width = r->height = 1024; r->steps = PS5_AI_MOSS_DEFAULT_REQUEST_STEPS; r->cfg_scale = PS5_AI_MOSS_DEFAULT_REQUEST_CFG; r->flow_shift = 5; r->seed = PS5_AI_MOSS_DEFAULT_REQUEST_SEED;
    r->manga_strength = 0;
    r->control_strength=1;
    strcpy(r->sampler, PS5_AI_MOSS_DEFAULT_REQUEST_SAMPLER);
    if (!parse_object(body, obj)) { snprintf(error, cap, "Malformed, duplicate, nested, or oversized JSON field"); goto done; }
    for (size_t i = 0; i < obj->count; ++i) {
        field *f = &obj->values[i];
        if(!strcmp(f->key,"init_image_id") || !strcmp(f->key,"reference_image_id") || !strcmp(f->key,"mask_image_id") ||
           !strcmp(f->key,"denoise_strength") || !strcmp(f->key,"ip_adapter_strength") || !strcmp(f->key,"controlnet") ||
           !strcmp(f->key,"control_image_id") || !strcmp(f->key,"control_strength")){
            snprintf(error,cap,"MOSS Image SFT currently supports text-to-image only; %.63s is unsupported",f->key);goto done;
        }
        if (!strcmp(f->key,"model")) {
            if(f->type!=1 || strcmp(f->value,"moss-image-sft")){snprintf(error,cap,"Only model moss-image-sft is available");goto done;}
            continue;
        }
        if (!strcmp(f->key, "prompt") || !strcmp(f->key, "negative_prompt")) {
            if (f->type != 1 || strlen(f->value) > 4096) { snprintf(error, cap, "Prompts must be strings up to 4096 UTF-8 bytes"); goto done; }
            strcpy(!strcmp(f->key, "prompt") ? r->prompt : r->negative_prompt, f->value);
        } else if (!strcmp(f->key, "width") || !strcmp(f->key, "height")) {
            if (!get_int(f, 64, 1024, &v) || v % 16) { snprintf(error, cap, "Dimensions must be integers 64..1024 divisible by 16"); goto done; }
            if (!strcmp(f->key, "width")) r->width = (int)v; else r->height = (int)v;
        } else if (!strcmp(f->key, "steps")) {
            if (!get_int(f, 1, 100, &v)) { snprintf(error, cap, "steps must be an integer 1..100"); goto done; } r->steps = (int)v;
        } else if (!strcmp(f->key, "cfg_scale")) {
            if (!get_double(f, 0, 30, &r->cfg_scale)) { snprintf(error, cap, "cfg_scale must be finite and 0..30"); goto done; }
        } else if (!strcmp(f->key, "flow_shift")) {
            if (!get_double(f, 0, FLT_MAX, &r->flow_shift) || (float)r->flow_shift <= 0) {
                snprintf(error, cap, "flow_shift must remain a positive finite float"); goto done;
            }
        } else if (!strcmp(f->key, "manga_strength")) {
            if (!get_double(f, 0, 0, &r->manga_strength)) { snprintf(error, cap, "The SDXL manga LoRA is incompatible with MOSS Image; manga_strength must be 0"); goto done; }
        } else if (!strcmp(f->key, "manga_monochrome")) {
            if (f->type != 4) { snprintf(error, cap, "The SDXL manga LoRA is incompatible with MOSS Image; manga_monochrome must be false"); goto done; }
        } else if (!strcmp(f->key, "init_image_id") || !strcmp(f->key, "reference_image_id") || !strcmp(f->key, "mask_image_id")) {
            if(f->type != 1 || !valid_id(f->value)){snprintf(error,cap,"Image IDs must be32 lowercase hexadecimal characters");goto done;}
            strcpy(!strcmp(f->key,"init_image_id")?r->init_image_id:!strcmp(f->key,"reference_image_id")?r->reference_image_id:r->mask_image_id,f->value);
        } else if (!strcmp(f->key, "denoise_strength")) {
            if(!get_double(f,0.01,1,&r->denoise_strength)){snprintf(error,cap,"denoise_strength must be finite0.01..1");goto done;}
        } else if (!strcmp(f->key, "ip_adapter_strength")) {
            if(!get_double(f,0.01,1.5,&r->ip_adapter_strength)){snprintf(error,cap,"ip_adapter_strength must be finite0.01..1.5");goto done;}
        } else if (!strcmp(f->key,"controlnet")) {
            if(f->type!=1 || (strcmp(f->value,"canny") && strcmp(f->value,"openpose"))){snprintf(error,cap,"controlnet must be canny or openpose");goto done;}
            strcpy(r->controlnet,f->value);
        } else if (!strcmp(f->key,"control_image_id")) {
            if(f->type!=1 || !valid_id(f->value)){snprintf(error,cap,"Invalid control image ID");goto done;}
            strcpy(r->control_image_id,f->value);
        } else if (!strcmp(f->key,"control_strength")) {
            if(!get_double(f,0.01,1.5,&r->control_strength)){snprintf(error,cap,"control_strength must be finite0.01..1.5");goto done;}
        } else if (!strcmp(f->key, "seed")) {
            if (!get_int(f, -1, INT64_MAX, &r->seed)) { snprintf(error, cap, "seed must be an integer -1..9223372036854775807"); goto done; }
        } else if (!strcmp(f->key, "sampler")) {
            if (f->type != 1 || !ps5_ai_moss_sampler_name_allowed(f->value)) {
                snprintf(error, cap, "sampler is not allowed by the compiled MOSS profile"); goto done;
            } strcpy(r->sampler, f->value);
        } else { snprintf(error, cap, "Unknown request field: %.63s", f->key); goto done; }
    }
    if (!r->prompt[0]) { snprintf(error, cap, "prompt is required and cannot be empty"); goto done; }
    if((find_field(obj,"denoise_strength") && !r->init_image_id[0]) ||
       (find_field(obj,"ip_adapter_strength") && !r->reference_image_id[0])){
        snprintf(error,cap,"Image strength requires its matching input image ID");goto done;
    }
    if((r->controlnet[0]!=0)!=(r->control_image_id[0]!=0) || (find_field(obj,"control_strength") && !r->controlnet[0])){
        snprintf(error,cap,"ControlNet requires both its model name and prepared control image ID");goto done;
    }
    if(r->mask_image_id[0] && !r->init_image_id[0]){snprintf(error,cap,"Mask requires an init image");goto done;}
    if(!ps5_ai_moss_sampler_request_allowed(r->width,r->height,r->steps,r->cfg_scale,r->flow_shift,r->sampler,r->seed)){
        snprintf(error,cap,"Sampler trial requires256/1024 square,30steps,CFG4,shift5,res_multistep,seed20261020");goto done;
    }
    ok = 1;
done:
    free(obj); return ok;
}

static int quote_json(char *out, size_t size, const char *text) {
    size_t used = 0;
    if (size < 3) return 0;
    out[used++] = '"';
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p < 32) {
            if (used + 6 >= size) return 0;
            int n = snprintf(out + used, size - used, "\\u%04x", *p); used += (size_t)n;
        } else {
            if (used + 3 >= size) return 0;
            if (*p == '"' || *p == '\\') out[used++] = '\\';
            out[used++] = (char)*p;
        }
    }
    if (used + 2 > size) return 0;
    out[used++] = '"'; out[used] = 0; return 1;
}
static int serialize_job(const job *j, char *out, size_t cap) {
    char error[sizeof(j->error) * 6 + 3], url[80], cpu[32], gpu[32];
    if (!quote_json(error, sizeof(error), j->error)) return 0;
    if (!strcmp(j->status, "completed")) snprintf(url, sizeof(url), "\"/api/images/%s\"", j->id);
    else strcpy(url, "null");
    if (j->stats.peak_cpu_bytes) snprintf(cpu, sizeof(cpu), "%" PRIu64, j->stats.peak_cpu_bytes); else strcpy(cpu, "null");
    if (j->stats.peak_gpu_bytes) snprintf(gpu, sizeof(gpu), "%" PRIu64, j->stats.peak_gpu_bytes); else strcpy(gpu, "null");
    int n = snprintf(out, cap,
        "{\"job_id\":\"%s\",\"status\":\"%s\",\"progress\":%.4f,\"error\":%s,"
        "\"image_url\":%s,\"created_at\":%" PRId64 ",\"updated_at\":%" PRId64 ","
        "\"gpu_used\":%s,\"generation_seconds\":%.6f,\"peak_cpu_bytes\":%s,"
        "\"peak_gpu_bytes\":%s,\"flow_shift\":%.17g,\"mode\":\"%s\",\"manga_strength\":%.6f,\"manga_monochrome\":%s,"
        "\"init_image_sha256\":\"%s\",\"reference_image_sha256\":\"%s\","
        "\"denoise_strength\":%.6f,\"ip_adapter_strength\":%.6f,\"actual_steps\":%d,"
        "\"ip_adapter_tokens\":%d,\"gpu_ip_attention_verified\":%s,"
        "\"init_encode_seconds\":%.6f,\"ip_adapter_seconds\":%.6f,"
        "\"controlnet\":\"%s\",\"control_image_sha256\":\"%s\",\"control_strength\":%.6f,"
        "\"gpu_controlnet_verified\":%s,\"controlnet_evals\":%d,\"controlnet_tensors\":%d,\"controlnet_seconds\":%.6f,"
        "\"inpainting\":%s,\"mask_prepared\":%s,\"mask_image_sha256\":\"%s\",\"mask_denoise_evaluations\":%d,"
        "\"mask_preserved_pixels\":%" PRIu64 ",\"mask_changed_pixels\":%" PRIu64 "}", j->id, j->status, j->progress,
        error, url, j->created, j->updated, j->stats.gpu_used ? "true" : "false",
        j->stats.generation_seconds, cpu, gpu, j->request.flow_shift,
        j->request.init_image_id[0]?(j->request.reference_image_id[0]?"combined":"img2img"):(j->request.reference_image_id[0]?"ip_adapter":"txt2img"),
        j->request.manga_strength,j->request.manga_monochrome?"true":"false",j->request.init_image_sha256,j->request.reference_image_sha256,
        j->request.init_image_id[0]?j->request.denoise_strength:0,j->request.reference_image_id[0]?j->request.ip_adapter_strength:0,
        j->stats.actual_steps,j->stats.ip_adapter_tokens,j->stats.gpu_ip_attention_verified?"true":"false",
        j->stats.init_encode_seconds,j->stats.ip_adapter_seconds,
        j->request.controlnet,j->request.control_image_sha256,j->request.controlnet[0]?j->request.control_strength:0,
        j->stats.gpu_controlnet_verified?"true":"false",j->stats.controlnet_evals,j->stats.controlnet_tensors,j->stats.controlnet_seconds,
        j->request.mask_image_id[0]?"true":"false",j->stats.mask_prepared?"true":"false",j->request.mask_image_sha256,
        j->stats.mask_denoise_evaluations,j->stats.mask_preserved_pixels,j->stats.mask_changed_pixels);
    return n >= 0 && (size_t)n < cap;
}
static int atomic_write(const char *folder, const char *name, const char *data, size_t len) {
    char path[MAX_PATH_SIZE], temp[MAX_PATH_SIZE];
    if (!path_for(path, sizeof(path), folder, name)) return 0;
    int n = snprintf(temp, sizeof(temp), "%s.tmp", path);
    if (n < 0 || (size_t)n >= sizeof(temp)) return 0;
#ifdef _WIN32
    int fd = _open(temp, _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
    int fd = open(temp, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
#endif
    if (fd < 0) return 0;
    size_t sent = 0; int ok = 1;
    while (sent < len) {
#ifdef _WIN32
        int wrote = _write(fd, data + sent, (unsigned)(len - sent));
#else
        ssize_t wrote = write(fd, data + sent, len - sent);
#endif
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) { ok = 0; break; }
        sent += (size_t)wrote;
    }
#ifdef _WIN32
    if (_commit(fd)) ok = 0;
    if (_close(fd)) ok = 0;
#else
    if (fsync(fd)) ok = 0;
    if (close(fd)) ok = 0;
#endif
    if (!ok) { remove(temp); return 0; }
#ifdef _WIN32
    return MoveFileExA(temp, path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(temp, path) == 0;
#endif
}
static int persist_job(const job *j) {
    char data[JOB_JSON_SIZE], name[64];
    if (!serialize_job(j, data, sizeof(data))) return 0;
    snprintf(name, sizeof(name), "%s.json", j->id);
    return atomic_write("jobs", name, data, strlen(data));
}
static int valid_id(const char *id) {
    if (strlen(id) != 32) return 0;
    for (int i = 0; i < 32; ++i)
        if (!((id[i] >= '0' && id[i] <= '9') || (id[i] >= 'a' && id[i] <= 'f'))) return 0;
    return 1;
}
static int read_small_file(const char *path, char *out, size_t cap) {
    if (!path || !out || cap < 2) return 0;
    int fd=open(path,O_RDONLY|O_NOFOLLOW); if(fd<0) return 0;
    struct stat st; int ok=!fstat(fd,&st) && S_ISREG(st.st_mode) && st.st_size>=0 && (uint64_t)st.st_size<cap;
    size_t n=0;
    while(ok && n<cap-1) {
        ssize_t got=read(fd,out+n,cap-1-n);
        if(got<0 && errno==EINTR) continue;
        if(got<0) {ok=0;break;}
        if(!got) break;
        n+=(size_t)got;
    }
    char extra; ssize_t end;
    do {end=read(fd,&extra,1);} while(end<0 && errno==EINTR);
    if(end!=0) ok=0;
    if(close(fd)) ok=0;
    out[n]=0;
    return ok && !memchr(out,0,n);
}
/* Native path metadata: open the owned target, never follow symlinks, then
 * inspect its descriptor. No FILE layout macros or path stat/lstat calls. */
static int path_metadata(const char *path,struct stat *out,int directory) {
#ifdef _WIN32
    (void)directory;DWORD attrs=GetFileAttributesA(path);if(attrs==INVALID_FILE_ATTRIBUTES || (attrs&FILE_ATTRIBUTE_REPARSE_POINT))return -1;return stat(path,out);
#else
    int fd=open(path,O_RDONLY|O_NOFOLLOW|(directory?O_DIRECTORY:0));if(fd<0)return -1;
    struct {uint64_t pre[2];struct stat value;uint64_t post[2];} g;
    memset(&g,0,sizeof(g));g.pre[0]=g.pre[1]=g.post[0]=g.post[1]=UINT64_C(0x9e6c2503d171a845);
    int ok=!fstat(fd,&g.value) && g.pre[0]==UINT64_C(0x9e6c2503d171a845) && g.pre[1]==g.pre[0] && g.post[0]==g.pre[0] && g.post[1]==g.pre[0];
    if(ok)*out=g.value;if(close(fd))ok=0;return ok?0:-1;
#endif
}
static int load_job_json(const char *id, char *out, size_t cap) {
    char path[MAX_PATH_SIZE], name[64];
    if (!valid_id(id)) return 0;
    snprintf(name, sizeof(name), "%s.json", id);
    return path_for(path, sizeof(path), "jobs", name) && read_small_file(path, out, cap);
}
static void recover_one(const char *name) {
    size_t len = strlen(name);
    if (len != 37 || strcmp(name + 32, ".json")) return;
    char id[33], data[JOB_JSON_SIZE]; memcpy(id, name, 32); id[32] = 0;
    if (!load_job_json(id, data, sizeof(data))) return;
    object *obj = malloc(sizeof(*obj));
    if (!obj) return;
    if (parse_job_object(data, obj)) {
        field *state = find_field(obj, "status");
        if (state && (!strcmp(state->value, "running") || !strcmp(state->value, "queued"))) {
            job j; memset(&j, 0, sizeof(j)); strcpy(j.id, id); strcpy(j.status, "failed");
            strcpy(j.error, "Server restarted while job was pending; submit a new request");
            field *created = find_field(obj, "created_at");
            get_int(created, 0, INT64_MAX, &j.created); j.updated = (int64_t)time(NULL);
            if (!persist_job(&j)) log_line("Recovery persistence failed for %s", id);
            else log_line("Recovered interrupted job %s as failed", id);
        }
    }
    free(obj);
}
static void recover_jobs(void) {
    char path[MAX_PATH_SIZE]; if (!path_for(path, sizeof(path), "jobs", "")) return;
#ifdef _WIN32
    char pattern[MAX_PATH_SIZE]; snprintf(pattern, sizeof(pattern), "%s*.json", path);
    WIN32_FIND_DATAA data; HANDLE h = FindFirstFileA(pattern, &data);
    if (h == INVALID_HANDLE_VALUE) return;
    do { recover_one(data.cFileName); } while (FindNextFileA(h, &data));
    FindClose(h);
#else
    DIR *d = opendir(path); if (!d) return; struct dirent *entry;
    while ((entry = readdir(d))) recover_one(entry->d_name);
    closedir(d);
#endif
}

static void progress_changed(double progress, void *user) {
    job *j = user;
    if (!finite_double(progress)) return;
    if (progress < 0) progress = 0;
    if (progress > 0.999) progress = 0.999;
    lock_state();
    /* Library progress restarts for each phase; accept phase resets as well as advances. */
    if (progress > j->progress + 0.005 || progress < j->progress - 0.005) {
        j->progress = progress; j->updated = (int64_t)time(NULL);
        if (!persist_job(j)) log_line("Progress persistence failed for %s", j->id);
    }
    unlock_state();
}
static int verify_png(const char *path, int width, int height) {
    static const unsigned char sig[8] = {137,80,78,71,13,10,26,10};
    unsigned char head[24]; FILE *f = fopen(path, "rb"); if (!f) return 0;
    size_t n = fread(head, 1, sizeof(head), f); fclose(f);
    if (n != sizeof(head) || memcmp(sig, head, 8) || memcmp(head + 12, "IHDR", 4)) return 0;
    uint32_t w = (uint32_t)head[16] << 24 | (uint32_t)head[17] << 16 | (uint32_t)head[18] << 8 | head[19];
    uint32_t h = (uint32_t)head[20] << 24 | (uint32_t)head[21] << 16 | (uint32_t)head[22] << 8 | head[23];
    return w == (uint32_t)width && h == (uint32_t)height;
}
#ifdef _WIN32
static DWORD WINAPI worker(void *unused)
#else
static void *worker(void *unused)
#endif
{
    (void)unused;
    while (!stopping) {
        job *j = NULL; lock_state();
        for (int i = 0; i < MAX_JOBS; ++i)
            if (jobs[i].occupied && !strcmp(jobs[i].status, "queued") &&
                (!j || jobs[i].created < j->created)) j = &jobs[i];
        if (j) {
            strcpy(j->status, "running"); j->updated = (int64_t)time(NULL);
            if (!persist_job(j)) {
                strcpy(j->status, "failed"); strcpy(j->error, "Unable to persist running state");
                log_line("Job %s failed to persist", j->id); input_unpin_request(&j->request); j = NULL;
            }
        }
        unlock_state();
        if (!j) { pause_ms(100); continue; }
        char output[MAX_PATH_SIZE], name[64], error[768] = {0};
        ps5_ai_stats stats; memset(&stats, 0, sizeof(stats));
        snprintf(name, sizeof(name), "%s.png", j->id);
        int ok = path_for(output, sizeof(output), "output", name);
        log_line("Job %s running %dx%d steps=%d", j->id, j->request.width, j->request.height, j->request.steps);
        ps5_ai_native_ui_job_begin(j->id);
        if (ok) ok = ps5_ai_engine_generate(&j->request, output, &stats, progress_changed, j, error, sizeof(error)) == 0;
        if (ok && !stats.gpu_used) { ok = 0; snprintf(error, sizeof(error), "Engine did not verify GPU inference dispatch"); }
        if (ok && !verify_png(output, j->request.width, j->request.height)) {
            ok = 0; snprintf(error, sizeof(error), "Engine output is missing or PNG dimensions do not match request");
        }
        if (!finite_double(stats.generation_seconds) || stats.generation_seconds < 0) stats.generation_seconds = 0;
        lock_state();
        input_unpin_request(&j->request);
        j->stats = stats; j->updated = (int64_t)time(NULL);
        strcpy(j->status, ok ? "completed" : "failed");
        if (ok) j->progress = 1;
        else snprintf(j->error, sizeof(j->error), "%s", error[0] ? error : "GPU inference failed");
        if (!persist_job(j)) {
            strcpy(j->status, "failed"); strcpy(j->error, "Unable to persist final state; see server log");
            log_line("Job %s final persistence failed", j->id);
        }
        log_line("Job %s %s gpu_used=%d seconds=%.6f peak_cpu=%" PRIu64 " peak_gpu=%" PRIu64 " error=%s",
            j->id, j->status, stats.gpu_used, stats.generation_seconds, stats.peak_cpu_bytes, stats.peak_gpu_bytes, j->error);
        ps5_ai_native_ui_phase(!strcmp(j->status,"completed")?PS5_AI_STATUS_COMPLETED:PS5_AI_STATUS_FAILED);
        ps5_ai_native_ui_log(!strcmp(j->status,"completed")?"PNG validated and job completed":"Job failed; see authenticated API result");
        unlock_state();
    }
    return 0;
}

static int send_all(socket_t fd, const void *data, size_t size) {
    const char *p = data;
    while (size) {
        int n = send(fd, p, (int)(size > 65536 ? 65536 : size), 0);
        if (n <= 0) return 0;
        p += n; size -= (size_t)n;
    }
    return 1;
}
static const char *status_text(int code) {
    switch (code) {
    case 201: return "Created"; case 200: return "OK"; case 202: return "Accepted"; case 400: return "Bad Request";
    case 401: return "Unauthorized"; case 403: return "Forbidden"; case 404: return "Not Found";
    case 405: return "Method Not Allowed"; case 408: return "Request Timeout";
    case 413: return "Content Too Large";
    case 415: return "Unsupported Media Type"; case 429: return "Too Many Requests";
    case 500: return "Internal Server Error"; case 503: return "Service Unavailable";
    default: return "Error";
    }
}
static int send_headers(socket_t fd, int code, const char *content_type, uint64_t length) {
    char header[768];
    int n = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %" PRIu64 "\r\n"
        "Connection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
        "X-Frame-Options: DENY\r\n"
        "%s\r\n", code, status_text(code), content_type, length,
        code == 401 ? "WWW-Authenticate: Bearer\r\n" : "");
    return n > 0 && (size_t)n < sizeof(header) && send_all(fd, header, (size_t)n);
}
static void send_json(socket_t fd, int code, const char *body) {
    if (send_headers(fd, code, "application/json; charset=utf-8", strlen(body))) send_all(fd, body, strlen(body));
}
static void send_error(socket_t fd, int code, const char *message) {
    char quoted[4608], body[4800];
    if (!quote_json(quoted, sizeof(quoted), message)) strcpy(quoted, "\"Error\"");
    snprintf(body, sizeof(body), "{\"error\":%s}", quoted); send_json(fd, code, body);
}
static int authorized(const char *header) {
    const char *prefix = "Bearer ";
    if (strncmp(header, prefix, 7) || strlen(header + 7) != token_len) return 0;
    unsigned diff = 0;
    for (size_t i = 0; i < token_len; ++i) diff |= (unsigned char)header[i + 7] ^ (unsigned char)api_token[i];
    return diff == 0;
}
static int equal_ci(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a++) != tolower((unsigned char)*b++)) return 0;
    }
    return !*a && !*b;
}
static void health(socket_t fd) {
    ps5_ai_engine_status s; memset(&s, 0, sizeof(s)); ps5_ai_engine_get_status(&s);
    if(retained_engine_fault()){s.available=0;s.gpu_available=0;s.gpu_verified=0;snprintf(s.reason,sizeof(s.reason),"Native engine stopped; reload required: %.450s",retained_fault_reason);}
    /* Engine hooks must return NUL-terminated bounded strings. Clamp defensively. */
    s.backend[sizeof(s.backend)-1] = s.device[sizeof(s.device)-1] = 0;
    s.driver[sizeof(s.driver)-1] = s.model[sizeof(s.model)-1] = s.reason[sizeof(s.reason)-1] = 0;
    s.model_initialization[sizeof(s.model_initialization)-1] = 0;
    char backend[400], device[1600], driver[1600], model[3200], reason[3200], body[12288];
    quote_json(backend, sizeof(backend), s.backend); quote_json(device, sizeof(device), s.device);
    quote_json(driver, sizeof(driver), s.driver); quote_json(model, sizeof(model), s.model);
    quote_json(reason, sizeof(reason), s.reason);
    int active = 0, queued = 0; lock_state();
    for (int i = 0; i < MAX_JOBS; ++i) if (jobs[i].occupied) {
        active += !strcmp(jobs[i].status, "running"); queued += !strcmp(jobs[i].status, "queued");
    }
    unlock_state();
    char screen_json[256]="null";
    char lora_json[4096]="null";
#ifndef _WIN32
#if PS5_AI_MOSS_TURBO_TRIAL
    if(ps5_ai_moss_turbo_identity_json && ps5_ai_moss_turbo_identity_json(lora_json,sizeof(lora_json)))
        snprintf(lora_json,sizeof(lora_json),"null");
#else
    struct ps5_ai_moss_lora_receipt lora={0};
    if(ps5_ai_moss_lora_snapshot && !ps5_ai_moss_lora_snapshot(&lora))
        snprintf(lora_json,sizeof(lora_json),"{\"adapter\":\"axiomofmind/Moss-image-nsfw-uncensored-lora\",\"path\":\"/app0/models/moss-image-nsfw-uncensored-r16-f32.safetensors\",\"configured\":%s,\"strength\":%.9g,\"loaded_tensors\":%u,\"matched_tensors\":%u,\"used_tensors\":%u,\"skipped_tensors\":%u,\"application_verified\":%s}",
            lora.configured?"true":"false",(double)lora.strength,lora.loaded,lora.matched,lora.used,lora.skipped,
            lora.application_verified?"true":"false");
#endif
    struct ps5_ai_status_display_state_v1 display={0};
    if(!ps5_ai_status_display_state(&display))
        snprintf(screen_json,sizeof(screen_json),"{\"ready\":%s,\"heartbeat\":%" PRIu64 ",\"last_presented_frame\":%" PRIu64 ",\"fault\":%s,\"stopped\":%s}",
            display.ready?"true":"false",display.heartbeat,display.completed_flip_argument,
            display.display_fault?"true":"false",display.stopped?"true":"false");
#endif
    char lifetime_json[1024]="null";
#if defined(__PROSPERO__) || defined(PS5_AI_LIFECYCLE_HOST_TEST)
    lifecycle_json(lifetime_json,sizeof(lifetime_json));
#endif
    snprintf(body, sizeof(body),
        "{\"status\":\"%s\",\"server\":\"ps5-ai-moss-image\",\"model_id\":\"moss-image-sft\",\"features\":{\"text_to_image\":true,\"img2img\":false,\"inpainting\":false,\"ip_adapter\":false,\"controlnet\":false,\"sdxl_manga_lora\":false,\"selected_moss_lora\":true},\"selected_moss_lora\":%s,\"backend\":%s,\"engine_available\":%s,"
        "\"gpu_available\":%s,\"gpu_verified\":%s,\"gpu_device\":%s,\"gpu_driver\":%s,"
        "\"model_loaded\":%s,\"model\":%s,\"reason\":%s,\"running_jobs\":%d,\"queued_jobs\":%d,\"port\":%d,"
        "\"process_id\":%" PRId64 ",\"started_at\":%" PRId64 ",\"listener_generation\":%" PRIu32 ",\"tv_display\":%s,\"lifecycle\":%s,\"model_initialization\":%s,\"moss_sampler_trial\":%s,\"moss_sampler_policy\":\"%s\",\"trial_sampler\":\"%s\",\"trial_scheduler\":\"%s\",\"trial_steps\":%d,\"trial_cfg_scale\":%.1f,\"moss_turbo_trial\":%s,\"moss_turbo_policy\":\"%s\",\"moss_turbo_adapter\":\"%s\",\"moss_turbo_rank\":%d,\"moss_turbo_runtime_multiplier\":%.1f}",
        s.available && s.gpu_available ? "ready" : "degraded", lora_json, backend,
        s.available ? "true" : "false", s.gpu_available ? "true" : "false",
        s.gpu_verified ? "true" : "false", device, driver, s.model_loaded ? "true" : "false",
        model, reason, active, queued, listen_port, (int64_t)get_pid(), server_started_at, listener_generation, screen_json,lifetime_json,s.model_initialization[0] ? s.model_initialization : "null",
        PS5_AI_MOSS_SAMPLER_TRIAL?"true":"false",PS5_AI_MOSS_SAMPLER_POLICY,PS5_AI_MOSS_DEFAULT_REQUEST_SAMPLER,PS5_AI_MOSS_REQUEST_SCHEDULER,PS5_AI_MOSS_DEFAULT_REQUEST_STEPS,PS5_AI_MOSS_DEFAULT_REQUEST_CFG,
        PS5_AI_MOSS_TURBO_TRIAL?"true":"false",PS5_AI_MOSS_TURBO_POLICY,PS5_AI_MOSS_TURBO_TRIAL?PS5_AI_MOSS_TURBO_ADAPTER_BASENAME:"disabled",PS5_AI_MOSS_TURBO_TRIAL?144:0,PS5_AI_MOSS_TURBO_TRIAL?1.0:0.0);
    send_json(fd, 200, body);
}
static void submit(socket_t fd, const char *body) {
    if(retained_engine_fault()){send_error(fd,503,"Native engine retained after failure; reload this app before generation");return;}
    ps5_ai_request request; char error[768];
    if (!validate_request(body, &request, error, sizeof(error))) { send_error(fd, 400, error); return; }
    ps5_ai_engine_status status; memset(&status, 0, sizeof(status)); ps5_ai_engine_get_status(&status);
    if (!status.available || !status.gpu_available) {
        status.reason[sizeof(status.reason)-1] = 0;
        send_error(fd, 503, status.reason[0] ? status.reason : "GPU inference engine unavailable"); return;
    }
    lock_state(); job *j = NULL;
    for (int i = 0; i < MAX_JOBS; ++i) {
        if (!jobs[i].occupied) { j = &jobs[i]; break; }
        if (strcmp(jobs[i].status, "queued") && strcmp(jobs[i].status, "running") &&
            (!j || jobs[i].updated < j->updated)) j = &jobs[i];
    }
    if (!j) { unlock_state(); send_error(fd, 429, "Job queue is full"); return; }
    int pinned=input_pin_request(&request);
    if(pinned!=1){unlock_state();send_error(fd,pinned<0?400:404,pinned<0?"Input/mask kinds or required input/output dimensions do not match":"Referenced input image is missing or expired");return;}
    memset(j, 0, sizeof(*j)); j->occupied = 1; j->request = request;
    j->created = j->updated = (int64_t)time(NULL);
    char existing[JOB_JSON_SIZE];
    do {
        ++sequence;
        snprintf(j->id, sizeof(j->id), "%08" PRIx32 "%08" PRIx32 "%08" PRIx32 "%08" PRIx32,
            (uint32_t)j->created, (uint32_t)get_pid(), sequence, sequence ^ (uint32_t)(j->created >> 32));
    } while (load_job_json(j->id, existing, sizeof(existing)));
    strcpy(j->status, "queued");
    char name[64]; snprintf(name, sizeof(name), "%s.request.json", j->id);
    if (!atomic_write("jobs", name, body, strlen(body)) || !persist_job(j)) {
        input_unpin_request(&j->request); j->occupied = 0; unlock_state(); send_error(fd, 500, "Unable to persist job"); return;
    }
    char response[JOB_JSON_SIZE]; serialize_job(j, response, sizeof(response));
    log_line("Job %s accepted", j->id); unlock_state(); send_json(fd, 202, response);
}
static void get_job(socket_t fd, const char *id) {
    if (!valid_id(id)) { send_error(fd, 404, "Job not found"); return; }
    char data[JOB_JSON_SIZE]; int found = 0; lock_state();
    for (int i = 0; i < MAX_JOBS; ++i) if (jobs[i].occupied && !strcmp(jobs[i].id, id)) {
        found = serialize_job(&jobs[i], data, sizeof(data)); break;
    }
    if (!found) found = load_job_json(id, data, sizeof(data));
    unlock_state();
    if (found) send_json(fd, 200, data); else send_error(fd, 404, "Job not found");
}
static void get_image(socket_t fd, const char *id) {
    if (!valid_id(id)) { send_error(fd, 404, "Image not found"); return; }
    char data[JOB_JSON_SIZE]; int completed = 0; lock_state();
    for (int i = 0; i < MAX_JOBS; ++i) if (jobs[i].occupied && !strcmp(jobs[i].id, id)) {
        completed = !strcmp(jobs[i].status, "completed"); break;
    }
    if (!completed && load_job_json(id, data, sizeof(data))) {
        object *obj = malloc(sizeof(*obj));
        if (obj) {
            if (parse_job_object(data, obj)) {
                field *s = find_field(obj, "status"), *g = find_field(obj, "gpu_used");
                completed = s && s->type == 1 && !strcmp(s->value, "completed") && g && g->type == 3;
            }
            free(obj);
        }
    }
    unlock_state();
    if (!completed) { send_error(fd, 404, "Completed GPU image not found"); return; }
    char path[MAX_PATH_SIZE], name[64]; snprintf(name, sizeof(name), "%s.png", id);
    if (!path_for(path, sizeof(path), "output", name)) { send_error(fd, 500, "Path too long"); return; }
    FILE *f = fopen(path, "rb"); if (!f) { send_error(fd, 404, "Image file not found"); return; }
    struct stat st;
    if (path_metadata(path, &st, 0) || !S_ISREG(st.st_mode) || st.st_size < 24 || st.st_size > 64 * 1024 * 1024) {
        fclose(f); send_error(fd, 500, "Invalid image size"); return;
    }
    if (send_headers(fd, 200, "image/png", (uint64_t)st.st_size)) {
        char buffer[16384]; size_t n;
        while ((n = fread(buffer, 1, sizeof(buffer), f))) if (!send_all(fd, buffer, n)) break;
    }
    fclose(f);
}
static void get_ui(socket_t fd) {
    char path[MAX_PATH_SIZE];
    if (!path_for(path, sizeof(path), "ui", "index.html")) { send_error(fd, 500, "Path too long"); return; }
    struct stat st;
    if (path_metadata(path, &st, 0) || !S_ISREG(st.st_mode) || st.st_size <= 0 || st.st_size > 1024 * 1024) {
        send_error(fd, 404, "UI file is not installed"); return;
    }
    FILE *f = fopen(path, "rb");
    if (!f) { send_error(fd, 404, "UI file is not installed"); return; }
    if (send_headers(fd, 200, "text/html; charset=utf-8", (uint64_t)st.st_size)) {
        char buffer[16384]; size_t n;
        while ((n = fread(buffer, 1, sizeof(buffer), f))) if (!send_all(fd, buffer, n)) break;
    }
    fclose(f);
}
#include "reference_store.inc"

static void admin_shutdown(socket_t fd) {
    if(retained_engine_fault()){send_error(fd,409,"Retained GPU resources require a verified native app close; cooperative worker join is unavailable");return;}
#if defined(__PROSPERO__) || defined(PS5_AI_LIFECYCLE_HOST_TEST)
    struct ps5_ai_native_lifecycle lifetime;
    ps5_ai_native_lifecycle_snapshot(&lifetime);
    if(!ps5_ai_lifecycle_checked_shutdown_allowed(&lifetime)) {
        char state[1024],response[1536];lifecycle_json(state,sizeof(state));
        snprintf(response,sizeof(response),"{\"error\":\"Initialization did not complete the checked GPU path; a separately verified normal OS title close is required. No in-app GPU cleanup or OS close was attempted.\",\"recovery\":%s}",state);
        send_json(fd,409,response);return;
    }
#endif
    int running = 0, queued = 0;
    lock_state();
    for (int i = 0; i < MAX_JOBS; ++i) if (jobs[i].occupied) {
        running += !strcmp(jobs[i].status, "running");
        queued += !strcmp(jobs[i].status, "queued");
    }
    if (stopping || running || queued) {
        unlock_state();
        send_error(fd, 409, "Shutdown requires an idle server with no queued or running jobs");
        return;
    }
    /* Main request thread, authenticated by handle_client; lock prevents a
     * worker transition between the idle check and the stop publication. */
    ps5_ai_v3_shutdown_request_accepted(running, queued);
    ps5_ai_native_ui_phase(PS5_AI_STATUS_STOPPING);
    stopping = 1;
    unlock_state();
    log_line("Authenticated idle shutdown accepted; listener close and worker join pending");
    send_json(fd, 202, "{\"status\":\"shutdown_requested\",\"exit_verified\":false}");
}
/* One monotonic deadline covers headers, body and rejected-request draining.
 * Per-read idle timeouts alone allow a peer to keep this serial receiver busy. */
static int monotonic_ms(uint64_t *value) {
#ifdef _WIN32
    *value = (uint64_t)GetTickCount64();
#else
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) || now.tv_sec < 0 ||
        now.tv_nsec < 0 || now.tv_nsec >= 1000000000L ||
        (uint64_t)now.tv_sec > (UINT64_MAX - 999) / 1000) return 0;
    *value = (uint64_t)now.tv_sec * 1000 + (uint64_t)now.tv_nsec / 1000000;
#endif
    return 1;
}
#define RECEIVE_EXPIRED (-2)
#define RECEIVE_SETUP_ERROR (-3)
static int recv_before_deadline(socket_t fd, char *buffer, int size,
                                uint64_t deadline, uint64_t idle_ms) {
    uint64_t now;
    if (!monotonic_ms(&now)) return RECEIVE_SETUP_ERROR;
    if (now >= deadline) return RECEIVE_EXPIRED;
    uint64_t remaining = deadline - now;
    if (remaining > idle_ms) remaining = idle_ms;
#ifdef _WIN32
    DWORD timeout = (DWORD)remaining;
#else
    struct timeval timeout = {(long)(remaining / 1000), (long)(remaining % 1000) * 1000};
#endif
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&timeout, sizeof(timeout)))
        return RECEIVE_SETUP_ERROR;
    int received = recv(fd, buffer, size, 0);
    if (!monotonic_ms(&now)) return RECEIVE_SETUP_ERROR;
    return now >= deadline ? RECEIVE_EXPIRED : received;
}
static int receive_failed(socket_t fd, int received) {
    if (received == RECEIVE_EXPIRED) send_error(fd, 408, "Request receive deadline exceeded");
    else if (received == RECEIVE_SETUP_ERROR) send_error(fd, 500, "Unable to enforce receive deadline");
    return received <= 0;
}
static void handle_client(socket_t fd, uint64_t deadline) {
    apply_retained_engine_fault();
    char *buffer = malloc(MAX_HEADER + MAX_BODY + 1);
    if (!buffer) { send_error(fd, 500, "Out of memory"); return; }
    size_t received = 0; char *end = NULL;
    while (received < MAX_HEADER) {
        int n = recv_before_deadline(fd, buffer + received, (int)(MAX_HEADER - received), deadline, 5000);
        if (receive_failed(fd, n)) goto done;
        received += (size_t)n; buffer[received] = 0; end = strstr(buffer, "\r\n\r\n");
        if (end) break;
        if (memchr(buffer, 0, received)) { send_error(fd, 400, "NUL byte in headers"); goto done; }
    }
    if (!end) { send_error(fd, 413, "Request headers too large"); goto done; }
    size_t header_size = (size_t)(end - buffer) + 4;
    if (memchr(buffer, 0, header_size)) { send_error(fd, 400, "NUL byte in headers"); goto done; }
    *end = 0;
    char *line_end = strstr(buffer, "\r\n");
    if (!line_end) { send_error(fd, 400, "Malformed request line"); goto done; }
    *line_end = 0;
    char method[16], path[256], version[16], extra;
    if (sscanf(buffer, "%15s %255s %15s %c", method, path, version, &extra) != 3 ||
        (strcmp(version, "HTTP/1.1") && strcmp(version, "HTTP/1.0"))) {
        send_error(fd, 400, "Malformed request line"); goto done;
    }
    char authorization[320] = {0}, content_type[128] = {0};
    int has_length = 0, has_auth = 0, has_type = 0; size_t content_length = 0;
    char *line = line_end + 2;
    while (*line) {
        char *next = strstr(line, "\r\n"); if (next) *next = 0;
        char *colon = strchr(line, ':');
        if (!colon || colon == line || *line == ' ' || *line == '\t') { send_error(fd, 400, "Malformed header"); goto done; }
        *colon = 0; char *value = colon + 1;
        for (const char *p = line; *p; ++p)
            if (!isalnum((unsigned char)*p) && *p != '-') { send_error(fd, 400, "Invalid header name"); goto done; }
        while (*value == ' ' || *value == '\t') ++value;
        char *tail = value + strlen(value);
        while (tail > value && (tail[-1] == ' ' || tail[-1] == '\t')) *--tail = 0;
        for (const char *p = value; *p; ++p)
            if ((unsigned char)*p < 32 && *p != '\t') { send_error(fd, 400, "Invalid header value"); goto done; }
        if (equal_ci(line, "Content-Length")) {
            if (has_length++ || !*value) { send_error(fd, 400, "Invalid or duplicate Content-Length"); goto done; }
            for (const char *p = value; *p; ++p) {
                if (!isdigit((unsigned char)*p) || content_length > PS5_AI_INPUT_MAX_BYTES) { send_error(fd, 413, "Request body too large or invalid length"); goto done; }
                content_length = content_length * 10 + (size_t)(*p - '0');
            }
        } else if (equal_ci(line, "Authorization")) {
            if (has_auth++ || strlen(value) >= sizeof(authorization)) { send_error(fd, 400, "Invalid Authorization header"); goto done; }
            strcpy(authorization, value);
        } else if (equal_ci(line, "Content-Type")) {
            if (has_type++ || strlen(value) >= sizeof(content_type)) { send_error(fd, 400, "Invalid Content-Type header"); goto done; }
            strcpy(content_type, value);
        } else if (equal_ci(line, "Transfer-Encoding")) { send_error(fd, 400, "Transfer-Encoding is unsupported"); goto done; }
        if (!next) break;
        line = next + 2;
    }
    int public_ui = !strcmp(method, "GET") && (!strcmp(path, "/") || !strcmp(path, "/index.html"));
    if (!public_ui && !authorized(authorization)) { send_error(fd, 401, "Valid Bearer token required"); goto done; }
    int mask_upload=!strcmp(method,"POST") && (!strcmp(path,"/api/masks") || !strcmp(path,"/api/masks/grayscale"));
    int input_upload=mask_upload || (!strcmp(method,"POST") && !strcmp(path,"/api/inputs"));
    size_t body_limit=mask_upload?PS5_AI_MASK_MAX_BYTES-1:input_upload?PS5_AI_INPUT_MAX_BYTES:MAX_BODY;
    if(content_length>body_limit){send_error(fd,413,"Request body too large");goto done;}
    if(input_upload){
        char *larger=realloc(buffer,MAX_HEADER+content_length+1);
        if(!larger){send_error(fd,500,"Out of memory");goto done;}buffer=larger;
    }
    if (received > header_size + content_length) { send_error(fd, 400, "Pipelining or extra request data unsupported"); goto done; }
    while (received < header_size + content_length) {
        int n = recv_before_deadline(fd, buffer + received, (int)(header_size + content_length - received), deadline, 5000);
        if (receive_failed(fd, n)) goto done;
        received += (size_t)n;
    }
    buffer[received] = 0;
    char *body = buffer + header_size;
    if (!input_upload && memchr(body, 0, content_length)) { send_error(fd, 400, "NUL byte in body"); goto done; }
    if (!strcmp(method, "GET") && content_length) { send_error(fd, 400, "GET body unsupported"); goto done; }
    if (public_ui) {
        get_ui(fd);
    } else if (!strcmp(path, "/health")) {
        if (strcmp(method, "GET")) send_error(fd, 405, "Use GET"); else health(fd);
    } else if (!strcmp(path, "/api/admin/shutdown")) {
        if (strcmp(method, "POST")) send_error(fd, 405, "Use POST");
        else if (!has_length || content_length) send_error(fd, 400, "Shutdown requires Content-Length: 0 and an empty body");
        else admin_shutdown(fd);
    } else if (!strcmp(path,"/api/inputs") || !strcmp(path,"/api/masks") || !strcmp(path,"/api/masks/grayscale")) {
        if(strcmp(method,"POST"))send_error(fd,405,"Use POST");
        else if(!MOSS_IMAGE_INPUTS_SUPPORTED)send_error(fd,400,"MOSS Image SFT currently supports text-to-image only; image inputs and masks are unsupported");
        else upload_input(fd,(const unsigned char *)body,content_length,content_type,!strcmp(path,"/api/masks")?1:!strcmp(path,"/api/masks/grayscale")?2:0);
    } else if (!strncmp(path,"/api/inputs/",12)) {
        if(strcmp(method,"GET") && strcmp(method,"DELETE"))send_error(fd,405,"Use GET or DELETE");
        else if(content_length)send_error(fd,400,"Input metadata/deletion requires empty body");
        else input_metadata(fd,path+12,!strcmp(method,"DELETE"));
    } else if (!strcmp(path, "/api/generate")) {
        if (strcmp(method, "POST")) send_error(fd, 405, "Use POST");
        else if (strcmp(content_type, "application/json") && strcmp(content_type, "application/json; charset=utf-8")) send_error(fd, 415, "Content-Type must be application/json");
        else if (!has_length || !content_length) send_error(fd, 400, "JSON body with Content-Length required");
        else submit(fd, body);
    } else if (!strncmp(path, "/api/jobs/", 10)) {
        if (strcmp(method, "GET")) send_error(fd, 405, "Use GET"); else get_job(fd, path + 10);
    } else if (!strncmp(path, "/api/images/", 12)) {
        if (strcmp(method, "GET")) send_error(fd, 405, "Use GET"); else get_image(fd, path + 12);
    } else send_error(fd, 404, "Endpoint not found");
done:
    free(buffer);
}

/* Send FIN before closing a rejected request with unread bytes, so clients can
 * receive the HTTP error instead of a TCP reset. Drain only a bounded amount. */
static void finish_client(socket_t fd, uint64_t deadline) {
    shutdown(fd, 1);
    char discard[2048]; size_t total = 0;
    while (total < MAX_HEADER + MAX_BODY) {
        int n = recv_before_deadline(fd, discard, sizeof(discard), deadline, 100);
        if (n <= 0) break;
        total += (size_t)n;
    }
    close_socket(fd);
}

static int setup_paths(void) {
    STARTUP_STAGE("setup_paths_enter",0);
    struct stat st;
    if (path_metadata(root_path, &st, 1) || !S_ISDIR(st.st_mode)) {
        STARTUP_STAGE("root_metadata_failed",0);
        fprintf(stderr, "Root directory must already exist: %s\n", root_path); return 0;
    }
    const char *folders[] = {"config", "logs", "jobs", "output", "inputs"};
    for (size_t i = 0; i < sizeof(folders)/sizeof(folders[0]); ++i) {
        STARTUP_STAGE("folder_begin",(int)i);
        char path[MAX_PATH_SIZE]; if (!path_for(path, sizeof(path), folders[i], "")) return 0;
        if (make_dir(path) && errno != EEXIST) { fprintf(stderr, "Cannot create %s\n", path); return 0; }
        if (path_metadata(path, &st, 1) || !S_ISDIR(st.st_mode)) return 0;
    }
    STARTUP_STAGE("recover_inputs_before",0);
    if(!recover_inputs()){STARTUP_STAGE("recover_inputs_failed",0);fprintf(stderr,"Input image storage recovery failed\n");return 0;}
    STARTUP_STAGE("recover_inputs_passed",1);
    if (!path_for(bounded_api_log_path, sizeof(bounded_api_log_path), "logs", "api.log")) return 0;
    log_file = fopen(bounded_api_log_path, "ab");
    STARTUP_STAGE("api_log_open",log_file!=NULL);
    if (!log_file) { fprintf(stderr, "Cannot open API log\n"); return 0; }
    return 1;
}
static int consume_token(void) {
    char token_path[MAX_PATH_SIZE], token_file[1024] = {0}; int ok = 0;
    if (!path_for(token_path, sizeof(token_path), "config", "api-token") ||
        !read_small_file(token_path, token_file, sizeof(token_file))) {
        fprintf(stderr, "Missing/invalid startup token: root/config/api-token\n"); goto done;
    }
    size_t n = strlen(token_file);
    while (n && (token_file[n-1] == '\r' || token_file[n-1] == '\n')) token_file[--n] = 0;
    if (n < 32 || n > 256) { fprintf(stderr, "Token must contain 32..256 ASCII non-space bytes\n"); goto done; }
    for (size_t i = 0; i < n; ++i) if ((unsigned char)token_file[i] <= 32 || (unsigned char)token_file[i] > 126) goto done;
    /* The existing WebSrv exposes /fs paths. Do not leave the token on disk.
     * Only the process which has acquired the listener may consume this file. */
    if (remove(token_path)) { fprintf(stderr, "Cannot consume startup token; refusing API startup\n"); goto done; }
    memcpy(api_token, token_file, n + 1); token_len = n; ok = 1;
    log_line("Startup token consumed; authentication secret retained in process memory");
done:
    wipe_secret(token_file, sizeof(token_file));
    if (!ok) { wipe_secret(api_token, sizeof(api_token)); token_len = 0; }
    return ok;
}
static int private_address(uint32_t host) {
    return (host >> 24) == 10 || (host >> 20) == 0xac1 || (host >> 16) == 0xc0a8 || (host >> 24) == 127;
}
static void stop_signal(int signal_number) { (void)signal_number; stopping = 1; }
static void reload_signal(int signal_number) { (void)signal_number; listener_reload = 1; }
static int socket_error(void) {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
}
static socket_t open_listener(void) {
    socket_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == BAD_SOCKET) { log_line("Listener socket failed error=%d", socket_error()); return BAD_SOCKET; }
    int yes = 1;
#ifdef _WIN32
    if (setsockopt(fd, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *)&yes, sizeof(yes))) {
        log_line("Exclusive listener option failed error=%d", socket_error()); close_socket(fd); return BAD_SOCKET;
    }
#else
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes))) {
        log_line("Listener reuse option failed error=%d", socket_error()); close_socket(fd); return BAD_SOCKET;
    }
#endif
    struct sockaddr_in address; memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET; address.sin_port = htons((uint16_t)listen_port); address.sin_addr = bind_address;
    if (bind(fd, (struct sockaddr *)&address, sizeof(address)) || listen(fd, 16)) {
        log_line("Bind/listen failed on %s:%d error=%d", bind_ip, listen_port, socket_error());
        close_socket(fd); return BAD_SOCKET;
    }
    return fd;
}

int main(int argc, char **argv) {
    STARTUP_STAGE("server_main_enter",0);
#ifdef _WIN32
    InitializeCriticalSection(&state_mutex); InitializeCriticalSection(&log_mutex);
    WSADATA ws; if (WSAStartup(MAKEWORD(2,2), &ws)) return 1;
#endif
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) { fprintf(stderr, "Options require a value\n"); return 1; }
        const char *opt = argv[i], *value = argv[++i];
        if (!strcmp(opt, "--root")) {
            if (strlen(value) >= sizeof(root_path) - 100) return 1;
            strcpy(root_path, value);
        } else if (!strcmp(opt, "--bind")) {
            if (strlen(value) >= sizeof(bind_ip)) return 1;
            strcpy(bind_ip, value);
        } else if (!strcmp(opt, "--port")) {
            char *end; long p = strtol(value, &end, 10);
            if (*end || p < 1024 || p > 65535) return 1;
            listen_port = (int)p;
        } else { fprintf(stderr, "Unknown option %s\n", opt); return 1; }
    }
    if (inet_pton(AF_INET, bind_ip, &bind_address) != 1 || !private_address(ntohl(bind_address.s_addr))) {
        fprintf(stderr, "Bind address must be a specific RFC1918 or loopback IPv4 address\n"); return 1;
    }
    if(listen_port==8188 || listen_port==8189 || listen_port==8190 || listen_port==8191 || listen_port==8193){fprintf(stderr,"Ports8188/8189 are reserved for WAI;8190/8191 for Anima;8193 is the MOSS Image gateway\n");return 1;}
    if (!setup_paths()) {STARTUP_STAGE("setup_paths_failed",0);return 1;}
    STARTUP_STAGE("setup_paths_passed",1);
    /* Windows uses EXCLUSIVEADDRUSE because its REUSEADDR can admit a second
     * live listener. POSIX REUSEADDR does not enable duplicate listeners. */
    socket_t listener = open_listener();
    STARTUP_STAGE("listener_open",listener!=BAD_SOCKET);
    if (listener == BAD_SOCKET) { fclose(log_file); return 1; }
    listener_generation = 1; server_started_at = (int64_t)time(NULL);
    if (!consume_token()) {
        STARTUP_STAGE("consume_token_failed",0);
        log_line("Startup token unavailable or could not be consumed; API refused to start");
        close_socket(listener); fclose(log_file); return 1;
    }
    STARTUP_STAGE("consume_token_passed",1);
    /* Acquire the port before changing persisted job states or initializing a
     * GPU engine. A rejected duplicate launch must leave existing jobs alone. */
    STARTUP_STAGE("recover_jobs_before",0);recover_jobs();STARTUP_STAGE("recover_jobs_after",1);
    char config[MAX_PATH_SIZE], error[768] = {0};
    snprintf(config, sizeof(config), "%s/config.json", root_path);
    STARTUP_STAGE("engine_init_before",0);
    if (ps5_ai_engine_init(config, error, sizeof(error))) log_line("Engine unavailable: %s", error);
    STARTUP_STAGE("engine_init_after",1);
#ifdef _WIN32
    HANDLE thread = CreateThread(NULL, 0, worker, NULL, 0, NULL);
    if (!thread) { STARTUP_STAGE("worker_create_failed",0);log_line("Worker thread creation failed"); close_socket(listener); return 1; }
#else
    pthread_t thread;
    if (pthread_create(&thread, NULL, worker, NULL)) { STARTUP_STAGE("worker_create_failed",0);log_line("Worker thread creation failed"); close_socket(listener); return 1; }
    signal(SIGPIPE, SIG_IGN);
#endif
    signal(SIGINT, stop_signal); signal(SIGTERM, stop_signal);
#ifdef SIGCONT
    signal(SIGCONT, reload_signal);
#else
    (void)reload_signal;
#endif
    log_line("API listening at %s:%d; authenticated; same /24 peers only; no GPU claim without engine evidence", bind_ip, listen_port);
    int exit_code = 0;
    while (!stopping) {
        apply_retained_engine_fault();
        if (listener_reload) {
            listener_reload = 0;
            log_line("Listener reload requested by SIGCONT; recreating on main thread");
            close_socket(listener); listener = open_listener();
            if (listener == BAD_SOCKET) { stopping = 1; exit_code = 1; break; }
            ++listener_generation;
            log_line("Listener recreated at %s:%d generation=%" PRIu32 "; existing token and jobs retained", bind_ip, listen_port, listener_generation);
        }
        fd_set set; FD_ZERO(&set); FD_SET(listener, &set); struct timeval tv = {1,0};
        int ready = select((int)listener + 1, &set, NULL, NULL, &tv);
        if (ready < 0) {
            int error = socket_error();
#ifdef _WIN32
            if (error == WSAEINTR) continue;
#else
            if (error == EINTR) continue;
#endif
            log_line("Listener select failed error=%d; stopping API for explicit restart", error);
            stopping = 1; exit_code = 1; break;
        }
        if (!ready) continue;
        struct sockaddr_in peer;
#ifdef _WIN32
        int peer_size = sizeof(peer);
#else
        socklen_t peer_size = sizeof(peer);
#endif
        socket_t fd = accept(listener, (struct sockaddr *)&peer, &peer_size);
        if (fd == BAD_SOCKET) continue;
        uint32_t peer_host = ntohl(peer.sin_addr.s_addr), bind_host = ntohl(bind_address.s_addr);
        if (!private_address(peer_host) || (peer_host & 0xffffff00u) != (bind_host & 0xffffff00u)) {
            close_socket(fd); log_line("Rejected peer outside bind /24"); continue;
        }
#ifdef _WIN32
        DWORD timeout = 5000;
#else
        struct timeval timeout = {5,0};
#endif
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&timeout, sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&timeout, sizeof(timeout));
        uint64_t started;
        if (!monotonic_ms(&started) || started > UINT64_MAX - 10000) {
            send_error(fd, 500, "Unable to establish receive deadline"); close_socket(fd); continue;
        }
        uint64_t deadline = started + 10000;
        handle_client(fd, deadline); finish_client(fd, deadline);
    }
    int listener_closed = listener == BAD_SOCKET;
    if (listener != BAD_SOCKET) {
        listener_closed = close_socket(listener) == 0;
        listener = BAD_SOCKET;
    }
    int join_rc;
#ifdef _WIN32
    DWORD waited = WaitForSingleObject(thread, INFINITE);
    int handle_closed = CloseHandle(thread) != 0;
    join_rc = waited == WAIT_OBJECT_0 && handle_closed ? 0 : -1;
    WSACleanup();
#else
    join_rc = pthread_join(thread, NULL);
#endif
    int remaining_running = 0, remaining_queued = 0;
    lock_state();
    for (int i = 0; i < MAX_JOBS; ++i) if (jobs[i].occupied) {
        remaining_running += !strcmp(jobs[i].status, "running");
        remaining_queued += !strcmp(jobs[i].status, "queued");
    }
    unlock_state();
    ps5_ai_v3_worker_join_receipt(join_rc, remaining_running, remaining_queued, listener_closed);
    ps5_ai_engine_shutdown(); wipe_secret(api_token, sizeof(api_token)); token_len = 0;
    ps5_ai_native_ui_log("API listener stopped; native TV display remains resident");
    log_line("API stopped"); if(log_file)fclose(log_file);
    return exit_code;
}

#ifdef PS5_AI_RETAIN_FAULT_TEST
#include "test_retained_engine.inc"
#endif
