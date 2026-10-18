/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "probe.h"
#include "precision_mode.h"
#include "engine.h"
#include "arena.h"
#include "v3_shutdown.h"
#include "shutdown_policy.h"
#include "journal_metrics.h"
#include "pool.h"
#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <ps5platform/kernel.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <locale.h>
#include <limits.h>
#include "native_gate.h"
#include "direct_pool.h"
#include "memory_plan.h"
#include "native_constructor_expect.h"
#include "ui_hooks.h"
#include "bounded_journal.h"
void ps5_ai_native_api_retained_failure(const char *);
static struct ps5_ai_bounded_journal startup_journal_cap;

/* Native SDK has no debugger subprocess contract. Unexpected library
 * backtraces may not launch a process even before environment setup. */
pid_t __wrap_fork(void) { ps5_ai_ggml_retain("native_subprocess_fork_rejected_before_call"); }
int execlp(const char *file, const char *argument, ...) {
    (void)file; (void)argument;
    ps5_ai_ggml_retain("native_subprocess_exec_rejected_before_call");
}
__attribute__((noreturn)) void __wrap__Exit(int status) {
    (void)status;ps5_ai_ggml_retain("native_library_immediate_exit_rejected");
}
__attribute__((noreturn)) void __wrap_exit(int status) {
    (void)status;ps5_ai_ggml_retain("native_library_exit_rejected");
}
#if !defined(__PROSPERO__)
#error "Only the real PS5 target is supported"
#endif
_Static_assert(sizeof(struct ps5_ai_arena_stats)==72, "Legacy owned-driver statistics ABI");
#define EXPECTED_FW UINT32_C(0x12400009)
#define LOG_PATH "/download0/ai-native-bounded-logs-v2/native-startup.log"
/* These three exclusive files belong to this launch only. The bootstrap
 * captures one public REALTIME value and PID before publishing any path. */
static char native_session_bootstrap_path[192];
static char native_session_witness_path[192];
static char native_session_stderr_path[192];
static struct timespec native_session_realtime;
static int native_session_pid;
static bool native_session_paths_ready;
/* Native portability only: preserve the public stat result unless it reports
 * EPERM. Read descriptor metadata on that exact failure; never read file data,
 * change permissions, log or allocate from this interposed function. */
extern int __real_stat(const char *,struct stat *);
static atomic_uint_fast64_t native_stat_fallback_attempts;
static atomic_uint_fast64_t native_stat_fallback_successes;
int __wrap_stat(const char *path,struct stat *out) {
    int rc=__real_stat(path,out),saved=errno;
    if(rc!=-1 || saved!=EPERM)return rc;
    atomic_fetch_add_explicit(&native_stat_fallback_attempts,1,memory_order_relaxed);
    int fd=open(path,O_RDONLY|O_NONBLOCK|O_CLOEXEC);if(fd<0)return -1;
    struct {uint64_t pre[2];struct stat value;uint64_t post[2];} g;
    memset(&g,0,sizeof(g));g.pre[0]=g.pre[1]=g.post[0]=g.post[1]=UINT64_C(0x728dc6b5f04a913e);
    rc=fstat(fd,&g.value);saved=errno;
    if(g.pre[0]!=UINT64_C(0x728dc6b5f04a913e) || g.pre[1]!=g.pre[0] || g.post[0]!=g.pre[0] || g.post[1]!=g.pre[0]) {rc=-1;saved=EIO;}
    if(!rc)*out=g.value;
    if(close(fd) && !rc) {rc=-1;saved=errno;}
    if(!rc)atomic_fetch_add_explicit(&native_stat_fallback_successes,1,memory_order_relaxed);
    errno=saved;return rc;
}
bool radv_ps5_platform_runs_gpu(void);
bool radv_ps5_platform_init(void);
int32_t __real_sceAgcInit(uint32_t);
int ps5_ai_native_gpu_data_gate(char *, size_t);
static int log_fd = -1;
static atomic_bool log_healthy = true;
static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t metadata_mutex = PTHREAD_MUTEX_INITIALIZER;
static int failures;
static unsigned agc_init_requests, agc_init_attempts;
static atomic_bool backend_quiescent;
static struct ps5_ai_arena_kick_stats last_kick;
static int32_t agc_init_result = -1;
static volatile unsigned constructor_witness_count;
static int engine_init_calls;
static atomic_uint native_config_stage;
static atomic_uint_fast64_t native_locale_queries, native_locale_nulls;
static atomic_uintptr_t native_locale_last_real;
static atomic_uint_fast64_t native_locale_empty_points, native_locale_empty_repairs;
static atomic_uint_fast64_t native_locale_dot_probe_attempts, native_locale_dot_probe_successes;
static atomic_uint_fast64_t native_locale_last_decimal_byte=UINT64_MAX;
static atomic_uint_fast64_t native_locale_dot_value_bits, native_locale_dot_end_offset=UINT64_MAX;
static atomic_int native_locale_dot_errno;
static atomic_bool native_stderr_redirected;
static atomic_bool native_driver_prepare_entered, native_data_prepare_entered;
extern struct lconv *__real_localeconv(void);
/* SDK platform/src/locale.c documents an empty reported decimal point while
 * the public strtod reads '.'. Correct that exact mismatch only after the
 * same known 0.5/full-consumption check; valid nonempty public data is unchanged.
 * The SDK's ps5_localeconv_l calls localeconv itself and cannot serve here. */
struct lconv *__wrap_localeconv(void) {
    static char dot[]=".", empty[]="";
    static struct lconv c_locale={
        .decimal_point=dot,.thousands_sep=empty,.grouping=empty,
        .int_curr_symbol=empty,.currency_symbol=empty,.mon_decimal_point=empty,
        .mon_thousands_sep=empty,.mon_grouping=empty,.positive_sign=empty,.negative_sign=empty,
        .int_frac_digits=CHAR_MAX,.frac_digits=CHAR_MAX,.p_cs_precedes=CHAR_MAX,
        .p_sep_by_space=CHAR_MAX,.n_cs_precedes=CHAR_MAX,.n_sep_by_space=CHAR_MAX,
        .p_sign_posn=CHAR_MAX,.n_sign_posn=CHAR_MAX,.int_p_cs_precedes=CHAR_MAX,
        .int_n_cs_precedes=CHAR_MAX,.int_p_sep_by_space=CHAR_MAX,.int_n_sep_by_space=CHAR_MAX,
        .int_p_sign_posn=CHAR_MAX,.int_n_sign_posn=CHAR_MAX};
    struct lconv *result=__real_localeconv();
    atomic_fetch_add_explicit(&native_locale_queries,1,memory_order_relaxed);
    atomic_store_explicit(&native_locale_last_real,(uintptr_t)result,memory_order_relaxed);
    if(!result) {
        atomic_store_explicit(&native_locale_last_decimal_byte,UINT64_MAX,memory_order_relaxed);
        atomic_fetch_add_explicit(&native_locale_nulls,1,memory_order_relaxed);
        return &c_locale;
    }
    const uint64_t point=result->decimal_point?(unsigned char)result->decimal_point[0]:UINT64_MAX;
    atomic_store_explicit(&native_locale_last_decimal_byte,point,memory_order_relaxed);
    if(point!=0)return result;
    atomic_fetch_add_explicit(&native_locale_empty_points,1,memory_order_relaxed);
    atomic_fetch_add_explicit(&native_locale_dot_probe_attempts,1,memory_order_relaxed);
    static const char sample[]="0.5";char *end=NULL;int saved=errno;
    errno=0;union {double value;uint64_t bits;} probe={.value=strtod(sample,&end)};int probe_error=errno;errno=saved;
    uintptr_t begin=(uintptr_t)(const void*)sample,finish=(uintptr_t)(void*)end;
    uint64_t offset=finish>=begin&&finish<=begin+3?(uint64_t)(finish-begin):UINT64_MAX;
    atomic_store_explicit(&native_locale_dot_value_bits,probe.bits,memory_order_relaxed);
    atomic_store_explicit(&native_locale_dot_end_offset,offset,memory_order_relaxed);
    atomic_store_explicit(&native_locale_dot_errno,probe_error,memory_order_relaxed);
    if(probe.value!=0.5||end!=sample+3||probe_error!=0)return result;
    atomic_fetch_add_explicit(&native_locale_dot_probe_successes,1,memory_order_relaxed);
    atomic_fetch_add_explicit(&native_locale_empty_repairs,1,memory_order_relaxed);
    return &c_locale;
}
void ps5_ai_native_locale_snapshot(uint64_t out[11]) {
    out[0]=atomic_load_explicit(&native_locale_queries,memory_order_relaxed);
    out[1]=atomic_load_explicit(&native_locale_nulls,memory_order_relaxed);
    out[2]=(uint64_t)atomic_load_explicit(&native_locale_last_real,memory_order_relaxed);
    out[3]=atomic_load_explicit(&native_locale_empty_points,memory_order_relaxed);
    out[4]=atomic_load_explicit(&native_locale_empty_repairs,memory_order_relaxed);
    out[5]=atomic_load_explicit(&native_locale_last_decimal_byte,memory_order_relaxed);
    out[6]=atomic_load_explicit(&native_locale_dot_value_bits,memory_order_relaxed);
    out[7]=atomic_load_explicit(&native_locale_dot_end_offset,memory_order_relaxed);
    out[8]=atomic_load_explicit(&native_locale_dot_probe_successes,memory_order_relaxed);
    out[9]=atomic_load_explicit(&native_locale_dot_probe_attempts,memory_order_relaxed);
    out[10]=(uint64_t)atomic_load_explicit(&native_locale_dot_errno,memory_order_relaxed);
}
void ps5_ai_native_config_mark(unsigned stage,const char *label) {
    atomic_store_explicit(&native_config_stage,stage,memory_order_relaxed);
    ps5_ai_ggml_report("NATIVE_CONFIG stage=%u phase=%s metadata_only=1 config_body_logged=0",stage,label);
}
static bool engine_started;
static pid_t probe_pid;
static pthread_t controller_main_thread;
static bool controller_main_thread_valid;
static atomic_bool generation_active;
static atomic_bool retention_entered;
static atomic_bool shutdown_requested, shutdown_proven;
static bool worker_joined, listener_closed, model_cleanup_checked;
static bool pre_cleanup_quiescent, post_cleanup_quiescent;
static unsigned shutdown_request_calls, worker_join_calls, shutdown_calls;
typedef void (*initializer_fn)(void);
extern initializer_fn __init_array_start[] __attribute__((weak));
extern initializer_fn __init_array_end[] __attribute__((weak));

/* Only own-data writes, no SCE calls or allocations before the persistent log. */
__attribute__((constructor)) static void constructor_witness(void) { constructor_witness_count++; }

/* Journal policy changes diagnostics only, never GPU completion/ownership.
 * The same mutex protects writes and bounded unsynced counters. The 1s limit
 * is checked on report calls; there is no timer thread or idle-time guarantee. */
#define ROUTINE_SYNC_RECORDS 1024u
#define ROUTINE_SYNC_BYTES (512u * 1024u)
#define ROUTINE_SYNC_NS UINT64_C(1000000000)
static unsigned routine_unsynced_records;
static size_t routine_unsynced_bytes;
static uint64_t journal_last_sync_ns;
static bool journal_sync_time_valid;
static atomic_bool journal_debug_enabled, journal_policy_configured;
static atomic_uint_fast64_t journal_routine_suppressed;
bool ps5_ai_native_journal_debug_enabled(void) {
    return atomic_load_explicit(&journal_debug_enabled,memory_order_relaxed);
}
bool ps5_ai_native_journal_configure_debug(bool enabled) {
    bool expected=false;
    if(engine_init_calls!=1 || agc_init_attempts ||
       !atomic_compare_exchange_strong(&journal_policy_configured,&expected,true))return false;
    atomic_store_explicit(&journal_debug_enabled,enabled,memory_order_relaxed);
    ps5_ai_ggml_report("journal_policy mode=%s routine_sync_records=%u routine_sync_bytes=%u routine_sync_ns=%llu critical_sync_immediate=1 GPU_checks_unchanged=1",
        enabled?"debug":"normal",ROUTINE_SYNC_RECORDS,ROUTINE_SYNC_BYTES,(unsigned long long)ROUTINE_SYNC_NS);
    return atomic_load(&log_healthy);
}
static bool monotonic_ns(uint64_t *out);
/* Optional UI producer: scalar phase/progress are atomic; bounded text copies
 * use a separate try-lock. No engine/SD/driver lock is acquired by callbacks. */
static atomic_uint ui_phase=PS5_AI_STATUS_IDLE, ui_progress;
static atomic_bool ui_progress_known, ui_job_known;
static atomic_uint ui_state_writers;
static atomic_uint_fast64_t ui_state_revision;
static pthread_mutex_t ui_text_mutex=PTHREAD_MUTEX_INITIALIZER;
static char ui_job[33],ui_lines[4][96];
static pthread_t ui_publisher_thread;
static void native_ui_phase_store(unsigned phase) {
    atomic_store(&ui_progress_known,false);
    if(phase<=PS5_AI_STATUS_STOPPING)atomic_store(&ui_phase,phase);
}
void ps5_ai_native_ui_phase(unsigned phase) {
    atomic_fetch_add(&ui_state_writers,1);
    native_ui_phase_store(phase);
    atomic_fetch_add(&ui_state_revision,1);atomic_fetch_sub(&ui_state_writers,1);
}
void ps5_ai_native_ui_job_begin(const char *id) {
    atomic_fetch_add(&ui_state_writers,1);
    native_ui_phase_store(PS5_AI_STATUS_LOADING);
    atomic_store(&ui_job_known,false);
    if(id&&strnlen(id,sizeof(ui_job))==32&&!pthread_mutex_trylock(&ui_text_mutex)) {
        memcpy(ui_job,id,sizeof(ui_job));
        if(!pthread_mutex_unlock(&ui_text_mutex))atomic_store(&ui_job_known,true);
    }
    atomic_fetch_add(&ui_state_revision,1);atomic_fetch_sub(&ui_state_writers,1);
}
void ps5_ai_native_ui_sampler_progress(unsigned value) {
    atomic_fetch_add(&ui_state_writers,1);
    if(value<=10000&&atomic_load(&ui_phase)==PS5_AI_STATUS_GENERATING) {
        atomic_store(&ui_progress,value);atomic_store(&ui_progress_known,true);
    }
    atomic_fetch_add(&ui_state_revision,1);atomic_fetch_sub(&ui_state_writers,1);
}
static int ui_private_line(const char *text) {
    static const char *words[]={"token","bearer","authorization","secret"};
    for(size_t i=0;i<1024&&text[i];i++)for(unsigned w=0;w<4;w++) {
        size_t j=0;while(words[w][j]) {
            if(i+j>=1024||!text[i+j])break;
            unsigned char c=(unsigned char)text[i+j];if(c>='A'&&c<='Z')c+=32;
            if(c!=(unsigned char)words[w][j])break;j++;
        }
        if(!words[w][j])return 1;
    }
    return 0;
}
void ps5_ai_native_ui_log(const char *text) {
    if(!text||ui_private_line(text)||pthread_mutex_trylock(&ui_text_mutex))return;
    memmove(ui_lines[0],ui_lines[1],sizeof(ui_lines)-sizeof(ui_lines[0]));
    size_t n=0;for(;n<95&&text[n];n++) {
        unsigned char c=(unsigned char)text[n];ui_lines[3][n]=c>=32&&c<127?(char)c:' ';
    }
    ui_lines[3][n]=0;(void)pthread_mutex_unlock(&ui_text_mutex);
}
static void *native_ui_publish_loop(void *unused) {
    (void)unused;
    for(;;) {
        struct ps5_ai_status_display_state_v1 screen={0};
        if(!ps5_ai_status_display_state(&screen)&&screen.stopped)return NULL;
        struct ps5_ai_status_snapshot_v1 view={0};
        view.abi_bytes=sizeof(view);view.abi_version=PS5_AI_STATUS_ABI;view.process_id=(uint32_t)getpid();
        /* Timestamp acquisition starts here; slow metadata never refreshes an
         * older sample's label. Renderer expires it after three seconds. */
        int sample_time_valid=monotonic_ns(&view.monotonic_ns);
        uint64_t revision=atomic_load(&ui_state_revision);
        int phase_coherent=!atomic_load(&ui_state_writers);
        view.valid=PS5_AI_STATUS_MODEL|PS5_AI_STATUS_PHASE;
        memcpy(view.model,"WAI-Illustrious-SDXL v17.0",sizeof("WAI-Illustrious-SDXL v17.0"));
        view.phase=atomic_load(&ui_phase);
        if(view.phase==PS5_AI_STATUS_GENERATING&&atomic_load(&ui_progress_known)) {
            view.progress_basis_points=atomic_load(&ui_progress);view.valid|=PS5_AI_STATUS_PROGRESS;
        } else if(view.phase==PS5_AI_STATUS_COMPLETED) {
            view.progress_basis_points=10000;view.valid|=PS5_AI_STATUS_PROGRESS;
        }
        if(!pthread_mutex_trylock(&ui_text_mutex)) {
            if(atomic_load(&ui_job_known)){memcpy(view.job_id,ui_job,sizeof(view.job_id));view.valid|=PS5_AI_STATUS_JOB;}
            memcpy(view.logs,ui_lines,sizeof(view.logs));view.valid|=PS5_AI_STATUS_LOGS;
            (void)pthread_mutex_unlock(&ui_text_mutex);
        }
        if(!phase_coherent||atomic_load(&ui_state_writers)||revision!=atomic_load(&ui_state_revision))
            view.valid&=~(PS5_AI_STATUS_PHASE|PS5_AI_STATUS_JOB|PS5_AI_STATUS_PROGRESS);
        /* UI mutex is released before these metadata-only existing snapshots.
         * No Vulkan call, engine status mutex, filesystem I/O or GPU wait here. */
        struct ps5_ai_flex_stats data={0};ps5_ai_flex_snapshot(&data);
        if(data.mapping_owned&&data.gpu_access_granted&&data.reserved_bytes&&
           data.used_bytes<=data.peak_used_bytes&&data.peak_used_bytes<=data.reserved_bytes) {
            view.direct_reserved_bytes=data.reserved_bytes;view.direct_used_bytes=data.used_bytes;
            view.direct_peak_bytes=data.peak_used_bytes;view.valid|=PS5_AI_STATUS_DIRECT_POOL;
        }
        /* Ordinary per-BO owned bytes are not a bounded arena capacity.
         * Keep this capacity row unavailable; the journal reports ownership. */
        struct ps5_ai_ggml_dispatch_stats counts={0};ps5_ai_ggml_dispatch_snapshot(&counts);
        view.dispatches=counts.dispatches;view.submits=counts.submits;view.fences=counts.fenced_completions;
        view.valid|=PS5_AI_STATUS_COUNTERS;
        /* No current RSS, utilization or temperature provider: all bits clear.
         * Counter values are process cumulative, not per-job GPU percentages. */
        if(sample_time_valid)(void)ps5_ai_status_publish(&view);
        for(unsigned n=0;n<10;n++)(void)usleep(100000);
    }
}
static void native_ui_start_after_real_gate(void) {
    const struct ps5_ai_status_gate_v1 gate={sizeof(gate),PS5_AI_STATUS_ABI,(uint32_t)probe_pid,1,
        UINT64_C(67108864),agc_init_attempts,0};
    ps5_ai_native_ui_phase(PS5_AI_STATUS_IDLE);
    int display_rc=ps5_ai_status_start_after_gate(&gate),publisher_rc=-1;
    if(!display_rc)publisher_rc=pthread_create(&ui_publisher_thread,NULL,native_ui_publish_loop,NULL);
    /* Optional diagnostic only; UI failure must never poison inference guards. */
    char line[160];int n=snprintf(line,sizeof(line),"native_TV_UI optional=1 display_thread_rc=%d publisher_thread_rc=%d hardware_ready_unverified=1\n",display_rc,publisher_rc);
    if(n>0&&(size_t)n<sizeof(line))(void)write(2,line,(size_t)n);
}
struct journal_metrics {
    uint64_t records, checked_write_ns, fsync_calls, fsync_ns;
    uint64_t routine_suppressed;
};
static struct journal_metrics journal_totals;

static void journal_add_elapsed(uint64_t *total, uint64_t begin) {
    uint64_t end;
    if (!monotonic_ns(&end) || end < begin || UINT64_MAX - *total < end - begin) {
        log_healthy = false;
        return;
    }
    *total += end - begin;
}

static bool journal_sync_locked(uint64_t now) {
    if (log_fd < 0) { log_healthy = false; return false; }
    uint64_t started = 0;
    bool measured = monotonic_ns(&started);
    if (!measured) log_healthy = false;
    journal_totals.fsync_calls++;
    int failed = fsync(log_fd);
    if (measured) journal_add_elapsed(&journal_totals.fsync_ns, started);
    if (failed) {
        log_healthy = false;
        return false;
    }
    routine_unsynced_records = 0;
    routine_unsynced_bytes = 0;
    journal_last_sync_ns = now;
    journal_sync_time_valid = true;
    return true;
}

static void report_impl(bool routine, const char *fmt, va_list args) {
    if (pthread_mutex_lock(&log_mutex)) { log_healthy = false; return; }
    char line[1024];
    int length = vsnprintf(line, sizeof(line)-1, fmt, args);
    if (length < 0 || (size_t)length >= sizeof(line)-1) {
        log_healthy = false;
        pthread_mutex_unlock(&log_mutex);
        return;
    }
    ps5_ai_native_ui_log(line);
    uint64_t now = 0;
    bool clock_valid = monotonic_ns(&now);
    if (!clock_valid || (journal_sync_time_valid && now < journal_last_sync_ns)) {
        /* No guessed elapsed time: fail closed and still try durable output. */
        log_healthy = false;
        routine = false;
    }
    size_t record_bytes = (size_t)length + 1u;
    if (routine && log_fd >= 0 && journal_sync_time_valid &&
        (routine_unsynced_records >= ROUTINE_SYNC_RECORDS ||
         record_bytes > ROUTINE_SYNC_BYTES - routine_unsynced_bytes ||
         now - journal_last_sync_ns >= ROUTINE_SYNC_NS)) {
        if (!journal_sync_locked(now)) routine = false;
    }
    /* Critical stderr remains a diagnostic mirror. Routine records retain
     * checked file writes and UI lines but omit this unchecked duplicate. */
    line[length] = '\n'; if (!routine) (void)write(2, line, record_bytes);
    if (log_fd >= 0) {
        uint64_t started = 0;
        bool measured = monotonic_ns(&started);
        if (!measured) log_healthy = false;
        int failed = !ps5_ai_journal_room(&log_fd,LOG_PATH,&startup_journal_cap,record_bytes,UINT64_C(4194304)); size_t sent = 0;
        while (!failed && sent < record_bytes) {
            ssize_t wrote = write(log_fd, line + sent, record_bytes - sent);
            if (wrote < 0 && errno == EINTR) continue;
            if (wrote <= 0) { failed = 1; break; }
            sent += (size_t)wrote;startup_journal_cap.bytes+=(size_t)wrote;
        }
        if (measured) journal_add_elapsed(&journal_totals.checked_write_ns, started);
        if (failed) {
            log_healthy = false;
            const char diagnostic[]="native_WAI_journal_write_failed sticky=1\n"; (void)write(2,diagnostic,sizeof(diagnostic)-1);
        } else {
            journal_totals.records++;
            if (!routine) {
                (void)journal_sync_locked(now);
            } else {
                routine_unsynced_records++;
                routine_unsynced_bytes += record_bytes;
                if (!journal_sync_time_valid ||
                    routine_unsynced_records >= ROUTINE_SYNC_RECORDS ||
                    routine_unsynced_bytes >= ROUTINE_SYNC_BYTES)
                    (void)journal_sync_locked(now);
            }
        }
    } else log_healthy = false;
    if (pthread_mutex_unlock(&log_mutex)) log_healthy = false;
}

void ps5_ai_ggml_report(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    report_impl(false, fmt, args);
    va_end(args);
}

void ps5_ai_ggml_report_routine(const char *fmt, ...) {
    if(!ps5_ai_native_journal_debug_enabled()) {
        atomic_fetch_add_explicit(&journal_routine_suppressed,1,memory_order_relaxed);
        return;
    }
    va_list args;
    va_start(args, fmt);
    report_impl(true, fmt, args);
    va_end(args);
}

static bool journal_metrics_snapshot(struct journal_metrics *out) {
    if (pthread_mutex_lock(&log_mutex)) { log_healthy = false; return false; }
    *out = journal_totals;
    out->routine_suppressed=atomic_load_explicit(&journal_routine_suppressed,memory_order_relaxed);
    if (pthread_mutex_unlock(&log_mutex)) { log_healthy = false; return false; }
    return log_healthy;
}

static void report_journal_metrics(const struct journal_metrics *before) {
    struct journal_metrics after;
    if (!journal_metrics_snapshot(&after)) return;
    if (after.records < before->records || after.checked_write_ns < before->checked_write_ns ||
        after.fsync_calls < before->fsync_calls || after.fsync_ns < before->fsync_ns) {
        log_healthy = false;
        return;
    }
    ps5_ai_ggml_report("startup_journal_metrics phase=API_after_generate records=%llu checked_write_us=%llu fsync_calls=%llu fsync_us=%llu routine_suppressed=%llu scope=generation_delta_before_summary startup_journal_only=1 stderr_mirror_and_lock_wait_excluded=1 native_platform_pool_engine_journals_excluded=1 not_total_generation_time=1",
        (unsigned long long)(after.records - before->records),
        (unsigned long long)((after.checked_write_ns - before->checked_write_ns) / 1000u),
        (unsigned long long)(after.fsync_calls - before->fsync_calls),
        (unsigned long long)((after.fsync_ns - before->fsync_ns) / 1000u),
        (unsigned long long)(after.routine_suppressed-before->routine_suppressed));
}

static void report_arena_journal_metrics(const struct ps5_ai_native_arena_journal_stats *before,uint64_t suppressed_before) {
    if(!ps5_ai_native_arena_journal_checkpoint())
        ps5_ai_ggml_retain("API_arena_journal_checkpoint_failed");
    struct ps5_ai_native_arena_journal_stats after={0};
    ps5_ai_native_arena_journal_snapshot(&after);
    if(before->abi_bytes!=sizeof(*before)||after.abi_bytes!=sizeof(after)||!before->log_healthy||!after.log_healthy||
       after.records<before->records||after.checked_write_ns<before->checked_write_ns||after.fsync_calls<before->fsync_calls||
       after.fsync_ns<before->fsync_ns||after.lock_wait_ns<before->lock_wait_ns)
        ps5_ai_ggml_retain("API_arena_journal_metrics_invalid");
    ps5_ai_ggml_report("arena_journal_metrics phase=API_after_generate records=%llu checked_write_us=%llu fsync_calls=%llu fsync_us=%llu lock_wait_us=%llu routine_suppressed=%llu scope=generation_delta_including_final_checkpoint arena_journal_only=1 stderr_timing_available=0 formatting_and_stderr_excluded=1 not_total_generation_time=1",
        (unsigned long long)(after.records-before->records),
        (unsigned long long)((after.checked_write_ns-before->checked_write_ns)/1000u),
        (unsigned long long)(after.fsync_calls-before->fsync_calls),
        (unsigned long long)((after.fsync_ns-before->fsync_ns)/1000u),
        (unsigned long long)((after.lock_wait_ns-before->lock_wait_ns)/1000u),
        (unsigned long long)(ps5_ai_native_arena_journal_suppressed()-suppressed_before));
}

/* Observe the existing known-size model file mappings, without replacing the
 * public provider or touching mapped data. A blocked call is not retried.
 * The atomic guard also suppresses recursive diagnostics from journal code.
 * Exact sizes identify scope only; this wrapper does not authenticate an fd. */
_Static_assert(sizeof(size_t) == 8 && sizeof(off_t) == 8, "Public mmap uses 64-bit size and offset");
extern void *__real_mmap(void *, size_t, int, int, int, off_t);
static atomic_flag native_mmap_diagnostic_active = ATOMIC_FLAG_INIT;
__attribute__((visibility("hidden")))
void *__wrap_mmap(void *address, size_t length, int protection, int flags, int fd, off_t offset) {
    int incoming_errno = errno;
    bool known_size = length == UINT64_C(6938040682) || length == UINT64_C(9793292);
    if (!atomic_load_explicit(&generation_active, memory_order_acquire) || !known_size ||
        fd < 0 || offset != 0 || (flags & MAP_ANON) ||
        atomic_flag_test_and_set_explicit(&native_mmap_diagnostic_active, memory_order_acquire))
        return __real_mmap(address, length, protection, flags, fd, offset);
    uint64_t began = 0, ended = 0;
    bool start_valid = monotonic_ns(&began);
    ps5_ai_ggml_report("BEFORE native_model_mmap address=%p length=%llu protection=%#x flags=%#x fd=%d offset=%lld monotonic_start_ns=%llu start_time_valid=%d public_provider=1 known_model_size_scope=1 fd_identity_authenticated=0 retries=0 mapped_data_touched=0",
        address, (unsigned long long)length, protection, flags, fd, (long long)offset,
        (unsigned long long)began, start_valid);
    if (!log_healthy) ps5_ai_ggml_retain("native_model_mmap_BEFORE_journal_failed");
    errno = incoming_errno;
    void *result = __real_mmap(address, length, protection, flags, fd, offset);
    int result_errno = errno;
    bool elapsed_valid = monotonic_ns(&ended) && start_valid && ended >= began;
    ps5_ai_ggml_report("RETURN native_model_mmap pointer=%p map_failed=%d errno=%d length=%llu protection=%#x flags=%#x fd=%d offset=%lld elapsed_ns=%llu elapsed_valid=%d public_provider=1 retries=0 mapped_data_touched=0 result_and_errno_unchanged=1",
        result, result == MAP_FAILED, result_errno, (unsigned long long)length,
        protection, flags, fd, (long long)offset,
        (unsigned long long)(elapsed_valid ? ended - began : 0), elapsed_valid);
    atomic_flag_clear_explicit(&native_mmap_diagnostic_active, memory_order_release);
    if (!log_healthy) ps5_ai_ggml_retain("native_model_mmap_RETURN_journal_failed");
    errno = result_errno;
    return result;
}

/* LLD --wrap=sceAgcInit is mandatory. RADV's call_once is the only owner of
 * initialization; main never makes a second manual call. SCE constructor-
 * internal calls/allocations are unknown and are not counted as zero here.
 */
int32_t __wrap_sceAgcInit(uint32_t version) {
    agc_init_requests++;
    ps5_ai_ggml_report("BEFORE RADV_sceAgcInit version=%u requests=%u explicit_attempts=%u hard_timeout=none constructor_internal_calls=unknown",
           version, agc_init_requests, agc_init_attempts);
    if (log_fd < 0 || !log_healthy || version != 8 || agc_init_attempts) {
        failures++; ps5_ai_ggml_report("FAIL AGC_init_guard no_actual_call=1"); return -1;
    }
    agc_init_attempts++;
    errno = 0;
    agc_init_result = __real_sceAgcInit(version);
    int saved = errno;
    ps5_ai_ggml_report("RETURN RADV_sceAgcInit signed_rc=%d hex_rc=0x%08x errno=%d explicit_attempts=%u",
           agc_init_result, (unsigned)agc_init_result, saved, agc_init_attempts);
    return agc_init_result;
}

static struct ps5_ai_arena_stats report_arena(const char *phase) {
    struct ps5_ai_arena_stats stats;
    memset(&stats, 0, sizeof(stats)); ps5_ai_arena_snapshot(&stats);
    ps5_ai_ggml_report("arena phase=%s reserved_bytes=%llu used_bytes=%llu peak_used_bytes=%llu live_allocations=%llu attempts=%llu denied=%llu physical_owned=%d mapping_owned=%d mapping_uncertain=%d fault_retained=%d global_capacity_measured=%d SCE_constructor_allocations=unknown",
           phase, (unsigned long long)stats.reserved_bytes, (unsigned long long)stats.used_bytes,
           (unsigned long long)stats.peak_used_bytes, (unsigned long long)stats.live_allocations,
           (unsigned long long)stats.allocation_attempts, (unsigned long long)stats.denied_allocations,
           stats.physical_owned, stats.mapping_owned, stats.mapping_uncertain, stats.gpu_fault_retained, stats.global_capacity_measured);
    return stats;
}

static struct ps5_ai_flex_stats report_direct(const char *phase) {
    struct ps5_ai_flex_stats stats = {0};
    ps5_ai_flex_snapshot(&stats);
    ps5_ai_ggml_report("owned_direct_data phase=%s reserved_bytes=%llu used_bytes=%llu peak_used_bytes=%llu free_bytes=%llu allocations=%llu active_imports=%llu mapping_owned=%d GPU_access_granted=%d fault_retained=%d log_healthy=%d global_capacity_measured=0",
        phase, (unsigned long long)stats.reserved_bytes, (unsigned long long)stats.used_bytes,
        (unsigned long long)stats.peak_used_bytes, (unsigned long long)stats.free_bytes,
        (unsigned long long)stats.allocations, (unsigned long long)stats.active_imports,
        stats.mapping_owned, stats.gpu_access_granted, stats.fault_retained, stats.log_healthy);
    return stats;
}

static bool monotonic_ns(uint64_t *out) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) || now.tv_sec < 0 || now.tv_nsec < 0 ||
        now.tv_nsec >= 1000000000 || (uint64_t)now.tv_sec > UINT64_MAX / UINT64_C(1000000000)) return false;
    *out = (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
    return true;
}

static bool await_backend_quiescence(const char *phase, bool expect_work) {
    backend_quiescent = false;
    /* Only these exact successful housekeeping phases may defer fsync.
     * Queries, deadlines, faults and the failure record below are unchanged. */
    const bool routine = phase && (!strcmp(phase, "GGML_internal_cleanup") ||
        !strcmp(phase, "GGML_after_fence") || !strcmp(phase, "flex_owned_buffer_free"));
    void (*success_report)(const char *, ...) = routine ? ps5_ai_ggml_report_routine : ps5_ai_ggml_report;
    success_report("BEFORE backend_quiescence phase=%s requested_timeout_ns=10000000000 metadata_poll_only=1 hard_timeout_guaranteed=0", phase);
    uint64_t begin, now;
    const char *reason = "persistent_log_unhealthy";
    if (!log_healthy) goto failed;
    reason = "monotonic_clock_failed";
    if (!monotonic_ns(&begin)) goto failed;
    for (;;) {
        reason = "backend_state_query_failed";
        int query = ps5_ai_arena_kick_snapshot(&last_kick);
        if (query < 0 || query > 1) goto failed;
        /* 1 is a busy try-lock, not a GPU failure or a valid new snapshot.
         * Poll again within the same deadline; this never retries GPU work. */
        if (!query) {
            reason = "backend_fault_or_log_failed";
            if (last_kick.fault_retained || !last_kick.log_healthy || !log_healthy) goto failed;
            reason = "backend_epoch_invalid";
            if (last_kick.completed_epoch > last_kick.requested_epoch) goto failed;
            if (!last_kick.pending && !last_kick.active && !last_kick.submit_inflight &&
                last_kick.requested_epoch == last_kick.completed_epoch) {
                reason = "submitted_work_without_started_kick";
                if (expect_work && (!last_kick.initialized || !last_kick.thread_started || !last_kick.requested_epoch)) goto failed;
                backend_quiescent = true;
                success_report("PASS backend_quiescence phase=%s initialized=%d thread_started=%d pending=%d active=%d submit_inflight=%d requested_epoch=%llu completed_epoch=%llu backend_fault_retained=0 backend_log_healthy=1",
                       phase, last_kick.initialized, last_kick.thread_started, last_kick.pending, last_kick.active,
                       last_kick.submit_inflight, (unsigned long long)last_kick.requested_epoch, (unsigned long long)last_kick.completed_epoch);
                if (log_healthy) return true;
                backend_quiescent = false; reason = "quiescence_log_failed"; goto failed;
            }
        }
        reason = "monotonic_clock_failed_or_reversed";
        if (!monotonic_ns(&now) || now < begin) goto failed;
        reason = "backend_quiescence_timeout";
        if (now - begin >= UINT64_C(10000000000)) goto failed;
        reason = "metadata_poll_sleep_failed";
        if (usleep(1000) && errno != EINTR) goto failed;
    }
failed:
    failures++;
    ps5_ai_ggml_report("FAIL backend_quiescence phase=%s reason=%s pending=%d active=%d submit_inflight=%d requested_epoch=%llu completed_epoch=%llu backend_fault_retained=%d backend_log_healthy=%d retain_all_resources=1 no_retry=1",
           phase, reason, last_kick.pending, last_kick.active, last_kick.submit_inflight,
           (unsigned long long)last_kick.requested_epoch, (unsigned long long)last_kick.completed_epoch,
           last_kick.fault_retained, last_kick.log_healthy);
    return false;
}

int ps5_ai_ggml_log_healthy(void) { return atomic_load(&log_healthy); }
__attribute__((noreturn)) void ps5_ai_ggml_cpu_allocation_failure(void) {
    if (log_fd < 0) { const char message[] = "GGML CPU allocation failed before persistent BEGIN; GPU not entered\n"; (void)write(2, message, sizeof(message)-1); for (;;) sleep(1); }
    ps5_ai_ggml_retain("CPU_CPP_allocation_failure_no_unwind");
}
int ps5_ai_ggml_wait_backend(const char *phase, int expect_work) {
    if (pthread_mutex_lock(&metadata_mutex)) return 0;
    int ready = await_backend_quiescence(phase, expect_work != 0);
    if (pthread_mutex_unlock(&metadata_mutex)) return 0;
    return ready;
}
__attribute__((noreturn)) void ps5_ai_ggml_retain(const char *reason) {
    ps5_ai_native_ui_phase(PS5_AI_STATUS_FAILED);
    ps5_ai_native_ui_log(reason);
    retention_entered = true;
    ps5_ai_native_api_retained_failure(reason);
    shutdown_proven = false;
    ps5_ai_arena_retain_gpu_fault();
    ps5_ai_ggml_report("END status=failed API_GPU_inference_verified=0 reason=%s all_GPU_resources_retained=1 process_exit=forbidden no_retry=1 image_result=unverified", reason);
    for (;;) sleep(1);
}

int __real_ps5_ai_engine_init(const char *, char *, size_t);
int __real_ps5_ai_engine_generate(const ps5_ai_request *, const char *, ps5_ai_stats *,
    ps5_ai_progress_cb, void *, char *, size_t);
void __real_ps5_ai_engine_shutdown(void);

static bool on_controller_main_thread(void) {
    return controller_main_thread_valid && pthread_equal(pthread_self(), controller_main_thread);
}

void ps5_ai_v3_shutdown_request_accepted(int running, int queued) {
    if (!on_controller_main_thread() || ++shutdown_request_calls != 1 || shutdown_requested ||
        running || queued || generation_active || retention_entered || !log_healthy)
        ps5_ai_ggml_retain("V3_admin_shutdown_request_guard_failed");
    ps5_ai_ggml_report("V3_ADMIN_SHUTDOWN_ACCEPTED pid=%d running=0 queued=0 authenticated_route=1 main_thread=1 signals_used=0", (int)probe_pid);
    if (!log_healthy) ps5_ai_ggml_retain("V3_admin_shutdown_receipt_failed");
    shutdown_requested = true;
}

void ps5_ai_v3_worker_join_receipt(int join_rc, int running, int queued, int closed) {
    if (!on_controller_main_thread() || ++worker_join_calls != 1 || !shutdown_requested ||
        join_rc || running || queued || closed != 1 || generation_active || retention_entered || !log_healthy)
        ps5_ai_ggml_retain("V3_listener_or_worker_join_unproven");
    ps5_ai_ggml_report("V3_WORKER_JOINED pid=%d join_rc=0 listener_closed=1 running=0 queued=0 generation_active=0", (int)probe_pid);
    if (!log_healthy) ps5_ai_ggml_retain("V3_worker_join_receipt_failed");
    listener_closed = true;
    worker_joined = true;
}

static struct ps5_ai_v3_exit_state exit_state(int server_rc) {
    struct ps5_ai_ggml_dispatch_stats dispatch = {0};
    struct ps5_ai_arena_stats arena = {0};
    ps5_ai_ggml_dispatch_snapshot(&dispatch);
    ps5_ai_arena_snapshot(&arena);
    return (struct ps5_ai_v3_exit_state) {
        shutdown_requested, listener_closed, worker_joined, model_cleanup_checked,
        pre_cleanup_quiescent, post_cleanup_quiescent, !generation_active,
        dispatch.work_pending, backend_quiescent, arena.gpu_fault_retained,
        arena.mapping_uncertain, retention_entered, log_healthy, server_rc
    };
}

static int init_error(char *error, size_t size, const char *message) {
    ps5_ai_native_ui_phase(PS5_AI_STATUS_FAILED);
    if (error && size) snprintf(error, size, "%s", message);
    ps5_ai_ggml_report("API_engine_unavailable reason=%s startup_failures=%d backend_quiescent=%d model_loaded=0 image_generated=0", message, failures, (int)backend_quiescent);
    struct ps5_ai_arena_stats stats = report_arena("init_failure");
    struct ps5_ai_flex_stats flex = report_direct("init_failure");
    if (stats.mapping_uncertain || stats.gpu_fault_retained || flex.fault_retained || !flex.log_healthy)
        ps5_ai_ggml_retain("API_init_ownership_uncertain");
    return -1;
}

static int native_capture_stderr(void) {
    /* Checked startup/driver journals and structured engine events already
     * retain errors. Avoid an unbounded duplicate raw stderr file. */
    ps5_ai_ggml_report("native_stderr_capture disabled=1 duplicate_raw_file=0 checked_bounded_journals_retained=1");
    return atomic_load(&log_healthy)?0:-1;
}

int __wrap_ps5_ai_engine_init(const char *config_path, char *error, size_t error_size) {
    if (++engine_init_calls != 1 || !config_path || strcmp(config_path,"/app0/ai-runtime/config.json")) {
        if (error && error_size) snprintf(error,error_size,"Invalid/repeated fixed native API configuration");
        return -1;
    }
    log_fd=open(LOG_PATH,O_WRONLY|O_CREAT|O_APPEND|O_NOFOLLOW,0600);
    struct stat st;
    if(log_fd<0 || fstat(log_fd,&st) || !S_ISREG(st.st_mode)) {
        log_healthy=false;
        if(error&&error_size)snprintf(error,error_size,"Native persistent raw-fd journal unavailable");
        return -1;
    }
    probe_pid=getpid();
    size_t linked=__init_array_start&&__init_array_end?(size_t)(__init_array_end-__init_array_start):0;
    ps5_ai_ggml_report("BEGIN pid=%d unix_time=%lld code=ps5-ai-native-wai-api-v1 native_CRT=1 payload_CRT=0 private_FW_or_credential_queries=0 manual_constructor_replay=0 config=%s",(int)probe_pid,(long long)time(NULL),config_path);
    ps5_ai_ggml_report("constructors linked_initializer_slots=%zu own_witness_executed=%u SCE_constructor_effects=unknown",linked,constructor_witness_count);
    if(probe_pid<=0||linked!=PS5_AI_NATIVE_EXPECTED_INITIALIZER_SLOTS||constructor_witness_count!=1||!log_healthy||!radv_ps5_platform_runs_gpu())
        return init_error(error,error_size,"Native constructor/hardware/log guard failed");
    if(native_capture_stderr())return init_error(error,error_size,"Native own stderr capture unavailable");
    struct ps5_ai_memory_plan plan = {0};
    ps5_ai_native_config_mark(1,"memory_plan_before");
    if (ps5_ai_v12_read_memory_plan(config_path,&plan,error,error_size) ||
        plan.gpu_pool_bytes != UINT64_C(6442450944) || plan.working_reserve_bytes != UINT64_C(1073741824))
        return init_error(error,error_size,"Fixed native Direct6GiB/CPUreserve1GiB configuration required");
    ps5_ai_native_config_mark(16,"memory_plan_returned");
    ps5_ai_ggml_report("native_Direct_plan target_bytes=%llu region_max_bytes=268435456 CPU_Direct_reserve_bytes=%llu CPU_whole_pool_prefault=0 Mprotect_calls=0 capacity_execution_receipt_required_before_deployment=1",
        (unsigned long long)plan.gpu_pool_bytes,(unsigned long long)plan.working_reserve_bytes);
    if (!log_healthy || ps5_ai_native_arena_open_log(1))
        return init_error(error,error_size,"Native arena persistent journal unavailable");
    if (setenv("GGML_NO_BACKTRACE","1",1) || setenv("MESA_SHADER_CACHE_DISABLE","true",1) ||
        setenv("MESA_SHADER_CACHE_DIR","/app0/ai-runtime/logs",1) || setenv("TMPDIR","/app0/ai-runtime",1) ||
        (PS5_AI_ENABLE_FP16 ? unsetenv("GGML_VK_DISABLE_F16") : setenv("GGML_VK_DISABLE_F16","1",1)) || setenv("GGML_VK_DISABLE_ASYNC","1",1) ||
        setenv("GGML_VK_SERIALIZE_SUBMISSIONS","1",1) || setenv("GGML_VK_DISABLE_FUSION","1",1) ||
        setenv("GGML_VK_DISABLE_COOPMAT","1",1) || setenv("GGML_VK_DISABLE_COOPMAT2","1",1) ||
        setenv("GGML_VK_SUBALLOCATION_BLOCK_SIZE","67108864",1) || setenv("GGML_VK_VISIBLE_DEVICES","0",1))
        return init_error(error,error_size,"Native GPU environment guard failed");
    ps5_ai_ggml_report("BEFORE native_RADV_platform_init sole_Init8_owner=1 hard_timeout=none");
    if (!log_healthy) return init_error(error,error_size,"Native Init8 BEFORE log failed");
    bool ready=radv_ps5_platform_init();
    ps5_ai_ggml_report("RETURN native_RADV_platform_init ready=%d requests=%u attempts=%u init_rc=0x%08x",ready,agc_init_requests,agc_init_attempts,(unsigned)agc_init_result);
    if (!ready || agc_init_requests!=1 || agc_init_attempts!=1 || agc_init_result || !log_healthy)
        return init_error(error,error_size,"Native sole Init8 failed");
    native_driver_prepare_entered=true;
    if (ps5_ai_arena_prepare()) return init_error(error,error_size,"Native per-BO owner prepare failed");
    struct ps5_ai_arena_stats arena=report_arena("native_owner_prepared");
    /* Zero owned BOs is valid before Vulkan creates its first internal object. */
    if (!ps5_ai_driver_owner_ready() || arena.reserved_bytes || arena.used_bytes || arena.live_allocations ||
        arena.physical_owned || arena.mapping_owned || arena.mapping_uncertain ||
        arena.gpu_fault_retained || arena.global_capacity_measured || !log_healthy)
        return init_error(error,error_size,"Native per-BO empty-owner guard failed");
    native_data_prepare_entered=true;
    if (ps5_ai_direct_pool_configure(plan.gpu_pool_bytes,UINT64_C(268435456),plan.working_reserve_bytes) ||
        ps5_ai_flex_prepare()) return init_error(error,error_size,"Native actual Direct data admission failed");
    struct ps5_ai_direct_pool_stats direct={0}; ps5_ai_direct_pool_snapshot(&direct);
    struct ps5_ai_flex_stats pool=report_direct("native_prepared");
    if (!direct.ready || direct.ownership_uncertain || direct.physical_owned_bytes!=plan.gpu_pool_bytes ||
        direct.mapped_bytes!=plan.gpu_pool_bytes || direct.largest_region_bytes!=UINT64_C(268435456) ||
        direct.region_count!=24 || pool.reserved_bytes!=plan.gpu_pool_bytes || !pool.mapping_owned ||
        !pool.gpu_access_granted || pool.fault_retained || !pool.log_healthy || !log_healthy)
        return init_error(error,error_size,"Native owned region accounting incomplete");
    int rc=__real_ps5_ai_engine_init(config_path,error,error_size);
    if (rc) return init_error(error,error_size,"Native real engine initialization failed");
    ps5_ai_ggml_graph_phase(1);
    int gate=ps5_ai_native_gpu_data_gate(error,error_size);
    ps5_ai_ggml_graph_phase(0);
    struct ps5_ai_ggml_dispatch_stats state={0}; ps5_ai_ggml_dispatch_snapshot(&state);
    if (state.work_pending || !ps5_ai_ggml_wait_backend("native_data_gate_final",state.submits!=0))
        ps5_ai_ggml_retain("native_data_gate_completion_unknown");
    if (!ps5_ai_driver_retire_quiet(last_kick.completed_epoch))
        ps5_ai_ggml_retain("native_data_gate_perBO_retirement_unconfirmed");
    if(!ps5_ai_native_arena_journal_checkpoint())
        ps5_ai_ggml_retain("native_data_gate_arena_journal_checkpoint_failed");
    pool=report_direct("native_data_gate_final");
    if (pool.fault_retained || !pool.log_healthy || !log_healthy)
        ps5_ai_ggml_retain("native_data_gate_ownership_or_log_unknown");
    if (gate) return init_error(error,error_size,"Native same-context Direct GPU gate failed");
    engine_started=true;
    native_ui_start_after_real_gate();
    ps5_ai_ggml_report("NATIVE_ENGINE_READY pid=%d Direct_data_reserved=%llu single_SCALE_tensor_verified_bytes=67108864 whole_pool_GPU_verified=0 model_loaded=0 GPU_UNet_image_verified=0 engine_ready=1",
        (int)probe_pid,(unsigned long long)pool.reserved_bytes);
    if (!log_healthy) ps5_ai_ggml_retain("native_ready_receipt_failed");
    return 0;
}

int __wrap_ps5_ai_engine_generate(const ps5_ai_request *request, const char *path, ps5_ai_stats *stats,
    ps5_ai_progress_cb progress, void *user, char *error, size_t error_size) {
    if (!engine_started || shutdown_requested || retention_entered || !log_healthy || !stats || !ps5_ai_ggml_wait_backend("API_before_generate", 0)) {
        if (error && error_size) snprintf(error, error_size, "GPU engine/startup/backend unavailable");
        return -1;
    }
    struct ps5_ai_ggml_dispatch_stats before = {0}, after = {0};
    ps5_ai_ggml_dispatch_snapshot(&before);
    if (before.work_pending) ps5_ai_ggml_retain("API_pending_before_generate");
    bool inactive = false;
    if (!atomic_compare_exchange_strong(&generation_active, &inactive, true))
        ps5_ai_ggml_retain("V3_overlapping_generation_rejected");
    struct journal_metrics journal_before;
    if (!journal_metrics_snapshot(&journal_before))
        ps5_ai_ggml_retain("API_journal_metrics_snapshot_failed");
    struct ps5_ai_native_arena_journal_stats arena_journal_before={0};
    ps5_ai_native_arena_journal_snapshot(&arena_journal_before);
    const uint64_t arena_suppressed_before=ps5_ai_native_arena_journal_suppressed();
    if(arena_journal_before.abi_bytes!=sizeof(arena_journal_before)||!arena_journal_before.log_healthy)
        ps5_ai_ggml_retain("API_arena_journal_metrics_snapshot_failed");
    ps5_ai_ggml_report("BEFORE API_generate width=%d height=%d steps=%d seed=%lld Windows_inference=0 model_policy_from_config=1",
        request ? request->width : 0, request ? request->height : 0, request ? request->steps : 0,
        (long long)(request ? request->seed : 0));
    if (!log_healthy) ps5_ai_ggml_retain("API_generate_BEFORE_log_failed");
    ps5_ai_ggml_graph_phase(1);
    int rc = __real_ps5_ai_engine_generate(request, path, stats, progress, user, error, error_size);
    ps5_ai_ggml_graph_phase(0);
    ps5_ai_ggml_dispatch_snapshot(&after);
    if (after.work_pending || !ps5_ai_ggml_wait_backend("API_after_generate", after.submits != 0))
        ps5_ai_ggml_retain("API_generate_completion_uncertain");
    bool actual_gpu_math = after.hardware_verified && after.dispatches > before.dispatches && after.submits > before.submits &&
        after.fenced_completions > before.fenced_completions && stats->gpu_used;
    if (!ps5_ai_driver_retire_quiet(last_kick.completed_epoch))
        ps5_ai_ggml_retain("API_generate_perBO_retirement_unconfirmed");
    struct ps5_ai_arena_stats arena = report_arena("API_after_generate");
    struct ps5_ai_flex_stats flex = report_direct("API_after_generate");
    if (arena.gpu_fault_retained || flex.fault_retained || !flex.log_healthy)
        ps5_ai_ggml_retain("API_generate_backend_fault");
    stats->gpu_used = actual_gpu_math;
    stats->peak_gpu_bytes = arena.peak_used_bytes + flex.peak_used_bytes;
    report_journal_metrics(&journal_before);
    report_arena_journal_metrics(&arena_journal_before,arena_suppressed_before);
    if (!rc && (!actual_gpu_math || !log_healthy)) {
        if (error && error_size) snprintf(error, error_size, "Image rejected without actual completed GPU math receipt");
        rc = -1;
    }
    ps5_ai_ggml_report("API_generate_return rc=%d completed_GPU_UNet_math=%d dispatch_delta=%llu command_submit_delta=%llu fence_delta=%llu owned_GPU_peak_upper_bound_bytes=%llu measurement=sum_of_owned_pool_peaks_not_global_VRAM GPU_scope=UNet_weight_math_required",
        rc, actual_gpu_math, (unsigned long long)(after.dispatches-before.dispatches),
        (unsigned long long)(after.submits-before.submits), (unsigned long long)(after.fenced_completions-before.fenced_completions),
        (unsigned long long)stats->peak_gpu_bytes);
    generation_active = false;
    return rc;
}

void __wrap_ps5_ai_engine_shutdown(void) {
    if (!on_controller_main_thread() || ++shutdown_calls != 1 || !shutdown_requested ||
        !worker_joined || !listener_closed || generation_active || retention_entered || !log_healthy)
        ps5_ai_ggml_retain("V3_shutdown_without_proven_idle_worker_join");
    struct ps5_ai_ggml_dispatch_stats state = {0}; ps5_ai_ggml_dispatch_snapshot(&state);
    if (state.work_pending || !ps5_ai_ggml_wait_backend("V3_shutdown_before_model_cleanup", state.submits != 0))
        ps5_ai_ggml_retain("API_shutdown_pending_or_uncertain");
    pre_cleanup_quiescent = true;
    ps5_ai_ggml_report("V3_BEFORE_CHECKED_MODEL_CLEANUP pid=%d own_worker_joined=1 hard_timeout=none", (int)probe_pid);
    if (!log_healthy) ps5_ai_ggml_retain("V3_cleanup_BEFORE_log_failed");
    char error[256] = {0};
    int cleanup_rc = ps5_ai_engine_shutdown_checked(error, sizeof(error));
    ps5_ai_ggml_report("V3_CHECKED_MODEL_CLEANUP_RETURN pid=%d rc=%d", (int)probe_pid, cleanup_rc);
    if (cleanup_rc || !log_healthy) ps5_ai_ggml_retain("V3_model_cleanup_failed_or_unproven");
    model_cleanup_checked = true;
    ps5_ai_ggml_dispatch_snapshot(&state);
    if (state.work_pending || !ps5_ai_ggml_wait_backend("V3_shutdown_after_model_cleanup", state.submits != 0))
        ps5_ai_ggml_retain("V3_post_cleanup_GPU_state_uncertain");
    if (!ps5_ai_driver_retire_quiet(last_kick.completed_epoch))
        ps5_ai_ggml_retain("V3_post_cleanup_perBO_retirement_unconfirmed");
    if(!ps5_ai_native_arena_journal_checkpoint())
        ps5_ai_ggml_retain("V3_post_cleanup_arena_journal_checkpoint_failed");
    post_cleanup_quiescent = true;
    (void)report_arena("V3_shutdown_after_model_cleanup");
    struct ps5_ai_flex_stats flex = report_direct("native_shutdown_after_model_cleanup");
    if (flex.fault_retained || !flex.log_healthy) ps5_ai_ggml_retain("native_shutdown_direct_state_uncertain");
    struct ps5_ai_v3_exit_state proof = exit_state(0);
    if (!ps5_ai_v3_exit_allowed(&proof)) ps5_ai_ggml_retain("V3_shutdown_health_proof_failed");
    ps5_ai_ggml_report("V3_SHUTDOWN_PROVEN pid=%d listener_closed=1 worker_joined=1 model_cleanup_checked=1 pre_cleanup_quiet=1 post_cleanup_quiet=1 pending=0 fault=0 durable_log=1 global_queue_retirement=unimplemented SCE_ownership_and_OS_reclaim=unmeasured", (int)probe_pid);
    if (!log_healthy) ps5_ai_ggml_retain("V3_shutdown_proof_log_failed");
    shutdown_proven = true;
    engine_started = false;
}

static atomic_bool native_abort_diagnostic_entered;
__attribute__((noreturn)) void __wrap_abort(void);
static void native_abort_diagnostic(const char *kind,uintptr_t caller,
                                    const char *function,const char *file,int line,const char *expression) {
    bool expected=false;
    if(!atomic_compare_exchange_strong(&native_abort_diagnostic_entered,&expected,true))return;
    char text[2048];
    uintptr_t wrapper=(uintptr_t)(void *)&__wrap_abort;
    int n=snprintf(text,sizeof(text),"NATIVE_ABORT kind=%s caller_pc=%#llx wrapper_pc=%#llx signed_relative=%lld config_stage=%u engine_init_calls=%d explicit_AGC_requests=%u explicit_AGC_attempts=%u driver_prepare_entered=%d data_prepare_entered=%d GPU_quiescence_claimed=0 function=%.100s file=%.250s line=%d expression=%.250s\n",
        kind,(unsigned long long)caller,(unsigned long long)wrapper,(long long)((int64_t)caller-(int64_t)wrapper),
        atomic_load_explicit(&native_config_stage,memory_order_relaxed),engine_init_calls,
        agc_init_requests,agc_init_attempts,(int)native_driver_prepare_entered,(int)native_data_prepare_entered,
        function?function:"unavailable",file?file:"unavailable",line,expression?expression:"unavailable");
    if(n>0&&(size_t)n<sizeof(text)&&log_fd>=0) {
        size_t sent=0;while(sent<(size_t)n){ssize_t done=write(log_fd,text+sent,(size_t)n-sent);if(done<0&&errno==EINTR)continue;if(done<=0||(size_t)done>(size_t)n-sent)break;sent+=(size_t)done;}
        if(sent!=(size_t)n||fsync(log_fd))log_healthy=false;
    }
    if(native_stderr_redirected)(void)fsync(STDERR_FILENO);
}
__attribute__((noreturn)) void __wrap_abort(void) {
    native_abort_diagnostic("abort",(uintptr_t)__builtin_return_address(0),NULL,NULL,0,NULL);
    if (log_fd < 0) {
        const char message[] = "SD/API library abort before persistent startup; own GPU not entered\n";
        (void)write(2, message, sizeof(message)-1); for (;;) sleep(1);
    }
    ps5_ai_ggml_retain("SD_or_GGML_library_abort_no_implicit_teardown");
}
__attribute__((noreturn)) void ps5_ai_native_assert(const char *function,const char *file,int line,const char *expression) {
    native_abort_diagnostic("assert",(uintptr_t)__builtin_return_address(0),function,file,line,expression);
    __wrap_abort();
}

int ps5_ai_server_main(int, char **);
/* Public typed API used by pinned cobanov6323 ps5/app/main.cpp95,240. */
int sceSystemServiceLoadExec(const char *path, const char *const *args);
#include "bootstrap.inc"
int main(int argc, char **argv) {
    controller_main_thread = pthread_self();
    controller_main_thread_valid = true;
    if(!native_bootstrap()) {
        const char message[]="native_WAI_bootstrap_failed_or_timeout native_resident=1 GPU_not_started=1 SCE_constructor_effects=unknown\n";
        (void)write(2,message,sizeof(message)-1);for(;;)sleep(1);
    }
    (void)argc; (void)argv;
    /* Native LaunchApp argv semantics are not assumed. Own fixed defaults are
     * compiled in; this does not accept or reinterpret arbitrary OS args. */
    char program[]="native-wai-direct"; char *fixed_argv[]={program,NULL};
    int rc = ps5_ai_server_main(1, fixed_argv);
    ps5_ai_native_ui_phase(PS5_AI_STATUS_STOPPING);
    ps5_ai_native_ui_log("API stopped; cooperative exit requires checked shutdown");
    /* Also cover API startup failures after engine initialization, such as a
     * failed worker allocation. Returning from main must not retire live GPU
     * mappings or the detached global worker implicitly. */
    struct ps5_ai_arena_stats arena = {0}; ps5_ai_arena_snapshot(&arena);
    struct ps5_ai_flex_stats flex = {0}; ps5_ai_flex_snapshot(&flex);
    /* A failed ownership attempt may leave no positive owned flag. Fault or
     * uncertain state must still forbid every fallback OS-exit path. */
    if (retention_entered || generation_active || arena.mapping_uncertain ||
        arena.gpu_fault_retained || flex.fault_retained || !flex.log_healthy || (log_fd >= 0 && !log_healthy))
        ps5_ai_ggml_retain("V3_main_return_fault_or_ownership_uncertain");
    if (!rc && shutdown_proven) {
        struct ps5_ai_v3_exit_state proof=exit_state(rc);
        if (!ps5_ai_v3_exit_allowed(&proof) || !ps5_ai_driver_owner_ready() ||
            !ps5_ai_ggml_wait_backend("native_cooperative_exit",1))
            ps5_ai_ggml_retain("native_cooperative_exit_health_unconfirmed");
        ps5_ai_ggml_report("BEFORE native_cooperative_LoadExec path=exit args_NULL=1 attempt=1 shutdown_proven=1 manual_GPU_release=0 SCE_retirement_unmeasured=1");
        if (!log_healthy) ps5_ai_ggml_retain("native_cooperative_exit_BEFORE_log_failed");
        int exit_rc=sceSystemServiceLoadExec("exit",NULL);
        ps5_ai_ggml_report("RETURN native_cooperative_LoadExec rc=%d rc_hex=0x%08x exit_unconfirmed=1 no_retry=1 native_resident=1",exit_rc,(unsigned)exit_rc);
    }
    ps5_ai_ggml_report("NATIVE_API_MAIN_RESIDENT rc=%d shutdown_proven=%d automatic_process_exit=0 manual_GPU_release=0 SCE_retirement_unmeasured=1",rc,(int)shutdown_proven);
    /* Native process lifetime is retained; the root may later select a normal
     * trusted OS title-close operation using fresh identity and idle proof. */
    for (;;) sleep(1);
}
