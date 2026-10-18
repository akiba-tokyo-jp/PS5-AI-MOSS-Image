/* Match the already deployed startup journal's bounded routine policy.
 * Routine means checked write now; only fsync may be deferred. Every hardware
 * BEFORE, failure, retention and final ownership record remains critical. */
#include "bounded_journal.h"
static struct ps5_ai_bounded_journal arena_journal_cap;
#define ARENA_ROUTINE_SYNC_RECORDS UINT64_C(1024)
#define ARENA_ROUTINE_SYNC_BYTES UINT64_C(524288)
#define ARENA_ROUTINE_SYNC_NS UINT64_C(1000000000)
static struct ps5_ai_native_arena_journal_stats arena_journal_totals;
_Static_assert(sizeof(struct ps5_ai_native_arena_journal_stats)==48,"Owned arena journal POD ABI");
static uint64_t arena_routine_records, arena_routine_bytes, arena_last_sync_ns;
static bool arena_sync_time_valid;
static atomic_uint_fast64_t arena_routine_suppressed;

static void arena_lock(atomic_flag *lock) {
    while (atomic_flag_test_and_set_explicit(lock, memory_order_acquire)) __builtin_ia32_pause();
}
static void arena_unlock(atomic_flag *lock) { atomic_flag_clear_explicit(lock, memory_order_release); }

static bool arena_journal_now(uint64_t *out) {
    struct timespec now;
    if(clock_gettime(CLOCK_MONOTONIC,&now)||now.tv_sec<0||now.tv_nsec<0||now.tv_nsec>=1000000000L||
       (uint64_t)now.tv_sec>(UINT64_MAX-(uint64_t)now.tv_nsec)/UINT64_C(1000000000))return false;
    *out=(uint64_t)now.tv_sec*UINT64_C(1000000000)+(uint64_t)now.tv_nsec;return true;
}
static bool arena_journal_add(uint64_t *total,uint64_t value) {
    if(UINT64_MAX-*total<value)return false;*total+=value;return true;
}
static bool arena_journal_elapsed(uint64_t *total,uint64_t began) {
    uint64_t end;return arena_journal_now(&end)&&end>=began&&arena_journal_add(total,end-began);
}
static bool arena_journal_sync(int64_t *failure_rc,int *saved,const char **stage) {
    uint64_t began;*stage="fsync_clock";
    if(!arena_journal_now(&began)||!arena_journal_add(&arena_journal_totals.fsync_calls,1))return false;
    errno=0;int rc=fsync(arena_fd);*failure_rc=rc;*saved=errno;*stage="fsync";
    if(!arena_journal_elapsed(&arena_journal_totals.fsync_ns,began)||rc)return false;
    arena_routine_records=0;arena_routine_bytes=0;
    if(!arena_journal_now(&arena_last_sync_ns)){*stage="fsync_completion_clock";return false;}
    arena_sync_time_valid=true;return true;
}
static bool ps5_ai_arena_trace_impl(bool routine,const char *format,va_list arguments) {
    char line[768];int count=vsnprintf(line,sizeof(line)-1,format,arguments);
    if(count<0 || (size_t)count>=sizeof(line)-1) { atomic_store(&arena_log_healthy,false);return false; }
    line[count++]='\n';uint64_t lock_began=0;bool lock_clock=arena_journal_now(&lock_began);arena_lock(&arena_log_mutex);
    bool failed=arena_fd<0 || !atomic_load(&arena_log_healthy);int saved=0;int64_t failure_rc=-1;const char *stage="sticky_or_missing_fd";
    if(!failed&&(!lock_clock||!arena_journal_elapsed(&arena_journal_totals.lock_wait_ns,lock_began))) {failed=true;stage="lock_clock";}
    uint64_t now=0;
    if(!failed&&(!arena_journal_now(&now)||(arena_sync_time_valid&&now<arena_last_sync_ns))) {failed=true;stage="record_clock";}
    if(!failed&&routine&&arena_sync_time_valid&&
       (arena_routine_records>=ARENA_ROUTINE_SYNC_RECORDS||
        (uint64_t)count>ARENA_ROUTINE_SYNC_BYTES-arena_routine_bytes||now-arena_last_sync_ns>=ARENA_ROUTINE_SYNC_NS))
        failed=!arena_journal_sync(&failure_rc,&saved,&stage);
    uint64_t began=0;
    if(!failed&&!arena_journal_now(&began)) {failed=true;stage="write_clock";}
    if(!failed && !ps5_ai_journal_room(&arena_fd,"/download0/ai-native-bounded-logs-v2/native-driver.log",&arena_journal_cap,(size_t)count,UINT64_C(4194304))){failed=true;stage="rotation";saved=errno;}
    size_t sent=0;
    while(!failed && sent<(size_t)count) {
        errno=0;ssize_t n=write(arena_fd,line+sent,(size_t)count-sent);saved=errno;
        if(n<0 && saved==EINTR)continue;
        if(n<=0) { failed=true;failure_rc=n;stage="write";break; }
        sent+=(size_t)n;arena_journal_cap.bytes+=(size_t)n;
    }
    if(!failed&&(!arena_journal_elapsed(&arena_journal_totals.checked_write_ns,began)||
       !arena_journal_add(&arena_journal_totals.records,1))) {failed=true;stage="write_metric";}
    if(!failed) {
        if(!routine)failed=!arena_journal_sync(&failure_rc,&saved,&stage);
        else {
            arena_routine_records++;arena_routine_bytes+=(uint64_t)count;
            if(!arena_sync_time_valid||arena_routine_records>=ARENA_ROUTINE_SYNC_RECORDS||arena_routine_bytes>=ARENA_ROUTINE_SYNC_BYTES)
                failed=!arena_journal_sync(&failure_rc,&saved,&stage);
        }
    }
    if(failed) {
        atomic_store(&arena_log_healthy,false);
        char diagnostic[192];int n=snprintf(diagnostic,sizeof(diagnostic),"native_arena_journal_failure stage=%s rc=%lld errno=%d log_healthy=0\n",stage,(long long)failure_rc,saved);
        if(n>0 && (size_t)n<sizeof(diagnostic)) {
            (void)write(2,diagnostic,(size_t)n);
            if(arena_fd>=0) { (void)write(arena_fd,diagnostic,(size_t)n);(void)fsync(arena_fd); }
        }
    }
    arena_unlock(&arena_log_mutex);return !failed;
}
static bool ps5_ai_arena_trace(const char *format,...) {
    va_list arguments;va_start(arguments,format);bool ok=ps5_ai_arena_trace_impl(false,format,arguments);va_end(arguments);return ok;
}
static bool ps5_ai_arena_trace_routine(const char *format,...) {
    if(!ps5_ai_native_journal_debug_enabled()) {
        atomic_fetch_add_explicit(&arena_routine_suppressed,1,memory_order_relaxed);
        return atomic_load(&arena_log_healthy);
    }
    va_list arguments;va_start(arguments,format);bool ok=ps5_ai_arena_trace_impl(true,format,arguments);va_end(arguments);return ok;
}
void ps5_ai_native_arena_journal_snapshot(struct ps5_ai_native_arena_journal_stats *out) {
    if(!out)return;arena_lock(&arena_log_mutex);*out=arena_journal_totals;
    out->log_healthy=atomic_load(&arena_log_healthy);out->abi_bytes=sizeof(*out);arena_unlock(&arena_log_mutex);
}
bool ps5_ai_native_arena_journal_checkpoint(void) {
    return ps5_ai_arena_trace("arena_journal_checkpoint critical=1 routine_success_flush_requested=1");
}
uint64_t ps5_ai_native_arena_journal_suppressed(void) {
    return atomic_load_explicit(&arena_routine_suppressed,memory_order_relaxed);
}

