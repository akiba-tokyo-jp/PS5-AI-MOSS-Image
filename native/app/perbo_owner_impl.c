/* SPDX-License-Identifier: MIT
 * Ordinary BO placement follows pinned PS5_Mesa 1ba9874f, platform.c290–351.
 * Public classic-six Direct(type12)/Map33 only. No fixed physical arena.
 * Free callbacks only retire CPU metadata. Kernel retirement is performed
 * solely by the frontend's existing positive-quiet, single-producer boundary.
 */
#include "arena.h"
#include "placement.h"
#include "radv_ps5_platform.h"
#include <stdatomic.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "journal_metrics.h"
#ifndef PS5_AI_PERBO_HOST_TEST
#include <ps5platform/kernel.h>
#endif

_Static_assert(sizeof(struct radv_ps5_memory)==40,"Existing winsys memory ABI");
_Static_assert(sizeof(struct ps5_ai_arena_stats)==72,"Existing owner statistics ABI");
static struct ps5_ai_arena_stats arena_state;
static atomic_flag arena_mutex=ATOMIC_FLAG_INIT;
static atomic_flag driver_operations=ATOMIC_FLAG_INIT;
static atomic_flag arena_log_mutex=ATOMIC_FLAG_INIT;
static atomic_bool arena_log_healthy=ATOMIC_VAR_INIT(true);
static int arena_fd=-1;
#ifndef PS5_AI_PERBO_HOST_TEST
#include "driver_journal_impl.inc"
#else
static void arena_lock(atomic_flag *l){while(atomic_flag_test_and_set_explicit(l,memory_order_acquire)){} }
static void arena_unlock(atomic_flag *l){atomic_flag_clear_explicit(l,memory_order_release);}
static bool ps5_ai_arena_trace(const char *fmt,...){(void)fmt;return atomic_load(&arena_log_healthy);}
static bool ps5_ai_arena_trace_routine(const char *fmt,...){(void)fmt;return atomic_load(&arena_log_healthy);}
#endif

enum driver_phase { DRIVER_ALLOCATING,DRIVER_LIVE,DRIVER_RETIRED,DRIVER_RELEASING,DRIVER_UNKNOWN };
struct driver_bo_owner {
    struct driver_bo_owner *next;
    uintptr_t live_slot;
    uint64_t serial;
    struct radv_ps5_memory memory;
    enum driver_phase phase;
    bool physical_owned,mapped;
};
static struct driver_bo_owner *driver_owners;
static uint64_t driver_nodes,driver_node_limit,driver_serial;
static bool driver_prepared,driver_prepare_attempted;
#define DRIVER_GUARD UINT64_C(0xb710d89ce45123af)

/* Statistics count every still-owned physical extent, including retired BOs.
 * They never report the public Direct quota as owned or as physically pinned.
 */
static void driver_stats_locked(void){
    uint64_t physical=0,mapped=0,live=0;
    for(struct driver_bo_owner *n=driver_owners;n;n=n->next){
        if(n->physical_owned){
            if(UINT64_MAX-physical<n->memory.bytes){arena_state.mapping_uncertain=arena_state.gpu_fault_retained=1;return;}
            physical+=n->memory.bytes;
        }
        if(n->mapped){
            if(UINT64_MAX-mapped<n->memory.bytes){arena_state.mapping_uncertain=arena_state.gpu_fault_retained=1;return;}
            mapped+=n->memory.bytes;
        }
        if(n->phase==DRIVER_LIVE)live++;
    }
    arena_state.reserved_bytes=arena_state.used_bytes=physical;
    if(physical>arena_state.peak_used_bytes)arena_state.peak_used_bytes=physical;
    arena_state.live_allocations=live;
    arena_state.physical_owned=physical!=0;
    arena_state.mapping_owned=physical!=0&&mapped==physical;
    arena_state.global_capacity_measured=0;
}
void ps5_ai_arena_snapshot(struct ps5_ai_arena_stats *out){
    if(!out)return;arena_lock(&arena_mutex);*out=arena_state;arena_unlock(&arena_mutex);
}
void ps5_ai_arena_retain_gpu_fault(void){
    arena_lock(&arena_mutex);arena_state.gpu_fault_retained=1;arena_unlock(&arena_mutex);
    (void)ps5_ai_arena_trace("driver_BO_fault_retained=1 irreversible=1 no_kernel_free=1");
}
static bool driver_fault(const char *why,bool uncertain){
    arena_lock(&arena_mutex);arena_state.gpu_fault_retained=1;
    if(uncertain)arena_state.mapping_uncertain=1;
    arena_unlock(&arena_mutex);
    (void)ps5_ai_arena_trace("driver_BO_retained reason=%s ownership_uncertain=%d no_retry=1",why,uncertain);
    return false;
}
static void driver_unlink_locked(struct driver_bo_owner *node){
    struct driver_bo_owner **at=&driver_owners;
    while(*at&&*at!=node)at=&(*at)->next;
    if(*at){*at=node->next;driver_nodes--;}
}
static bool driver_tuple(const struct radv_ps5_memory *a,const struct radv_ps5_memory *b){
    return a->cpu==b->cpu&&a->bytes==b->bytes&&a->physical==b->physical&&
        a->granule==b->granule&&a->granules==b->granules&&a->window_replay==b->window_replay;
}

static int native_log_attempted;
static const char *native_log_path="unavailable";
const char *ps5_ai_native_arena_log_path(void){return native_log_path;}
int ps5_ai_native_arena_open_log(int sandbox){
    if(native_log_attempted++||(sandbox!=0&&sandbox!=1))return -1;
#ifndef PS5_AI_PERBO_HOST_TEST
    const char *paths[2]={"/data/ai/logs/radv-native-perbo-v2.log","/download0/ai-native-bounded-logs-v2/native-driver.log"};
    for(int i=sandbox;i<2;i++){
        int fd=open(paths[i],O_WRONLY|O_CREAT|O_APPEND|O_NOFOLLOW,0600);if(fd<0)continue;
        struct stat st;if(fstat(fd,&st)||!S_ISREG(st.st_mode)){close(fd);continue;}
        arena_fd=fd;native_log_path=paths[i];
        return ps5_ai_arena_trace("NATIVE_LOG_READY pid=%d path=%s durable_before_explicit_AGC=1 GPU_allocation_attempts=0",(int)getpid(),native_log_path)?0:-1;
    }
    atomic_store(&arena_log_healthy,false);return -1;
#else
    arena_fd=1;return 0;
#endif
}
int ps5_ai_arena_prepare(void){
    arena_lock(&arena_mutex);
    if(driver_prepare_attempted){bool ok=driver_prepared;arena_unlock(&arena_mutex);return ok?0:-1;}
    driver_prepare_attempted=true;arena_unlock(&arena_mutex);
    if(arena_fd<0||!atomic_load(&arena_log_healthy))return -1;
    if(!ps5_ai_arena_trace("BEFORE perBO_GetDirectMemorySize query_only=1 allocation_calls=0"))return -1;
    int64_t total=sceKernelGetDirectMemorySize();
    if(!ps5_ai_arena_trace("RETURN perBO_GetDirectMemorySize bytes=%lld accounting_not_physical_proof=1",(long long)total)||total<=0)return -1;
    uint64_t limit=(uint64_t)total/RADV_PS5_PAGE_BYTES;
    if(!limit||limit>SIZE_MAX/sizeof(struct driver_bo_owner))return -1;
    arena_lock(&arena_mutex);driver_node_limit=limit;driver_prepared=true;arena_unlock(&arena_mutex);
    return ps5_ai_arena_trace("BEGIN native-perBO-Direct-v2 pid=%d owned_bytes=0 metadata_ready=1 physical_byte_cap=none node_limit_from_Direct_pages=%llu allocation_calls=0 Map_calls=0 Mprotect_calls=0",
        (int)getpid(),(unsigned long long)limit)?0:-1;
}
int ps5_ai_driver_owner_ready(void){
    arena_lock(&arena_mutex);bool ok=driver_prepared&&!arena_state.mapping_uncertain&&!arena_state.gpu_fault_retained&&atomic_load(&arena_log_healthy);arena_unlock(&arena_mutex);return ok;
}

bool radv_ps5_memory_alloc(uint64_t bytes,uint64_t alignment,bool window32,struct radv_ps5_memory *out){
    if(!out)return false;
    arena_lock(&driver_operations);
    arena_lock(&arena_mutex);arena_state.allocation_attempts++;
    bool duplicate=false;
    for(struct driver_bo_owner *q=driver_owners;q;q=q->next)
        if(q->phase==DRIVER_LIVE&&q->live_slot==(uintptr_t)out)duplicate=true;
    bool ready=driver_prepared&&!arena_state.mapping_uncertain&&!arena_state.gpu_fault_retained&&atomic_load(&arena_log_healthy);
    arena_unlock(&arena_mutex);
    if(duplicate){arena_unlock(&driver_operations);return driver_fault("allocation_over_live_owner_slot",false);}
    *out=(struct radv_ps5_memory){.physical=-1};
    uint64_t requested=bytes;
    if(!bytes)bytes=1;
    if(bytes>UINT64_MAX-(RADV_PS5_PAGE_BYTES-1)||alignment>(UINT64_C(1)<<62))goto denied;
    bytes=(bytes+RADV_PS5_PAGE_BYTES-1)&~(RADV_PS5_PAGE_BYTES-1);
    alignment=MAX2(alignment,PS5_KERNEL_DIRECT_ALIGNMENT);
    if(bytes>=RADV_PS5_LARGE_BYTES)alignment=MAX2(alignment,RADV_PS5_LARGE_BYTES);
    alignment=util_next_power_of_two64(alignment);
    if(!ready||!alignment||alignment>=RADV_PS5_GPU_ADDRESS_LIMIT||bytes>SIZE_MAX||
        (window32&&(bytes>RADV_PS5_WINDOW_BYTES||alignment>RADV_PS5_WINDOW_BYTES)))goto denied;
    /* Metadata exists before any allocation syscall can create an owner. */
    struct driver_bo_owner *n=calloc(1,sizeof(*n));
    if(!n)goto denied;
    n->memory.physical=-1;n->memory.bytes=bytes;n->phase=DRIVER_ALLOCATING;
    arena_lock(&arena_mutex);
    if(driver_nodes>=driver_node_limit||driver_serial==UINT64_MAX){arena_unlock(&arena_mutex);free(n);goto denied;}
    n->serial=++driver_serial;n->next=driver_owners;driver_owners=n;driver_nodes++;
    arena_unlock(&arena_mutex);
    int64_t total=sceKernelGetDirectMemorySize();
    if(total<=0||bytes>(uint64_t)total)goto no_owner;
    struct {uint64_t pre[2];int64_t physical;uint64_t post[2];} p={{DRIVER_GUARD,DRIVER_GUARD},-1,{DRIVER_GUARD,DRIVER_GUARD}};
    if(!ps5_ai_arena_trace("BEFORE perBO_AllocateDirect serial=%llu bytes=%llu alignment=%llu search_start=0 search_end=%lld type=12 attempt=1",
        (unsigned long long)n->serial,(unsigned long long)bytes,(unsigned long long)alignment,(long long)total))goto unknown;
    int32_t rc=sceKernelAllocateDirectMemory(0,total,(size_t)bytes,(size_t)alignment,12,&p.physical);
    bool canary=p.pre[0]==DRIVER_GUARD&&p.pre[1]==DRIVER_GUARD&&p.post[0]==DRIVER_GUARD&&p.post[1]==DRIVER_GUARD;
    bool known=rc==0&&canary&&p.physical>=0&&(uint64_t)p.physical%alignment==0&&p.physical<=total&&bytes<=(uint64_t)(total-p.physical);
    if(known){
        arena_lock(&arena_mutex);n->memory.physical=p.physical;n->physical_owned=true;driver_stats_locked();
        for(struct driver_bo_owner *q=driver_owners;q;q=q->next)if(q!=n&&q->physical_owned&&
            (uint64_t)p.physical<(uint64_t)q->memory.physical+q->memory.bytes&&
            (uint64_t)q->memory.physical<(uint64_t)p.physical+bytes)known=false;
        arena_unlock(&arena_mutex);
    }
    if(!ps5_ai_arena_trace("RETURN perBO_AllocateDirect serial=%llu rc=0x%08x physical=%lld canary=%d known_owner=%d",
        (unsigned long long)n->serial,(unsigned)rc,(long long)p.physical,canary,known))goto unknown;
    if(rc!=0&&canary&&p.physical==-1)goto no_owner;
    if(!known)goto unknown;
    /* Reference ordinary placement: window pointers are kernel-chosen; full
     * address BOs get a non-fixed hint from the 256GiB VA metadata range.
     * No retry on a failed or uncertain mapping in this guarded candidate. */
    void *hint=NULL;
    if(!window32){
        uint64_t count=DIV_ROUND_UP(bytes,RADV_PS5_LARGE_BYTES);
        if(count>UINT32_MAX)goto unknown;
        uint32_t first=radv_ps5_granules_take(&radv_ps5_region,(uint32_t)count,(uint32_t)MAX2(alignment/RADV_PS5_LARGE_BYTES,1));
        if(first==UINT32_MAX)goto unknown; /* No NULL-map fallback into shared low window. */
        if(first!=UINT32_MAX){n->memory.granule=first;n->memory.granules=(uint32_t)count;
            hint=(void *)(uintptr_t)(RADV_PS5_REGION_BASE+(uint64_t)first*RADV_PS5_LARGE_BYTES);}
    }
    struct {uint64_t pre[2];void *cpu;uint64_t post[2];} m={{DRIVER_GUARD,DRIVER_GUARD},hint,{DRIVER_GUARD,DRIVER_GUARD}};
    if(!ps5_ai_arena_trace("BEFORE perBO_MapDirect serial=%llu bytes=%llu protection=0x33 flags=0 hint=%p window32=%d attempt=1 Mprotect_calls=0",
        (unsigned long long)n->serial,(unsigned long long)bytes,hint,window32))goto unknown;
    rc=sceKernelMapDirectMemory(&m.cpu,(size_t)bytes,0x33,0,p.physical,(size_t)alignment);
    canary=m.pre[0]==DRIVER_GUARD&&m.pre[1]==DRIVER_GUARD&&m.post[0]==DRIVER_GUARD&&m.post[1]==DRIVER_GUARD;
    uint64_t va=(uint64_t)(uintptr_t)m.cpu;
    known=rc==0&&canary&&va&&va!=UINTPTR_MAX&&va%alignment==0&&va<RADV_PS5_GPU_ADDRESS_LIMIT&&bytes<=RADV_PS5_GPU_ADDRESS_LIMIT-va;
    bool placed=known&&(window32?radv_ps5_window_contains(va,bytes):
        hint&&m.cpu==hint&&va>=PS5_AI_REFERENCE_VA_BASE&&va<PS5_AI_DATA_VA_BASE&&bytes<=PS5_AI_DATA_VA_BASE-va);
    if(known){
        arena_lock(&arena_mutex);n->memory.cpu=m.cpu;n->mapped=true;
        for(struct driver_bo_owner *q=driver_owners;q;q=q->next)if(q!=n&&q->mapped&&
            va<(uint64_t)(uintptr_t)q->memory.cpu+q->memory.bytes&&
            (uint64_t)(uintptr_t)q->memory.cpu<va+bytes)known=false;
        driver_stats_locked();arena_unlock(&arena_mutex);
    }
    if(!ps5_ai_arena_trace("RETURN perBO_MapDirect serial=%llu rc=0x%08x pointer=%p canary=%d known_mapping=%d placement_valid=%d GPU_readback_proven=0",
        (unsigned long long)n->serial,(unsigned)rc,m.cpu,canary,known,placed))goto unknown;
    if(!known||!placed)goto unknown;
    arena_lock(&arena_mutex);
    bool publish=!arena_state.gpu_fault_retained&&!arena_state.mapping_uncertain&&atomic_load(&arena_log_healthy);
    if(publish){n->live_slot=(uintptr_t)out;n->phase=DRIVER_LIVE;*out=n->memory;driver_stats_locked();}
    arena_unlock(&arena_mutex);
    if(!publish)goto unknown;
    if(!ps5_ai_arena_trace_routine("alloc_family=perBO denied=0 serial=%llu requested=%llu bytes=%llu pointer=%p window32=%d",
        (unsigned long long)n->serial,(unsigned long long)requested,(unsigned long long)bytes,out->cpu,window32))goto unknown;
    arena_unlock(&driver_operations);return true;
unknown:
    arena_lock(&arena_mutex);n->phase=DRIVER_UNKNOWN;arena_state.denied_allocations++;arena_unlock(&arena_mutex);
    (void)driver_fault("allocation_or_mapping_or_log_unknown",true);
    arena_unlock(&driver_operations);return false;
no_owner:
    arena_lock(&arena_mutex);driver_unlink_locked(n);driver_stats_locked();arena_unlock(&arena_mutex);free(n);
denied:
    arena_lock(&arena_mutex);arena_state.denied_allocations++;arena_unlock(&arena_mutex);
    (void)ps5_ai_arena_trace("alloc_family=perBO denied=1 requested=%llu no_physical_owner_published=1",(unsigned long long)requested);
    arena_unlock(&driver_operations);return false;
}
void radv_ps5_memory_free(struct radv_ps5_memory *memory){
    if(!memory||!memory->cpu)return;
    arena_lock(&arena_mutex);struct driver_bo_owner *found=NULL;
    for(struct driver_bo_owner *n=driver_owners;n;n=n->next)if(n->phase==DRIVER_LIVE&&n->live_slot==(uintptr_t)memory&&driver_tuple(&n->memory,memory)){found=n;break;}
    bool ok=found&&!arena_state.gpu_fault_retained&&!arena_state.mapping_uncertain&&atomic_load(&arena_log_healthy);
    uint64_t serial=found?found->serial:0;
    if(ok){found->phase=DRIVER_RETIRED;found->live_slot=0;*memory=(struct radv_ps5_memory){.physical=-1};driver_stats_locked();}
    arena_unlock(&arena_mutex);
    if(!ok){(void)driver_fault("free_owner_or_health_mismatch",false);return;}
    if(!ps5_ai_arena_trace_routine("free_family=perBO serial=%llu retired=1 kernel_unmap_calls=0 kernel_release_calls=0 drain_at_positive_quiet_only=1",(unsigned long long)serial))
        (void)driver_fault("retirement_log_failed",false);
}
static bool driver_quiet(uint64_t epoch){
    struct ps5_ai_arena_kick_stats s={0};
    return epoch>0&&ps5_ai_arena_kick_snapshot(&s)==0&&s.initialized&&s.thread_started&&
        !s.pending&&!s.active&&!s.submit_inflight&&!s.fault_retained&&s.log_healthy&&
        s.requested_epoch==epoch&&s.completed_epoch==epoch&&ps5_ai_driver_owner_ready();
}
int ps5_ai_driver_retire_quiet(uint64_t epoch){
    /* The caller is outside RADV locks and owns the sole-producer boundary.
     * This function queries metadata only; it never waits for GPU progress. */
    arena_lock(&driver_operations);
    if(!driver_quiet(epoch)){arena_unlock(&driver_operations);return driver_fault("retirement_without_positive_quiet",false);}
    for(;;){
        arena_lock(&arena_mutex);struct driver_bo_owner *n=driver_owners;
        while(n&&n->phase!=DRIVER_RETIRED)n=n->next;
        if(n)n->phase=DRIVER_RELEASING;
        arena_unlock(&arena_mutex);if(!n)break;
        if(!driver_quiet(epoch)||!n->physical_owned||!n->mapped||n->memory.physical<0)goto failed;
        if(!ps5_ai_arena_trace("BEFORE perBO_retired_Munmap serial=%llu pointer=%p bytes=%llu epoch=%llu exact_owner=1",
            (unsigned long long)n->serial,n->memory.cpu,(unsigned long long)n->memory.bytes,(unsigned long long)epoch))goto failed;
        int32_t rc=sceKernelMunmap(n->memory.cpu,(size_t)n->memory.bytes);
        if(!rc){arena_lock(&arena_mutex);n->mapped=false;driver_stats_locked();arena_unlock(&arena_mutex);}
        if(!ps5_ai_arena_trace("RETURN perBO_retired_Munmap serial=%llu rc=0x%08x",(unsigned long long)n->serial,(unsigned)rc)||rc||!driver_quiet(epoch))goto failed;
        if(!ps5_ai_arena_trace("BEFORE perBO_retired_ReleaseDirect serial=%llu physical=%lld bytes=%llu epoch=%llu exact_owner=1",
            (unsigned long long)n->serial,(long long)n->memory.physical,(unsigned long long)n->memory.bytes,(unsigned long long)epoch))goto failed;
        rc=sceKernelReleaseDirectMemory(n->memory.physical,(size_t)n->memory.bytes);
        if(!rc){arena_lock(&arena_mutex);n->physical_owned=false;driver_stats_locked();arena_unlock(&arena_mutex);}
        if(!ps5_ai_arena_trace("RETURN perBO_retired_ReleaseDirect serial=%llu rc=0x%08x",(unsigned long long)n->serial,(unsigned)rc)||rc)goto failed;
        radv_ps5_granules_give(&radv_ps5_region,n->memory.granule,n->memory.granules);
        arena_lock(&arena_mutex);driver_unlink_locked(n);driver_stats_locked();arena_unlock(&arena_mutex);free(n);
    }
    arena_unlock(&driver_operations);return 1;
failed:
    (void)driver_fault("retirement_kernel_or_journal_unknown",true);
    arena_unlock(&driver_operations);return 0;
}
uint64_t radv_ps5_memory_pool_bytes(void){
    int64_t n=sceKernelGetDirectMemorySize();return n>0?(uint64_t)n:0;
}
uint64_t radv_ps5_memory_available_bytes(void){
    int64_t total=sceKernelGetDirectMemorySize();if(total<=0)return 0;
    struct {uint64_t pre[2];int64_t start;uint64_t mid[2];size_t bytes;uint64_t post[2];} q={{DRIVER_GUARD,DRIVER_GUARD},-1,{DRIVER_GUARD,DRIVER_GUARD},0,{DRIVER_GUARD,DRIVER_GUARD}};
    int32_t rc=sceKernelAvailableDirectMemorySize(0,total,RADV_PS5_PAGE_BYTES,&q.start,&q.bytes);
    bool valid=!rc&&q.pre[0]==DRIVER_GUARD&&q.pre[1]==DRIVER_GUARD&&q.mid[0]==DRIVER_GUARD&&q.mid[1]==DRIVER_GUARD&&q.post[0]==DRIVER_GUARD&&q.post[1]==DRIVER_GUARD&&q.start>=0&&q.start<=total&&q.bytes<=(uint64_t)(total-q.start);
    return valid?q.bytes:0;
}
int ps5_ai_arena_release_if_idle(void){
    arena_lock(&arena_mutex);bool retained=driver_owners||arena_state.gpu_fault_retained||arena_state.mapping_uncertain;arena_unlock(&arena_mutex);return retained?1:0;
}
bool radv_ps5_memory_alloc_replayable(uint64_t bytes,uint64_t alignment,bool window32,uint64_t replay_va,struct radv_ps5_memory *out){
    (void)bytes;(void)alignment;(void)window32;(void)replay_va;if(out)*out=(struct radv_ps5_memory){.physical=-1};return false;
}
bool radv_ps5_vrange_reserve(uint64_t bytes,bool window32,bool replayable,uint64_t replay_va,struct radv_ps5_memory *out){
    (void)bytes;(void)window32;(void)replayable;(void)replay_va;if(out)*out=(struct radv_ps5_memory){.physical=-1};return false;
}
void radv_ps5_vrange_release(struct radv_ps5_memory *range){(void)range;}
bool radv_ps5_vrange_bind(void *at,uint64_t bytes,const struct radv_ps5_memory *m,uint64_t offset){(void)at;(void)bytes;(void)m;(void)offset;return false;}
bool radv_ps5_vrange_unbind(void *at,uint64_t bytes){(void)at;(void)bytes;return false;}
uint64_t radv_ps5_vrange_space_bytes(void){return 0;}
