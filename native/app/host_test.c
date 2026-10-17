/* Actual perBO owner implementation under deterministic kernel mocks only. */
#define PS5_AI_PERBO_HOST_TEST 1
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "arena.h"
#include "radv_ps5_platform.h"
#define PS5_KERNEL_DIRECT_ALIGNMENT UINT64_C(0x10000)
#define MAX2(a,b) ((a)>(b)?(a):(b))
#define DIV_ROUND_UP(a,b) (((a)+(b)-1)/(b))
#define RADV_PS5_REGION_BASE UINT64_C(0x4000000000)
struct radv_ps5_granules {unsigned unused;};
static struct radv_ps5_granules radv_ps5_region;
static uint32_t next_granule;
static bool fail_granules;
static uint32_t radv_ps5_granules_take(struct radv_ps5_granules *g,uint32_t n,uint32_t alignment){
    (void)g;if(fail_granules)return UINT32_MAX;next_granule=(next_granule+alignment-1)/alignment*alignment;
    uint32_t result=next_granule;next_granule+=n;return result;
}
static void radv_ps5_granules_give(struct radv_ps5_granules *g,uint32_t first,uint32_t count){(void)g;(void)first;(void)count;}
static uint64_t util_next_power_of_two64(uint64_t n){uint64_t r=1;while(r<n)r<<=1;return r;}
static bool radv_ps5_window_contains(uint64_t p,uint64_t n){return p>=RADV_PS5_WINDOW_BASE&&n<=RADV_PS5_WINDOW_BYTES&&p-RADV_PS5_WINDOW_BASE<=RADV_PS5_WINDOW_BYTES-n;}
int64_t sceKernelGetDirectMemorySize(void);
int32_t sceKernelAllocateDirectMemory(int64_t,int64_t,size_t,size_t,int,int64_t *);
int32_t sceKernelMapDirectMemory(void **,size_t,int,int,int64_t,size_t);
int32_t sceKernelMunmap(void *,size_t);
int32_t sceKernelReleaseDirectMemory(int64_t,size_t);
int32_t sceKernelAvailableDirectMemorySize(int64_t,int64_t,size_t,int64_t *,size_t *);
static bool fail_calloc;
static void *owner_test_calloc(size_t n,size_t bytes);
#define calloc owner_test_calloc
#include "perbo_owner_impl.c"
#undef calloc

enum mode {OK,ALLOC_REFUSE,ALLOC_CHANGED,ALLOC_CANARY,MAP_FAIL,MAP_BAD_VA,MAP_CANARY,UNMAP_FAIL,RELEASE_FAIL,LOG_AFTER_ALLOC,FAULT_AFTER_MAP,LOG_AFTER_RELEASE,AVAILABLE_CANARY,MAP_DRIFT};
static enum mode mode;
static unsigned allocs,maps,unmaps,releases,tests;
static uint64_t next_physical;
static bool metadata_before_allocation;
static struct ps5_ai_arena_kick_stats kick;
static void *owner_test_calloc(size_t n,size_t bytes){if(fail_calloc){fail_calloc=false;return NULL;}return calloc(n,bytes);}
int64_t sceKernelGetDirectMemorySize(void){return INT64_C(12)<<30;}
int32_t sceKernelAllocateDirectMemory(int64_t begin,int64_t end,size_t bytes,size_t alignment,int type,int64_t *p){
    assert(begin==0&&end==sceKernelGetDirectMemorySize()&&type==12&&alignment>=65536);
    allocs++;metadata_before_allocation=driver_nodes>0&&driver_owners->phase==DRIVER_ALLOCATING;
    if(mode==ALLOC_REFUSE)return -35;
    if(mode==ALLOC_CHANGED){*p=0;return -35;}
    next_physical=(next_physical+alignment-1)&~(uint64_t)(alignment-1);*p=(int64_t)next_physical;next_physical+=bytes;
    if(mode==ALLOC_CANARY)((uint64_t *)(p+1))[0]^=1;
    if(mode==LOG_AFTER_ALLOC)atomic_store(&arena_log_healthy,false);
    return 0;
}
int32_t sceKernelMapDirectMemory(void **p,size_t bytes,int prot,int flags,int64_t physical,size_t alignment){
    assert(prot==0x33&&flags==0&&physical>=0&&bytes>0);maps++;
    if(mode==MAP_FAIL)return -35;
    if(!*p)*p=(void *)(uintptr_t)(RADV_PS5_WINDOW_BASE+UINT64_C(0x200000)+(uint64_t)physical);
    assert((uintptr_t)*p%alignment==0);
    if(mode==MAP_BAD_VA)*p=(void *)(uintptr_t)RADV_PS5_GPU_ADDRESS_LIMIT;
    if(mode==MAP_DRIFT)*p=(void *)((uintptr_t)*p+UINT64_C(2097152));
    if(mode==MAP_CANARY)((uint64_t *)(p+1))[0]^=1;
    if(mode==FAULT_AFTER_MAP)arena_state.gpu_fault_retained=1;
    return 0;
}
int32_t sceKernelMunmap(void *p,size_t bytes){assert(p&&bytes);unmaps++;return mode==UNMAP_FAIL?-1:0;}
int32_t sceKernelReleaseDirectMemory(int64_t p,size_t bytes){assert(p>=0&&bytes);releases++;if(mode==LOG_AFTER_RELEASE)atomic_store(&arena_log_healthy,false);return mode==RELEASE_FAIL?-1:0;}
int32_t sceKernelAvailableDirectMemorySize(int64_t begin,int64_t end,size_t alignment,int64_t *p,size_t *bytes){
    assert(begin==0&&end==(INT64_C(12)<<30)&&alignment==16384);*p=0;*bytes=UINT64_C(6)<<30;
    if(mode==AVAILABLE_CANARY)((uint64_t *)(bytes+1))[0]^=1;return 0;
}
int ps5_ai_arena_kick_snapshot(struct ps5_ai_arena_kick_stats *s){*s=kick;s->fault_retained|=arena_state.gpu_fault_retained;s->log_healthy=atomic_load(&arena_log_healthy);return 0;}
static void reset(void){
    while(driver_owners){struct driver_bo_owner *n=driver_owners;driver_owners=n->next;free(n);}
    memset(&arena_state,0,sizeof(arena_state));driver_nodes=driver_node_limit=driver_serial=0;driver_prepared=driver_prepare_attempted=false;
    atomic_flag_clear(&arena_mutex);atomic_flag_clear(&driver_operations);atomic_flag_clear(&arena_log_mutex);atomic_store(&arena_log_healthy,true);
    arena_fd=1;mode=OK;allocs=maps=unmaps=releases=0;next_physical=next_granule=0;metadata_before_allocation=false;fail_calloc=fail_granules=false;
    kick=(struct ps5_ai_arena_kick_stats){.requested_epoch=5,.completed_epoch=5,.initialized=1,.thread_started=1,.log_healthy=1};
    assert(ps5_ai_arena_prepare()==0&&ps5_ai_driver_owner_ready());
}
/* Returning a memory struct would change the live_slot. Use this macro so the
 * exact actual owner structure passed to alloc is also passed to free. */
#define OWN(m) struct radv_ps5_memory m={0}; assert(radv_ps5_memory_alloc(16384,16384,true,&m)); assert(metadata_before_allocation)
#define CHECK(c) do{assert(c);tests++;}while(0)
int main(void){
    reset();CHECK(!allocs&&!maps&&!arena_state.reserved_bytes&&!arena_state.physical_owned);
    OWN(a);CHECK(a.physical==0&&arena_state.reserved_bytes==16384&&arena_state.mapping_owned&&allocs==1&&maps==1);
    reset();OWN(b);struct radv_ps5_memory before=b;CHECK(!radv_ps5_memory_alloc(16384,16384,true,&b)&&driver_tuple(&b,&before)&&allocs==1&&arena_state.gpu_fault_retained);
    reset();mode=ALLOC_REFUSE;struct radv_ps5_memory c;CHECK(!radv_ps5_memory_alloc(16384,16384,true,&c)&&!driver_nodes&&!maps&&!arena_state.gpu_fault_retained&&!releases);
    reset();mode=ALLOC_CHANGED;CHECK(!radv_ps5_memory_alloc(16384,16384,true,&c)&&driver_nodes==1&&!maps&&arena_state.mapping_uncertain&&!releases);
    reset();mode=ALLOC_CANARY;CHECK(!radv_ps5_memory_alloc(16384,16384,true,&c)&&driver_nodes==1&&!maps&&arena_state.mapping_uncertain);
    reset();mode=MAP_FAIL;CHECK(!radv_ps5_memory_alloc(16384,16384,true,&c)&&arena_state.physical_owned&&arena_state.mapping_uncertain&&!unmaps&&!releases);
    reset();mode=MAP_BAD_VA;CHECK(!radv_ps5_memory_alloc(16384,16384,true,&c)&&arena_state.physical_owned&&arena_state.mapping_uncertain&&!releases);
    reset();mode=MAP_CANARY;CHECK(!radv_ps5_memory_alloc(16384,16384,true,&c)&&arena_state.mapping_uncertain&&!releases);
    reset();mode=LOG_AFTER_ALLOC;CHECK(!radv_ps5_memory_alloc(16384,16384,true,&c)&&arena_state.physical_owned&&!maps&&!releases);
    reset();mode=FAULT_AFTER_MAP;CHECK(!radv_ps5_memory_alloc(16384,16384,true,&c)&&!c.cpu&&arena_state.physical_owned&&arena_state.gpu_fault_retained&&!releases);
    reset();fail_calloc=true;CHECK(!radv_ps5_memory_alloc(16384,16384,true,&c)&&!allocs&&!driver_nodes&&!arena_state.gpu_fault_retained);
    reset();OWN(d);radv_ps5_memory_free(&d);CHECK(!d.cpu&&!unmaps&&!releases&&arena_state.reserved_bytes==16384&&!arena_state.live_allocations);CHECK(ps5_ai_driver_retire_quiet(5)==1&&unmaps==1&&releases==1&&!driver_nodes&&!arena_state.reserved_bytes);
    reset();OWN(e);radv_ps5_memory_free(&e);kick.pending=1;CHECK(!ps5_ai_driver_retire_quiet(5)&&!unmaps&&!releases&&arena_state.gpu_fault_retained);
    reset();OWN(f);radv_ps5_memory_free(&f);CHECK(!ps5_ai_driver_retire_quiet(4)&&!unmaps&&!releases);
    reset();OWN(g);radv_ps5_memory_free(&g);mode=UNMAP_FAIL;CHECK(!ps5_ai_driver_retire_quiet(5)&&unmaps==1&&!releases&&arena_state.physical_owned&&arena_state.mapping_owned);
    reset();OWN(h);radv_ps5_memory_free(&h);mode=RELEASE_FAIL;CHECK(!ps5_ai_driver_retire_quiet(5)&&unmaps==1&&releases==1&&arena_state.physical_owned&&!arena_state.mapping_owned);CHECK(!ps5_ai_driver_retire_quiet(5)&&unmaps==1&&releases==1);
    reset();OWN(i);i.bytes+=16384;radv_ps5_memory_free(&i);CHECK(arena_state.gpu_fault_retained&&arena_state.live_allocations==1&&!unmaps);
    reset();OWN(j);struct radv_ps5_memory copy=j;radv_ps5_memory_free(&copy);CHECK(arena_state.gpu_fault_retained&&arena_state.live_allocations==1&&!unmaps);
    reset();OWN(k);copy=k;radv_ps5_memory_free(&k);k=copy;radv_ps5_memory_free(&k);CHECK(arena_state.gpu_fault_retained&&!unmaps&&!releases&&driver_nodes==1);
    reset();struct radv_ps5_memory high;CHECK(radv_ps5_memory_alloc(16384,16384,false,&high)&&(uintptr_t)high.cpu>=RADV_PS5_REGION_BASE);
    CHECK(radv_ps5_memory_pool_bytes()==(UINT64_C(12)<<30)&&radv_ps5_memory_available_bytes()==(UINT64_C(6)<<30)&&arena_state.reserved_bytes==16384);
    mode=AVAILABLE_CANARY;CHECK(radv_ps5_memory_available_bytes()==0);
    reset();fail_granules=true;CHECK(!radv_ps5_memory_alloc(16384,16384,false,&c)&&allocs==1&&!maps&&arena_state.physical_owned&&arena_state.mapping_uncertain&&!releases);
    reset();mode=MAP_DRIFT;CHECK(!radv_ps5_memory_alloc(16384,16384,false,&c)&&allocs==1&&maps==1&&arena_state.physical_owned&&arena_state.mapping_owned&&arena_state.mapping_uncertain&&!releases);
    reset();OWN(l);radv_ps5_memory_free(&l);mode=LOG_AFTER_RELEASE;CHECK(!ps5_ai_driver_retire_quiet(5)&&releases==1&&!arena_state.physical_owned&&driver_nodes==1&&arena_state.gpu_fault_retained);CHECK(!ps5_ai_driver_retire_quiet(5)&&releases==1);
    reset();CHECK(!radv_ps5_memory_alloc(UINT64_MAX,16384,true,&c)&&!allocs&&!driver_nodes);
    reset();CHECK(!radv_ps5_memory_alloc(16384,UINT64_MAX,true,&c)&&!allocs&&!driver_nodes);
    reset();CHECK(!radv_ps5_memory_alloc(16384,RADV_PS5_GPU_ADDRESS_LIMIT,false,&c)&&!allocs&&!driver_nodes);
    reset();struct radv_ps5_memory many[768]={0};
    for(unsigned z=0;z<768;z++)assert(radv_ps5_memory_alloc(16384,16384,true,&many[z]));
    CHECK(driver_nodes==768&&arena_state.live_allocations==768&&arena_state.reserved_bytes==UINT64_C(768)*16384);
    for(unsigned z=0;z<768;z++)radv_ps5_memory_free(&many[z]);
    CHECK(!unmaps&&!releases&&arena_state.live_allocations==0&&arena_state.reserved_bytes==UINT64_C(768)*16384);
    CHECK(ps5_ai_driver_retire_quiet(5)==1&&unmaps==768&&releases==768&&!driver_nodes&&!arena_state.reserved_bytes);
    reset();printf("PASS %u actual-owner lifecycle checks; no real Direct/GPU calls\n",tests);return 0;
}
