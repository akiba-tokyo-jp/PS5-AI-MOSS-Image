/* SPDX-License-Identifier: MIT -- ownership/metadata tests, no GPU or inference. */
#include <assert.h>
#define PS5_AI_DIRECT_POOL_HOST_TEST 1
#include "pool.c"
#define MIB (UINT64_C(1) << 20)
static int allocs, maps, unmaps, frees, held, pending, wait_ok = 1;
static int allocation_mode, map_mode, query_corrupt, query_fail, protect_corrupt, sync_fail, fail_sync_after_map;
static int fail_unmap, fail_release, short_writes, write_calls;
static uint64_t mock_owned;
static struct { int64_t physical; void *address; uint64_t bytes; int mapped, owned; } mock[32];
int host_open_journal(void) { return 42; }
ssize_t host_write(int fd, const void *p, size_t n) {
    assert(fd == 42 && p); ++write_calls;
    if (short_writes && write_calls == 1) { errno = EINTR; return -1; }
    if (short_writes && write_calls == 2 && n > 1) return (ssize_t)(n / 2);
    return (ssize_t)n;
}
int host_fsync(int fd) { assert(fd == 42); return sync_fail ? -1 : 0; }
int64_t sceKernelGetDirectMemorySize(void) { return INT64_C(12) << 30; }
int32_t sceKernelAvailableDirectMemorySize(int64_t start, int64_t end, size_t alignment, int64_t *physical, size_t *available) {
    assert(!start && end == (INT64_C(12) << 30) && alignment == UNIT);
    *physical = UNIT; *available = (UINT64_C(12) << 30) - mock_owned - UNIT;
    if (query_corrupt) available[1] = 0;
    return query_fail ? -1 : 0;
}
int32_t sceKernelAllocateDirectMemory(int64_t start, int64_t end, size_t bytes, size_t alignment, int type, int64_t *physical) {
    assert(!start && end == (INT64_C(12) << 30) && alignment == UNIT && type == 12 && bytes % UNIT == 0);
    int index = allocs++; assert(index < 32);
    if (allocation_mode == 1 && index == 1) return -1;
    if (allocation_mode == 2) { *physical = 0; return 0; }
    if (allocation_mode == 3) { *physical = UNIT; return -1; }
    if (allocation_mode == 4) { *physical = UNIT; physical[1] = 0; return 0; }
    if (allocation_mode == 5 && index == 1) *physical = mock[0].physical;
    else *physical = (int64_t)(UNIT + (uint64_t)index * bytes);
    mock[index].physical = *physical; mock[index].bytes = bytes; mock[index].owned = 1; mock_owned += bytes;
    return 0;
}
int32_t sceKernelMapDirectMemory(void **address, size_t bytes, int protection, int flags, int64_t physical, size_t alignment) {
    assert((uintptr_t)*address == PS5_AI_DATA_VA_BASE+(uint64_t)maps*bytes && protection == 0x33 && !flags && alignment == UNIT);
    int index = maps++; assert(index < 32 && physical == mock[index].physical && bytes == mock[index].bytes);
    if (map_mode == 1) { *address = NULL; return 0; }
    if (map_mode == 2) { *address = (void *)UINTPTR_MAX; return 0; }
    if (map_mode == 3) { *address = (void *)(uintptr_t)VA_LIMIT; return 0; }
    if (map_mode == 4 && index == 1) *address = mock[0].address;
    else if(map_mode==5)*address=(void *)(uintptr_t)UINT64_C(0x200000000);
    mock[index].address = *address; mock[index].mapped = 1;
    if (fail_sync_after_map) sync_fail = 1;
    return 0;
}
int32_t sceKernelQueryMemoryProtection(void *address, void **low, void **high, uint32_t *flags) {
    for (int i = 0; i < maps; ++i) if (mock[i].address == address) {
        *low = address; *high = (void *)((uintptr_t)address + mock[i].bytes); *flags = 0x33;
        if (protect_corrupt) flags[1] = 1;
        return 0;
    }
    assert(0); return -1;
}
int32_t sceKernelMunmap(void *address, size_t bytes) {
    ++unmaps;
    if (fail_unmap) return -1;
    for (int i = 0; i < maps; ++i) if (mock[i].address == address && mock[i].mapped) {
        assert(bytes == mock[i].bytes); mock[i].mapped = 0; return 0;
    }
    assert(0); return -1;
}
int32_t sceKernelReleaseDirectMemory(int64_t physical, size_t bytes) {
    ++frees;
    if (fail_release) return -1;
    for (int i = 0; i < allocs; ++i) if (mock[i].physical == physical && mock[i].owned) {
        assert(!mock[i].mapped && bytes == mock[i].bytes); mock[i].owned = 0; mock_owned -= bytes; return 0;
    }
    assert(0); return -1;
}
void ps5_ai_arena_retain_gpu_fault(void) { held = 1; }
void ps5_ai_arena_snapshot(struct ps5_ai_arena_stats *out) { memset(out, 0, sizeof(*out)); out->gpu_fault_retained = held; }
void ps5_ai_ggml_dispatch_snapshot(struct ps5_ai_ggml_dispatch_stats *out) { memset(out, 0, sizeof(*out)); out->work_pending = pending; }
int ps5_ai_ggml_log_healthy(void) { return 1; }
int ps5_ai_ggml_wait_backend(const char *phase, int expect) { assert(phase && !expect); return wait_ok; }
int main(int argc, char **argv) {
    assert(argc == 2); const char *name = argv[1];
    if (!strcmp(name, "arithmetic")) {
        assert(!ps5_ai_direct_pool_configure(UINT64_C(6) << 30, UINT64_C(512) << 20, UINT64_C(1) << 30));
        assert(direct.target_bytes > UINT32_MAX && !allocs && !maps);
        assert(ps5_ai_direct_pool_configure(UNIT, UNIT, MIN_RESERVE) == -1); return 0;
    }
    if (!strcmp(name, "unconfigured")) { assert(ps5_ai_flex_prepare() == -1 && !allocs); return 0; }
    if (!strcmp(name, "invalid_config")) {
        assert(ps5_ai_direct_pool_configure(UINT64_MAX, UNIT, MIN_RESERVE) == -1);
        assert(ps5_ai_direct_pool_configure(UNIT, UNIT, MIN_RESERVE - 1) == -1);
        assert(ps5_ai_direct_pool_configure(UNIT, UNIT * 2, MIN_RESERVE) == -1);
        assert(!allocs); return 0;
    }
    if (!strcmp(name, "refusal_rollback")) allocation_mode = 1;
    if (!strcmp(name, "physical_zero")) allocation_mode = 2;
    if (!strcmp(name, "changed_failure")) allocation_mode = 3;
    if (!strcmp(name, "allocation_canary")) allocation_mode = 4;
    if (!strcmp(name, "physical_overlap")) allocation_mode = 5;
    if (!strcmp(name, "map_null")) map_mode = 1;
    if (!strcmp(name, "map_failed")) map_mode = 2;
    if (!strcmp(name, "VA_range")) map_mode = 3;
    if (!strcmp(name, "VA_overlap")) map_mode = 4;
    if (!strcmp(name, "hint_ignored")) map_mode = 5;
    if (!strcmp(name, "protection_canary")) protect_corrupt = 1;
    if (!strcmp(name, "query_canary")) query_corrupt = 1;
    if (!strcmp(name, "query_error")) query_fail = 1;
    if (!strcmp(name, "map_log_failure")) fail_sync_after_map = 1;
    if (!strcmp(name, "short_write_EINTR")) short_writes = 1;
    uint64_t target = PS5_AI_DATA_VA_BYTES;
    uint64_t piece = UINT64_C(256) << 20;
    assert(!ps5_ai_direct_pool_configure(target, piece, MIN_RESERVE));
    int rc = ps5_ai_flex_prepare();
    if (query_corrupt || query_fail) { assert(rc == -1 && !allocs && !maps); return 0; }
    if (allocation_mode == 1) { assert(rc == -1 && allocs == 2 && maps == 1 && frees == 1 && unmaps == 1 && !mock_owned && !held); return 0; }
    if (allocation_mode || map_mode || protect_corrupt || fail_sync_after_map) {
        assert(rc == -1 && held && state.fault_retained && direct.ownership_uncertain && !unmaps && !frees);
        if(map_mode==5)assert(direct.mapped_bytes==piece&&regions[0].mapped&&regions[0].physical_owned&&
                             (uintptr_t)regions[0].cpu==UINT64_C(0x200000000));
        assert(!ps5_ai_flex_pool_bytes() && !ps5_ai_flex_alloc(PAGE, PAGE)); return 0;
    }
    assert(!rc && direct.ready && ps5_ai_flex_pool_bytes() == target && direct.region_count == 24);
    assert((uintptr_t)mock[0].address==PS5_AI_DATA_VA_BASE && (uintptr_t)mock[23].address+piece==PS5_AI_DATA_VA_END);
    assert(ps5_ai_flex_prepare() == -1 && allocs == 24);
    assert(!ps5_ai_flex_alloc(piece + PAGE, PAGE)); /* Cannot span two mappings. */
    if (!strcmp(name, "claim_growth")) {
        for (unsigned i = 0; i < 4097; ++i) assert(ps5_ai_flex_alloc(PAGE, PAGE));
        assert(claim_capacity == 8192 && state.allocations == 4097); return 0;
    }
    if (!strcmp(name, "region_fragmentation")) {
        void *a[24];for(unsigned i=0;i<24;i++){a[i]=ps5_ai_flex_alloc(piece-UNIT,PAGE);assert(a[i]==mock[i].address);}
        assert(state.free_bytes==24*UNIT && !ps5_ai_flex_alloc(3*MIB,PAGE));
        for(unsigned i=0;i<24;i++)assert(!ps5_ai_flex_free(a[i],piece-UNIT));
    }
    void *p = ps5_ai_flex_alloc(PAGE, PAGE); assert(p == mock[0].address);
    if (!strcmp(name, "pending")) pending = 1;
    if (!strcmp(name, "quiet_failure")) wait_ok = 0;
    if (pending || !wait_ok) {
        assert(ps5_ai_flex_free(p, PAGE) == -1 && held && state.allocations == 1 && !unmaps); return 0;
    }
    if (!strcmp(name, "bad_free")) {
        assert(ps5_ai_flex_free((char *)p + PAGE, PAGE) == -1 && held && state.allocations == 1); return 0;
    }
    assert(!ps5_ai_flex_grant_gpu((char *)p + PAGE, PAGE));
    assert(ps5_ai_flex_grant_gpu(p, PAGE)); assert(!ps5_ai_flex_grant_gpu(p, PAGE));
    assert(!ps5_ai_flex_import_committed(p, PAGE) && state.active_imports == 1);
    assert(!ps5_ai_flex_free(p, PAGE) && !state.active_imports && !state.used_bytes);
    if (!strcmp(name, "unmap_failure")) fail_unmap = 1;
    if (!strcmp(name, "release_failure")) fail_release = 1;
    if (fail_unmap || fail_release) {
        assert(ps5_ai_direct_pool_release_if_idle() == -1 && held && direct.ownership_uncertain && mock_owned == target);
        assert(unmaps == 1 && frees == (fail_release ? 1 : 0)); return 0;
    }
    assert(!ps5_ai_direct_pool_release_if_idle() && !mock_owned && unmaps == 24 && frees == 24 && !ps5_ai_flex_pool_bytes());
    return 0;
}
