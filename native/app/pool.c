/* SPDX-License-Identifier: MIT
 * Native application-owned Direct regions, not anonymous flexible memory.
 * Historical ps5_ai_flex_* ABI is retained for the frozen GGML host importer.
 * No fixed VA, Mprotect, raw syscall, budget/credential or foreign process API.
 */
#include "direct_pool.h"
#include "placement.h"
#include "arena.h"
#include "probe.h"
#include <ps5platform/kernel.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef PS5_AI_DIRECT_POOL_HOST_TEST
#include "bounded_journal.h"
static struct ps5_ai_bounded_journal pool_journal_cap;
#endif
#ifdef PS5_AI_DIRECT_POOL_HOST_TEST
#include "host_shim.h"
#endif

#define UNIT UINT64_C(2097152)
#define PAGE PS5_AI_FLEX_PAGE_BYTES
#define VA_LIMIT (UINT64_C(1) << 47)
#define GUARD UINT64_C(0x857b43c1fdf5621a)
#define MIN_RESERVE (UINT64_C(256) << 20)
#define CLAIM_ACTIVE 1u
#define CLAIM_GRANTED 2u
#define CLAIM_IMPORTED 4u
struct region {
    int64_t physical;
    void *cpu;
    uint64_t bytes;
    uint32_t page_first, pages;
    unsigned physical_owned, mapped;
};
struct claim { uint32_t region, first, pages, flags; uint64_t bytes; };
static struct region *regions;
static struct claim *claims;
static uint64_t *bitmap;
static uint32_t total_pages, region_capacity, claim_capacity;
static struct ps5_ai_flex_stats state;
static struct ps5_ai_direct_pool_stats direct;
static struct ps5_ai_flex_prepare_stats preparation;
static atomic_flag lock = ATOMIC_FLAG_INIT, journal_lock = ATOMIC_FLAG_INIT;
static atomic_bool log_ok = ATOMIC_VAR_INIT(true);
static int journal = -1;
static const char *journal_path = "host_mock";
static int data_open_errno;
_Static_assert(sizeof(size_t) == 8, "64-bit addresses and requests");
_Static_assert(sizeof(struct ps5_ai_flex_stats) == 72, "Frozen GGML snapshot ABI");

static void acquire(atomic_flag *value) {
    while (atomic_flag_test_and_set_explicit(value, memory_order_acquire)) __builtin_ia32_pause();
}
static void release(atomic_flag *value) { atomic_flag_clear_explicit(value, memory_order_release); }
static int trace(const char *format, ...) {
    char line[1024];
    va_list args; va_start(args, format);
    int count = vsnprintf(line, sizeof(line) - 1, format, args); va_end(args);
    if (count < 0 || (size_t)count >= sizeof(line) - 1) { atomic_store(&log_ok, false); return 0; }
    line[count++] = '\n';
    acquire(&journal_lock);
    int failed = journal < 0;
#ifndef PS5_AI_DIRECT_POOL_HOST_TEST
    if(!failed && !ps5_ai_journal_room(&journal,journal_path,&pool_journal_cap,(size_t)count,UINT64_C(4194304)))failed=1;
#endif
    for (size_t done = 0; !failed && done < (size_t)count;) {
        ssize_t wrote = write(journal, line + done, (size_t)count - done);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0 || (size_t)wrote > (size_t)count - done) failed = 1;
        else {done += (size_t)wrote;
#ifndef PS5_AI_DIRECT_POOL_HOST_TEST
            pool_journal_cap.bytes+=(size_t)wrote;
#endif
        }
    }
    if (!failed && fsync(journal)) failed = 1;
    release(&journal_lock);
    if (failed) atomic_store(&log_ok, false);
    return !failed && atomic_load(&log_ok);
}
static int open_journal(void) {
#ifdef PS5_AI_DIRECT_POOL_HOST_TEST
    return host_open_journal();
#else
    const char *paths[] = { "/data/ai/logs/native-direct-data-pool-v1.log",
                            "/download0/ai-native-bounded-logs-v2/native-pool.log" };
    for (unsigned i = 0; i < 2; ++i) {
        int fd = open(paths[i], O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW, 0600);
        if (!i && fd < 0) data_open_errno = errno;
        if (fd < 0) continue;
        struct stat info;
        if (!fstat(fd, &info) && S_ISREG(info.st_mode)) { journal_path = paths[i]; return fd; }
        (void)close(fd);
    }
    return -1;
#endif
}
void ps5_ai_flex_snapshot(struct ps5_ai_flex_stats *out) {
    if (!out) return;
    acquire(&lock); *out = state; out->log_healthy = atomic_load(&log_ok); release(&lock);
}
void ps5_ai_direct_pool_snapshot(struct ps5_ai_direct_pool_stats *out) {
    if (!out) return;
    acquire(&lock); *out = direct; out->claim_capacity = claim_capacity; release(&lock);
}
void ps5_ai_flex_prepare_snapshot(struct ps5_ai_flex_prepare_stats *out) {
    if (!out) return;
    acquire(&lock); *out = preparation; release(&lock);
}
int ps5_ai_flex_cancel_cpu_prefault(void) { return 1; } /* No anonymous prefault. */
void ps5_ai_flex_retain_fault(const char *reason) {
    acquire(&lock); state.fault_retained = 1; preparation.phase = PS5_AI_FLEX_RETAINED; release(&lock);
    ps5_ai_arena_retain_gpu_fault();
    (void)trace("fault_retained=1 reason=%s reuse=forbidden Direct_release=forbidden", reason ? reason : "unknown");
}
static int uncertain(const char *reason) {
    acquire(&lock); direct.ownership_uncertain = 1; preparation.ownership_uncertain = 1; release(&lock);
    ps5_ai_flex_retain_fault(reason); return -1;
}
uint64_t ps5_ai_flex_pool_bytes(void) {
    acquire(&lock);
    uint64_t result = direct.ready && state.mapping_owned && state.gpu_access_granted &&
        !state.fault_retained && atomic_load(&log_ok) ? state.reserved_bytes : 0;
    release(&lock); return result;
}
uint64_t ps5_ai_direct_pool_max_claim_bytes(void) {
    acquire(&lock);
    uint64_t result = direct.ready && !state.fault_retained && atomic_load(&log_ok) ? direct.largest_region_bytes : 0;
    release(&lock); return result;
}
int ps5_ai_direct_pool_configure(uint64_t target, uint64_t piece, uint64_t reserve) {
    if (target != PS5_AI_DATA_VA_BYTES || target % UNIT || !piece || piece % UNIT || piece > target ||
        reserve < MIN_RESERVE || reserve % PAGE || target > INT64_MAX ||
        target > SIZE_MAX || piece > SIZE_MAX || reserve > SIZE_MAX || target / PAGE > UINT32_MAX)
        return -1;
    acquire(&lock);
    if (direct.configured || direct.attempted) { release(&lock); return -1; }
    direct.target_bytes = target; direct.region_max_bytes = piece; direct.cpu_direct_reserve_bytes = reserve;
    direct.configured = 1;
    release(&lock); return 0;
}
static int query_available(int64_t *total, uint64_t *available) {
    if (!trace("BEFORE GetDirectMemorySize query_only=1")) return -1;
    *total = sceKernelGetDirectMemorySize();
    if (!trace("RETURN GetDirectMemorySize bytes=%lld", (long long)*total) || *total <= 0) return -1;
    struct { uint64_t pre[2]; int64_t start; uint64_t mid[2]; size_t bytes; uint64_t post[2]; } q = {
        {GUARD, GUARD}, -1, {GUARD, GUARD}, 0, {GUARD, GUARD}
    };
    if (!trace("BEFORE AvailableDirectMemorySize search_start=0 search_end=%lld alignment=2097152", (long long)*total)) return -1;
    int32_t rc = sceKernelAvailableDirectMemorySize(0, *total, UNIT, &q.start, &q.bytes);
    int canary = q.pre[0] == GUARD && q.pre[1] == GUARD && q.mid[0] == GUARD &&
        q.mid[1] == GUARD && q.post[0] == GUARD && q.post[1] == GUARD;
    int valid = !rc && canary && q.start >= 0 && q.start <= *total && q.bytes <= (uint64_t)(*total - q.start);
    if (!trace("RETURN AvailableDirectMemorySize rc=0x%08x canary=%d start=%lld bytes=%llu valid=%d total_physical_free_proven=0",
        (unsigned)rc, canary, (long long)q.start, (unsigned long long)q.bytes, valid) || !valid) return -1;
    *available = q.bytes; return 0;
}
static int prepare_rollback(const char *reason) {
    /* No Vulkan imports have been exposed until ready. Any uncertain outcome
     * or failed durable journal retains every known and possibly-owned extent. */
    acquire(&lock);
    int safe = !state.allocations && !state.active_imports && !state.fault_retained && !direct.ready &&
        !direct.ownership_uncertain && atomic_load(&log_ok);
    release(&lock);
    if (!safe) return uncertain("prepare_rollback_not_safe");
    for (uint64_t n = direct.region_count; n; --n) {
        struct region *r = &regions[n - 1];
        if (r->mapped) {
            if (!trace("BEFORE prepare_rollback_Munmap region=%llu pointer=%p bytes=%llu reason=%s no_imports=1", (unsigned long long)(n - 1), r->cpu, (unsigned long long)r->bytes, reason)) return uncertain("rollback_BEFORE_log_failed");
            int32_t rc = sceKernelMunmap(r->cpu, (size_t)r->bytes);
            if (!rc) { r->mapped = 0; acquire(&lock); direct.mapped_bytes -= r->bytes; release(&lock); }
            if (!trace("RETURN prepare_rollback_Munmap rc=0x%08x", (unsigned)rc) || rc) return uncertain("rollback_unmap_unknown");
        }
        if (!r->physical_owned || r->physical <= 0) return uncertain("rollback_physical_owner_invalid");
        if (!trace("BEFORE prepare_rollback_ReleaseDirectMemory physical=%lld bytes=%llu exact_owned_extent=1", (long long)r->physical, (unsigned long long)r->bytes)) return uncertain("rollback_release_BEFORE_log_failed");
        int32_t rc = sceKernelReleaseDirectMemory(r->physical, (size_t)r->bytes);
        if (!rc) { r->physical_owned = 0; acquire(&lock); direct.physical_owned_bytes -= r->bytes; release(&lock); }
        if (!trace("RETURN prepare_rollback_ReleaseDirectMemory rc=0x%08x", (unsigned)rc) || rc) return uncertain("rollback_release_unknown");
    }
    acquire(&lock);
    struct region *old_regions = regions; uint64_t *old_bitmap = bitmap; struct claim *old_claims = claims;
    regions = NULL; bitmap = NULL; claims = NULL;
    state.mapping_owned = 0; state.gpu_access_granted = 0; state.reserved_bytes = state.free_bytes = 0;
    direct.region_count = direct.largest_region_bytes = 0;
    total_pages = region_capacity = claim_capacity = 0; preparation.phase = PS5_AI_FLEX_CPU_ABORTED;
    release(&lock);
    free(old_regions); free(old_bitmap); free(old_claims);
    (void)trace("KNOWN_DATA_REGIONS_RELEASED reason=%s owned_physical_bytes=0 owned_mapping_bytes=0 target_ready=0 automatic_retry=0", reason);
    return -1;
}
static int add_region(uint64_t bytes, int64_t total) {
    if (direct.region_count >= region_capacity) return uncertain("region_metadata_bounds");
    struct { uint64_t pre[2]; int64_t physical; uint64_t post[2]; } p = {{GUARD, GUARD}, -1, {GUARD, GUARD}};
    if (!trace("BEFORE AllocateDirectMemory region=%llu bytes=%llu alignment=2097152 type=12 search_start=0 search_end=%lld attempt=1 fallback=0",
        (unsigned long long)direct.region_count, (unsigned long long)bytes, (long long)total)) return uncertain("allocation_BEFORE_log_failed");
    int32_t rc = sceKernelAllocateDirectMemory(0, total, (size_t)bytes, UNIT, 12, &p.physical);
    int canary = p.pre[0] == GUARD && p.pre[1] == GUARD && p.post[0] == GUARD && p.post[1] == GUARD;
    int known = !rc && canary && p.physical > 0 && (uint64_t)p.physical % UNIT == 0 &&
        (uint64_t)p.physical <= (uint64_t)INT64_MAX - bytes && p.physical <= total && bytes <= (uint64_t)(total - p.physical);
    int refusal = rc && canary && p.physical == -1;
    if (known) {
        acquire(&lock);
        struct region *r = &regions[direct.region_count++];
        *r = (struct region){.physical = p.physical, .bytes = bytes, .physical_owned = 1,
                            .page_first = total_pages, .pages = (uint32_t)(bytes / PAGE)};
        direct.physical_owned_bytes += bytes;
        release(&lock);
    }
    if (!trace("RETURN AllocateDirectMemory rc=0x%08x physical=%lld canary=%d known_owner=%d unchanged_refusal=%d refusal_not_aggregate_limit=1",
        (unsigned)rc, (long long)p.physical, canary, known, refusal)) return uncertain("allocation_RETURN_log_failed");
    if (refusal) return 1;
    if (!known) return uncertain("physical_unknown_no_map_no_release");
    struct region *r = &regions[direct.region_count - 1];
    for (uint64_t i = 0; i + 1 < direct.region_count; ++i) {
        struct region *old = &regions[i];
        if ((uint64_t)r->physical < (uint64_t)old->physical + old->bytes &&
            (uint64_t)old->physical < (uint64_t)r->physical + r->bytes) return uncertain("physical_alias_or_overlap");
    }
    if (direct.mapped_bytes > PS5_AI_DATA_VA_BYTES || bytes > PS5_AI_DATA_VA_BYTES-direct.mapped_bytes)
        return uncertain("high_VA_interval_bounds");
    uint64_t expected_va=PS5_AI_DATA_VA_BASE+direct.mapped_bytes;
    struct { uint64_t pre[2]; void *cpu; uint64_t post[2]; } m = {{GUARD, GUARD}, (void *)(uintptr_t)expected_va, {GUARD, GUARD}};
    if (!trace("BEFORE MapDirectMemory region=%llu bytes=%llu protection=0x33 flags=0 alignment=2097152 hint=%p no_fixed_VA=1 no_Mprotect=1 disjoint_driver_window=1",
        (unsigned long long)(direct.region_count - 1), (unsigned long long)bytes, m.cpu)) return uncertain("map_BEFORE_log_failed");
    rc = sceKernelMapDirectMemory(&m.cpu, (size_t)bytes, 0x33, 0, r->physical, UNIT);
    uint64_t va = (uint64_t)(uintptr_t)m.cpu;
    canary = m.pre[0] == GUARD && m.pre[1] == GUARD && m.post[0] == GUARD && m.post[1] == GUARD;
    known = !rc && canary && va && va != UINTPTR_MAX && va % UNIT == 0 && va < VA_LIMIT && bytes <= VA_LIMIT - va;
    if (known) {
        r->cpu = m.cpu; r->mapped = 1;
        acquire(&lock); direct.mapped_bytes += bytes; state.mapping_owned = 1; release(&lock);
    }
    if (!trace("RETURN MapDirectMemory rc=0x%08x pointer=%p canary=%d mapping_known=%d high_hint_exact=%d GPU_readback_proven=0", (unsigned)rc, m.cpu, canary, known, known&&va==expected_va)) return uncertain("map_RETURN_log_failed");
    if (!known) return uncertain("mapping_unknown_no_release");
    if (va != expected_va) return uncertain("high_hint_ignored_known_mapping_retained");
    for (uint64_t i = 0; i + 1 < direct.region_count; ++i) {
        uint64_t old = (uint64_t)(uintptr_t)regions[i].cpu;
        if (va < old + regions[i].bytes && old < va + bytes) return uncertain("VA_alias_or_overlap");
    }
    struct { uint64_t pre[2]; void *low, *high; uint32_t flags, pad; uint64_t post[2]; } q = {
        {GUARD, GUARD}, NULL, NULL, 0, 0, {GUARD, GUARD}
    };
    if (!trace("BEFORE QueryMemoryProtection region=%llu pointer=%p", (unsigned long long)(direct.region_count - 1), r->cpu)) return uncertain("protection_BEFORE_log_failed");
    rc = sceKernelQueryMemoryProtection(r->cpu, &q.low, &q.high, &q.flags);
    canary = q.pre[0] == GUARD && q.pre[1] == GUARD && q.post[0] == GUARD && q.post[1] == GUARD && !q.pad;
    known = !rc && canary && (uintptr_t)q.low <= va && (uintptr_t)q.high >= va + bytes && (q.flags & 0x33) == 0x33;
    if (!trace("RETURN QueryMemoryProtection rc=0x%08x canary=%d low=%p high=%p flags=0x%x contains_owned_region=%d GPU_readback_proven=0",
        (unsigned)rc, canary, q.low, q.high, q.flags, known)) return uncertain("protection_RETURN_log_failed");
    if (!known) return uncertain("CPU_range_or_protection_unknown");
    total_pages += r->pages;
    acquire(&lock);
    if (bytes > direct.largest_region_bytes) direct.largest_region_bytes = bytes;
    release(&lock);
    return 0;
}
int ps5_ai_flex_prepare(void) {
    acquire(&lock);
    if (!direct.configured || direct.attempted) { release(&lock); return -1; }
    direct.attempted = 1; preparation.phase = PS5_AI_FLEX_CHECKING;
    preparation.requested_bytes = direct.target_bytes;
    preparation.working_reserve_bytes = direct.cpu_direct_reserve_bytes;
    release(&lock);
    journal = open_journal();
    if (journal < 0) { atomic_store(&log_ok, false); return -1; }
    if (!trace("BEGIN owned_native_Direct_data_pool_v1 pid=%d target_bytes=%llu region_max_bytes=%llu CPU_Direct_reserve_bytes=%llu log_path=%s data_open_errno=%d ordinary_driver_perBO=1 high_VA_partition=1 fixed_VA=0 Mprotect_calls=0 GPU_readback_proven=0",
        (int)getpid(), (unsigned long long)direct.target_bytes, (unsigned long long)direct.region_max_bytes,
        (unsigned long long)direct.cpu_direct_reserve_bytes, journal_path, data_open_errno)) return -1;
    int64_t total = 0; uint64_t available = 0;
    if (query_available(&total, &available) || direct.cpu_direct_reserve_bytes > (uint64_t)total ||
        direct.target_bytes > (uint64_t)total - direct.cpu_direct_reserve_bytes) return prepare_rollback("initial_query_or_total_admission");
    uint64_t pages = direct.target_bytes / PAGE;
    uint64_t count = direct.target_bytes / direct.region_max_bytes + (direct.target_bytes % direct.region_max_bytes != 0);
    uint64_t words = (pages + 63) / 64;
    if (!count || count > UINT32_MAX || count > SIZE_MAX / sizeof(*regions) || words > SIZE_MAX / sizeof(*bitmap) ||
        pages > UINT32_MAX || pages > SIZE_MAX / sizeof(*claims)) return prepare_rollback("metadata_overflow");
    uint32_t slots = (uint32_t)(pages > 4096 ? 4096 : pages);
    struct region *new_regions = calloc((size_t)count, sizeof(*regions));
    uint64_t *new_bitmap = calloc((size_t)words, sizeof(*bitmap));
    struct claim *new_claims = calloc(slots, sizeof(*claims));
    if (!new_regions || !new_bitmap || !new_claims) {
        free(new_regions); free(new_bitmap); free(new_claims); return prepare_rollback("metadata_allocation_failed");
    }
    acquire(&lock);
    region_capacity = (uint32_t)count; claim_capacity = slots;
    regions = new_regions; bitmap = new_bitmap; claims = new_claims;
    preparation.pages = pages; preparation.claim_capacity = claim_capacity;
    preparation.metadata_bytes = count * sizeof(*regions) + words * sizeof(*bitmap) + (uint64_t)claim_capacity * sizeof(*claims);
    preparation.configured_bytes = (uint64_t)total; preparation.available_bytes = available;
    preparation.phase = PS5_AI_FLEX_CPU_MAPPING;
    release(&lock);
    for (uint64_t done = 0; done < direct.target_bytes;) {
        uint64_t request = direct.target_bytes - done;
        if (request > direct.region_max_bytes) request = direct.region_max_bytes;
        if (query_available(&total, &available) || direct.cpu_direct_reserve_bytes > available ||
            request > available - direct.cpu_direct_reserve_bytes) return prepare_rollback("fresh_available_reserve_admission");
        int result = add_region(request, total);
        if (result < 0) return -1;
        if (result > 0) return prepare_rollback("single_request_refused_no_retry");
        done += request;
    }
    acquire(&lock);
    state.reserved_bytes = state.free_bytes = direct.mapped_bytes;
    state.mapping_owned = 1; state.gpu_access_granted = 1; direct.ready = 1;
    preparation.phase = PS5_AI_FLEX_GPU_READY;
    release(&lock);
    if (!trace("READY actual_owned_Direct_mapped_bytes=%llu physical_owned_bytes=%llu regions=%llu largest_claim_region_bytes=%llu CPU_Direct_reserve_bytes=%llu capacity_is_actual_owned_not_global_limit=1 GPU_readback_extent_bytes=0 imports=0 claim_cross_region_forbidden=1 whole_pool_contiguous_VA=0",
        (unsigned long long)state.reserved_bytes, (unsigned long long)direct.physical_owned_bytes,
        (unsigned long long)direct.region_count, (unsigned long long)direct.largest_region_bytes,
        (unsigned long long)direct.cpu_direct_reserve_bytes)) return uncertain("ready_journal_failed");
    return 0;
}
int ps5_ai_flex_prepare_sized(uint64_t requested, uint64_t reserve) {
    acquire(&lock); int same = direct.configured && requested == direct.target_bytes && reserve == direct.cpu_direct_reserve_bytes; release(&lock);
    return same ? ps5_ai_flex_prepare() : -1;
}
static int bit(uint32_t page) { return (int)((bitmap[page / 64] >> (page % 64)) & 1); }
static struct claim *find(void *address, uint64_t bytes) {
    for (uint32_t i = 0; i < claim_capacity; ++i) {
        struct claim *c = &claims[i];
        if (!(c->flags & CLAIM_ACTIVE) || c->region >= direct.region_count) continue;
        struct region *r = &regions[c->region];
        if ((uint8_t *)r->cpu + (uint64_t)c->first * PAGE == address && c->bytes == bytes) return c;
    }
    return NULL;
}
static uint32_t unused_claim(void) {
    for (uint32_t i = 0; i < claim_capacity; ++i) if (!(claims[i].flags & CLAIM_ACTIVE)) return i;
    if (claim_capacity == total_pages) return UINT32_MAX;
    uint32_t old = claim_capacity;
    uint32_t next = old > total_pages / 2 ? total_pages : old * 2;
    if (next <= old) return UINT32_MAX; /* total_pages already checked against SIZE_MAX/sizeof(claim). */
    struct claim *grown = realloc(claims, (size_t)next * sizeof(*claims));
    if (!grown) return UINT32_MAX;
    memset(grown + old, 0, (size_t)(next - old) * sizeof(*claims));
    claims = grown; claim_capacity = next; return old;
}
void *ps5_ai_flex_alloc(uint64_t bytes, uint64_t alignment) {
    acquire(&lock);
    int valid = bytes && bytes <= UINT64_MAX - (PAGE - 1) && alignment && !(alignment & (alignment - 1)) &&
        alignment <= UNIT && direct.ready && state.mapping_owned && state.gpu_access_granted &&
        !state.fault_retained && atomic_load(&log_ok);
    if (!valid) { state.denied_allocations++; release(&lock); return NULL; }
    uint64_t rounded = (bytes + PAGE - 1) & ~(PAGE - 1);
    if (rounded > direct.largest_region_bytes || rounded > state.free_bytes) { state.denied_allocations++; release(&lock); return NULL; }
    uint32_t pages = (uint32_t)(rounded / PAGE), slot = unused_claim();
    uint32_t found_region = UINT32_MAX, found_page = 0;
    if (slot != UINT32_MAX) for (uint32_t n = 0; n < direct.region_count; ++n) {
        struct region *r = &regions[n];
        if (pages > r->pages) continue;
        for (uint32_t first = 0; first <= r->pages - pages; ++first) {
            uint64_t address = (uint64_t)(uintptr_t)r->cpu + (uint64_t)first * PAGE;
            if (address % alignment) continue;
            uint32_t run = 0;
            while (run < pages && !bit(r->page_first + first + run)) ++run;
            if (run == pages) { found_region = n; found_page = first; break; }
            first += run;
        }
        if (found_region != UINT32_MAX) break;
    }
    if (found_region == UINT32_MAX) { state.denied_allocations++; release(&lock); return NULL; }
    struct region *r = &regions[found_region];
    for (uint32_t i = 0; i < pages; ++i) {
        uint32_t page = r->page_first + found_page + i; bitmap[page / 64] |= UINT64_C(1) << (page % 64);
    }
    claims[slot] = (struct claim){found_region, found_page, pages, CLAIM_ACTIVE, bytes};
    state.used_bytes += rounded; state.free_bytes -= rounded; state.allocations++;
    if (state.used_bytes > state.peak_used_bytes) state.peak_used_bytes = state.used_bytes;
    void *address = (uint8_t *)r->cpu + (uint64_t)found_page * PAGE;
    release(&lock);
    if (!trace("alloc pointer=%p bytes=%llu rounded_bytes=%llu region=%u claim_inside_one_region=1", address,
        (unsigned long long)bytes, (unsigned long long)rounded, found_region)) { ps5_ai_flex_retain_fault("claim_allocation_log_failed"); return NULL; }
    return address;
}
bool ps5_ai_flex_grant_gpu(void *address, uint64_t bytes) {
    acquire(&lock); struct claim *c = direct.ready ? find(address, bytes) : NULL;
    int valid = c && c->flags == CLAIM_ACTIVE && direct.ready && !state.fault_retained && atomic_load(&log_ok) &&
        bytes % PAGE == 0 && regions[c->region].mapped && regions[c->region].physical_owned;
    if (valid) c->flags |= CLAIM_GRANTED;
    release(&lock);
    if (!trace("import_grant allowlisted=%d pointer=%p bytes=%llu permission_calls=0 owned_Direct_Map33=1 physical_ownership_transferred=0",
        valid, address, (unsigned long long)bytes)) { ps5_ai_flex_retain_fault("grant_journal_failed"); return false; }
    return valid;
}
int ps5_ai_flex_import_committed(void *address, uint64_t bytes) {
    acquire(&lock); struct claim *c = direct.ready ? find(address, bytes) : NULL;
    int valid = c && c->flags == (CLAIM_ACTIVE | CLAIM_GRANTED) && !state.fault_retained && atomic_load(&log_ok);
    if (valid) { c->flags |= CLAIM_IMPORTED; state.active_imports++; }
    release(&lock);
    if (!valid) { ps5_ai_flex_retain_fault("import_commit_ownership_invalid"); return -1; }
    if (!trace("import_committed pointer=%p bytes=%llu", address, (unsigned long long)bytes)) { ps5_ai_flex_retain_fault("commit_journal_failed"); return -1; }
    return 0;
}
static int positive_quiet(const char *where) {
    struct ps5_ai_ggml_dispatch_stats dispatch = {0}; struct ps5_ai_arena_stats driver = {0};
    ps5_ai_ggml_dispatch_snapshot(&dispatch); ps5_ai_arena_snapshot(&driver);
    if (dispatch.work_pending || driver.gpu_fault_retained || !ps5_ai_ggml_log_healthy() ||
        !ps5_ai_ggml_wait_backend(where, dispatch.submits != 0)) { ps5_ai_flex_retain_fault("without_positive_GPU_idle_receipt"); return 0; }
    return 1;
}
int ps5_ai_flex_free(void *address, uint64_t bytes) {
    if (!positive_quiet("native_Direct_owned_buffer_free")) return -1;
    acquire(&lock); struct claim *c = direct.ready ? find(address, bytes) : NULL;
    if (!c || state.fault_retained || !atomic_load(&log_ok)) { release(&lock); ps5_ai_flex_retain_fault("free_claim_invalid"); return -1; }
    uint64_t rounded = (uint64_t)c->pages * PAGE; struct region *r = &regions[c->region];
    for (uint32_t i = 0; i < c->pages; ++i) {
        uint32_t page = r->page_first + c->first + i; bitmap[page / 64] &= ~(UINT64_C(1) << (page % 64));
    }
    if (c->flags & CLAIM_IMPORTED) state.active_imports--;
    memset(c, 0, sizeof(*c)); state.used_bytes -= rounded; state.free_bytes += rounded; state.allocations--;
    release(&lock);
    if (!trace("free pointer=%p bytes=%llu data_Direct_regions_retained=1", address, (unsigned long long)bytes)) { ps5_ai_flex_retain_fault("free_journal_failed"); return -1; }
    return 0;
}
int ps5_ai_direct_pool_release_if_idle(void) {
    if (!positive_quiet("before_native_Direct_data_pool_release")) return -1;
    acquire(&lock);
    int safe = direct.ready && !state.allocations && !state.active_imports && !state.used_bytes &&
        !state.fault_retained && !direct.ownership_uncertain && atomic_load(&log_ok);
    if (safe) { direct.ready = 0; state.gpu_access_granted = 0; }
    release(&lock);
    if (!safe) return 1;
    /* This is an explicit successful release, not rollback or self exit. */
    int rc = prepare_rollback("explicit_all_imported_objects_destroyed_idle_release");
    return !direct.physical_owned_bytes && !direct.mapped_bytes && !state.fault_retained && atomic_load(&log_ok) ? 0 : rc;
}
