/*
 * Copyright ﾂｩ 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_ps5_platform.h"
#include "arena.h"
#include "placement.h"
static bool ps5_ai_arena_trace(const char *format, ...);

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "c11/threads.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"
#include "util/u_math.h"

#if defined(__PROSPERO__)
#include <ps5platform/agc.h>
#include <ps5platform/kernel.h>
#else
#include <sys/mman.h>
#endif

/* A first-fit allocator over a range of fixed-size granules, one bit each. The
 * console places buffers the GPU reaches through full addresses in a region it
 * maps at the address asked for (PS5_Vulkan R88); the host model places the
 * address window's buffers the same way. */
struct radv_ps5_granules {
   simple_mtx_t lock;
   uint64_t base;
   uint64_t granule_bytes;
   uint32_t count;
   uint64_t *used;
};

static bool
radv_ps5_granules_init(struct radv_ps5_granules *g, uint64_t base, uint64_t bytes, uint64_t granule_bytes)
{
   simple_mtx_init(&g->lock, mtx_plain);
   g->base = base;
   g->granule_bytes = granule_bytes;
   g->count = (uint32_t)(bytes / granule_bytes);
   g->used = calloc(DIV_ROUND_UP(g->count, 64), sizeof(uint64_t));
   return g->used != NULL;
}

static void
radv_ps5_granules_mark(struct radv_ps5_granules *g, uint32_t first, uint32_t count, bool used)
{
   for (uint32_t i = first; i < first + count; i++) {
      const uint64_t bit = UINT64_C(1) << (i % 64);
      if (used)
         g->used[i / 64] |= bit;
      else
         g->used[i / 64] &= ~bit;
   }
}

/* count free granules in a row, aligned to align_granules; UINT32_MAX if none. */
static uint32_t
radv_ps5_granules_take(struct radv_ps5_granules *g, uint32_t count, uint32_t align_granules)
{
   if (count == 0 || count > g->count)
      return UINT32_MAX;
   align_granules = MAX2(align_granules, 1);
   simple_mtx_lock(&g->lock);
   uint32_t start = 0;
   while (start + count <= g->count) {
      uint32_t run = 0;
      while (run < count && !(g->used[(start + run) / 64] & (UINT64_C(1) << ((start + run) % 64))))
         run++;
      if (run == count) {
         radv_ps5_granules_mark(g, start, count, true);
         simple_mtx_unlock(&g->lock);
         return start;
      }
      start = align(start + run + 1, align_granules);
   }
   simple_mtx_unlock(&g->lock);
   return UINT32_MAX;
}

static bool
radv_ps5_granules_free(const struct radv_ps5_granules *g, uint32_t first, uint32_t count)
{
   for (uint32_t i = first; i < first + count; i++) {
      if (g->used[i / 64] & (UINT64_C(1) << (i % 64)))
         return false;
   }
   return true;
}

/* The same from the top down: where captured buffers go, away from everything
 * else, so their addresses are still free when a replay asks for them. */
static uint32_t
radv_ps5_granules_take_top(struct radv_ps5_granules *g, uint32_t count, uint32_t align_granules)
{
   if (count == 0 || count > g->count)
      return UINT32_MAX;
   align_granules = MAX2(align_granules, 1);
   simple_mtx_lock(&g->lock);
   for (uint32_t start = (g->count - count) / align_granules * align_granules;;
        start -= align_granules) {
      if (radv_ps5_granules_free(g, start, count)) {
         radv_ps5_granules_mark(g, start, count, true);
         simple_mtx_unlock(&g->lock);
         return start;
      }
      if (start < align_granules)
         break;
   }
   simple_mtx_unlock(&g->lock);
   return UINT32_MAX;
}

/* Exactly the granules from first on, if they are all free. */
static bool
radv_ps5_granules_take_at(struct radv_ps5_granules *g, uint32_t first, uint32_t count)
{
   if (count == 0 || first >= g->count || count > g->count - first)
      return false;
   simple_mtx_lock(&g->lock);
   const bool free = radv_ps5_granules_free(g, first, count);
   if (free)
      radv_ps5_granules_mark(g, first, count, true);
   simple_mtx_unlock(&g->lock);
   return free;
}

static void
radv_ps5_granules_give(struct radv_ps5_granules *g, uint32_t first, uint32_t count)
{
   if (count == 0)
      return;
   simple_mtx_lock(&g->lock);
   radv_ps5_granules_mark(g, first, count, false);
   simple_mtx_unlock(&g->lock);
}

/* The device-memory region: 256 GiB at 0x4000000000, handed out in 2 MiB
 * granules. The GPU reads and writes the whole direct-memory pool there
 * (PS5_Vulkan R86-R88). */
#define RADV_PS5_REGION_BASE UINT64_C(0x4000000000)
#define RADV_PS5_REGION_BYTES (UINT64_C(256) << 30)
_Static_assert(RADV_PS5_REGION_BASE==PS5_AI_REFERENCE_VA_BASE&&RADV_PS5_REGION_BYTES==PS5_AI_REFERENCE_VA_BYTES,"Shared reference VA partition must match native manager");

static struct radv_ps5_granules radv_ps5_region;
static once_flag radv_ps5_once = ONCE_FLAG_INIT;
static bool radv_ps5_ready;

uint64_t
radv_ps5_now_ns(void)
{
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC, &now);
   return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

void
radv_ps5_sleep_us(unsigned microseconds)
{
   const struct timespec delay = {
      .tv_sec = microseconds / 1000000u,
      .tv_nsec = (long)(microseconds % 1000000u) * 1000l,
   };
   nanosleep(&delay, NULL);
}

void
radv_ps5_cpu_flush(const void *address, size_t bytes)
{
   if (bytes == 0)
      return;
   const uintptr_t line = 64;
   uintptr_t at = (uintptr_t)address & ~(line - 1);
   const uintptr_t end = (uintptr_t)address + bytes;
   /* CLFLUSHOPT (Zen 2 has it): the lines are written back and dropped as
    * CLFLUSH does, but not one after another; the closing MFENCE orders them
    * before every later store and load (a submission, or a read of what the
    * GPU wrote). A submission's words took one serialising CLFLUSH a line. */
   __builtin_ia32_mfence();
   for (; at < end; at += line)
      __asm__ volatile("clflushopt %0" : "+m"(*(volatile char *)at));
   __builtin_ia32_mfence();
}

static bool
radv_ps5_window_contains(uint64_t address, uint64_t bytes)
{
   return address >= RADV_PS5_WINDOW_BASE && bytes <= RADV_PS5_WINDOW_BYTES &&
          address - RADV_PS5_WINDOW_BASE <= RADV_PS5_WINDOW_BYTES - bytes;
}

#if defined(__PROSPERO__)

/* ------------------------------------------------------------------ console */

/* GPU-visible direct memory as ps5vk and the test runner allocate it: type 12,
 * mapped for CPU and GPU read and write. */
#define RADV_PS5_DIRECT_TYPE 12
#define RADV_PS5_PROTECTION                                                                        \
   (PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE | PS5_KERNEL_PROT_GPU_READ |              \
    PS5_KERNEL_PROT_GPU_WRITE)

static int32_t radv_ps5_agc_result = -1;

bool
radv_ps5_window_replayable(void)
{
   return false; /* no replay VA reservation in the owned-arena test */
}

static void
radv_ps5_platform_once(void)
{
   setenv("MESA_SHADER_CACHE_DISABLE", "true", 1);
   /* Order-only comparison: init before the first real arena allocation.
    * arena_log is not open yet. The mandatory frontend --wrap=sceAgcInit
    * records durable BEFORE/RETURN, enforces one version8 call, and main
    * requires attempts==1/result==0/log healthy before arena_prepare.
    * No ordinary/replay VA or direct-memory reservation occurs here.
    */
   radv_ps5_agc_result = sceAgcInit(PS5_AGC_INIT_VERSION);
   radv_ps5_ready = radv_ps5_agc_result == 0 &&
      radv_ps5_granules_init(&radv_ps5_region, RADV_PS5_REGION_BASE, PS5_AI_DRIVER_VA_BYTES, RADV_PS5_LARGE_BYTES);
}

bool
radv_ps5_platform_runs_gpu(void)
{
   return true;
}

#include "perbo_owner_impl.c"

/* Where radv_ps5_submit's time went (radv_ps5_submit_times); [2] is the kick
 * thread's. */
static uint64_t radv_ps5_submit_ns[3];

/* Without a suspend point after it the console starts a submission up to a
 * refresh late (PS5_Vulkan R68), and the suspend point took about 0.3 ms a
 * submission of the submitting thread (RPCS3's renderer: 2,000 of them in
 * 10 s). A thread of its own makes it instead, as soon as a submission is
 * made; submissions made while it is at one share the next. */
#include "kick_impl32.c"

int
radv_ps5_submit(uint32_t *words, uint32_t count, volatile uint32_t *marker, uint32_t marker_value)
{
   (void)marker;
   (void)marker_value;
   uint64_t epoch = 0;
   if (!radv_ps5_kick_submit_begin(&epoch)) {
      ps5_ai_arena_retain_gpu_fault();
      return -1;
   }
   const uint64_t started = os_time_get_nano();
   radv_ps5_cpu_flush(words, (size_t)count * sizeof(uint32_t));
   const uint64_t flushed = os_time_get_nano();
   struct ps5_agc_submit_description description = {
      .words = words,
      .word_count = count,
   };
   int32_t result = -1;
   bool failed = !ps5_ai_arena_trace_routine("BEFORE sceAgcDriverSubmitDcb epoch=%llu word_count=%u hard_timeout=none", (unsigned long long)epoch, count);
   if (!failed) {
      result = sceAgcDriverSubmitDcb(&description);
      const bool logged = (result == 0 ? ps5_ai_arena_trace_routine : ps5_ai_arena_trace)("RETURN sceAgcDriverSubmitDcb epoch=%llu rc=0x%08x", (unsigned long long)epoch, (unsigned)result);
      failed = !logged || result != 0;
   }
   const uint64_t submitted = os_time_get_nano();
   radv_ps5_submit_ns[0] += flushed - started;
   radv_ps5_submit_ns[1] += submitted - flushed;
   if (failed)
      ps5_ai_arena_retain_gpu_fault();
   if (!radv_ps5_kick_submit_finish(epoch, failed))
      return result != 0 ? result : -1;
   return 0;
}

void
radv_ps5_submit_times(uint64_t times[3])
{
   for (unsigned i = 0; i < 3; i++)
      times[i] = p_atomic_xchg(&radv_ps5_submit_ns[i], 0);
}

int
radv_ps5_set_tess_factor_ring(uint64_t va, uint32_t size)
{
   return sceAgcDriverSetTFRing((uintptr_t)va, size);
}

int
radv_ps5_set_hs_offchip_param(uint32_t granularity, uint32_t buffering)
{
   return sceAgcDriverSetHsOffchipParam(granularity, buffering);
}

bool
radv_ps5_memory_map_at(const struct radv_ps5_memory *memory, void *address)
{
   (void)memory; (void)address;
   ps5_ai_arena_trace("map_family=placed denied=1 fallback=0");
   return false;
}

bool
radv_ps5_memory_grant_gpu(void *address, uint64_t bytes)
{
   /* The owner only grants one exact live Direct claim. It performs no
    * permission mutation; each separate region was already MapDirect33. */
   extern bool ps5_ai_flex_grant_gpu(void *, uint64_t);
   return ps5_ai_flex_grant_gpu(address, bytes);
}

void
radv_ps5_memory_unmap_at(void *address, uint64_t bytes, bool reserve)
{
   (void)address; (void)bytes; (void)reserve;
   ps5_ai_arena_trace("unmap_family=placed refused=unsupported no_kernel_unmap=1");
}

#else

/* --------------------------------------------------------------- host model */

/* The window is reserved at the console's address, so 32-bit pointers carry
 * the same high word on both; buffers outside it are ordinary mappings. */
static struct radv_ps5_granules radv_ps5_window;
#define RADV_PS5_WINDOW_GRANULE UINT64_C(0x10000)

static void
radv_ps5_platform_once(void)
{
   void *const window = mmap((void *)(uintptr_t)RADV_PS5_WINDOW_BASE, RADV_PS5_WINDOW_BYTES, PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
   if (window != (void *)(uintptr_t)RADV_PS5_WINDOW_BASE) {
      fprintf(stderr, "radv/ps5: the host model could not reserve the address window: %s\n", strerror(errno));
      if (window != MAP_FAILED)
         munmap(window, RADV_PS5_WINDOW_BYTES);
      return;
   }
   radv_ps5_ready = radv_ps5_granules_init(&radv_ps5_window, RADV_PS5_WINDOW_BASE, RADV_PS5_WINDOW_BYTES,
                                           RADV_PS5_WINDOW_GRANULE);
}

/* The Vulkan loader may unload the driver once its instances are gone and load
 * it again (the CTS does): the window goes with the library, or the next load
 * finds it taken. */
static void __attribute__((destructor))
radv_ps5_platform_unload(void)
{
   if (!radv_ps5_ready)
      return;
   munmap((void *)(uintptr_t)RADV_PS5_WINDOW_BASE, RADV_PS5_WINDOW_BYTES);
   free(radv_ps5_window.used);
   simple_mtx_destroy(&radv_ps5_window.lock);
}

bool
radv_ps5_platform_runs_gpu(void)
{
   return false;
}

uint64_t
radv_ps5_memory_pool_bytes(void)
{
   /* The console's pool is about 12 GiB; the model reports the same. */
   return UINT64_C(12) << 30;
}

uint64_t
radv_ps5_memory_available_bytes(void)
{
   return radv_ps5_memory_pool_bytes();
}

bool
radv_ps5_memory_alloc(uint64_t bytes, uint64_t alignment, bool window32, struct radv_ps5_memory *out)
{
   *out = (struct radv_ps5_memory){.physical = -1};
   bytes = align64(MAX2(bytes, 1), RADV_PS5_PAGE_BYTES);
   if (window32) {
      const uint32_t count = (uint32_t)DIV_ROUND_UP(bytes, RADV_PS5_WINDOW_GRANULE);
      const uint32_t first = radv_ps5_granules_take(
         &radv_ps5_window, count, (uint32_t)MAX2(util_next_power_of_two64(alignment) / RADV_PS5_WINDOW_GRANULE, 1));
      if (first == UINT32_MAX)
         return false;
      uint8_t *const cpu = (uint8_t *)(uintptr_t)(RADV_PS5_WINDOW_BASE + (uint64_t)first * RADV_PS5_WINDOW_GRANULE);
      const uint64_t span = (uint64_t)count * RADV_PS5_WINDOW_GRANULE;
      if (mprotect(cpu, span, PROT_READ | PROT_WRITE) != 0) {
         radv_ps5_granules_give(&radv_ps5_window, first, count);
         return false;
      }
      *out = (struct radv_ps5_memory){.cpu = cpu, .bytes = span, .physical = -1, .granule = first, .granules = count};
      return true;
   }
   alignment = util_next_power_of_two64(MAX2(alignment, RADV_PS5_PAGE_BYTES));
   /* Over-allocate so the start can be aligned, then trim. */
   const uint64_t span = bytes + alignment;
   uint8_t *const raw = mmap(NULL, span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
   if (raw == MAP_FAILED)
      return false;
   uint8_t *const cpu = (uint8_t *)align64((uint64_t)(uintptr_t)raw, alignment);
   if (cpu > raw)
      munmap(raw, cpu - raw);
   const uint8_t *const end = raw + span;
   if (cpu + bytes < end)
      munmap(cpu + bytes, end - (cpu + bytes));
   *out = (struct radv_ps5_memory){.cpu = cpu, .bytes = bytes, .physical = -1};
   return true;
}

bool
radv_ps5_memory_alloc_replayable(uint64_t bytes, uint64_t alignment, bool window32, uint64_t replay_va,
                                 struct radv_ps5_memory *out)
{
   if (!replay_va)
      return radv_ps5_memory_alloc(bytes, alignment, window32, out);
   *out = (struct radv_ps5_memory){.physical = -1};
   bytes = align64(MAX2(bytes, 1), RADV_PS5_PAGE_BYTES);
   if (window32) {
      const uint32_t count = (uint32_t)DIV_ROUND_UP(bytes, RADV_PS5_WINDOW_GRANULE);
      if (!radv_ps5_window_contains(replay_va, bytes) || (replay_va - RADV_PS5_WINDOW_BASE) % RADV_PS5_WINDOW_GRANULE)
         return false;
      const uint32_t first = (uint32_t)((replay_va - RADV_PS5_WINDOW_BASE) / RADV_PS5_WINDOW_GRANULE);
      if (!radv_ps5_granules_take_at(&radv_ps5_window, first, count))
         return false;
      uint8_t *const cpu = (uint8_t *)(uintptr_t)replay_va;
      const uint64_t span = (uint64_t)count * RADV_PS5_WINDOW_GRANULE;
      if (mprotect(cpu, span, PROT_READ | PROT_WRITE) != 0) {
         radv_ps5_granules_give(&radv_ps5_window, first, count);
         return false;
      }
      *out = (struct radv_ps5_memory){.cpu = cpu, .bytes = span, .physical = -1, .granule = first, .granules = count};
      return true;
   }
   if (replay_va % RADV_PS5_PAGE_BYTES)
      return false;
   void *const cpu = mmap((void *)(uintptr_t)replay_va, bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
   if (cpu == MAP_FAILED)
      return false;
   if (cpu != (void *)(uintptr_t)replay_va) {
      munmap(cpu, bytes);
      return false;
   }
   *out = (struct radv_ps5_memory){.cpu = cpu, .bytes = bytes, .physical = -1};
   return true;
}

/* The model places every window buffer itself (radv_ps5_window). */
bool
radv_ps5_window_replayable(void)
{
   return true;
}

bool
radv_ps5_memory_grant_gpu(void *address, uint64_t bytes)
{
   /* The model runs no GPU work: the memory stays the application's. */
   (void)address;
   (void)bytes;
   return true;
}

/* Sparse ranges in the model: plain memory, which binds leave alone (the
 * model runs no GPU work to read through them). */
bool
radv_ps5_vrange_reserve(uint64_t bytes, bool window32, bool replayable, uint64_t replay_va,
                        struct radv_ps5_memory *out)
{
   const uint64_t span = align64(MAX2(bytes, 1), RADV_PS5_LARGE_BYTES);
   return replayable ? radv_ps5_memory_alloc_replayable(span, RADV_PS5_LARGE_BYTES, window32, replay_va, out)
                     : radv_ps5_memory_alloc(span, RADV_PS5_LARGE_BYTES, window32, out);
}

void
radv_ps5_vrange_release(struct radv_ps5_memory *range)
{
   radv_ps5_memory_free(range);
}

bool
radv_ps5_vrange_bind(void *at, uint64_t bytes, const struct radv_ps5_memory *memory, uint64_t offset)
{
   (void)at;
   return offset <= memory->bytes && bytes <= memory->bytes - offset;
}

bool
radv_ps5_vrange_unbind(void *at, uint64_t bytes)
{
   (void)at;
   (void)bytes;
   return true;
}

uint64_t
radv_ps5_vrange_space_bytes(void)
{
   return UINT64_C(256) << 30;
}

void
radv_ps5_memory_free(struct radv_ps5_memory *memory)
{
   if (!memory->cpu)
      return;
   if (radv_ps5_window_contains((uint64_t)(uintptr_t)memory->cpu, memory->bytes)) {
      /* Back to reserved, and zero when it is handed out again. */
      mmap(memory->cpu, memory->bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);
      radv_ps5_granules_give(&radv_ps5_window, memory->granule, memory->granules);
   } else {
      munmap(memory->cpu, memory->bytes);
   }
   *memory = (struct radv_ps5_memory){.physical = -1};
}

int
radv_ps5_submit(uint32_t *words, uint32_t count, volatile uint32_t *marker, uint32_t marker_value)
{
   (void)words;
   (void)count;
   /* Nothing runs the words on a PC; the submission completes at once. */
   if (marker)
      *marker = marker_value;
   return 0;
}

void
radv_ps5_submit_times(uint64_t times[3])
{
   times[0] = times[1] = times[2] = 0;
}

int
radv_ps5_set_tess_factor_ring(uint64_t va, uint32_t size)
{
   (void)va;
   (void)size;
   return 0;
}

int
radv_ps5_set_hs_offchip_param(uint32_t granularity, uint32_t buffering)
{
   (void)granularity;
   (void)buffering;
   return 0;
}

bool
radv_ps5_memory_map_at(const struct radv_ps5_memory *memory, void *address)
{
   /* The host model's memory is anonymous: nothing maps it twice. */
   (void)memory;
   (void)address;
   return false;
}

void
radv_ps5_memory_unmap_at(void *address, uint64_t bytes, bool reserve)
{
   (void)address;
   (void)bytes;
   (void)reserve;
}

#endif

bool
radv_ps5_platform_init(void)
{
   call_once(&radv_ps5_once, radv_ps5_platform_once);
   return radv_ps5_ready;
}


/* Opt-in diagnostic query readback. This is not a general Vulkan driver change.
 * Only the internally owned timestamp pool is accepted, after the same existing
 * positive fence/quiescence contract used by graph execution. CPU never stores
 * into query storage: reset is a GPU command and writes are GPU timestamps.
 */
#include "moss_gpu_profile.h"
#if PS5_AI_MOSS_GPU_PROFILE
#include "probe.h"
#include "radv_query.h"
/* Exact layouts independently observed in the adopted staticdriver disassembly.
 * Refuse an incompatible Mesaheader/driver ABI before touching opaquehandles. */
_Static_assert(offsetof(struct radv_query_pool, vk.query_type) == 0x40, "query type ABI");
_Static_assert(offsetof(struct radv_query_pool, stride) == 0x58, "query stride ABI");
_Static_assert(offsetof(struct radv_query_pool, size) == 0x60, "query size ABI");
_Static_assert(offsetof(struct radv_query_pool, ptr) == 0x68, "query pointer ABI");
int ps5_ai_moss_gpu_profile_query_cache_readback(uint64_t handle, uint32_t first,
                                                uint32_t count, uint64_t before_fences);
int
ps5_ai_moss_gpu_profile_query_cache_readback(uint64_t handle, uint32_t first,
                                            uint32_t count, uint64_t before_fences)
{
   struct ps5_ai_ggml_dispatch_stats completed = {0};
   ps5_ai_ggml_dispatch_snapshot(&completed);
   if (!handle || !completed.hardware_verified || completed.work_pending ||
       completed.fenced_completions <= before_fences)
      return 0;
   struct radv_query_pool *pool = radv_query_pool_from_handle((VkQueryPool)(uintptr_t)handle);
   if (!pool || pool->vk.base.type != VK_OBJECT_TYPE_QUERY_POOL ||
       pool->vk.query_type != VK_QUERY_TYPE_TIMESTAMP || pool->stride != 8 || !pool->ptr ||
       !ps5_ai_moss_gpu_range_valid(first, count, pool->vk.query_count))
      return 0;
   const uint64_t offset = (uint64_t)first * pool->stride;
   const uint64_t bytes = (uint64_t)count * pool->stride;
   if (offset > pool->size || bytes > pool->size - offset || bytes > SIZE_MAX)
      return 0;
   radv_ps5_cpu_flush(pool->ptr + offset, (size_t)bytes);
   return 1;
}
#endif
