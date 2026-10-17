/* SPDX-License-Identifier: MIT
 * Test-only completion accounting for the original asynchronous RADV kick.
 * Included after arena_impl.c and radv_ps5_submit_ns in the platform variant.
 * All kick state is protected by the same mutex; initialization is published
 * separately so read-only queries never initialize a worker themselves.
 */
#include <limits.h>
static mtx_t radv_ps5_kick_lock;
static cnd_t radv_ps5_kick_wanted;
static once_flag radv_ps5_kick_once = ONCE_FLAG_INIT;
static atomic_bool radv_ps5_kick_initialized = ATOMIC_VAR_INIT(false);
static bool radv_ps5_kick_thread_started;
static bool radv_ps5_kick_pending;
static bool radv_ps5_kick_active;
static bool radv_ps5_kick_failed;
static unsigned radv_ps5_kick_submit_inflight;
static uint64_t radv_ps5_kick_requested_epoch;
static uint64_t radv_ps5_kick_ready_epoch;
static uint64_t radv_ps5_kick_completed_epoch;

static void
radv_ps5_kick_fault(const char *reason)
{
   ps5_ai_arena_retain_gpu_fault();
   ps5_ai_arena_trace("kick_fault=%s irreversible=1", reason);
}

int
ps5_ai_arena_kick_snapshot(struct ps5_ai_arena_kick_stats *out)
{
   if (!out)
      return -1;
   struct ps5_ai_arena_kick_stats copy = {0};
   if (atomic_load_explicit(&radv_ps5_kick_initialized, memory_order_acquire)) {
      const int locked = mtx_trylock(&radv_ps5_kick_lock);
      if (locked == thrd_busy)
         return 1;
      if (locked != thrd_success)
         return -1;
      copy.initialized = 1;
      copy.thread_started = radv_ps5_kick_thread_started;
      copy.pending = radv_ps5_kick_pending;
      copy.active = radv_ps5_kick_active;
      copy.submit_inflight = radv_ps5_kick_submit_inflight != 0;
      copy.requested_epoch = radv_ps5_kick_requested_epoch;
      copy.completed_epoch = radv_ps5_kick_completed_epoch;
      copy.fault_retained = radv_ps5_kick_failed;
      if (mtx_unlock(&radv_ps5_kick_lock) != thrd_success)
         return -1;
   }
   struct ps5_ai_arena_stats owned;
   ps5_ai_arena_snapshot(&owned);
   copy.fault_retained |= owned.gpu_fault_retained;
   copy.log_healthy = atomic_load_explicit(&arena_log_healthy, memory_order_acquire);
   *out = copy;
   return 0;
}

static int
radv_ps5_kick_body(void *unused)
{
   (void)unused;
   if (mtx_lock(&radv_ps5_kick_lock) != thrd_success) {
      radv_ps5_kick_fault("worker_lock_failed");
      return -1;
   }
   for (;;) {
      while (!radv_ps5_kick_pending) {
         if (cnd_wait(&radv_ps5_kick_wanted, &radv_ps5_kick_lock) != thrd_success) {
            radv_ps5_kick_failed = true;
            radv_ps5_kick_active = true;
            mtx_unlock(&radv_ps5_kick_lock);
            radv_ps5_kick_fault("worker_wait_failed");
            return -1;
         }
      }
      const uint64_t target = radv_ps5_kick_ready_epoch;
      const bool already_failed = radv_ps5_kick_failed;
      /* No pending=false/active=false gap visible to the query. */
      radv_ps5_kick_active = true;
      radv_ps5_kick_pending = false;
      if (mtx_unlock(&radv_ps5_kick_lock) != thrd_success) {
         radv_ps5_kick_fault("worker_unlock_failed");
         return -1;
      }
      bool failed = already_failed;
      struct ps5_ai_arena_stats owned;
      ps5_ai_arena_snapshot(&owned);
      failed |= owned.gpu_fault_retained || !atomic_load(&arena_log_healthy);
      if (!failed) {
         failed = !ps5_ai_arena_trace_routine("BEFORE sceAgcSuspendPoint epoch=%llu hard_timeout=none", (unsigned long long)target);
         if (!failed) {
            const uint64_t started = os_time_get_nano();
            const int32_t result = sceAgcSuspendPoint();
            p_atomic_add(&radv_ps5_submit_ns[2], os_time_get_nano() - started);
            const bool logged = (result == 0 ? ps5_ai_arena_trace_routine : ps5_ai_arena_trace)("RETURN sceAgcSuspendPoint epoch=%llu rc=0x%08x", (unsigned long long)target, (unsigned)result);
            failed = !logged || result != 0;
         }
      }
      /* Retention + its durable log are complete before publishing completion. */
      if (failed)
         radv_ps5_kick_fault("suspend_point_or_log_failed");
      if (mtx_lock(&radv_ps5_kick_lock) != thrd_success) {
         radv_ps5_kick_fault("worker_completion_lock_failed");
         return -1;
      }
      radv_ps5_kick_failed |= failed;
      if (target > radv_ps5_kick_completed_epoch)
         radv_ps5_kick_completed_epoch = target;
      radv_ps5_kick_active = false;
      /* Keep the lock when returning to cnd_wait, as in the original worker. */
   }
}

static void
radv_ps5_kick_start(void)
{
   if (!ps5_ai_arena_trace("BEFORE kick_worker_init hard_timeout=none") ||
       mtx_init(&radv_ps5_kick_lock, mtx_plain) != thrd_success) {
      radv_ps5_kick_fault("mutex_init_or_log_failed");
      return;
   }
   if (cnd_init(&radv_ps5_kick_wanted) != thrd_success) {
      radv_ps5_kick_fault("condition_init_failed");
      return;
   }
   atomic_store_explicit(&radv_ps5_kick_initialized, true, memory_order_release);
   thrd_t thread;
   const int created = thrd_create(&thread, radv_ps5_kick_body, NULL);
   const bool logged = ps5_ai_arena_trace("RETURN kick_worker_create rc=%d", created);
   const int detached = created == thrd_success ? thrd_detach(thread) : thrd_error;
   const bool detach_logged = ps5_ai_arena_trace("RETURN kick_worker_detach rc=%d attempted=%d", detached, created == thrd_success);
   const bool failed = !logged || !detach_logged || created != thrd_success || detached != thrd_success;
   if (failed)
      radv_ps5_kick_fault("thread_create_detach_or_log_failed");
   if (mtx_lock(&radv_ps5_kick_lock) != thrd_success) {
      radv_ps5_kick_fault("init_publish_lock_failed");
      return;
   }
   radv_ps5_kick_thread_started = created == thrd_success;
   radv_ps5_kick_failed |= failed;
   if (mtx_unlock(&radv_ps5_kick_lock) != thrd_success)
      radv_ps5_kick_fault("init_publish_unlock_failed");
}

/* Register the request BEFORE entering SubmitDcb, not after a GPU marker can
 * complete. submit_inflight covers the hardware-call/publication interval. */
static bool
radv_ps5_kick_submit_begin(uint64_t *epoch)
{
   call_once(&radv_ps5_kick_once, radv_ps5_kick_start);
   struct ps5_ai_arena_stats owned;
   ps5_ai_arena_snapshot(&owned);
   if (!atomic_load_explicit(&radv_ps5_kick_initialized, memory_order_acquire) ||
       owned.gpu_fault_retained || !atomic_load(&arena_log_healthy))
      return false;
   if (mtx_lock(&radv_ps5_kick_lock) != thrd_success) {
      radv_ps5_kick_fault("submit_begin_lock_failed");
      return false;
   }
   if (!radv_ps5_kick_thread_started || radv_ps5_kick_failed ||
       radv_ps5_kick_requested_epoch == UINT64_MAX || radv_ps5_kick_submit_inflight == UINT_MAX) {
      radv_ps5_kick_failed = true;
      mtx_unlock(&radv_ps5_kick_lock);
      radv_ps5_kick_fault("submit_begin_unready_or_overflow");
      return false;
   }
   *epoch = ++radv_ps5_kick_requested_epoch;
   radv_ps5_kick_submit_inflight++;
   if (mtx_unlock(&radv_ps5_kick_lock) != thrd_success) {
      radv_ps5_kick_fault("submit_begin_unlock_failed");
      return false;
   }
   return true;
}

static bool
radv_ps5_kick_submit_finish(uint64_t epoch, bool failed)
{
   /* Caller has finished RETURN logging, and retention if it failed. */
   if (mtx_lock(&radv_ps5_kick_lock) != thrd_success) {
      radv_ps5_kick_fault("submit_finish_lock_failed");
      return false;
   }
   radv_ps5_kick_failed |= failed;
   if (!radv_ps5_kick_submit_inflight || !epoch || epoch > radv_ps5_kick_requested_epoch) {
      radv_ps5_kick_failed = true;
      mtx_unlock(&radv_ps5_kick_lock);
      radv_ps5_kick_fault("submit_finish_invalid_epoch");
      return false;
   }
   radv_ps5_kick_submit_inflight--;
   if (!failed) {
      if (epoch > radv_ps5_kick_ready_epoch)
         radv_ps5_kick_ready_epoch = epoch;
      radv_ps5_kick_pending = true;
      if (cnd_signal(&radv_ps5_kick_wanted) != thrd_success) {
         radv_ps5_kick_failed = true;
         failed = true;
      }
   }
   if (mtx_unlock(&radv_ps5_kick_lock) != thrd_success) {
      radv_ps5_kick_fault("submit_finish_unlock_failed");
      return false;
   }
   if (failed)
      radv_ps5_kick_fault("submit_or_signal_failed");
   return !failed;
}
