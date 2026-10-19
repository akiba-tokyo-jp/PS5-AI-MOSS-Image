/* SPDX-License-Identifier: MIT -- included once by native startup.c.
 * One atomic word publishes stage + first failure boundary + readiness. No
 * allocation, lock, filesystem, GPU call, exit or privileged operation.
 */
_Static_assert(ATOMIC_INT_LOCK_FREE == 2 && sizeof(unsigned)==sizeof(uint32_t),
    "Lifecycle publication requires lock-free32-bit native atomics");
static atomic_uint ps5_ai_lifecycle_word = ATOMIC_VAR_INIT(PS5_AI_LIFE_ENTRY);
static void native_lifecycle_mark(unsigned phase) {
    if (phase >= PS5_AI_LIFE_FAILED) return;
    unsigned before = atomic_load_explicit(&ps5_ai_lifecycle_word, memory_order_relaxed);
    for (;;) {
        if ((before & PS5_AI_LIFE_FAILURE_MASK) || phase < (before & UINT32_C(255))) return;
        unsigned after = (before & ~UINT32_C(255)) | phase;
        if (phase == PS5_AI_LIFE_READY) after |= PS5_AI_LIFE_READY_FLAG;
        if (atomic_compare_exchange_weak_explicit(&ps5_ai_lifecycle_word, &before, after,
                memory_order_release, memory_order_relaxed)) return;
    }
}
static void native_lifecycle_failed(unsigned kind) {
    if (!kind || (kind & ~PS5_AI_LIFE_FAILURE_MASK)) return;
    unsigned before = atomic_load_explicit(&ps5_ai_lifecycle_word, memory_order_relaxed);
    for (;;) {
        unsigned after = before | kind;
        if (!(before & PS5_AI_LIFE_FAILURE_MASK))
            after = (after & ~UINT32_C(0xff00)) | ((before & UINT32_C(255)) << 8);
        after = (after & ~UINT32_C(255)) |
            ((after & PS5_AI_LIFE_RETAINED_FLAG) ? PS5_AI_LIFE_RETAINED : PS5_AI_LIFE_FAILED);
        if (atomic_compare_exchange_weak_explicit(&ps5_ai_lifecycle_word, &before, after,
                memory_order_release, memory_order_relaxed)) return;
    }
}
void ps5_ai_native_lifecycle_snapshot(struct ps5_ai_native_lifecycle *out) {
    if (!out) return;
    unsigned word = atomic_load_explicit(&ps5_ai_lifecycle_word, memory_order_acquire);
    *out = (struct ps5_ai_native_lifecycle){sizeof(*out), 1, word & UINT32_C(255),
        (word >> 8) & UINT32_C(255), !!(word & PS5_AI_LIFE_READY_FLAG),
        !!(word & PS5_AI_LIFE_FAILURE_MASK), !!(word & PS5_AI_LIFE_RETAINED_FLAG), 1};
}
