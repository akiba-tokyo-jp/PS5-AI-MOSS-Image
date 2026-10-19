/* SPDX-License-Identifier: MIT -- actual state/policy, no GPU/PS5 operations. */
#include <assert.h>
#include <stdatomic.h>
#include <stdio.h>
#include "lifecycle.h"
#include "lifecycle_impl.inc"
static unsigned checks;
static void check(int value) { ++checks; assert(value); }
int main(void) {
    struct ps5_ai_native_lifecycle s;
    for(unsigned boundary=PS5_AI_LIFE_ENTRY;boundary<PS5_AI_LIFE_READY;++boundary) {
        atomic_store(&ps5_ai_lifecycle_word,PS5_AI_LIFE_ENTRY);
        native_lifecycle_mark(boundary);
        ps5_ai_native_lifecycle_snapshot(&s);
        check(!ps5_ai_lifecycle_checked_shutdown_allowed(&s));
        check(s.constructor_effects_unknown==1 && s.phase==boundary);
        native_lifecycle_failed(PS5_AI_LIFE_INIT_FAILED);
        ps5_ai_native_lifecycle_snapshot(&s);
        check(!ps5_ai_lifecycle_checked_shutdown_allowed(&s) && s.failed_phase==boundary);
        native_lifecycle_mark(PS5_AI_LIFE_READY);
        native_lifecycle_failed(PS5_AI_LIFE_RETAINED_FLAG);
        ps5_ai_native_lifecycle_snapshot(&s);
        check(s.failed_phase==boundary && s.fault_retention_entered && !s.engine_ready);
    }
    atomic_store(&ps5_ai_lifecycle_word,PS5_AI_LIFE_ENTRY);
    for(unsigned stage=0;stage<=PS5_AI_LIFE_READY;stage++)native_lifecycle_mark(stage);
    ps5_ai_native_lifecycle_snapshot(&s);
    check(ps5_ai_lifecycle_checked_shutdown_allowed(&s));
    native_lifecycle_mark(PS5_AI_LIFE_CONFIG);
    ps5_ai_native_lifecycle_snapshot(&s);
    check(s.phase==PS5_AI_LIFE_READY);
    native_lifecycle_failed(PS5_AI_LIFE_SERVER_FAILED);
    ps5_ai_native_lifecycle_snapshot(&s);
    check(s.failed_phase==PS5_AI_LIFE_READY && !ps5_ai_lifecycle_checked_shutdown_allowed(&s));
    for(unsigned kind=PS5_AI_LIFE_INIT_FAILED;kind<=PS5_AI_LIFE_BOOTSTRAP_FAILED;kind<<=1) {
        atomic_store(&ps5_ai_lifecycle_word,PS5_AI_LIFE_BOOTSTRAP);
        native_lifecycle_failed(kind);ps5_ai_native_lifecycle_snapshot(&s);
        check(s.failure_observed && s.failed_phase==PS5_AI_LIFE_BOOTSTRAP);
    }
    atomic_store(&ps5_ai_lifecycle_word,PS5_AI_LIFE_READY|PS5_AI_LIFE_READY_FLAG);
    ps5_ai_native_lifecycle_snapshot(&s);s.abi_version=0;
    check(!ps5_ai_lifecycle_checked_shutdown_allowed(&s));
    ps5_ai_native_lifecycle_snapshot(&s);s.abi_bytes=0;
    check(!ps5_ai_lifecycle_checked_shutdown_allowed(&s));
    ps5_ai_native_lifecycle_snapshot(&s);s.constructor_effects_unknown=0;
    check(!ps5_ai_lifecycle_checked_shutdown_allowed(&s));
    check(!ps5_ai_lifecycle_checked_shutdown_allowed(NULL));
    ps5_ai_native_lifecycle_snapshot(NULL);
    printf("PASS %u lifecycle rejection/normal-path/first-failure checks; PS5/GPU/OS-close calls=0\n",checks);
    return 0;
}
