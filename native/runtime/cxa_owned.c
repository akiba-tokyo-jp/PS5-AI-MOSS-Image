/* SPDX-License-Identifier: GPL-3.0-or-later
 * Adapted from PS5_PayloadSDK b83202be platform/src/cxa.c,
 * Copyright (C) 2026 Mihawk. Local addition exposes an explicit TLS cleanup
 * before ordinary pthread return, as newer upstream SDK threads require.
 * Uses only the payload's standard CPU libc and pthread APIs.
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>

struct thread_destructor {
    void (*destructor)(void *);
    void *object;
    struct thread_destructor *next;
};
static pthread_key_t destructors_key;
static pthread_once_t destructors_once = PTHREAD_ONCE_INIT;
static atomic_bool destructors_ready;

static void run_destructors(void *list) {
    struct thread_destructor *entry = list;
    while (entry) {
        struct thread_destructor *next = entry->next;
        entry->destructor(entry->object);
        free(entry);
        struct thread_destructor *added = pthread_getspecific(destructors_key);
        if (added) {
            pthread_setspecific(destructors_key, NULL);
            run_destructors(added);
        }
        entry = next;
    }
}

void ps5p_run_thread_destructors(void) {
    if (!atomic_load_explicit(&destructors_ready, memory_order_acquire)) return;
    void *list = pthread_getspecific(destructors_key);
    pthread_setspecific(destructors_key, NULL);
    run_destructors(list);
}

static void setup_destructors(void) {
    bool ready = pthread_key_create(&destructors_key, run_destructors) == 0 &&
                 atexit(ps5p_run_thread_destructors) == 0;
    atomic_store_explicit(&destructors_ready, ready, memory_order_release);
}

int ps5___cxa_thread_atexit_impl(void (*destructor)(void *), void *object, void *dso) {
    (void)dso;
    pthread_once(&destructors_once, setup_destructors);
    if (!atomic_load_explicit(&destructors_ready, memory_order_acquire)) return -1;
    struct thread_destructor *entry = malloc(sizeof(*entry));
    if (!entry) return -1;
    entry->destructor = destructor;
    entry->object = object;
    entry->next = pthread_getspecific(destructors_key);
    if (pthread_setspecific(destructors_key, entry) != 0) { free(entry); return -1; }
    return 0;
}
