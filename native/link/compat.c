/*
 * Native title public-libc compatibility for exactly five unavailable imports.
 * Copyright (C) 2026 Mihawk (PS5_PayloadSDK platform/src/posix.c portions).
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Source identities and adaptations are bound by build.py/manifest.json.
 * readlink/link/symlink: selected newer PS5 platform semantics, renamed only.
 * pathconf: pinned cobanov PS5LM native compat_app.c semantics, unchanged.
 * mkstemp: newer platform's bounded random/O_EXCL algorithm, restricted to
 * suffix zero and mode0600 as pinned PS5LM and POSIX mkstemp require. The
 * installed platform random helper is reused; no new random/runtime provider.
 * No Direct/GPU API, raw syscall, constructor, logging or process operation.
 */
#define _BSD_SOURCE 1
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <ps5platform/libc.h>

_Static_assert(sizeof(size_t) == 8 && sizeof(ssize_t) == 8 && sizeof(long) == 8,
               "Native public 64-bit libc ABI");

ssize_t readlink(const char *restrict path, char *restrict buffer, size_t size)
{
    (void)buffer;
    (void)size;
    struct stat status;
    if (lstat(path, &status) != 0)
        return -1;
    errno = EINVAL;
    return -1;
}

int link(const char *existing, const char *name)
{
    (void)existing;
    (void)name;
    errno = EOPNOTSUPP;
    return -1;
}

int symlink(const char *target, const char *name)
{
    (void)target;
    (void)name;
    errno = EPERM;
    return -1;
}

long pathconf(const char *path, int name)
{
    (void)path;
    (void)name;
    errno = EINVAL;
    return -1;
}

int mkstemp(char *path_template)
{
    static const char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const size_t length = path_template ? strlen(path_template) : 0;
    if (length < 6) {
        errno = EINVAL;
        return -1;
    }
    char *const x = path_template + length - 6;
    if (memcmp(x, "XXXXXX", 6) != 0) {
        errno = EINVAL;
        return -1;
    }
    for (int attempt = 0; attempt < 1000; attempt++) {
        for (int i = 0; i < 6; i++)
            x[i] = alphabet[ps5_arc4random_uniform(sizeof(alphabet) - 1)];
        const int fd = open(path_template, O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd >= 0 || errno != EEXIST)
            return fd;
    }
    memcpy(x, "XXXXXX", 6);
    errno = EEXIST;
    return -1;
}
