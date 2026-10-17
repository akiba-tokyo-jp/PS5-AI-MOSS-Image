/* SPDX-License-Identifier: MIT -- mocks only, never linked into native title. */
#ifndef PS5_AI_DIRECT_POOL_HOST_SHIM_H
#define PS5_AI_DIRECT_POOL_HOST_SHIM_H
#define write host_write
#define fsync host_fsync
ssize_t host_write(int, const void *, size_t);
int host_fsync(int);
int host_open_journal(void);
#endif
