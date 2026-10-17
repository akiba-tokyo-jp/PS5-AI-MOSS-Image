/* Copyright (C) 2024 John Törnblom

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3, or (at your option) any
later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; see the file COPYING. If not, see
<http://www.gnu.org/licenses/>.  */

#include "kernel.h"
#include "klog.h"
#include "payload.h"
#include "syscall.h"


/**
 * Patch syscall permissions of the currently running process.
 **/
static int
patch_syscall_permissions(void) {
  int pid = __crt_syscall(SYS_getpid);
  unsigned long kproc;
  unsigned long kaddr;
  unsigned long uaddr;

  if(!(kproc=kernel_get_proc(pid))) {
    return -1;
  }

  if(kernel_copyout(kproc + 0x3e8, &kaddr, sizeof(kaddr)) < 0) {
    return -1;
  }

  // set lowest address allowed to make syscalls
  uaddr = 0;
  if(kernel_copyin(&uaddr, kaddr + 0xf0, sizeof(uaddr)) < 0) {
    return -1;
  }

  // set highest address allowed to make syscalls
  uaddr = -1;
  if(kernel_copyin(&uaddr, kaddr + 0xf8, sizeof(uaddr)) < 0) {
    return -1;
  }

  return 0;
}


/* Preserve inherited credentials: only the unchanged SDK self syscall range remains. */
int __patch_init(void) { return patch_syscall_permissions(); }
const char ps5_ai_startup_mode[] = "credential-preserving-syscallrange-only";
const unsigned ps5_ai_credentials_preserved = 1;
