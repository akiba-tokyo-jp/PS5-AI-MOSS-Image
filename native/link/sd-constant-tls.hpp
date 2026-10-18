// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

// PS5's native ELF loader does not resolve the optional dynamic TLS init hooks.
// These two upstream definitions have constant nullptr/false initializers.
namespace sd {
class ParallelExecutor;
namespace parallel_detail {
extern constinit thread_local ParallelExecutor* executor;
extern constinit thread_local bool active;
}
}
