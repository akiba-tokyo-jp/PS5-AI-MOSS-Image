/* SPDX-License-Identifier: MIT
 * Grow-to-fit capped, byte-preserving staging for raw same-format weight uploads.
 * Design reference: Niko1221/Strata e88ee39d, src/core/weights.cpp (MIT).
 * No CUDA/HIP calls, host writes to device addresses, conversion or I/O retry.
 */
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>

namespace ps5_ai_weight_io {
constexpr size_t staging_bytes = size_t(8) << 20;
enum class Error { none, invalid_span, allocation, read, upload };
struct Result {
    bool success = false;
    Error error = Error::invalid_span;
    uint64_t bytes = 0;
    uint64_t chunks = 0;
    size_t staging_payload = 0;
};
class Scratch {
    std::unique_ptr<unsigned char[]> data_;
    size_t capacity_ = 0;
public:
    Scratch() = default;
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    size_t capacity() const noexcept { return capacity_; }
    unsigned char* acquire(size_t required) noexcept {
        if (!required || required > staging_bytes) return nullptr;
        if (required <= capacity_) return data_.get();
        // Free before growing: never hold old and replacement payload together.
        // Allocation failure leaves an empty, reusable Scratch; transfer_exact
        // does not retry this tensor or fall back to a different loading path.
        data_.reset(); capacity_ = 0;
        auto* replacement = new(std::nothrow) unsigned char[required];
        if (!replacement) return nullptr;
        data_.reset(replacement); capacity_ = required;
        return data_.get();
    }
};

// read_exact(absolute file offset, host destination, bytes) and
// upload_exact(relative tensor offset, host source, bytes) must be synchronous.
// The existing backend upload retains ownership/cache/fence handling. A failed
// prefix is never advertised as a successfully loaded tensor and is not retried.
template<class Read, class Upload>
Result transfer_exact(uint64_t source_offset, uint64_t total_bytes, Scratch& scratch,
                      Read&& read_exact, Upload&& upload_exact) {
    Result result;
    if (!total_bytes || source_offset > UINT64_MAX - total_bytes ||
        total_bytes > std::numeric_limits<size_t>::max()) return result;
    const size_t required = static_cast<size_t>(std::min<uint64_t>(total_bytes, staging_bytes));
    auto* memory = scratch.acquire(required);
    if (!memory) { result.error = Error::allocation; return result; }
    result.staging_payload = scratch.capacity();
    for (uint64_t done = 0; done < total_bytes;) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(result.staging_payload, total_bytes - done));
        if (!read_exact(source_offset + done, memory, n)) {
            result.error = Error::read; return result;
        }
        if (!upload_exact(static_cast<size_t>(done), memory, n)) {
            result.error = Error::upload; return result;
        }
        done += n; result.bytes = done; ++result.chunks;
    }
    result.success = true; result.error = Error::none; return result;
}
} // namespace ps5_ai_weight_io
