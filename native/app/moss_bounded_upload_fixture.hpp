/* SPDX-License-Identifier: MIT -- synthetic bytes, never learned inference. */
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace moss_bounded_upload_fixture {
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,
              "The bounded upload fixture requires IEEE754 binary32");
constexpr size_t tensor_bytes = (size_t(9) << 20) + 68;
constexpr size_t elements = tensor_bytes / sizeof(float);
constexpr uint64_t source_offset = UINT64_C(4294967424); // >32-bit offset; no file I/O.
constexpr size_t readback_elements = 65536;
constexpr uint64_t fnv_offset = UINT64_C(1469598103934665603);
constexpr uint64_t fnv_prime = UINT64_C(1099511628211);
static_assert(tensor_bytes % sizeof(float)==0 && tensor_bytes > (size_t(8)<<20),
              "Exercise a full8MiB fragment and a nonzero aligned tail");
inline int32_t numerator(uint64_t index) noexcept {
    return static_cast<int32_t>(((index%8191)*17+((index/4093)%8191)*3)%8191)-4095;
}
inline float input(uint64_t index) noexcept { return static_cast<float>(numerator(index))*0.0625f; }
inline float output(uint64_t index) noexcept { return static_cast<float>(numerator(index))*0.125f; }
inline bool read_exact(uint64_t absolute_offset, unsigned char* memory, size_t bytes) noexcept {
    if (!memory || !bytes || absolute_offset<source_offset) return false;
    const uint64_t relative=absolute_offset-source_offset;
    if (relative>tensor_bytes || bytes>tensor_bytes-relative || relative%sizeof(float) || bytes%sizeof(float)) return false;
    const uint64_t first=relative/sizeof(float);
    for(size_t i=0;i<bytes/sizeof(float);++i) {
        const float value=input(first+i);
        std::memcpy(memory+i*sizeof(float),&value,sizeof(value));
    }
    return true;
}
inline uint32_t bits(float value) noexcept { uint32_t result;std::memcpy(&result,&value,sizeof(result));return result; }
} // namespace moss_bounded_upload_fixture
