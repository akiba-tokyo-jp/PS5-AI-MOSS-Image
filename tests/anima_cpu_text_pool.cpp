// SPDX-License-Identifier: GPL-3.0-or-later
// Exercise the actual borrowed-buffer helper with mock pool/CPU descriptors.
// No model payload, Direct memory, Vulkan or PS5 connection is allocated here.
#include "cpu_clip_pool_buffer.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static uint64_t maximum = UINT64_C(512) << 20;
static size_t allocations, descriptors, frees, last_bytes;
static int allocation_failure, descriptor_failure, free_failure;
alignas(64) static unsigned char claim_token[64];
struct Event { std::string action; uint64_t bytes; int success; };
static std::vector<Event> events;
static unsigned checks;
static void check(bool condition, const char* message) {
    ++checks;
    if (!condition) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
static void reset() {
    maximum=UINT64_C(512)<<20;allocations=descriptors=frees=last_bytes=0;
    allocation_failure=descriptor_failure=free_failure=0;events.clear();
}
extern "C" uint64_t ps5_ai_direct_pool_max_claim_bytes(void) { return maximum; }
extern "C" void* ps5_ai_flex_alloc(uint64_t bytes, uint64_t alignment) {
    ++allocations;last_bytes=static_cast<size_t>(bytes);
    check(alignment==64, "borrow claim preserves alignment");
    return allocation_failure ? nullptr : claim_token;
}
extern "C" int ps5_ai_flex_free(void* pointer, uint64_t bytes) {
    ++frees;check(pointer==claim_token && bytes==last_bytes, "return exact owned claim only");
    return free_failure ? -1 : 0;
}
extern "C" void ps5_ai_cpu_clip_pool_event(const char* action, uint64_t bytes, int success) noexcept {
    events.push_back({action,bytes,success});
}
extern "C" ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void* pointer, size_t size) {
    ++descriptors;check(pointer==claim_token && size==last_bytes, "CPU descriptor aliases exact claim");
    if (descriptor_failure==1) return nullptr;
    if (descriptor_failure==2) throw std::runtime_error("mock descriptor failure");
    auto* buffer=new ggml_backend_buffer{};buffer->context=pointer;buffer->size=size;return buffer;
}
static void release(ggml_backend_buffer_t buffer) {
    check(buffer && buffer->iface.free_buffer, "claim-return callback installed");
    buffer->iface.free_buffer(buffer);delete buffer;
}
int main() {
    constexpr size_t qwen_embedding=311164928;
    reset();check(!ps5_ai_cpu_clip_pool_buffer(0), "zero rejected");
    check(!ps5_ai_cpu_clip_pool_buffer((size_t(512)<<20)+1), "over actual 512 MiB rejected");
    check(!allocations && !descriptors && !frees, "rejected claims perform no allocations");
    maximum=0;check(!ps5_ai_cpu_clip_pool_buffer(qwen_embedding) && !allocations, "unavailable pool rejected");
    maximum=UINT64_C(256)<<20;
    check(!ps5_ai_cpu_clip_pool_buffer(qwen_embedding) && !allocations, "smaller actual region remains a hard bound");
    reset();auto* embedding=ps5_ai_cpu_clip_pool_buffer(qwen_embedding);
    check(embedding && allocations==1 && descriptors==1 && !frees, "296.75 MiB Qwen embedding admitted once");
    check(events.size()==1 && events[0].action=="allocated" && events[0].bytes==qwen_embedding && events[0].success, "successful loan accounted");
    release(embedding);
    check(frees==1 && events.size()==2 && events.back().action=="free" && events.back().success, "successful loan returned once");
    reset();auto* boundary=ps5_ai_cpu_clip_pool_buffer(size_t(512)<<20);
    check(boundary && allocations==1, "actual 512 MiB boundary admitted");release(boundary);
    reset();allocation_failure=1;
    check(!ps5_ai_cpu_clip_pool_buffer(qwen_embedding), "failed pool allocation propagated");
    check(allocations==1 && !descriptors && !frees && events.size()==1 && events[0].action=="allocation_denied", "pool failure has no retry or hidden fallback");
    reset();descriptor_failure=1;
    check(!ps5_ai_cpu_clip_pool_buffer(qwen_embedding), "null descriptor propagated");
    check(allocations==1 && descriptors==1 && frees==1 && events.size()==1 && events[0].action=="descriptor_failure_cleanup" && events[0].success, "null descriptor returns loan exactly once");
    reset();descriptor_failure=2;bool threw=false;
    try { (void)ps5_ai_cpu_clip_pool_buffer(qwen_embedding); } catch (const std::runtime_error&) { threw=true; }
    check(threw && allocations==1 && descriptors==1 && frees==1 && events.size()==1 && events[0].action=="descriptor_failure_cleanup", "throwing descriptor returns loan and propagates");
    reset();descriptor_failure=1;free_failure=1;
    check(!ps5_ai_cpu_clip_pool_buffer(qwen_embedding), "descriptor failure with retained loan returns null");
    check(frees==1 && events.size()==1 && !events[0].success, "failed return is not reported as released");
    reset();auto* retained=ps5_ai_cpu_clip_pool_buffer(qwen_embedding);free_failure=1;release(retained);
    check(frees==1 && events.size()==2 && events.back().action=="free" && !events.back().success, "normal failed return preserves failure accounting");
    std::printf("PASS %u Anima CPU text pool boundary/ownership checks; model inference=0, PS5 calls=0\n", checks);
}
