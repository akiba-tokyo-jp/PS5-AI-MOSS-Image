/* SPDX-License-Identifier: MIT. Included once by engine_sdcpp.cpp.
 * One lock-free word publishes stage+state+clock together. No reader retry or
 * lock, and no odd seqlock generation remains held while an I/O call blocks.
 */
#ifndef PS5_AI_INIT_STAGE_IMPL_HPP
#define PS5_AI_INIT_STAGE_IMPL_HPP
#include "init_stage.h"
#include "probe.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cerrno>
#include <limits>
static_assert(std::atomic<uint64_t>::is_always_lock_free,"Init snapshot must be lock-free");
static_assert(std::atomic<uint32_t>::is_always_lock_free,"Init counters must be lock-free");
namespace ps5_ai_init_trace {
static std::atomic<uint64_t> current{0};
static std::atomic<uint32_t> reserved{0},completed{0},suppressed{0};
static constexpr uint64_t max_ms=UINT64_MAX>>12;
static const char* const names[]={"not_started","generation_preamble","pre_context_event","new_sd_ctx",
    "model_headers","DiT_header","Qwen_text_header","VAE_header","backend_init","file_io_prepare",
    "core_runner_build","VAE_runner_build","runner_validation","TE_flash_probe","VAE_flash_probe",
    "DiT_flash_selection","denoiser_build"};
static const char* const states[]={"not_started","log_pending","operation_started","operation_returned","operation_exception"};
static_assert(sizeof(names)/sizeof(names[0])==PS5_AI_INIT_STAGE_COUNT,"Fixed stage table");
static uint64_t clock_ms(bool *valid) noexcept {
#ifdef PS5_AI_INIT_STAGE_HOST_TEST
    extern uint64_t test_clock_ms(bool *) noexcept;
    return test_clock_ms(valid);
#else
    const auto value=std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    *valid=value>=0&&static_cast<uint64_t>(value)<=max_ms;
    return *valid?static_cast<uint64_t>(value):0;
#endif
}
static void publish(unsigned stage,unsigned state) noexcept {
    bool valid=false;uint64_t ms=clock_ms(&valid);
    valid=valid&&ms<=max_ms;if(!valid)ms=0;
    current.store((ms<<12)|(uint64_t(valid)<<11)|(uint64_t(state)<<8)|stage,std::memory_order_release);
}
static bool reserve_record(void) noexcept {
    uint32_t count=reserved.load(std::memory_order_relaxed);
    while(count<PS5_AI_INIT_MAX_RECORDS){
        if(reserved.compare_exchange_weak(count,count+1,std::memory_order_relaxed))return true;
    }
    uint32_t n=suppressed.load(std::memory_order_relaxed);
    while(n<UINT32_MAX&&!suppressed.compare_exchange_weak(n,n+1,std::memory_order_relaxed)){}
    return false;
}
static void log_record(unsigned stage,const char *boundary) noexcept {
    if(!reserve_record())return;
    const uint64_t packed=current.load(std::memory_order_acquire);
    char line[PS5_AI_INIT_RECORD_BYTES];
    const int n=std::snprintf(line,sizeof(line),
        "INIT_STAGE id=%u stage=%s boundary=%s state=%s changed_ms=%llu clock_valid=%u model_or_GPU_completion_claimed=0",
        stage,names[stage],boundary,states[(packed>>8)&7u],
        static_cast<unsigned long long>(packed>>12),unsigned((packed>>11)&1u));
    /* report adds one LF; every accepted record is at most384 bytes including LF. */
    if(n>0&&static_cast<size_t>(n)<sizeof(line)-1){
        ps5_ai_ggml_report("%s",line);
        completed.fetch_add(1,std::memory_order_relaxed);
    }
}
}
extern "C" void ps5_ai_init_stage_enter(unsigned stage) noexcept {
    if(!stage||stage>=PS5_AI_INIT_STAGE_COUNT)return;
    const int saved=errno;
    ps5_ai_init_trace::publish(stage,PS5_AI_INIT_LOG_PENDING);
    ps5_ai_init_trace::log_record(stage,"before");
    ps5_ai_init_trace::publish(stage,PS5_AI_INIT_OPERATION_STARTED);
    errno=saved;
}
extern "C" void ps5_ai_init_stage_returned(unsigned stage,int exception) noexcept {
    if(!stage||stage>=PS5_AI_INIT_STAGE_COUNT)return;
    const int saved=errno;
    ps5_ai_init_trace::publish(stage,exception?PS5_AI_INIT_OPERATION_EXCEPTION:PS5_AI_INIT_OPERATION_RETURNED);
    ps5_ai_init_trace::log_record(stage,exception?"exception":"return");
    errno=saved;
}
extern "C" void ps5_ai_init_stage_snapshot(struct ps5_ai_init_stage_snapshot *out) noexcept {
    if(!out)return;const int saved=errno;
    const uint64_t packed=ps5_ai_init_trace::current.load(std::memory_order_acquire);
    const uint32_t done=ps5_ai_init_trace::completed.load(std::memory_order_relaxed);
    const uint32_t count=ps5_ai_init_trace::reserved.load(std::memory_order_relaxed);
    *out={packed>>12,static_cast<uint32_t>(packed&255u),static_cast<uint32_t>((packed>>8)&7u),
        static_cast<uint32_t>((packed>>11)&1u),count,done,ps5_ai_init_trace::suppressed.load(std::memory_order_relaxed)};
    errno=saved;
}
extern "C" int ps5_ai_init_stage_json(char *out,size_t bytes) noexcept {
    if(!out||!bytes)return 0;const int saved=errno;out[0]=0;
    struct ps5_ai_init_stage_snapshot s={};ps5_ai_init_stage_snapshot(&s);
    bool clock_valid=false;const uint64_t now=ps5_ai_init_trace::clock_ms(&clock_valid);
    char age[96]={0};
    if(s.clock_valid&&clock_valid&&now>=s.state_changed_ms)
        std::snprintf(age,sizeof(age),",\"state_age_ms\":%llu",static_cast<unsigned long long>(now-s.state_changed_ms));
    const int n=std::snprintf(out,bytes,
        "{\"stage\":\"%s\",\"state\":\"%s\"%s,\"journal_records_reserved\":%u,\"journal_calls_completed\":%u,\"journal_records_suppressed\":%u,\"journal_record_limit\":32,\"journal_record_max_bytes\":384,\"snapshot_lock_free\":true,\"model_or_GPU_completion_claimed\":false}",
        ps5_ai_init_trace::names[s.stage],ps5_ai_init_trace::states[s.state],age,
        s.journal_records_reserved,s.journal_calls_completed,s.journal_records_suppressed);
    const bool ok=n>=0&&static_cast<size_t>(n)<bytes;if(!ok)out[0]=0;errno=saved;return ok?1:0;
}
#endif
