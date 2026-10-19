/* Synthetic host tests of the exact staged tracer, no PS5/files/tokens. */
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#define PS5_AI_INIT_STAGE_HOST_TEST 1
#include "init_stage_impl.hpp"
static std::atomic<uint64_t> synthetic_ms{1000};
static std::atomic<bool> synthetic_clock_valid{true};
namespace ps5_ai_init_trace {
uint64_t test_clock_ms(bool *valid) noexcept {errno=EINTR;*valid=synthetic_clock_valid.load();return synthetic_ms.load();}
}
static std::mutex gate;
static std::condition_variable changed;
static unsigned logger_calls,logger_bytes,max_record_bytes;
static bool block[40],entered[40],released[40],operation_entered,operation_released;
extern "C" void ps5_ai_ggml_report(const char *format,...) {
    char text[1024];va_list ap;va_start(ap,format);int n=std::vsnprintf(text,sizeof(text),format,ap);va_end(ap);
    assert(n>0&&static_cast<size_t>(n)<sizeof(text));
    assert(!std::strncmp(text,"INIT_STAGE ",11));
    std::unique_lock<std::mutex> lock(gate);
    unsigned call=++logger_calls;logger_bytes+=unsigned(n)+1;
    if(unsigned(n)+1>max_record_bytes)max_record_bytes=unsigned(n)+1;
    assert(call<40);entered[call]=true;changed.notify_all();
    changed.wait(lock,[&]{return !block[call]||released[call];});
    errno=EIO;
}
static void reset(void) {
    ps5_ai_init_trace::current.store(0);ps5_ai_init_trace::reserved.store(0);
    ps5_ai_init_trace::completed.store(0);ps5_ai_init_trace::suppressed.store(0);
    std::lock_guard<std::mutex> lock(gate);logger_calls=logger_bytes=max_record_bytes=0;
    std::memset(block,0,sizeof(block));std::memset(entered,0,sizeof(entered));std::memset(released,0,sizeof(released));
    operation_entered=operation_released=false;synthetic_ms=1000;synthetic_clock_valid=true;
}
static void wait_for(bool& value) {
    std::unique_lock<std::mutex> lock(gate);
    assert(changed.wait_for(lock,std::chrono::seconds(2),[&]{return value;}));
}
static void release_log(unsigned i){std::lock_guard<std::mutex> lock(gate);released[i]=true;changed.notify_all();}
static struct ps5_ai_init_stage_snapshot observe(void) {
    /* A logger-held lock must never delay the exact health formatter/snapshot. */
    auto result=std::async(std::launch::async,[]{
        char json[PS5_AI_INIT_JSON_BYTES];struct ps5_ai_init_stage_snapshot s={};
        errno=E2BIG;assert(ps5_ai_init_stage_json(json,sizeof(json)));assert(errno==E2BIG);
        assert(std::strstr(json,"\"snapshot_lock_free\":true"));
        ps5_ai_init_stage_snapshot(&s);assert(errno==E2BIG);return s;
    });
    if(result.wait_for(std::chrono::seconds(1))!=std::future_status::ready){std::fputs("snapshot blocked behind synthetic logger\n",stderr);std::_Exit(3);}
    return result.get();
}
int main(void) {
    unsigned cases=0;
    reset();auto s=observe();assert(s.stage==PS5_AI_INIT_NOT_STARTED&&!s.clock_valid&&!s.journal_records_reserved);++cases;
    reset();block[1]=block[2]=true;int returned=0;
    std::thread worker([&]{returned=ps5_ai_init_stage_call(PS5_AI_INIT_VAE_FLASH,[&]{
        std::unique_lock<std::mutex> lock(gate);operation_entered=true;changed.notify_all();
        changed.wait(lock,[]{return operation_released;});return 23;
    });});
    wait_for(entered[1]);s=observe();assert(s.stage==PS5_AI_INIT_VAE_FLASH&&s.state==PS5_AI_INIT_LOG_PENDING&&s.journal_records_reserved==1&&!s.journal_calls_completed);++cases;
    release_log(1);wait_for(operation_entered);s=observe();assert(s.state==PS5_AI_INIT_OPERATION_STARTED&&s.journal_calls_completed==1);++cases;
    {std::lock_guard<std::mutex> lock(gate);operation_released=true;changed.notify_all();}
    wait_for(entered[2]);s=observe();assert(s.state==PS5_AI_INIT_OPERATION_RETURNED&&s.journal_records_reserved==2&&s.journal_calls_completed==1);++cases;
    release_log(2);worker.join();assert(returned==23&&logger_calls==2);++cases;
    reset();try{ps5_ai_init_stage_call(PS5_AI_INIT_QWEN_HEADER,[]{throw std::runtime_error("synthetic exception");});assert(false);}catch(const std::runtime_error& e){assert(!std::strcmp(e.what(),"synthetic exception"));}
    s=observe();assert(s.state==PS5_AI_INIT_OPERATION_EXCEPTION&&logger_calls==2);++cases;
    reset();int operation_calls=0;assert(!ps5_ai_init_stage_call(PS5_AI_INIT_HEADERS,[&]{++operation_calls;return false;}));
    ps5_ai_init_stage_call(PS5_AI_INIT_FILE_IO,[&]{++operation_calls;});assert(operation_calls==2&&logger_calls==4);++cases;
    reset();for(unsigned i=0;i<32;i++){ps5_ai_init_stage_enter(PS5_AI_INIT_BACKENDS);ps5_ai_init_stage_returned(PS5_AI_INIT_BACKENDS,0);}
    s=observe();assert(logger_calls==32&&s.journal_records_reserved==32&&s.journal_calls_completed==32&&s.journal_records_suppressed==32);
    assert(max_record_bytes<=384&&logger_bytes<=32*384&&s.stage==PS5_AI_INIT_BACKENDS&&s.state==PS5_AI_INIT_OPERATION_RETURNED);++cases;
    ps5_ai_init_stage_enter(PS5_AI_INIT_DIT_HEADER);s=observe();assert(logger_calls==32&&s.stage==PS5_AI_INIT_DIT_HEADER&&s.state==PS5_AI_INIT_OPERATION_STARTED&&s.journal_records_suppressed==33);++cases;
    auto before=s;ps5_ai_init_stage_enter(0);ps5_ai_init_stage_enter(PS5_AI_INIT_STAGE_COUNT);ps5_ai_init_stage_returned(UINT32_MAX,0);s=observe();assert(s.stage==before.stage&&s.state==before.state&&s.journal_records_suppressed==before.journal_records_suppressed);++cases;
    reset();synthetic_clock_valid=false;ps5_ai_init_stage_enter(PS5_AI_INIT_CONTEXT);char json[PS5_AI_INIT_JSON_BYTES];assert(ps5_ai_init_stage_json(json,sizeof(json)));s=observe();assert(!s.clock_valid&&!s.state_changed_ms&&!std::strstr(json,"state_age_ms"));++cases;
    reset();synthetic_ms=UINT64_MAX;ps5_ai_init_stage_enter(PS5_AI_INIT_CONTEXT);s=observe();assert(!s.clock_valid&&!s.state_changed_ms);++cases;
    reset();ps5_ai_init_stage_enter(PS5_AI_INIT_CONTEXT);synthetic_ms=900;assert(ps5_ai_init_stage_json(json,sizeof(json))&&!std::strstr(json,"state_age_ms"));++cases;
    synthetic_ms=1100;assert(ps5_ai_init_stage_json(json,sizeof(json))&&std::strstr(json,"\"state_age_ms\":100"));++cases;
    errno=E2BIG;char tiny[4]={'x','x','x','x'};assert(!ps5_ai_init_stage_json(tiny,sizeof(tiny))&&!tiny[0]&&errno==E2BIG);assert(!ps5_ai_init_stage_json(nullptr,0));ps5_ai_init_stage_snapshot(nullptr);++cases;
    std::printf("PASS %u synthetic host cases; logger/operation blocks isolated; journal max32x384; no PS5/files/tokens\n",cases);
}
