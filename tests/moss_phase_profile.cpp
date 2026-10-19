/* SPDX-License-Identifier: MIT -- counter scopes and strict precision parsing. */
#include "phase_profile.h"
#include "backend_precision_log.hpp"
#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>

static std::atomic<bool> enabled{true};
static std::array<std::atomic<unsigned>,PS5_AI_PROFILE_METRICS> calls{};
static thread_local unsigned runner=2;
extern "C" int ps5_ai_profile_enabled(void){return enabled.load();}
extern "C" void ps5_ai_profile_add(unsigned id,uint64_t,uint64_t,int valid){assert(id<PS5_AI_PROFILE_METRICS && valid);++calls[id];}
extern "C" unsigned ps5_ai_profile_set_runner(unsigned group){unsigned old=runner;runner=group;return old;}
extern "C" unsigned ps5_ai_profile_get_runner(void){return runner;}
int main(){
    assert(ps5_ai_backend_precision_token(nullptr)==-1);
    assert(ps5_ai_backend_precision_token("no field")==-1);
    assert(ps5_ai_backend_precision_token("ggml_vulkan: x | fp16: 0 | bf16: 0")==0);
    assert(ps5_ai_backend_precision_token("ggml_vulkan: x | fp16: 1 | bf16: 0")==1);
    assert(ps5_ai_backend_precision_token("ggml_vulkan: x | fp16: dot2 | bf16: 0")==1);
    for(const char* token:{"10","01","true","dot2bad","","2"}){
        std::string text="ggml_vulkan: x | fp16: ";text+=token;text+=" | bf16: 0";
        assert(ps5_ai_backend_precision_token(text.c_str())==-1);
    }
    assert(ps5_ai_profile_vulkan_metric(0)==37);
    {PS5AIProfileScope outer(4);
        assert(ps5_ai_profile_vulkan_metric(0)==32);
        {PS5AIProfileScope submit(ps5_ai_profile_vulkan_metric(0));}
        assert(runner==0 && calls[32]==1);
        {PS5AIProfileScope other(20);assert(ps5_ai_profile_vulkan_metric(1)==38);}
        assert(runner==0 && calls[20]==1);
        try{PS5AIProfileScope nested(20);throw 7;}catch(int){}
        assert(runner==0 && calls[20]==2);
        std::thread t([]{assert(runner==2);{PS5AIProfileScope scope(4);assert(runner==0);}assert(runner==2);});t.join();
        assert(runner==0);
    }
    assert(runner==2 && calls[4]==2);
    enabled=false;auto previous=calls[37].load();{PS5AIProfileScope inactive(37);}
    assert(calls[37]==previous);
    std::cout<<"PASS strict precision tokens, nested/exception/thread scope, inactive counters\n";
}
