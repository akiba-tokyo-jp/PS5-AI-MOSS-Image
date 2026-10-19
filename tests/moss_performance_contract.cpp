/* SPDX-License-Identifier: MIT -- real bounded performance helper, no model/GPU. */
#include "moss_trial_contract.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <thread>
#include <vector>

int main() {
    static_assert(!PS5_AI_MOSS_RUNTIME_ADMITTED);
    unsigned checks=0;
#if PS5_AI_MOSS_PERFORMANCE_TRIAL
    static_assert(PS5_AI_MOSS_CAN_RUN_ADMISSION);
    static_assert(!PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL && !PS5_AI_MOSS_1024_COLD_WARM_TRIAL);
    for(int size:{256,1024}) {
        assert(ps5_ai_moss_trial_request_valid(size,size,50,3,5,"euler",20261020));++checks;
    }
    std::atomic<unsigned> attempts{0};
    const auto claim=[&](int w,int h,int steps=50,double cfg=3,double shift=5,const char* sampler="euler",int64_t seed=20261020) {
        return ps5_ai_moss_performance_trial_claim(attempts,w,h,steps,cfg,shift,sampler,seed);
    };
    for(int size:{0,255,512,768,1023,1040}) {
        assert(!claim(size,size)&&attempts==0);++checks;
    }
    assert(!claim(256,1024)&&attempts==0);++checks;
    for(int steps:{4,49,51}) {assert(!claim(256,256,steps)&&attempts==0);++checks;}
    for(double cfg:{0.0,2.0,3.001,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        assert(!claim(256,256,50,cfg)&&attempts==0);++checks;
    }
    assert(!claim(256,256,50,3,4)&&attempts==0);++checks;
    assert(!claim(256,256,50,3,5,"euler_a")&&attempts==0);++checks;
    assert(!claim(256,256,50,3,5,nullptr)&&attempts==0);++checks;
    assert(!claim(256,256,50,3,5,"euler",20261019)&&attempts==0);++checks;
    assert(claim(256,256)&&attempts==1);++checks;
    /* A simulated failed generation still owns the first claim: no refund. */
    assert(claim(256,256)&&attempts==2);++checks;
    assert(claim(1024,1024)&&attempts==3);++checks;
    assert(claim(1024,1024)&&attempts==4);++checks;
    assert(!claim(256,256)&&attempts==4);++checks;
    assert(!claim(1024,1024)&&attempts==4);++checks;
    std::atomic<bool> legacy{false};
    assert(!ps5_ai_moss_trial_claim(legacy,256,256)&&!legacy);++checks;
    std::atomic<unsigned> old_benchmark{0};
    assert(!ps5_ai_moss_benchmark_trial_claim(old_benchmark,1024,1024)&&old_benchmark==0);++checks;
    std::atomic<unsigned> parallel{0},winners{0};std::vector<std::thread> threads;
    for(unsigned i=0;i<16;++i)threads.emplace_back([&]{
        if(ps5_ai_moss_performance_trial_claim(parallel,256,256,50,3,5,"euler",20261020))++winners;
    });
    for(auto& thread:threads)thread.join();
    assert(winners==4&&parallel==4);++checks;
#else
    static_assert(!PS5_AI_MOSS_CAN_RUN_ADMISSION);
    std::atomic<unsigned> attempts{0};
    assert(!ps5_ai_moss_performance_trial_claim(attempts,256,256,50,3,5,"euler",20261020)&&attempts==0);++checks;
    assert(!ps5_ai_moss_trial_request_valid(256,256,50,3,5,"euler",20261020));++checks;
#endif
    std::printf("PASS %u performance contract checks variant=%d GPU_calls=0 model_inference=0\n",checks,PS5_AI_MOSS_PERF_VARIANT);
}
