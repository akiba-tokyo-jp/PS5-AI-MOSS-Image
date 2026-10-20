/* SPDX-License-Identifier: MIT -- actual header/atomic contract; no inference. */
#include "moss_trial_contract.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <thread>
#include <vector>
int main(){
    static_assert(!PS5_AI_MOSS_RUNTIME_ADMITTED);unsigned checks=0;
    std::atomic<unsigned> attempts{0};
#if PS5_AI_MOSS_PERFORMANCE_TRIAL
    constexpr int steps=PS5_AI_MOSS_DEFAULT_REQUEST_STEPS;
    constexpr double cfg=PS5_AI_MOSS_DEFAULT_REQUEST_CFG;
    const char *sampler=PS5_AI_MOSS_DEFAULT_REQUEST_SAMPLER;
    auto claim=[&](int w,int h,int n,double c,double f,const char *s,int64_t seed){
        return ps5_ai_moss_performance_trial_claim(attempts,w,h,n,c,f,s,seed);
    };
    for(int size:{256,1024}){assert(ps5_ai_moss_trial_request_valid(size,size,steps,cfg,5,sampler,20261020));++checks;}
    for(int size:{0,255,512,768,1040}){assert(!claim(size,size,steps,cfg,5,sampler,20261020)&&attempts==0);++checks;}
    assert(!claim(256,1024,steps,cfg,5,sampler,20261020)&&attempts==0);++checks;
    assert(!claim(256,256,steps+1,cfg,5,sampler,20261020)&&attempts==0);++checks;
    for(double c:{cfg+1,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}){
        assert(!claim(256,256,steps,c,5,sampler,20261020)&&attempts==0);++checks;
    }
    assert(!claim(256,256,steps,cfg,4,sampler,20261020)&&attempts==0);++checks;
    assert(!claim(256,256,steps,cfg,5,nullptr,20261020)&&attempts==0);++checks;
    assert(!claim(256,256,steps,cfg,5,sampler,20261019)&&attempts==0);++checks;
    assert(!claim(256,256,steps,cfg,5,PS5_AI_MOSS_SAMPLER_TRIAL?"euler":"res_multistep",20261020)&&attempts==0);++checks;
    assert(!claim(256,256,PS5_AI_MOSS_SAMPLER_TRIAL?50:30,PS5_AI_MOSS_SAMPLER_TRIAL?3:4,5,
                  PS5_AI_MOSS_SAMPLER_TRIAL?"euler":"res_multistep",20261020)&&attempts==0);++checks;
    for(unsigned n=0;n<4;++n){assert(claim(n<2?256:1024,n<2?256:1024,steps,cfg,5,sampler,20261020)&&attempts==n+1);++checks;}
    // The first claimed attempt simulates a failure: claims are never refunded.
    assert(!claim(256,256,steps,cfg,5,sampler,20261020)&&attempts==4);++checks;
    std::atomic<unsigned> parallel{0},winners{0};std::vector<std::thread> threads;
    for(unsigned i=0;i<16;++i)threads.emplace_back([&]{
        if(ps5_ai_moss_performance_trial_claim(parallel,256,256,steps,cfg,5,sampler,20261020))++winners;
    });
    for(auto& thread:threads)thread.join();assert(parallel==4&&winners==4);++checks;
#else
    assert(!ps5_ai_moss_trial_request_valid(256,256,30,4,5,"res_multistep",20261020));++checks;
    assert(!ps5_ai_moss_performance_trial_claim(attempts,256,256,30,4,5,"res_multistep",20261020)&&attempts==0);++checks;
#endif
    std::printf("PASS %u sampler contract checks trial=%d GPU_calls=0 model_inference=0\n",checks,PS5_AI_MOSS_SAMPLER_TRIAL);
}
