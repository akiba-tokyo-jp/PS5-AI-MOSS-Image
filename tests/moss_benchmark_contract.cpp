/* SPDX-License-Identifier: MIT -- exact1024 two-attempt helper, no model/GPU. */
#include "moss_trial_contract.hpp"
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>
int main() {
    static_assert(PS5_AI_MOSS_1024_COLD_WARM_TRIAL && PS5_AI_MOSS_CAN_RUN_ADMISSION);
    static_assert(!PS5_AI_MOSS_RUNTIME_ADMITTED && !PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL);
    unsigned checks=0;
    assert(ps5_ai_moss_trial_request_valid(1024,1024,50,3,5,"euler",20261020));++checks;
    for(int width:{256,512,1008,1040}){assert(!ps5_ai_moss_trial_request_valid(width,1024,50,3,5,"euler",20261020));++checks;}
    for(int steps:{4,49,51}){assert(!ps5_ai_moss_trial_request_valid(1024,1024,steps,3,5,"euler",20261020));++checks;}
    assert(!ps5_ai_moss_trial_request_valid(1024,1024,50,3,5,"euler",20261019));++checks;
    assert(!ps5_ai_moss_trial_request_valid(1024,1024,50,2,5,"euler",20261020));++checks;
    assert(!ps5_ai_moss_trial_request_valid(1024,1024,50,3,4,"euler",20261020));++checks;
    assert(!ps5_ai_moss_trial_request_valid(1024,1024,50,3,5,"euler_a",20261020));++checks;
    assert(!ps5_ai_moss_trial_request_valid(1024,1024,50,3,5,nullptr,20261020));++checks;
    std::atomic<unsigned> count{0};
    assert(!ps5_ai_moss_benchmark_trial_claim(count,256,256)&&count==0);++checks;
    assert(ps5_ai_moss_benchmark_trial_claim(count,1024,1024)&&count==1);++checks;
    /* Simulated failure leaves count1; it cannot reset or refund a claim. */
    assert(ps5_ai_moss_benchmark_trial_claim(count,1024,1024)&&count==2);++checks;
    assert(!ps5_ai_moss_benchmark_trial_claim(count,1024,1024)&&count==2);++checks;
    std::atomic<bool> legacy{false};assert(!ps5_ai_moss_trial_claim(legacy,256,256)&&!legacy);++checks;
    std::atomic<unsigned> parallel{0},winners{0};std::vector<std::thread> threads;
    for(unsigned i=0;i<12;++i)threads.emplace_back([&]{if(ps5_ai_moss_benchmark_trial_claim(parallel,1024,1024))++winners;});
    for(auto&t:threads)t.join();assert(winners==2&&parallel==2);++checks;
    assert(!ps5_ai_moss_benchmark_trial_claim(parallel,1024,1024));++checks;
    std::printf("PASS %u benchmark contract checks;GPU_calls=0 model_inference=0\n",checks);
}
