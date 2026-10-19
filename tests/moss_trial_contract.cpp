/* SPDX-License-Identifier: MIT -- actual request-claim helper, no GPU/model. */
#include "moss_trial_contract.hpp"
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>
int main() {
    std::atomic<bool> claim{false};unsigned checks=0;
#if PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL
    static_assert(!PS5_AI_MOSS_RUNTIME_ADMITTED,"trial must not assert verified production");
    static_assert(PS5_AI_MOSS_CAN_RUN_ADMISSION,"explicit trial runs actual admission");
    const int valid_steps=4;
    assert(ps5_ai_moss_trial_request_valid(256,256,valid_steps,3.0,5.0,"euler",20261019));++checks;
    assert(!ps5_ai_moss_trial_request_valid(512,256,valid_steps,3.0,5.0,"euler"));++checks;
    assert(ps5_ai_moss_trial_request_valid(256,256,50,3.0,5.0,"euler")==bool(PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL));++checks;
    assert(!ps5_ai_moss_trial_request_valid(256,256,49,3.0,5.0,"euler"));++checks;
    assert(!ps5_ai_moss_trial_request_valid(256,256,51,3.0,5.0,"euler"));++checks;
#if PS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL
    assert(!ps5_ai_moss_trial_request_valid(256,256,50,3.0,5.0,"euler",20261018));++checks;
    assert(!ps5_ai_moss_trial_request_valid(256,256,4,3.0,5.0,"euler",20261018));++checks;
#endif
    assert(!ps5_ai_moss_trial_request_valid(256,256,valid_steps,1.0,5.0,"euler"));++checks;
    assert(!ps5_ai_moss_trial_request_valid(256,256,valid_steps,3.0,2.0,"euler"));++checks;
    assert(!ps5_ai_moss_trial_request_valid(256,256,valid_steps,3.0,5.0,"euler_a"));++checks;
    assert(!ps5_ai_moss_trial_request_valid(256,256,valid_steps,3.0,5.0,nullptr));++checks;
    assert(!ps5_ai_moss_trial_claim(claim,512,512)&&!claim.load());++checks;
    assert(!ps5_ai_moss_trial_claim(claim,256,1024)&&!claim.load());++checks;
    assert(ps5_ai_moss_trial_claim(claim,256,256)&&claim.load());++checks;
    assert(!ps5_ai_moss_trial_claim(claim,256,256)&&claim.load());++checks;
    /* A failed caller never resets the exact production helper's atomic. */
    assert(!ps5_ai_moss_trial_claim(claim,256,256));++checks;
    std::atomic<bool> parallel{false};std::atomic<unsigned> winners{0};std::vector<std::thread> threads;
    for(unsigned i=0;i<12;i++)threads.emplace_back([&]{if(ps5_ai_moss_trial_claim(parallel,256,256))winners.fetch_add(1);});
    for(auto& t:threads)t.join();assert(winners==1);++checks;
#else
    static_assert(!PS5_AI_MOSS_RUNTIME_ADMITTED&&!PS5_AI_MOSS_CAN_RUN_ADMISSION,"preparation remains closed");
    assert(!ps5_ai_moss_trial_request_valid(256,256,4,3.0,5.0,"euler"));++checks;
    assert(!ps5_ai_moss_trial_claim(claim,256,256)&&!claim.load());++checks;
    assert(!ps5_ai_moss_trial_claim(claim,1024,1024)&&!claim.load());++checks;
#endif
    std::printf("PASS %u actual MOSS trial-claim cases; production_admitted=%d experimental=%d GPU_calls=0 model_inference=0\n",checks,PS5_AI_MOSS_RUNTIME_ADMITTED,PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL);
}
