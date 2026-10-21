// SPDX-License-Identifier: GPL-3.0-or-later
// Pure policy/seed tests; no learned model, network or GPU.
#include "moss_trial_contract.hpp"
#include "moss_img2img_trial.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
int main(){
    unsigned checks=0;std::atomic<unsigned> claims{0};
    for(int i=0;i<10000;i++){
        int w=64+16*(i%61),h=64+16*((i*7)%61);
        assert(ps5_ai_moss_performance_trial_claim(claims,w,h,4,1,5,"euler",i-1));++checks;
    }
    assert(claims.load()==0);++checks;
    assert(ps5_ai_moss_trial_request_valid(1024,1024,4,1,5,"euler",-1));++checks;
    for(int w:{0,63,65,1032,1040}){assert(!ps5_ai_moss_trial_request_valid(w,768,4,1,5,"euler",17));++checks;}
    assert(!ps5_ai_moss_trial_request_valid(768,512,20,1,5,"euler",17));++checks;
    assert(!ps5_ai_moss_trial_request_valid(768,512,4,5,5,"euler",17));++checks;
    assert(!ps5_ai_moss_trial_request_valid(768,512,4,1,5,"euler_a",17));++checks;
    assert(!ps5_ai_moss_trial_request_valid(768,512,4,1,5,"euler",-2));++checks;
    assert(ps5_ai_moss_img2img_steps(.1)==1 && ps5_ai_moss_img2img_steps(.25)==2 && ps5_ai_moss_img2img_steps(.5)==3 && ps5_ai_moss_img2img_steps(.75)==4 && ps5_ai_moss_img2img_steps(1)==4);++checks;
    assert(ps5_ai_moss_img2img_sigma(.75)==1.0f);++checks;
    assert(!ps5_ai_moss_img2img_steps(1.1) && !ps5_ai_moss_img2img_steps(std::numeric_limits<double>::quiet_NaN()));++checks;
    static int64_t seeds[50000];
    for(unsigned i=0;i<50000;i++){auto v=ps5_ai_moss_new_image_seed(123456,77,i);assert(v>=0 && v==ps5_ai_moss_new_image_seed(123456,77,i));seeds[i]=v;++checks;}
    std::qsort(seeds,50000,sizeof(seeds[0]),[](const void*a,const void*b){auto x=*(const int64_t*)a,y=*(const int64_t*)b;return (x>y)-(x<y);});
    for(unsigned i=1;i<50000;i++)assert(seeds[i]!=seeds[i-1]);++checks;
    std::printf("{\"passed\":%u,\"claims_after10000_requests\":%u,\"inference\":false}\n",checks,claims.load());
}
