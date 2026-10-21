// SPDX-License-Identifier: GPL-3.0-or-later
// Existing SD.cpp sigma suffix semantics, without models or GPU inference.
#include "moss_img2img_trial.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <initializer_list>
int main(){
    unsigned checked=0;const float author[]={1,.910f,.717f,.024f,0};
    for(unsigned j=100;j<7500;++j){
        const double value=j/10000.0;const float strength=(float)value;
        const unsigned t_enc=(unsigned)(4*strength),start=4-t_enc-1;
        assert(start>0&&start<4);
        assert(ps5_ai_moss_img2img_steps(value)==int(4-start));
        assert(ps5_ai_moss_img2img_sigma(value)==author[start]);
        assert(1-author[start]>0&&168*(4-start)<=504);++checked;
    }
    for(double v:{-1.0,0.0,.009,.75,1.0,1e300,std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}){
        assert(!ps5_ai_moss_img2img_steps(v)&&!ps5_ai_moss_img2img_sigma(v));++checked;
    }
    // Double values that round across a boundary must follow actual float math.
    assert(ps5_ai_moss_img2img_steps(.249999999)==2);++checked;
    assert(ps5_ai_moss_img2img_steps(.499999999)==3);++checked;
    assert(ps5_ai_moss_img2img_steps(.749999999)==0);++checked;
    std::printf("{\"passed\":%u,\"model_inference\":false,\"PS5_connections\":0}\n",checked);
}
