/* SPDX-License-Identifier: GPL-3.0-or-later
 * Separate fixed FTZ probe. The first non-dyadic fixture is unchanged.
 * Half coefficients here are exactly representable in both storage formats.
 */
#pragma once
#include "moss_lora_f16_storage_fixture.hpp"
namespace moss_lora_subnormal_fixture {
using moss_lora_f16_storage_fixture::Fixture;
using moss_lora_f16_storage_fixture::half_bits;
using moss_lora_f16_storage_fixture::from_half;
constexpr uint16_t magnitudes[]={1,2,3,7,31,127,511,1023};
inline uint16_t subnormal_bits(int coordinate){
    return uint16_t(magnitudes[(coordinate/2)%8]|((coordinate&1)?0x8000u:0));
}
inline float down_value(int branch,int r){
    if(branch==0)return from_half(subnormal_bits(r));
    if(branch==1)return 1.f;
    return 1.f/1024.f;
}
inline float up_value(int branch,int row,int r){
    if(branch==0)return ((row&1)?-1.f:1.f)*((r&1)?-1.f:1.f);
    if(branch==1)return from_half(subnormal_bits(row));
    return 1.f/128.f;
}
inline Fixture make(){
    Fixture f;f.k=3072;f.m=3072;f.branches=3;
    f.name="separate_F16_subnormal_down_up_and_normal_control_K3072_R144_M3072_N9";
    f.input.assign(size_t(f.k)*9,16.f);f.chain.resize(3);
    for(int b=0;b<3;++b){
        auto& c=f.chain[b];c.down.resize(size_t(f.k)*144);c.down_half.resize(c.down.size());
        c.up.resize(size_t(144)*f.m);c.up_half.resize(c.up.size());
        c.baseline_down.resize(size_t(144)*9);c.rounded_down.resize(c.baseline_down.size());
        c.baseline_up.resize(size_t(f.m)*9);c.rounded_up.resize(c.baseline_up.size());
        for(int r=0;r<144;++r)for(int k=0;k<f.k;++k){
            const size_t i=size_t(r)*f.k+k;c.down[i]=down_value(b,r);c.down_half[i]=half_bits(c.down[i]);}
        for(int row=0;row<f.m;++row)for(int r=0;r<144;++r){
            const size_t i=size_t(row)*144+r;c.up[i]=up_value(b,row,r);c.up_half[i]=half_bits(c.up[i]);}
        // Independent closed-form FP64 oracle for these constant-K rows.
        // No GGML, no model, no assembled-vector multiplication.
        for(int t=0;t<9;++t){
            for(int r=0;r<144;++r){
                const double expected=double(down_value(b,r))*3072.0*16.0;
                c.baseline_down[size_t(t)*144+r]=expected;c.rounded_down[size_t(t)*144+r]=expected;}
            for(int row=0;row<f.m;++row){
                double expected=0;
                for(int r=0;r<144;++r)expected+=double(up_value(b,row,r))*(double(down_value(b,r))*3072.0*16.0);
                c.baseline_up[size_t(t)*f.m+row]=expected;c.rounded_up[size_t(t)*f.m+row]=expected;}
        }
    }
    f.baseline_concat.resize(size_t(f.m)*3*9);f.rounded_concat.resize(f.baseline_concat.size());
    for(int t=0;t<9;++t)for(int b=0;b<3;++b)for(int row=0;row<f.m;++row){
        const size_t i=size_t(t)*f.m*3+size_t(b)*f.m+row,j=size_t(t)*f.m+row;
        f.baseline_concat[i]=f.chain[b].baseline_up[j];f.rounded_concat[i]=f.baseline_concat[i];}
    return f;
}
}
