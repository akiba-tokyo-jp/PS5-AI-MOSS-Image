/* SPDX-License-Identifier: GPL-3.0-or-later
 * Fixed synthetic coefficient/matrix arithmetic, never model inference.
 * Each case contains current F32 and separately rounded F16 weight oracles.
 */
#pragma once
#include "moss_turbo_lora_fixture.hpp"
#include <cstring>

namespace moss_lora_f16_storage_fixture {
enum class Kind {qkv,ffn_up,ffn_down};
using Comparison=moss_turbo_lora_fixture::Comparison;
using moss_turbo_lora_fixture::compare;
using moss_turbo_lora_fixture::add;
constexpr int rank=144,tokens=9;
constexpr double absolute_tolerance=0.001,relative_tolerance=0.01;

/* IEEE754 binary32 -> binary16 RNE, with finite-only admission. These exact
 * integer conversions are tested independently against Python struct('e'). */
inline uint16_t half_bits(float value){
    uint32_t x;std::memcpy(&x,&value,4);
    const uint32_t sign=(x>>16)&0x8000u,mag=x&0x7fffffffu;
    if(mag>=0x7f800000u)throw std::runtime_error("F16 fixture nonfinite input");
    const int exponent=int((x>>23)&255)-127;
    if(exponent>15)throw std::runtime_error("F16 fixture overflow");
    uint32_t mantissa=x&0x7fffffu;
    if(exponent<-25)return uint16_t(sign);
    if(exponent<-14){
        mantissa|=0x800000u;const unsigned shift=unsigned(-exponent-1);
        const uint32_t base=mantissa>>shift,rest=mantissa&((1u<<shift)-1),half=1u<<(shift-1);
        return uint16_t(sign+base+(rest>half||(rest==half&&(base&1))));
    }
    uint32_t rounded=(uint32_t(exponent+15)<<10)+(mantissa>>13);
    const uint32_t rest=mantissa&8191u;
    rounded+=rest>4096u||(rest==4096u&&(rounded&1));
    if(rounded>=0x7c00u)throw std::runtime_error("F16 fixture rounded overflow");
    return uint16_t(sign|rounded);
}
inline float from_half(uint16_t h){
    uint32_t sign=uint32_t(h&0x8000u)<<16,exp=(h>>10)&31u,frac=h&1023u,bits;
    if(exp==31)throw std::runtime_error("F16 fixture nonfinite half");
    if(!exp){
        if(!frac)bits=sign;
        else {int e=-14;while(!(frac&1024u)){frac<<=1;--e;}bits=sign|(uint32_t(e+127)<<23)|((frac&1023u)<<13);}
    } else bits=sign|((exp+112u)<<23)|(frac<<13);
    float value;std::memcpy(&value,&bits,4);return value;
}
inline float rounded(float value){return from_half(half_bits(value));}
inline float input_value(int k,int token){
    return float(int(moss_turbo_lora_fixture::mix(uint32_t(k)*17u+uint32_t(token)*65537u+0x494e5055u)%65u)-32)/127.f;
}
inline float down_value(int k,int r,int branch){
    return float(int(moss_turbo_lora_fixture::mix(uint32_t(k)*131u+uint32_t(r)*8191u+uint32_t(branch)*104729u+0x444f574eu)%33u)-16)/113.f;
}
inline float up_value(int row,int r,int branch){
    const float value=float(int(moss_turbo_lora_fixture::mix(uint32_t(row)*257u+uint32_t(r)*4093u+uint32_t(branch)*130363u+0x55505052u)%33u)-16)/137.f;
    return value*(r<16?0.7f:1.0f);
}
struct Chain {
    std::vector<float> down,up;
    std::vector<uint16_t> down_half,up_half;
    std::vector<double> baseline_down,baseline_up,rounded_down,rounded_up;
};
struct Fixture {
    Kind kind{};int k=0,m=0,branches=0;const char *name=nullptr;
    std::vector<float> input;std::vector<Chain> chain;
    std::vector<double> baseline_concat,rounded_concat;
    uint64_t host_payload_bytes()const{
        uint64_t n=input.size()*sizeof(float)+(baseline_concat.size()+rounded_concat.size())*sizeof(double);
        for(const auto& c:chain)n+=(c.down.size()+c.up.size())*sizeof(float)+(c.down_half.size()+c.up_half.size())*sizeof(uint16_t)+
            (c.baseline_down.size()+c.baseline_up.size()+c.rounded_down.size()+c.rounded_up.size())*sizeof(double);
        return n;
    }
};
inline Fixture make(Kind kind){
    Fixture f;f.kind=kind;
    switch(kind){
    case Kind::qkv:f.k=3072;f.m=3072;f.branches=3;f.name="non_dyadic_QKV_K3072_M3072_N9";break;
    case Kind::ffn_up:f.k=3072;f.m=16384;f.branches=1;f.name="non_dyadic_FFN_up_K3072_M16384_N9";break;
    case Kind::ffn_down:f.k=8192;f.m=3072;f.branches=1;f.name="non_dyadic_FFN_down_K8192_M3072_N9";break;
    default:throw std::runtime_error("Invalid F16 storage fixture kind");
    }
    f.input.resize(size_t(f.k)*tokens);
    for(int t=0;t<tokens;++t)for(int k=0;k<f.k;++k)f.input[size_t(t)*f.k+k]=input_value(k,t);
    f.chain.resize(f.branches);
    for(int b=0;b<f.branches;++b){
        auto& c=f.chain[b];c.down.resize(size_t(f.k)*rank);c.down_half.resize(c.down.size());
        c.up.resize(size_t(rank)*f.m);c.up_half.resize(c.up.size());
        for(int r=0;r<rank;++r)for(int k=0;k<f.k;++k){
            const size_t i=size_t(r)*f.k+k;c.down[i]=down_value(k,r,b);c.down_half[i]=half_bits(c.down[i]);
        }
        for(int row=0;row<f.m;++row)for(int r=0;r<rank;++r){
            const size_t i=size_t(row)*rank+r;c.up[i]=up_value(row,r,b);c.up_half[i]=half_bits(c.up[i]);
        }
        c.baseline_down.resize(size_t(rank)*tokens);c.rounded_down.resize(c.baseline_down.size());
        c.baseline_up.resize(size_t(f.m)*tokens);c.rounded_up.resize(c.baseline_up.size());
        // Independent oracle regenerates coefficients by logical coordinates;
        // it does not multiply the assembled GPU vectors or use GGML helpers.
        for(int t=0;t<tokens;++t)for(int r=0;r<rank;++r){
            double baseline=0,half=0;
            for(int k=0;k<f.k;++k){const double x=double(input_value(k,t));
                baseline+=double(down_value(k,r,b))*x;half+=double(rounded(down_value(k,r,b)))*x;}
            c.baseline_down[size_t(t)*rank+r]=baseline;c.rounded_down[size_t(t)*rank+r]=half;
        }
        for(int t=0;t<tokens;++t)for(int row=0;row<f.m;++row){
            double baseline=0,half=0;
            for(int r=0;r<rank;++r){
                baseline+=double(up_value(row,r,b))*c.baseline_down[size_t(t)*rank+r];
                half+=double(rounded(up_value(row,r,b)))*c.rounded_down[size_t(t)*rank+r];}
            c.baseline_up[size_t(t)*f.m+row]=baseline;c.rounded_up[size_t(t)*f.m+row]=half;
        }
    }
    if(f.branches==3){
        f.baseline_concat.resize(size_t(f.m)*f.branches*tokens);f.rounded_concat.resize(f.baseline_concat.size());
        for(int t=0;t<tokens;++t)for(int b=0;b<3;++b)for(int row=0;row<f.m;++row){
            const size_t i=size_t(t)*f.m*3+size_t(b)*f.m+row,j=size_t(t)*f.m+row;
            f.baseline_concat[i]=f.chain[b].baseline_up[j];f.rounded_concat[i]=f.chain[b].rounded_up[j];}
    }
    return f;
}
} // namespace moss_lora_f16_storage_fixture
