/* SPDX-License-Identifier: GPL-3.0-or-later
 * Synthetic complete MOSS FFN oracle. Repeated row classes bound CPU oracle
 * cost while executing full3072->16384->8192->3072 GPU matrices and rank144.
 * No learned parameters, file I/O or CPU model backend.
 */
#pragma once
#include "ggml-quants.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace moss_ffn_chunk_fixture {
constexpr int K=3072,F=8192,U=16384,R=144;
constexpr double absolute_tolerance=.001,relative_tolerance=.01;
inline uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
inline int component(int r){return r<16?0:1;}
inline int local(int r){return r<16?r:r-16;}
inline int rank_class(int r){return component(r)*8+local(r)%8;}
inline float xvalue(int k,int t){return float(int(mix(uint32_t(k)*37u+uint32_t(t)*65537u+13u)%127)-63)/131.f;}
inline float qvalue(int k,int c,int salt){return float(int(mix(uint32_t(k)*17u+uint32_t(c)*8191u+uint32_t(salt))%63)-31)/4096.f;}
inline float avalue(int k,int rc,int salt){return float(int(mix(uint32_t(k)*31u+uint32_t(rc)*104729u+uint32_t(salt))%31)-15)/2047.f;}
inline float bvalue(int c,int r,int salt){
    const float raw=float(int(mix(uint32_t(c)*65537u+uint32_t(local(r))*257u+uint32_t(component(r))*1000003u+uint32_t(salt))%31)-15)/127.f;
    return raw*(component(r)==0?.7f:1.f);
}
inline int up_class(int row){return row%4+(row>=F?4:0);}
struct Fixture {
    int tokens;
    std::vector<block_q4_K> q_up,q_down;
    std::vector<float> input,a_up,b_up,a_down,b_down;
    std::array<std::vector<float>,8> decoded_up,decoded_down;
    std::vector<double> expected; // [8,N], repeated output rows by row%8.
    explicit Fixture(int n):tokens(n) {
        if(n<1||n>4202)throw std::runtime_error("FFN synthetic fixture token count rejected");
        input.resize(size_t(K)*n);
        for(int t=0;t<n;++t)for(int k=0;k<K;++k)input[size_t(t)*K+k]=xvalue(k,t);
        q_up.resize(size_t(U)*K/QK_K);q_down.resize(size_t(K)*F/QK_K);
        std::array<std::vector<block_q4_K>,8> up_rows,down_rows;
        std::vector<float> temporary(F);
        for(int c=0;c<8;++c) {
            up_rows[c].resize(K/QK_K);down_rows[c].resize(F/QK_K);
            for(int k=0;k<K;++k)temporary[k]=qvalue(k,c,17);
            quantize_row_q4_K_ref(temporary.data(),up_rows[c].data(),K);
            decoded_up[c].resize(K);dequantize_row_q4_K(up_rows[c].data(),decoded_up[c].data(),K);
            for(int k=0;k<F;++k)temporary[k]=qvalue(k,c,113);
            quantize_row_q4_K_ref(temporary.data(),down_rows[c].data(),F);
            decoded_down[c].resize(F);dequantize_row_q4_K(down_rows[c].data(),decoded_down[c].data(),F);
        }
        for(int row=0;row<U;++row)std::memcpy(q_up.data()+size_t(row)*K/QK_K,up_rows[up_class(row)].data(),K/QK_K*sizeof(block_q4_K));
        for(int row=0;row<K;++row)std::memcpy(q_down.data()+size_t(row)*F/QK_K,down_rows[row%8].data(),F/QK_K*sizeof(block_q4_K));
        a_up.resize(size_t(R)*K);b_up.resize(size_t(U)*R);
        a_down.resize(size_t(R)*F);b_down.resize(size_t(K)*R);
        for(int r=0;r<R;++r) {
            for(int k=0;k<K;++k)a_up[size_t(r)*K+k]=avalue(k,rank_class(r),401);
            // A structured but nonzero down matrix: exact four-class reduction.
            for(int k=0;k<F;++k)a_down[size_t(r)*F+k]=avalue(k%4,rank_class(r),919)/64.f;
            for(int row=0;row<U;++row)b_up[size_t(row)*R+r]=bvalue(up_class(row),r,719);
            for(int row=0;row<K;++row)b_down[size_t(row)*R+r]=bvalue(row%8,r,1213);
        }
        expected.resize(size_t(n)*8);
        std::array<std::array<double,4>,8> qsum{};
        std::array<std::array<double,4>,16> asum{};
        for(int c=0;c<8;++c)for(int k=0;k<F;++k)qsum[c][k%4]+=decoded_down[c][k];
        for(int rc=0;rc<16;++rc)for(int k=0;k<F;++k)asum[rc][k%4]+=double(avalue(k%4,rc,919)/64.f);
        for(int t=0;t<n;++t) {
            std::array<double,8> up{};
            std::array<double,16> low{},down{};
            for(int rc=0;rc<16;++rc)for(int k=0;k<K;++k)low[rc]+=double(avalue(k,rc,401))*input[size_t(t)*K+k];
            for(int c=0;c<8;++c) {
                for(int k=0;k<K;++k)up[c]+=double(decoded_up[c][k])*input[size_t(t)*K+k];
                for(int r=0;r<R;++r)up[c]+=double(bvalue(c,r,719))*low[rank_class(r)];
            }
            std::array<double,4> hidden{};
            for(int c=0;c<4;++c) {
                const double a=4.0*std::tanh(up[c+4]*double(.25f));
                const double b=1.0/(1.0+std::exp(-up[c+4]));
                const double d=25.0*std::tanh(up[c]*double(1.f/25.f));
                hidden[c]=(a*b)*d;
            }
            for(int rc=0;rc<16;++rc)for(int c=0;c<4;++c)down[rc]+=asum[rc][c]*hidden[c];
            for(int row=0;row<8;++row) {
                double value=0;
                for(int c=0;c<4;++c)value+=qsum[row][c]*hidden[c];
                for(int r=0;r<R;++r)value+=double(bvalue(row,r,1213))*down[rank_class(r)];
                if(!std::isfinite(value))throw std::runtime_error("FFN synthetic CPU oracle nonfinite");
                expected[size_t(t)*8+row]=value;
            }
        }
    }
    double oracle(size_t index)const{return expected[(index/K)*8+(index%K)%8];}
};
struct Comparison {uint64_t compared=0,nonfinite=0,mismatches=0;double max_absolute_error=0;};
template<class Expected>
Comparison compare(const std::vector<float>& actual,Expected&& expected) {
    if(actual.empty())throw std::runtime_error("FFN comparison empty");
    Comparison r;r.compared=actual.size();
    for(size_t i=0;i<actual.size();++i) {
        const double reference=expected(i);const bool finite=std::isfinite(actual[i])&&std::isfinite(reference);
        const double error=finite?std::fabs(double(actual[i])-reference):0;
        r.nonfinite+=!finite;r.mismatches+=!finite||error>absolute_tolerance+relative_tolerance*std::fabs(reference);
        r.max_absolute_error=std::max(r.max_absolute_error,error);
    }
    return r;
}
} // namespace moss_ffn_chunk_fixture
