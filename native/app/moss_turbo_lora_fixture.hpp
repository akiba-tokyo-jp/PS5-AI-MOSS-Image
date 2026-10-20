/* SPDX-License-Identifier: GPL-3.0-or-later
 * Fixed-data numerical fixture only. No model weights, CPU inference or I/O.
 * The reference independently evaluates rank128 and rank16 contributions;
 * the tested GPU representation concatenates those ranks to144 instead.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace moss_turbo_lora_fixture {
constexpr int turbo_rank=128, nsfw_rank=16, rank=turbo_rank+nsfw_rank, tokens=9;
constexpr double absolute_tolerance=0.001, relative_tolerance=0.01;
constexpr float turbo_baked_scale=1.0f, nsfw_baked_scale=0.7f;

enum class Kind { qkv=0, ffn_up=1, ffn_down=2 };
struct Chain {
    int branch=0, k=0, m=0;
    std::vector<float> down, up;
    std::vector<double> expected_down, expected_up;
};
struct Fixture {
    Kind kind{}; int k=0, m=0, branches=0;
    const char* name=nullptr;
    std::vector<float> input;
    std::vector<Chain> chain;
    std::vector<double> expected_concat;
    uint64_t f32_payload_bytes() const {
        uint64_t count=input.size();
        for (const auto& c:chain) count+=c.down.size()+c.up.size();
        return count*sizeof(float);
    }
    uint64_t oracle_payload_bytes() const {
        uint64_t count=expected_concat.size();
        for (const auto& c:chain) count+=c.expected_down.size()+c.expected_up.size();
        return count*sizeof(double);
    }
};

// Explicit unsigned32 mixing is reproducible on PS5/Windows. Separate salts
// make Q/K/V and rank128/rank16 inputs independent, including signed values.
inline uint32_t mix(uint32_t x) {
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x;
}
inline float input_value(int k,int token) {
    return float(int(mix(uint32_t(k)*17u+uint32_t(token)*65537u+0x494e5055u)%65u)-32)/128.f;
}
inline float down_value(int k,int local_rank,int branch,int component) {
    const uint32_t seed=uint32_t(k)*131u+uint32_t(local_rank)*8191u+
        uint32_t(branch)*104729u+uint32_t(component)*1000003u+0x444f574eu;
    return float(int(mix(seed)%33u)-16)/128.f;
}
inline float up_value(int row,int local_rank,int branch,int component) {
    const uint32_t seed=uint32_t(row)*257u+uint32_t(local_rank)*4093u+
        uint32_t(branch)*130363u+uint32_t(component)*1000033u+0x55505052u;
    const float raw=float(int(mix(seed)%33u)-16)/128.f;
    return raw*(component?nsfw_baked_scale:turbo_baked_scale);
}

inline Fixture make(Kind kind) {
    Fixture f; f.kind=kind;
    switch(kind) {
        case Kind::qkv:f.k=3072;f.m=3072;f.branches=3;f.name="F32_rank144_split16_128_Q_K_V_independent_concat_N9";break;
        case Kind::ffn_up:f.k=3072;f.m=16384;f.branches=1;f.name="F32_rank144_split16_128_FFN_up_K3072_M16384_N9";break;
        case Kind::ffn_down:f.k=8192;f.m=3072;f.branches=1;f.name="F32_rank144_split16_128_FFN_down_K8192_M3072_N9";break;
        default:throw std::runtime_error("Turbo rank144 fixture kind is invalid");
    }
    f.input.resize(size_t(f.k)*tokens);
    for(int t=0;t<tokens;++t)for(int d=0;d<f.k;++d)
        f.input[size_t(t)*f.k+d]=input_value(d,t);
    f.chain.resize(f.branches);
    for(int branch=0;branch<f.branches;++branch) {
        auto& c=f.chain[branch];c.branch=branch;c.k=f.k;c.m=f.m;
        c.down.resize(size_t(f.k)*rank);c.up.resize(size_t(rank)*f.m);
        // Exact converter representation: NSFW16 first, then Turbo128;
        // preweighted B contains NSFW0.7 and Turbo1.0 exactly once.
        for(int r=0;r<rank;++r) {
            const int component=r<nsfw_rank,local=component?r:r-nsfw_rank;
            for(int d=0;d<f.k;++d)c.down[size_t(r)*f.k+d]=down_value(d,local,branch,component);
            for(int row=0;row<f.m;++row)c.up[size_t(row)*rank+r]=up_value(row,local,branch,component);
        }
        c.expected_down.resize(size_t(rank)*tokens);
        c.expected_up.assign(size_t(f.m)*tokens,0.0);
        // Oracle: two independently indexed components, no concatenated
        // matrices and no GGML implementation used for reference products.
        for(int component=0;component<2;++component) {
            const int count=component?nsfw_rank:turbo_rank;
            std::vector<double> intermediate(size_t(count)*tokens);
            for(int t=0;t<tokens;++t)for(int r=0;r<count;++r) {
                double sum=0;
                for(int d=0;d<f.k;++d)
                    sum+=double(down_value(d,r,branch,component))*double(input_value(d,t));
                intermediate[size_t(t)*count+r]=sum;
                c.expected_down[size_t(t)*rank+(component?0:nsfw_rank)+r]=sum;
            }
            for(int t=0;t<tokens;++t)for(int row=0;row<f.m;++row) {
                double sum=0;
                for(int r=0;r<count;++r)
                    sum+=double(up_value(row,r,branch,component))*intermediate[size_t(t)*count+r];
                c.expected_up[size_t(t)*f.m+row]+=sum;
            }
        }
    }
    if(f.branches==3) {
        f.expected_concat.resize(size_t(f.m)*f.branches*tokens);
        for(int t=0;t<tokens;++t)for(int branch=0;branch<f.branches;++branch)for(int row=0;row<f.m;++row)
            f.expected_concat[size_t(t)*f.m*f.branches+size_t(branch)*f.m+row]=f.chain[branch].expected_up[size_t(t)*f.m+row];
    }
    return f;
}
struct Comparison { uint64_t compared=0, nonfinite=0, mismatches=0;double max_absolute_error=0; };
inline Comparison compare(const std::vector<float>& actual,const std::vector<double>& expected) {
    if(actual.empty()||actual.size()!=expected.size())throw std::runtime_error("Turbo fixture output size mismatch");
    Comparison c;c.compared=actual.size();
    for(size_t i=0;i<actual.size();++i) {
        const bool finite=std::isfinite(actual[i])&&std::isfinite(expected[i]);
        const double error=finite?std::fabs(double(actual[i])-expected[i]):0.0;
        c.nonfinite+=!finite;c.mismatches+=!finite||error>absolute_tolerance+relative_tolerance*std::fabs(expected[i]);
        c.max_absolute_error=std::max(c.max_absolute_error,error);
    }
    return c;
}
inline void add(Comparison& a,const Comparison& b) {
    a.compared+=b.compared;a.nonfinite+=b.nonfinite;a.mismatches+=b.mismatches;
    a.max_absolute_error=std::max(a.max_absolute_error,b.max_absolute_error);
}
} // namespace moss_turbo_lora_fixture
