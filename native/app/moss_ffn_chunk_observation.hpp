/* SPDX-License-Identifier: MIT
 * Bounded metadata accounting from the existing evaluation callback's asks.
 * No new callback requests/barriers, GPU timing claim, allocation or per-op I/O.
 * Completion is proven separately by four finite outputs and GPU Q4 receipts.
 */
#pragma once
#include <cstdint>
#include <cstring>
namespace moss_ffn_chunk_observation {
struct Count {uint64_t nodes=0,tokens=0;int64_t first_tokens=0,max_tokens=0;};
struct Proof {bool valid=false;int64_t main_tokens=0,pieces=0;uint64_t expected_q4=0;};
struct State {
    Count counts[42][4]{};bool invalid=false;
    void add(const char* name,int64_t k,int64_t m,int64_t n,bool f32) noexcept {
        constexpr char prefix[]="model.diffusion_model.layers.";
        if(!name||std::strncmp(name,prefix,sizeof(prefix)-1))return;
        const char* p=name+sizeof(prefix)-1;unsigned layer=0;
        if(*p<'0'||*p>'9')return;
        while(*p>='0'&&*p<='9'){layer=layer*10+unsigned(*p++-'0');if(layer>=42)return;}
        if(std::strncmp(p,".proj.",6))return;p+=6;
        int module=-1;
        if(!std::strcmp(p,"qkv.weight"))module=0;
        else if(!std::strcmp(p,"o.weight"))module=1;
        else if(!std::strcmp(p,"ffn_up_gate.weight"))module=2;
        else if(!std::strcmp(p,"ffn_down.weight"))module=3;
        else return;
        const int64_t wantk=module==3?8192:3072;
        const int64_t wantm=module==0?9216:module==2?16384:3072;
        if(!f32||k!=wantk||m!=wantm||n<1||n>4202){invalid=true;return;}
        auto& c=counts[layer][module];++c.nodes;c.tokens+=uint64_t(n);
        if(!c.first_tokens)c.first_tokens=n;
        if(n>c.max_tokens)c.max_tokens=n;
        if(module<2&&c.first_tokens!=n)invalid=true;
    }
    Proof proof(int64_t chunk)const noexcept {
        Proof p;
        if(invalid||(chunk!=512&&chunk!=1024&&chunk!=2048))return p;
        const int64_t n=counts[0][0].first_tokens;
        if(n<1||n>4202)return p;
        const int64_t pieces=(n+chunk-1)/chunk;
        for(unsigned layer=0;layer<42;++layer) {
            const auto& row=counts[layer];
            for(int m=0;m<4;++m){
                if(row[m].nodes!=uint64_t(4*(m<2?1:pieces))||row[m].tokens!=uint64_t(4*n))return p;
                if(m<2&&row[m].first_tokens!=n)return p;
                if(m>=2&&(row[m].first_tokens!=(n<chunk?n:chunk)||row[m].max_tokens!=(n<chunk?n:chunk)))return p;
            }
        }
        p.valid=true;p.main_tokens=n;p.pieces=pieces;p.expected_q4=4*42*(2+2*uint64_t(pieces));return p;
    }
};
} // namespace moss_ffn_chunk_observation
