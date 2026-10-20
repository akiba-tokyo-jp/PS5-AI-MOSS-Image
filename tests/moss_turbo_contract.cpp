#include "moss_trial_contract.hpp"
#include "moss_lora_contract.hpp"
#include "engine_config.h"
#include <cassert>
#include <cstdio>
#include <thread>
int main(){
    assert(!PS5_AI_MOSS_RUNTIME_ADMITTED);
    constexpr int steps=PS5_AI_MOSS_DEFAULT_REQUEST_STEPS;
    constexpr double cfg=PS5_AI_MOSS_DEFAULT_REQUEST_CFG;
    assert(steps==(PS5_AI_MOSS_TURBO_TRIAL?4:50));
    assert(cfg==(PS5_AI_MOSS_TURBO_TRIAL?1.:3.));
    const auto& shapes=ps5_ai_moss_lora::expected();assert(shapes.size()==504);
    for(const auto& row:shapes){
        const bool down=row.first.find(".lora_down")!=std::string::npos;
        assert((down?row.second.output:row.second.input)==(PS5_AI_MOSS_TURBO_TRIAL?144:16));
        ps5_ai_moss_lora::validate_tensor(row.first,true,2,row.second.input,row.second.output);
    }
    std::atomic<unsigned> attempts{0};
    assert(!ps5_ai_moss_performance_trial_claim(attempts,512,512,steps,cfg,5,"euler",20261020));
    assert(!ps5_ai_moss_performance_trial_claim(attempts,256,256,steps+1,cfg,5,"euler",20261020));
    assert(attempts==0);
    for(unsigned i=0;i<4;++i)assert(ps5_ai_moss_performance_trial_claim(attempts,i<2?256:1024,i<2?256:1024,steps,cfg,5,"euler",20261020));
    assert(!ps5_ai_moss_performance_trial_claim(attempts,256,256,steps,cfg,5,"euler",20261020)&&attempts==4);
    nlohmann::json c={{"gpu_pool_bytes",UINT64_C(10737418240)},{"working_reserve_bytes",UINT64_C(1073741824)},
        {"moss_lora_path",PS5AIEngineConfig::selected_moss_lora_path()},{"moss_lora_strength",PS5_AI_MOSS_TURBO_TRIAL?1.:.7}};
#if PS5_AI_MOSS_TURBO_TRIAL
    c["moss_lora_sha256"]=std::string(64,'a'); // synthetic syntax fixture, not an actual artifact hash.
#endif
    auto parsed=PS5AIEngineConfig::parse(c);assert(parsed.moss_lora_strength==(PS5_AI_MOSS_TURBO_TRIAL?1.:.7));
#if PS5_AI_MOSS_TURBO_TRIAL
    for(auto replacement:{nlohmann::json(nullptr),nlohmann::json(std::string(64,'0')),nlohmann::json("bad")}){
        auto bad=c;bad["moss_lora_sha256"]=replacement;bool failed=false;
        try{PS5AIEngineConfig::parse(bad);}catch(const std::runtime_error&){failed=true;}assert(failed);
    }
    auto bad=c;bad["moss_lora_strength"]=.7;bool failed=false;
    try{PS5AIEngineConfig::parse(bad);}catch(const std::runtime_error&){failed=true;}assert(failed);
    bad=c;bad["moss_lora_path"]="/app0/models/moss-image-nsfw-uncensored-r16-f32.safetensors";failed=false;
    try{PS5AIEngineConfig::parse(bad);}catch(const std::runtime_error&){failed=true;}assert(failed);
    float sigmas[5];ps5_ai_moss_turbo_sigmas(sigmas);
    assert(sigmas[0]==1.f&&sigmas[1]==.910f&&sigmas[2]==.717f&&sigmas[3]==.024f&&sigmas[4]==0.f);
#endif
    puts("PASS actual rank/atomic-attempt/config contract; synthetic only");
}
