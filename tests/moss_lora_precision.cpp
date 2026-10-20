/* SPDX-License-Identifier: GPL-3.0-or-later -- pure receipt tests, no GPU. */
#include <mutex>
#include <fstream>
#include <iterator>
#include <cstdio>
#include <stdexcept>
#include "json.hpp"
#include "moss_lora_precision_receipt.inc"
#include "moss_lora_f16_storage_contract.hpp"
static std::vector<unsigned char> read(const std::string& p){
    std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("module fixture missing");
    return {std::istreambuf_iterator<char>(f),{}};
}
int main(int argc,char** argv){
    if(argc!=2)return 2;unsigned checks=0;
    auto test=[&](bool ok){++checks;if(!ok)throw std::runtime_error("private LoRA receipt assertion "+std::to_string(checks));};
    const std::string root=argv[1];
    const std::vector<unsigned char> original[2]={read(root+"/matmul_f32_f32_fp32-0.spv"),read(root+"/matmul_f16_f32_fp32-0.spv")};
    uint32_t spec[12]={128,64,64,32,64,32,2,4,2,1,64,0};
    // All eight source-derived patched-byte identities and four private slots.
    for(unsigned mode=0;mode<4;++mode){
        moss_lora_storage_report={};
        for(unsigned slot=0;slot<4;++slot){
            auto actual=read(root+"/"+(slot<2?"matmul_f32_f32_fp32-":"matmul_f16_f32_fp32-")+std::to_string(mode)+".spv");
            spec[11]=slot%2;ps5_ai_vk_lora_storage_compile(slot,original[slot/2].data(),original[slot/2].size(),
                actual.data(),actual.size(),100+slot,64,slot%2,spec,12,mode&1,(mode>>1)&1);
        }
        ps5_ai_moss_lora_storage_receipt r{};ps5_ai_moss_lora_storage_snapshot(&r);
        test(moss_lora_storage_pipelines_valid(r));
        auto corrupted=r;corrupted.pipelines[0].specialization[3]=64;test(!moss_lora_storage_pipelines_valid(corrupted));
        corrupted=r;corrupted.pipelines[0].actual_sha256[0]^=1;test(!moss_lora_storage_pipelines_valid(corrupted));
    }
    auto good=moss_lora_storage_report;
    for(unsigned bad=0;bad<9;++bad){
        moss_lora_storage_report={};auto bytes=original[0];auto current=std::vector<uint32_t>(spec,spec+12);current[11]=0;
        if(bad==0)bytes[20]^=1;if(bad==1)current[3]=64;
        ps5_ai_vk_lora_storage_compile(bad==2?4:0,original[0].data(),original[0].size(),bytes.data(),bytes.size(),
            bad==3?0:100,bad==4?16:64,bad==5?1:0,current.data(),bad==6?11:12,bad==7?1:0,bad==8?2:0);
        test(moss_lora_storage_report.invalid==1);
    }
    moss_lora_storage_report=good;
    // Simulate only callback metadata, never compiled GPU work.
    for(unsigned type=0;type<2;++type)for(unsigned n=0;n<16;++n)
        ps5_ai_vk_lora_storage_record("moss_f16_storage_fixture.branch0.down",3072,144,9,type,!type,1,1,1,0,0,100+2*type,1,2*type);
    ps5_ai_moss_lora_storage_receipt before{},after{};ps5_ai_moss_lora_storage_snapshot(&before);
    unsigned names=0;for(const auto& [name,shape]:ps5_ai_moss_lora_f16_storage::expected()){
        ++names;const unsigned slot=PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL?2:0;
        for(unsigned step=0;step<4;++step)ps5_ai_vk_lora_storage_record(name.c_str(),shape.input,shape.output,362,
            PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL,!PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL,1,1,1,0,0,100+slot,1,slot);
    }
    ps5_ai_moss_lora_storage_snapshot(&after);test(names==504);test(moss_lora_storage_generation_delta_valid(before,after));
    test((PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL?after.runtime_f16:after.runtime_f32)==2016);
    auto missing=after;--missing.matrix_index[0];test(!moss_lora_storage_generation_delta_valid(before,missing));
    auto extra=after;++extra.matrix_index[0];test(!moss_lora_storage_generation_delta_valid(before,extra));
    auto wrong=after;++wrong.invalid;test(!moss_lora_storage_generation_delta_valid(before,wrong));
    wrong=after;++wrong.pipelines[0].pipeline;test(!moss_lora_storage_generation_delta_valid(before,wrong));
    wrong=after;++wrong.fixture_f32;test(!moss_lora_storage_generation_delta_valid(before,wrong));
    const auto json=moss_lora_storage_generation_json(before);test(json["exact_504_tensors_four_each"]==true&&json["image_quality_parity"]==false);
    for(const auto& badname:{"lora.model.diffusion_model.layers.00.proj.o.weight.lora_down","lora.model.diffusion_model.layers.42.proj.o.weight.lora_down","lora.model.diffusion_model.layers.0.proj.o.weight.lora_down.extra"})test(ps5_ai_moss_lora_storage_index(badname,3072,144)==-1);
    for(unsigned bad=0;bad<9;++bad){
        auto n=moss_lora_storage_report.invalid;const unsigned type=PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL,slot=2*type;
        ps5_ai_vk_lora_storage_record("lora.model.diffusion_model.layers.0.proj.o.weight.lora_down",bad==0?3073:3072,144,bad==1?9:362,
            type,!type,bad==2?0:1,bad==3?0:1,bad==4?0:1,bad==5?1:0,bad==6?1:0,bad==7?999:100+slot,bad==8?0:1,slot);
        test(moss_lora_storage_report.invalid==n+1);
    }
    std::printf("{\"success\":true,\"checks\":%u,\"canonical_names_checked\":%u,\"GPU_execution\":false}\n",checks,names);
}
