/* SPDX-License-Identifier: MIT
 * Actual header/selected504 metadata; no model values, GPU or Vulkan calls.
 */
#include "ggml.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "ggml-impl.h"
#pragma clang diagnostic pop
#include "moss_gpu_profile.h"
#include "moss_lora_contract.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <deque>

extern "C" unsigned ps5_ai_test_actual_node_class(const ggml_tensor *node);
extern "C" ps5_ai_moss_gpu_coverage ps5_ai_test_actual_graph_coverage(const ggml_cgraph *graph);

static unsigned checks;
static void need(bool ok, const char *what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL %u %s\n", checks, what); std::abort(); }
}
static bool ends(const std::string &s, const std::string &tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

static void classification() {
    static_assert(GGML_MAX_NAME == 160, "Match actual75core/backend name ABI");
    const auto &catalog = ps5_ai_moss_lora::expected();
    need(catalog.size() == 504, "actual selected504 contract");
    unsigned down = 0, up = 0;
    for (const auto &entry : catalog) {
        const std::string &name = entry.first;
        const auto &shape = entry.second;
        const bool is_down = ends(name, ".lora_down");
        const unsigned expected = is_down ? PS5_AI_GPU_LORA_DOWN : PS5_AI_GPU_LORA_UP;
        ggml_tensor actual{};
        ggml_set_name(&actual, name.c_str());
        actual.type = GGML_TYPE_F32;
        actual.ne[0] = shape.input; actual.ne[1] = shape.output; actual.ne[2] = actual.ne[3] = 1;
        ggml_tensor rhs{}, node{};
        rhs.type = GGML_TYPE_F32; rhs.ne[0] = shape.input; rhs.ne[1] = 304; rhs.ne[2] = rhs.ne[3] = 1;
        node.type = GGML_TYPE_F32; node.op = GGML_OP_MUL_MAT; node.src[0] = &actual; node.src[1] = &rhs;
        node.ne[0] = shape.output; node.ne[1] = 304; node.ne[2] = node.ne[3] = 1;
        need(name.size() < sizeof(actual.name), "full canonical name fits actual tensor ABI");
        need(name == actual.name, "actual ggml_set_name retains full canonical name");
        need(ps5_ai_moss_gpu_lora_class(actual.name, shape.input, shape.output) == expected,
             "all504 actual names and shapes classified");
        need(ps5_ai_moss_gpu_lora_class(actual.name, shape.input + 1, shape.output) == PS5_AI_GPU_OTHER_MM,
             "rank/input geometry mismatch refuses specific LoRA class");
        need(ps5_ai_moss_gpu_lora_class(actual.name, shape.input, shape.output + 1) == PS5_AI_GPU_OTHER_MM,
             "rank/output geometry mismatch refuses specific LoRA class");
        need(ps5_ai_moss_gpu_lora_class((name + ".extra").c_str(), shape.input, shape.output) == PS5_AI_GPU_OTHER_MM,
             "suffix alias refuses specific LoRA class");
        need(ps5_ai_moss_gpu_lora_class(("foreign." + name).c_str(), shape.input, shape.output) == PS5_AI_GPU_OTHER_MM,
             "other model namespace refuses specific LoRA class");
        need(ps5_ai_moss_gpu_lora_class(actual.name, 0, 0) == PS5_AI_GPU_OTHER_MM,
             "missing geometry refuses specific LoRA class");
        need(ps5_ai_test_actual_node_class(&node) == expected, "actual backend classifier accepts canonical F32 rank16 operator");
        rhs.type = GGML_TYPE_F16;
        need(ps5_ai_test_actual_node_class(&node) == PS5_AI_GPU_OTHER_MM, "actual backend refuses nonF32 LoRA RHS");
        rhs.type = GGML_TYPE_F32; node.type = GGML_TYPE_F16;
        need(ps5_ai_test_actual_node_class(&node) == PS5_AI_GPU_OTHER_MM, "actual backend refuses nonF32 LoRA result");
        node.type = GGML_TYPE_F32; actual.type = GGML_TYPE_F16;
        need(ps5_ai_test_actual_node_class(&node) == PS5_AI_GPU_OTHER_MM, "actual backend refuses nonF32 LoRA weight");
        actual.type = GGML_TYPE_F32; actual.ne[2] = 2;
        need(ps5_ai_test_actual_node_class(&node) == PS5_AI_GPU_OTHER_MM, "actual backend refuses batched LoRA weights");
        actual.ne[2] = 1; rhs.ne[0] = shape.input + 1;
        need(ps5_ai_test_actual_node_class(&node) == PS5_AI_GPU_OTHER_MM, "actual backend refuses mismatched LoRA input geometry");
        rhs.ne[0] = shape.input; node.ne[0] = shape.output + 1;
        need(ps5_ai_test_actual_node_class(&node) == PS5_AI_GPU_OTHER_MM, "actual backend refuses mismatched LoRA result geometry");
        node.ne[0] = shape.output; node.op = GGML_OP_ADD;
        need(ps5_ai_test_actual_node_class(&node) == PS5_AI_GPU_ADD, "actual backend uses operation before LoRA-like names");
        down += is_down; up += !is_down;
    }
    need(down == 252 && up == 252, "252down and252up weights independently classified");

    for (const std::string invalid : {"-1", "00", "01", "042", "42", "100", "", "1x", "1/", "1.0"}) {
        const std::string name = "lora.model.diffusion_model.layers." + invalid + ".proj.o.weight.lora_down";
        need(ps5_ai_moss_gpu_lora_class(name.c_str(), 3072, 16) == PS5_AI_GPU_OTHER_MM,
             "malformed/out of range/leadingzero layer grammar rejected");
    }
    need(ps5_ai_moss_gpu_lora_class(nullptr, 3072, 16) == PS5_AI_GPU_OTHER_MM, "null name rejected");
    need(ps5_ai_moss_gpu_lora_class("lora.model.diffusion_model.layers.0.proj.qkv.weight.0.lora_up", 16, 3072) == PS5_AI_GPU_OTHER_MM,
         "qkv first branch has no .0 alias");
    need(ps5_ai_moss_gpu_lora_class("lora.model.diffusion_model.layers.0.proj.qkv.weight.3.lora_up", 16, 3072) == PS5_AI_GPU_OTHER_MM,
         "qkv has no fourth slice");
    need(ps5_ai_moss_gpu_lora_class("model.diffusion_model.layers.0.proj.o.weight.lora_down", 3072, 16) == PS5_AI_GPU_OTHER_MM,
         "missing lora namespace rejected");
    need(ps5_ai_moss_gpu_lora_class("lora.cond_stage_model.layers.0.proj.o.weight.lora_down", 3072, 16) == PS5_AI_GPU_OTHER_MM,
         "text model LoRA not classified as MOSS");
    need(ps5_ai_moss_gpu_lora_class("lora.model.diffusion_model.transformer_blocks.0.proj.o.weight.lora_down", 3072, 16) == PS5_AI_GPU_OTHER_MM,
         "other architecture not classified as MOSS");

    const struct { const char *module; int64_t k, m; unsigned id; } q4[] = {
        {"qkv", 3072, 9216, PS5_AI_GPU_Q4_QKV}, {"o", 3072, 3072, PS5_AI_GPU_Q4_O},
        {"ffn_up_gate", 3072, 16384, PS5_AI_GPU_Q4_FFN_UP}, {"ffn_down", 8192, 3072, PS5_AI_GPU_Q4_FFN_DOWN}
    };
    for (unsigned layer = 0; layer < 42; ++layer) for (const auto &item : q4) {
        const auto name = std::string("model.diffusion_model.layers.") + std::to_string(layer) + ".proj." + item.module + ".weight";
        need(ps5_ai_moss_gpu_q4_class(name.c_str(), item.k, item.m) == item.id, "all168 actual base module names and geometry");
        need(ps5_ai_moss_gpu_q4_class(name.c_str(), item.k + 1, item.m) == PS5_AI_GPU_Q4_OTHER, "Q4 shape mismatch recorded other");
        need(ps5_ai_moss_gpu_q4_class((name + ".extra").c_str(), item.k, item.m) == PS5_AI_GPU_Q4_OTHER, "Q4 trailing alias recorded other");
        need(ps5_ai_moss_gpu_q4_class(("foreign." + name).c_str(), item.k, item.m) == PS5_AI_GPU_Q4_OTHER, "other namespace has no MOSS module attribution");
        ggml_tensor weight{}, rhs{}, node{};
        ggml_set_name(&weight, name.c_str()); weight.type = GGML_TYPE_Q4_K;
        weight.ne[0] = item.k; weight.ne[1] = item.m; weight.ne[2] = weight.ne[3] = 1;
        rhs.type = GGML_TYPE_F32; rhs.ne[0] = item.k; rhs.ne[1] = 4096; rhs.ne[2] = rhs.ne[3] = 1;
        node.type = GGML_TYPE_F32; node.op = GGML_OP_MUL_MAT; node.src[0] = &weight; node.src[1] = &rhs;
        node.ne[0] = item.m; node.ne[1] = 4096; node.ne[2] = node.ne[3] = 1;
        need(ps5_ai_test_actual_node_class(&node) == item.id, "actual Q4 backend operator classifier matches all168modules");
        rhs.type = GGML_TYPE_F16;
        need(ps5_ai_test_actual_node_class(&node) != item.id, "actual Q4 backend rejects nonF32 RHS attribution");
        rhs.type = GGML_TYPE_F32; node.type = GGML_TYPE_F16;
        need(ps5_ai_test_actual_node_class(&node) != item.id, "actual Q4 backend rejects nonF32 result attribution");
        node.type = GGML_TYPE_F32; weight.ne[2] = 2;
        need(ps5_ai_test_actual_node_class(&node) != item.id, "actual Q4 backend rejects batched weight attribution");
        weight.ne[2] = 1; rhs.ne[0] = item.k + 1;
        need(ps5_ai_test_actual_node_class(&node) != item.id, "actual Q4 backend rejects mismatched input geometry");
        rhs.ne[0] = item.k; node.ne[0] = item.m + 1;
        need(ps5_ai_test_actual_node_class(&node) != item.id, "actual Q4 backend rejects mismatched result geometry");
    }
    need(ps5_ai_moss_gpu_q4_class(nullptr, 3072, 16384) == PS5_AI_GPU_Q4_OTHER, "Q4 null name is other");
    need(ps5_ai_test_actual_node_class(nullptr) == PS5_AI_GPU_OTHER, "missing actual node has explicit other attribution");
    ggml_tensor other{};
    other.op = GGML_OP_MUL_MAT;
    need(ps5_ai_test_actual_node_class(&other) == PS5_AI_GPU_OTHER_MM, "missing actual matrix operands have explicit other attribution");
    const struct { ggml_op op; unsigned id; } other_ops[] = {
        {GGML_OP_FLASH_ATTN_EXT, PS5_AI_GPU_ATTENTION}, {GGML_OP_SCALE, PS5_AI_GPU_SCALE},
        {GGML_OP_ADD, PS5_AI_GPU_ADD}, {GGML_OP_CONT, PS5_AI_GPU_OTHER},
        {GGML_OP_NONE, PS5_AI_GPU_OTHER}, {GGML_OP_MUL_MAT_ID, PS5_AI_GPU_OTHER}
    };
    uint64_t attributed[PS5_AI_MOSS_GPU_PROFILE_CLASSES]{};
    for (const auto &item : other_ops) {
        other.op = item.op; const auto id = ps5_ai_test_actual_node_class(&other);
        need(id == item.id && id < PS5_AI_MOSS_GPU_PROFILE_CLASSES, "actual remaining operator has one bounded explicit class");
        ++attributed[id];
    }
    uint64_t total = 0; for (const auto count : attributed) total += count;
    need(total == sizeof(other_ops) / sizeof(other_ops[0]) && attributed[PS5_AI_GPU_OTHER] == 3,
         "other/unclassified operators explicitly counted exactly once");

    // Independently enumerate the actual42 positive/41 negative blocks over50steps.
    uint64_t down_ops = 0, up_ops = 0, block_count = 0;
    for (unsigned step = 0; step < 50; ++step) for (unsigned branch = 0; branch < 2; ++branch)
        for (unsigned layer = 0; layer < 42; ++layer) {
            if (branch == 1 && layer == 22) continue;
            ++block_count;
            const std::string marker = ".layers." + std::to_string(layer) + ".proj.";
            for (const auto &entry : catalog) {
                if (entry.first.find(marker) == std::string::npos) continue;
                const auto c = ps5_ai_moss_gpu_lora_class(entry.first.c_str(), entry.second.input, entry.second.output);
                down_ops += c == PS5_AI_GPU_LORA_DOWN;
                up_ops += c == PS5_AI_GPU_LORA_UP;
                need(c == PS5_AI_GPU_LORA_DOWN || c == PS5_AI_GPU_LORA_UP, "enumerated executed LoRA has classified attribution");
            }
        }
    need(block_count == 4150 && down_ops == 24900 && up_ops == 24900, "4150block full-job expected LoRA coverage");
    need(block_count * 3 == 12450, "three direct SCALE+ADD candidates per block, qkv concatenated");

    for (unsigned c = 0; c < PS5_AI_MOSS_GPU_PROFILE_CLASSES; ++c)
        need(std::strcmp(ps5_ai_moss_gpu_class_name(c), "invalid") != 0, "all class names explicit");
    need(std::strcmp(ps5_ai_moss_gpu_class_name(PS5_AI_MOSS_GPU_PROFILE_CLASSES), "invalid") == 0, "out of bounds class name rejected");
    need(std::strcmp(ps5_ai_moss_gpu_class_name(UINT32_MAX), "invalid") == 0, "large invalid class rejected");
}

static void timer_ranges_and_values() {
    need(ps5_ai_moss_gpu_range_valid(0, 2, PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES), "first pair bounded");
    need(ps5_ai_moss_gpu_range_valid(0, PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES, PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES), "whole fixed pool bounded");
    need(ps5_ai_moss_gpu_range_valid(PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES - 2, 2, PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES), "last pair bounded");
    need(!ps5_ai_moss_gpu_range_valid(0, 0, PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES), "empty range rejected");
    need(!ps5_ai_moss_gpu_range_valid(0, 1, 0), "missing pool rejected");
    need(!ps5_ai_moss_gpu_range_valid(PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES, 1, PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES), "past end rejected");
    need(!ps5_ai_moss_gpu_range_valid(PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES - 1, 2, PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES), "straddling range rejected");
    need(!ps5_ai_moss_gpu_range_valid(UINT64_MAX, 2, PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES), "first addition overflow refused");
    need(!ps5_ai_moss_gpu_range_valid(1, UINT64_MAX, PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES), "count addition overflow refused");
    need(!ps5_ai_moss_gpu_range_valid(0, 2, UINT64_MAX), "unbounded capacity refused");

    ps5_ai_moss_gpu_query_value a{100, 1}, b{130, 1};
    uint64_t ticks = 0xabc;
    need(ps5_ai_moss_gpu_pair_ticks(&a, &b, &ticks) && ticks == 30, "ordered available query pair duration");
    b.ticks = a.ticks;
    need(ps5_ai_moss_gpu_pair_ticks(&a, &b, &ticks) && ticks == 0, "equal nonzero query pair explicitly quantized zero");
    b.ticks = a.ticks - 1; ticks = 0xabc;
    need(!ps5_ai_moss_gpu_pair_ticks(&a, &b, &ticks) && ticks == 0xabc, "reversed pair refused without output mutation");
    b.ticks = 130;
    for (uint64_t unavailable : {uint64_t(0), uint64_t(2), UINT64_MAX}) {
        a.available = unavailable; ticks = 0xabc;
        need(!ps5_ai_moss_gpu_pair_ticks(&a, &b, &ticks) && ticks == 0xabc, "begin availability must be exactly1");
        a.available = 1; b.available = unavailable;
        need(!ps5_ai_moss_gpu_pair_ticks(&a, &b, &ticks) && ticks == 0xabc, "end availability must be exactly1");
        b.available = 1;
    }
    a.ticks = 0;
    need(!ps5_ai_moss_gpu_pair_ticks(&a, &b, &ticks), "missing/zero start timestamp rejected");
    a.ticks = 100; b.ticks = 0;
    need(!ps5_ai_moss_gpu_pair_ticks(&a, &b, &ticks), "missing/zero end timestamp rejected");
    a.ticks = UINT64_MAX; b.ticks = UINT64_MAX;
    need(!ps5_ai_moss_gpu_pair_ticks(&a, &b, &ticks), "notready sentinel rejected despite malformed availability1");
    a.ticks = 100; b.ticks = UINT64_MAX;
    need(!ps5_ai_moss_gpu_pair_ticks(&a, &b, &ticks), "end notready sentinel rejected despite malformed availability1");
    a.ticks = UINT64_MAX; b.ticks = 130;
    need(!ps5_ai_moss_gpu_pair_ticks(&a, &b, &ticks), "start notready sentinel rejected despite malformed availability1");
    a.ticks = 100;
    b.ticks = 130;
    need(!ps5_ai_moss_gpu_pair_ticks(nullptr, &b, &ticks), "null begin rejected");
    need(!ps5_ai_moss_gpu_pair_ticks(&a, nullptr, &ticks), "null end rejected");
    need(!ps5_ai_moss_gpu_pair_ticks(&a, &b, nullptr), "null duration output rejected");

    ps5_ai_moss_gpu_class_stat s{};
    need(ps5_ai_moss_gpu_accumulate(&s, 30), "first timer accepted");
    need(s.intervals == 1 && s.ticks == 30 && s.minimum_ticks == 30 && s.maximum_ticks == 30, "initial timer summary complete");
    need(ps5_ai_moss_gpu_accumulate(&s, 5) && ps5_ai_moss_gpu_accumulate(&s, 80) && ps5_ai_moss_gpu_accumulate(&s, 0), "varied and quantized-zero timers accepted");
    need(s.intervals == 4 && s.ticks == 115 && s.minimum_ticks == 0 && s.maximum_ticks == 80, "summary intervals/sum/min/max exact");
    need(!ps5_ai_moss_gpu_accumulate(nullptr, 1), "missing summary refused");
    s = {UINT64_MAX, 1, 1, 1}; auto old = s;
    need(!ps5_ai_moss_gpu_accumulate(&s, 2) && std::memcmp(&s, &old, sizeof(s)) == 0, "saturated count refused without mutation/wrap");
    s = {10, UINT64_MAX - 1, 1, 7}; old = s;
    need(!ps5_ai_moss_gpu_accumulate(&s, 2) && std::memcmp(&s, &old, sizeof(s)) == 0, "sum overflow refused without mutation/wrap");
    need(ps5_ai_moss_gpu_accumulate(&s, 1) && s.ticks == UINT64_MAX && s.intervals == 11, "exact sum boundary accepted");
    old = s;
    need(!ps5_ai_moss_gpu_accumulate(&s, 1) && std::memcmp(&s, &old, sizeof(s)) == 0, "next increment after saturated sum refused");
}

namespace complete_graph_metadata {
struct Op {ggml_tensor a{},b{},out{};};
struct Fixture {
    std::deque<Op> ops;
    std::vector<ggml_tensor*> nodes;
    Op& mm(const std::string& name,int64_t K,int64_t M,uint64_t N,ggml_type type=GGML_TYPE_Q4_K){
        ops.emplace_back();auto&o=ops.back();ggml_set_name(&o.a,name.c_str());o.a.type=type;
        o.a.ne[0]=K;o.a.ne[1]=M;o.a.ne[2]=o.a.ne[3]=1;
        o.b.type=GGML_TYPE_F32;o.b.ne[0]=K;o.b.ne[1]=N;o.b.ne[2]=o.b.ne[3]=1;
        o.out.type=GGML_TYPE_F32;o.out.op=GGML_OP_MUL_MAT;o.out.src[0]=&o.a;o.out.src[1]=&o.b;
        o.out.ne[0]=M;o.out.ne[1]=N;o.out.ne[2]=o.out.ne[3]=1;nodes.push_back(&o.out);return o;
    }
    void fa(){ops.emplace_back();ops.back().out.op=GGML_OP_FLASH_ATTN_EXT;nodes.push_back(&ops.back().out);}
    ps5_ai_moss_gpu_coverage coverage(){ggml_cgraph graph{};graph.n_nodes=int(nodes.size());graph.nodes=nodes.data();return ps5_ai_test_actual_graph_coverage(&graph);}
};
static const char* modules[]={"qkv","o","ffn_up_gate","ffn_down"};
static Fixture full(int skip,uint64_t N,bool lora=true){
    Fixture f;
    for(unsigned l=0;l<42;l++)if(int(l)!=skip)for(unsigned m=0;m<4;m++){
        auto base=std::string("model.diffusion_model.layers.")+std::to_string(l)+".proj."+modules[m]+".weight";
        f.mm(base,m==3?8192:3072,m==0?9216:m==2?16384:3072,N);
    }
    if(lora)for(const auto& entry:ps5_ai_moss_lora::expected()){
        auto marker=std::string(".layers.")+std::to_string(skip)+".proj.";
        if(skip>=0 && entry.first.find(marker)!=std::string::npos)continue;
        f.mm(entry.first,entry.second.input,entry.second.output,N,GGML_TYPE_F32);
    }
    // Complete forward also contains two text-refiner attention operations.
    for(unsigned i=0;i<(skip<0?44:43);i++)f.fa();
    return f;
}
static void qualification(){
    static_assert(GGML_MAX_NAME==160,"actual tensor name ABI");
    const uint64_t all=(UINT64_C(1)<<42)-1;
    for(uint64_t N:{uint64_t(256),uint64_t(304),uint64_t(4144)})for(int skip:{-1,22}){
        auto f=full(skip,N);auto c=f.coverage();unsigned count=skip<0?42:41;
        need(ps5_ai_moss_gpu_coverage_complete(&c),"actual complete positive/negative graph accepted");
        need(ps5_ai_moss_gpu_coverage_layers(&c)==count && c.tokens==N,"actual layer count/token metadata retained");
        need(c.masks[0]==(skip<0?all:all^(UINT64_C(1)<<22)),"exact layer mask");
        need(c.lora_down_nodes==6*count && c.lora_up_nodes==6*count,"LoRA GGMLnode coverage252/246 independent of dispatches");
        for(unsigned m=0;m<4;m++)need(c.q4_main_nodes[m]==count && c.masks[m]==c.masks[0],"all four module masks/counts complete");
        std::reverse(f.nodes.begin(),f.nodes.end());c=f.coverage();
        need(ps5_ai_moss_gpu_coverage_complete(&c),"graph ordering cannot spoof or lose metadata coverage");
        std::reverse(f.nodes.begin(),f.nodes.end());
        // Duplicated nodes cannot substitute for missing layers.
        for(unsigned m=0;m<4;m++){
            f.nodes.push_back(&f.ops[m].out);c=f.coverage();
            need(!ps5_ai_moss_gpu_coverage_complete(&c) && c.invalid>0,"duplicate canonical module rejected");f.nodes.pop_back();
        }
        auto&o=f.ops.front();
        const auto a=o.a,b=o.b,out=o.out;
        for(unsigned test=0;test<11;test++){
            o.a=a;o.b=b;o.out=out;
            switch(test){
                case 0:o.a.type=GGML_TYPE_F16;break;case 1:o.b.type=GGML_TYPE_F16;break;case 2:o.out.type=GGML_TYPE_F16;break;
                case 3:o.a.ne[2]=2;break;case 4:o.b.ne[2]=2;break;case 5:o.out.ne[3]=2;break;
                case 6:o.b.ne[0]++;break;case 7:o.out.ne[0]++;break;
                case 8:o.b.ne[1]++;break;case 9:o.b.ne[1]++;o.out.ne[1]++;break;case 10:o.out.op=GGML_OP_ADD;break;
            }
            c=f.coverage();need(!ps5_ai_moss_gpu_coverage_complete(&c),"wrong dtype/batch/geometry/token/op excludes incomplete main graph");
        }
        o.a=a;o.b=b;o.out=out;
        ggml_set_name(&o.a,"model.diffusion_model.layers.00.proj.qkv.weight");
        c=f.coverage();need(!ps5_ai_moss_gpu_coverage_complete(&c),"leadingzero canonical name cannot qualify");
        o.a=a;
        auto saved=f.nodes.front();f.nodes.erase(f.nodes.begin());c=f.coverage();
        need(!ps5_ai_moss_gpu_coverage_complete(&c),"third first-positive tail missing qkv layer0 rejected");f.nodes.insert(f.nodes.begin(),saved);
        c=f.coverage();need(ps5_ai_moss_gpu_coverage_complete(&c),"restoring exact graph restores eligibility");
    }
    auto wrong=full(21,304);auto c=wrong.coverage();
    need(!ps5_ai_moss_gpu_coverage_complete(&c),"41layers missing21 cannot impersonate negative skip22");
    Fixture prefix1;prefix1.mm("model.diffusion_model.text_embedder.blocks.0.qkv.weight",3072,9216,48);prefix1.fa();
    c=prefix1.coverage();need(!ps5_ai_moss_gpu_coverage_complete(&c)&&c.masks[0]==0&&c.lora_down_nodes==0,"first text-attention witness prefix rejected");
    Fixture prefix2;prefix2.mm("model.diffusion_model.text_embedder.blocks.1.qkv.weight",3072,9216,48);prefix2.fa();
    prefix2.mm("model.diffusion_model.layers.0.proj.qkv.weight",3072,9216,304);
    c=prefix2.coverage();need(!ps5_ai_moss_gpu_coverage_complete(&c)&&c.q4_main_nodes[0]==1&&c.lora_up_nodes==0,"second prefix through first baseqkv rejected");
    auto tail=full(-1,304);tail.nodes.erase(tail.nodes.begin());c=tail.coverage();
    need(!ps5_ai_moss_gpu_coverage_complete(&c)&&c.q4_main_nodes[0]==41&&c.q4_main_nodes[1]==42&&c.lora_up_nodes==252,"third tail rejected despite all252LoRA and remaining41qkv");
    auto nolora=full(-1,304,false);c=nolora.coverage();
    need(ps5_ai_moss_gpu_coverage_complete(&c)&&c.lora_up_nodes==0&&c.lora_down_nodes==0,"predicate is Q4 completeness; LoRA diagnostics are not silently asserted");
    Fixture empty;c=empty.coverage();need(!ps5_ai_moss_gpu_coverage_complete(&c),"empty graph rejected");
    need(!ps5_ai_moss_gpu_coverage_complete(nullptr),"null coverage rejected");
    ps5_ai_moss_gpu_coverage bad{};
    ps5_ai_moss_gpu_coverage_add(&bad,4,0,304);need(bad.invalid==1,"module out of range flagged");
    ps5_ai_moss_gpu_coverage_add(&bad,0,42,304);need(bad.invalid==2,"layer out of range flagged");
    ps5_ai_moss_gpu_coverage_add(&bad,0,0,0);need(bad.invalid==3,"empty token geometry flagged");
}
}

int main() {
    classification(); timer_ranges_and_values(); complete_graph_metadata::qualification();
#if !PS5_AI_MOSS_GPU_PROFILE
    ps5_ai_moss_gpu_profile_report r;
    std::memset(&r, 0xa5, sizeof(r));
    need(ps5_ai_moss_gpu_profile_gate_begin() == 0 && ps5_ai_moss_gpu_profile_job_begin() == 0, "default0 cannot arm Vulkan profiler");
    need(ps5_ai_moss_gpu_profile_gate_end(&r) == 0 && r.enabled == 0 && r.valid == 0, "default0 gate end clears report");
    std::memset(&r, 0xa5, sizeof(r));
    need(ps5_ai_moss_gpu_profile_job_end(&r) == 0 && r.enabled == 0 && r.valid == 0, "default0 job end clears report");
    std::memset(&r, 0xa5, sizeof(r)); ps5_ai_moss_gpu_profile_snapshot(&r);
    need(r.enabled == 0 && r.valid == 0 && r.active == 0, "default0 snapshot clears report");
    ps5_ai_moss_gpu_profile_snapshot(nullptr);
    need(ps5_ai_moss_gpu_profile_gate_end(nullptr) == 0 && ps5_ai_moss_gpu_profile_job_end(nullptr) == 0, "default0 null report safe");
#endif
    std::printf("PASS %u checks; selected504 metadata, exact160 name ABI, class counts, query ranges/availability/monotonicity/overflow; GPU=0 model=0 PS5=0 mode=%d\n", checks, PS5_AI_MOSS_GPU_PROFILE);
}
