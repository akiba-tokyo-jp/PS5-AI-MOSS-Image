/* SPDX-License-Identifier: MIT
 * Bounded diagnostic GPU timestamp intervals; not an exclusive whole-model profile.
 * No Vulkan calls in this header. The optional backend owns the query pool.
 */
#ifndef PS5_AI_MOSS_GPU_PROFILE_H
#define PS5_AI_MOSS_GPU_PROFILE_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "moss_dit_gpu_profile_policy.h"
#ifndef PS5_AI_MOSS_GPU_PROFILE
#define PS5_AI_MOSS_GPU_PROFILE 0
#endif
#if PS5_AI_MOSS_GPU_PROFILE != 0 && PS5_AI_MOSS_GPU_PROFILE != 1
#error PS5_AI_MOSS_GPU_PROFILE_must_be_zero_or_one
#endif
#if PS5_AI_MOSS_GPU_PROFILE && (!PS5_AI_MOSS_F32_ACCUMULATION || !PS5_AI_MOSS_PERFORMANCE_TRIAL || PS5_AI_MOSS_PERF_VARIANT != 1 || !PS5_AI_MOSS_TRIAL_10G || !PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL)
#error MOSS_GPU_profile_requires_explicit_FP16_F32acc_10GiB_performance_trial
#endif
#define PS5_AI_MOSS_GPU_PROFILE_MAX_NODES 8192u
#define PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES (2u * PS5_AI_MOSS_GPU_PROFILE_MAX_NODES)
#define PS5_AI_MOSS_GPU_PROFILE_GRAPHS 2u
#define PS5_AI_MOSS_GPU_PROFILE_JOBS 4u
#if PS5_AI_MOSS_DIT_GPU_PROFILE
#define PS5_AI_MOSS_GPU_PROFILE_CLASSES 25u
#define PS5_AI_MOSS_GPU_SHADER_CLASSES 8u
#else
#define PS5_AI_MOSS_GPU_PROFILE_CLASSES 12u
#endif
#ifdef __cplusplus
extern "C" {
#endif
enum ps5_ai_moss_gpu_class {
    PS5_AI_GPU_Q4_QKV, PS5_AI_GPU_Q4_O, PS5_AI_GPU_Q4_FFN_UP,
    PS5_AI_GPU_Q4_FFN_DOWN, PS5_AI_GPU_Q4_OTHER,
    PS5_AI_GPU_LORA_DOWN, PS5_AI_GPU_LORA_UP, PS5_AI_GPU_OTHER_MM,
    PS5_AI_GPU_ATTENTION, PS5_AI_GPU_SCALE, PS5_AI_GPU_ADD, PS5_AI_GPU_OTHER
#if PS5_AI_MOSS_DIT_GPU_PROFILE
    , PS5_AI_GPU_RMS, PS5_AI_GPU_CAST, PS5_AI_GPU_CONT, PS5_AI_GPU_COPY,
    PS5_AI_GPU_LAYOUT, PS5_AI_GPU_SITU, PS5_AI_GPU_NORM_MOD,
    PS5_AI_GPU_RESIDUAL, PS5_AI_GPU_QK_ROPE, PS5_AI_GPU_LORA_ADD,
    PS5_AI_GPU_LORA_PACK, PS5_AI_GPU_ROPE, PS5_AI_GPU_MUL
#endif
};
struct ps5_ai_moss_gpu_class_stat {
    uint64_t intervals, ticks, minimum_ticks, maximum_ticks;
};
struct ps5_ai_moss_gpu_graph_stat {
    uint64_t ordinal, nodes, recorded_nodes, intervals, queries_written, queries_available;
    uint64_t complete_main_graph, canonical_layer_count, canonical_layer_mask;
    uint64_t q4_main_nodes[4], lora_down_nodes, lora_up_nodes, tokens;
    uint64_t zero_intervals, invalid_intervals, overflow, dispatches, submits, fences;
    uint64_t cpu_wall_ns;
    struct ps5_ai_moss_gpu_class_stat classes[PS5_AI_MOSS_GPU_PROFILE_CLASSES];
#if PS5_AI_MOSS_DIT_GPU_PROFILE
    /* Secondary attribution of the SAME intervals, never an additional total. */
    struct ps5_ai_moss_gpu_class_stat shader_classes[PS5_AI_MOSS_GPU_PROFILE_CLASSES][PS5_AI_MOSS_GPU_SHADER_CLASSES];
#endif
};
struct ps5_ai_moss_gpu_profile_report {
    uint32_t schema, enabled, valid, startup_gate, active, gate_verified;
    uint64_t generation_ordinal, graphs_seen, graphs_sampled, unprofiled_graphs;
    uint64_t full_graphs_seen, skipped_partial_views;
    uint64_t errors, capacity_errors, unavailable_queries, invalid_intervals;
    uint64_t result_calls, cache_flush_calls, query_pool_queries, query_pool_bytes;
    uint64_t cpu_fixed_bytes, device_generation, query_reset_commands;
    uint32_t timestamp_valid_bits, reserved;
    double timestamp_period_ns;
    struct ps5_ai_moss_gpu_graph_stat graphs[PS5_AI_MOSS_GPU_PROFILE_GRAPHS];
};
struct ps5_ai_moss_gpu_query_value { uint64_t ticks, available; };
static inline const char *ps5_ai_moss_gpu_class_name(unsigned id) {
    static const char * const names[PS5_AI_MOSS_GPU_PROFILE_CLASSES] = {
        "q4_qkv", "q4_o", "q4_ffn_up_gate", "q4_ffn_down", "q4_other",
        "lora_down_f32", "lora_up_f32", "other_matmul", "attention",
        "scale", "add", "other"
#if PS5_AI_MOSS_DIT_GPU_PROFILE
        , "rms_norm", "cast", "contiguous", "copy", "layout", "fused_situ",
        "fused_norm_mod", "fused_residual", "fused_qk_rope", "fused_lora_add",
        "fused_lora_pack", "rope", "mul"
#endif
    };
    return id < PS5_AI_MOSS_GPU_PROFILE_CLASSES ? names[id] : "invalid";
}
#if PS5_AI_MOSS_DIT_GPU_PROFILE
static inline unsigned ps5_ai_moss_gpu_shader_class(const char *name) {
    if(!name)return 7;
    if(!strcmp(name,"split_k_reduce")||!strcmp(name,"fa_split_k_reduce"))return 3;
    if(!strncmp(name,"matmul_split_k_reduce",21)||!strncmp(name,"reduce_",7)||!strncmp(name,"sum_rows",8))return 3;
    if(!strncmp(name,"matmul",6)||!strncmp(name,"mul_mat",7))return 0;
    if(!strncmp(name,"flash_attn",10))return 1;
    if(!strncmp(name,"dequant",7)||!strncmp(name,"copy",4)||!strncmp(name,"cpy",3)||!strncmp(name,"cont",4))return 2;
    if(!strncmp(name,"rms_norm",8))return 4;
    if(!strcmp(name,"moss_dit_fused_f32"))return 5;
    if(!strncmp(name,"add",3)||!strncmp(name,"mul",3)||!strncmp(name,"scale",5)||
       !strncmp(name,"unary",5)||!strncmp(name,"silu",4)||!strncmp(name,"tanh",4)||
       !strncmp(name,"sigmoid",7)||!strncmp(name,"rope",4))return 6;
    return 7;
}
static inline const char *ps5_ai_moss_gpu_shader_name(unsigned id) {
    static const char *const names[PS5_AI_MOSS_GPU_SHADER_CLASSES]={"matrix","attention","conversion_copy","reduction","rms_norm","moss_fused","pointwise","other"};
    return id<PS5_AI_MOSS_GPU_SHADER_CLASSES?names[id]:"invalid";
}
#endif
/* Exact layer grammar: no partial names, out-of-range layers or leadingzero aliases. */
static inline const char *ps5_ai_moss_gpu_layer_tail(const char *name, int lora) {
    const char *prefix = lora ? "lora.model.diffusion_model.layers." : "model.diffusion_model.layers.";
    if (!name || strncmp(name,prefix,strlen(prefix))) return NULL;
    const char *p=name+strlen(prefix); unsigned layer=0, digits=0;
    if (*p=='0' && p[1]!='.') return NULL;
    while (*p>='0' && *p<='9') { if (++digits>2) return NULL; layer=layer*10+(unsigned)(*p++-'0'); }
    if (!digits || layer>=42 || strncmp(p,".proj.",6)) return NULL;
    return p+6;
}
/* Actual F32 src0 names and tensor geometry must agree with the selected504LoRAs.
 * caller must also check actual MUL_MAT op/src1+dstF32/2D weight, not a name alone.
 */
static inline unsigned ps5_ai_moss_gpu_lora_class(const char *name, int64_t k, int64_t m) {
    const char *p=ps5_ai_moss_gpu_layer_tail(name,1); if(!p) return PS5_AI_GPU_OTHER_MM;
    const int64_t rank=PS5_AI_MOSS_DIT_GPU_PROFILE?144:16;
    const char *tails[]={"qkv.weight", "qkv.weight.1", "qkv.weight.2", "o.weight", "ffn_up_gate.weight", "ffn_down.weight"};
    for(unsigned i=0;i<6;i++){
        size_t n=strlen(tails[i]); if(strncmp(p,tails[i],n))continue;
        const int64_t input=i==5?8192:3072, output=i==4?16384:3072;
        if(!strcmp(p+n,".lora_down") && k==input && m==rank) return PS5_AI_GPU_LORA_DOWN;
        if(!strcmp(p+n,".lora_up") && k==rank && m==output) return PS5_AI_GPU_LORA_UP;
    }
    return PS5_AI_GPU_OTHER_MM;
}
static inline unsigned ps5_ai_moss_gpu_q4_class(const char *name,int64_t k,int64_t m) {
    const char *p=ps5_ai_moss_gpu_layer_tail(name,0); if(!p)return PS5_AI_GPU_Q4_OTHER;
    if(!strcmp(p,"qkv.weight") && k==3072 && m==9216)return PS5_AI_GPU_Q4_QKV;
    if(!strcmp(p,"o.weight") && k==3072 && m==3072)return PS5_AI_GPU_Q4_O;
    if(!strcmp(p,"ffn_up_gate.weight") && k==3072 && m==16384)return PS5_AI_GPU_Q4_FFN_UP;
    if(!strcmp(p,"ffn_down.weight") && k==8192 && m==3072)return PS5_AI_GPU_Q4_FFN_DOWN;
    return PS5_AI_GPU_Q4_OTHER;
}
/* Graph completeness is metadata proof, separate from shaderdispatch counts.
 * Firstforward witness callbacks partition one runnergraph into three views.
 * Their prefixes/tail must not masquerade as complete model forwards.
 */
struct ps5_ai_moss_gpu_coverage {
    uint64_t masks[4], q4_main_nodes[4], lora_down_nodes, lora_up_nodes, tokens;
    uint64_t invalid;
};
static inline int ps5_ai_moss_gpu_layer_index(const char *name,int lora,unsigned *out) {
    if(!out||!ps5_ai_moss_gpu_layer_tail(name,lora))return 0;
    const char *p=name+strlen(lora?"lora.model.diffusion_model.layers.":"model.diffusion_model.layers.");
    unsigned layer=0;while(*p>='0'&&*p<='9')layer=layer*10+(unsigned)(*p++-'0');
    *out=layer;return 1;
}
static inline void ps5_ai_moss_gpu_coverage_add(struct ps5_ai_moss_gpu_coverage *c,
                                                unsigned module,unsigned layer,uint64_t tokens) {
    if(!c)return;
    if(module>=4||layer>=42||!tokens){++c->invalid;return;}
    const uint64_t bit=UINT64_C(1)<<layer;
    if(c->masks[module]&bit)++c->invalid;
    if(c->tokens&&c->tokens!=tokens)++c->invalid;
    c->tokens=tokens;c->masks[module]|=bit;++c->q4_main_nodes[module];
}
static inline unsigned ps5_ai_moss_gpu_coverage_layers(const struct ps5_ai_moss_gpu_coverage *c) {
    uint64_t mask=c?c->masks[0]:0;unsigned n=0;while(mask){n+=(unsigned)(mask&1);mask>>=1;}return n;
}
static inline int ps5_ai_moss_gpu_coverage_complete(const struct ps5_ai_moss_gpu_coverage *c) {
    const uint64_t all=(UINT64_C(1)<<42)-1,negative=all^(UINT64_C(1)<<22);
    if(!c||c->invalid||!c->tokens||(c->masks[0]!=all&&c->masks[0]!=negative))return 0;
    const unsigned layers=c->masks[0]==all?42:41;
#if PS5_AI_MOSS_DIT_GPU_PROFILE
    if(layers!=42||c->lora_down_nodes!=252||c->lora_up_nodes!=252)return 0;
#endif
    for(unsigned i=0;i<4;i++)if(c->masks[i]!=c->masks[0]||c->q4_main_nodes[i]!=layers)return 0;
    return 1;
}

static inline int ps5_ai_moss_gpu_range_valid(uint64_t first,uint64_t count,uint64_t capacity) {
    return count && first<=capacity && count<=capacity-first && capacity<=PS5_AI_MOSS_GPU_PROFILE_MAX_QUERIES;
}
static inline int ps5_ai_moss_gpu_pair_ticks(const struct ps5_ai_moss_gpu_query_value *a,
                                           const struct ps5_ai_moss_gpu_query_value *b,uint64_t *ticks) {
    if(!a||!b||!ticks||a->available!=1||b->available!=1||!a->ticks||!b->ticks||a->ticks==UINT64_MAX||b->ticks==UINT64_MAX||b->ticks<a->ticks)return 0;
    *ticks=b->ticks-a->ticks;return 1;
}
static inline int ps5_ai_moss_gpu_accumulate(struct ps5_ai_moss_gpu_class_stat *s,uint64_t ticks) {
    if(!s||s->intervals==UINT64_MAX||ticks>UINT64_MAX-s->ticks)return 0;
    if(!s->intervals||ticks<s->minimum_ticks)s->minimum_ticks=ticks;
    if(!s->intervals||ticks>s->maximum_ticks)s->maximum_ticks=ticks;
    s->intervals++;s->ticks+=ticks;return 1;
}
#if PS5_AI_MOSS_GPU_PROFILE
/* Serialized engine calls only. Begin returns1 on accepted arm; end returns1
 * only for complete validated sampling. End always disarms, including failure.
 * Gate samples one existing64MiB SCALEgraph. Job samples first2 actualDiTgraphs.
 */
int ps5_ai_moss_gpu_profile_gate_begin(void);
int ps5_ai_moss_gpu_profile_gate_end(struct ps5_ai_moss_gpu_profile_report *out);
int ps5_ai_moss_gpu_profile_job_begin(void);
int ps5_ai_moss_gpu_profile_job_end(struct ps5_ai_moss_gpu_profile_report *out);
void ps5_ai_moss_gpu_profile_snapshot(struct ps5_ai_moss_gpu_profile_report *out);
/* Called only immediately around real dispatch command recording. */
unsigned ps5_ai_moss_gpu_profile_dispatch_begin(uint64_t command_buffer);
#if PS5_AI_MOSS_DIT_GPU_PROFILE
unsigned ps5_ai_moss_gpu_profile_dispatch_begin_named(uint64_t command_buffer,const char *pipeline_name);
#endif
void ps5_ai_moss_gpu_profile_dispatch_end(uint64_t command_buffer,unsigned cookie);
#else
static inline int ps5_ai_moss_gpu_profile_gate_begin(void){return 0;}
static inline int ps5_ai_moss_gpu_profile_gate_end(struct ps5_ai_moss_gpu_profile_report *out){if(out)memset(out,0,sizeof(*out));return 0;}
static inline int ps5_ai_moss_gpu_profile_job_begin(void){return 0;}
static inline int ps5_ai_moss_gpu_profile_job_end(struct ps5_ai_moss_gpu_profile_report *out){if(out)memset(out,0,sizeof(*out));return 0;}
static inline void ps5_ai_moss_gpu_profile_snapshot(struct ps5_ai_moss_gpu_profile_report *out){if(out)memset(out,0,sizeof(*out));}
#endif
#ifdef __cplusplus
}
#endif
#endif
