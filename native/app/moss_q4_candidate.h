#ifndef PS5_AI_MOSS_Q4_CANDIDATE_H
#define PS5_AI_MOSS_Q4_CANDIDATE_H
#include <stdint.h>
#include <string.h>
#ifndef PS5_AI_MOSS_Q4_CANDIDATE_A
#define PS5_AI_MOSS_Q4_CANDIDATE_A 0
#endif
#if PS5_AI_MOSS_Q4_CANDIDATE_A != 0 && PS5_AI_MOSS_Q4_CANDIDATE_A != 1
#error MOSS Q4 candidate flag must be 0 or 1
#endif
#if PS5_AI_MOSS_Q4_CANDIDATE_A
#if PS5_AI_MOSS_F32_ACCUMULATION != 1 || PS5_AI_MOSS_PERFORMANCE_TRIAL != 1 || PS5_AI_MOSS_PERF_VARIANT != 1 || PS5_AI_MOSS_TRIAL_10G != 1 || PS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL != 1
#error MOSS Q4 candidate requires the preserved F32 accumulation/performance variant1/10GiB trial
#endif
#endif
/* Counts mean command recording; GPU completion/readback remain separate gates. */
struct ps5_ai_moss_q4_candidate_stat {
    char name[64];uint32_t format;uint64_t value_bits;
};
struct ps5_ai_moss_q4_candidate_pipeline {
    uint64_t pipeline, original_module_bytes, actual_module_bytes, compile_count;
    uint32_t specialization[13], required_subgroup, aligned, stats_available;
    uint32_t stats_recorded,stats_total;
    struct ps5_ai_moss_q4_candidate_stat stats[16];
};
struct ps5_ai_moss_q4_candidate_report {
    uint32_t schema,enabled,active;
    uint64_t selected_calls,layer_calls[42],invalid_receipts,device_generation;
    uint64_t baseline_calls,baseline_layer_calls[42];
    uint64_t token_calls[4],baseline_token_calls[4]; /* actual N:270,362,4110,4202 */
    struct ps5_ai_moss_q4_candidate_pipeline pipelines[4]; /* 0/1 A un/aligned, 2/3 baseline medium64 un/aligned */
};
static inline int ps5_ai_moss_q4_ffn_layer(const char *name,unsigned *out) {
    static const char prefix[]="model.diffusion_model.layers.";
    static const char suffix[]=".proj.ffn_up_gate.weight";
    unsigned n=0,digits=0;
    if(!name || strncmp(name,prefix,sizeof(prefix)-1))return 0;
    const char *p=name+sizeof(prefix)-1;
    if(*p=='0' && p[1]>='0' && p[1]<='9')return 0;
    while(*p>='0' && *p<='9') {if(++digits>2)return 0;n=10*n+(unsigned)(*p++-'0');}
    if(!digits || n>=42 || strcmp(p,suffix))return 0;
    if(out)*out=n;return 1;
}
#ifdef __cplusplus
extern "C" {
#endif
#if PS5_AI_MOSS_Q4_CANDIDATE_A
void ps5_ai_moss_q4_candidate_snapshot(struct ps5_ai_moss_q4_candidate_report*);
/* No wait or submit; rejects non-idle hardware. Root restores active1 after numeric baseline gate. */
int ps5_ai_moss_q4_candidate_set_enabled(unsigned);
#else
static inline int ps5_ai_moss_q4_candidate_set_enabled(unsigned value) {return value==0;}
static inline void ps5_ai_moss_q4_candidate_snapshot(struct ps5_ai_moss_q4_candidate_report *out) {if(out)memset(out,0,sizeof(*out));}
#endif
#ifdef __cplusplus
}
#endif
#endif
