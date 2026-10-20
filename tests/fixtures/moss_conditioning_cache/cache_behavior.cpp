#include <cstdlib>
#include <iostream>
#include "conditioning_cache.h"
unsigned checks=0;
void need(bool value,const char*why){++checks;if(!value){std::cerr<<"FAIL "<<why<<"\n";std::abort();}}
ConditionerParams input(){ConditionerParams p;p.text="same MOSS prompt";p.width=256;p.height=256;return p;}
void key_miss(const std::function<void(ConditionerParams&)>&change){
 LLMEmbedder m;ConditioningCache c;auto p=input();c.get(m,1,p);change(p);p.width=1024;p.height=768;c.get(m,1,p);need(m.calls==2,"changed preserved key must miss");}
void foreign_output(const std::function<void(SDCondition&)>&change){
 LLMEmbedder m;m.mutate=change;ConditioningCache c;auto p=input();c.get(m,1,p);c.get(m,1,p);need(m.calls==1,"foreign output retains normal same-size caching");
 p.width=1024;c.get(m,1,p);need(m.calls==2,"foreign condition cannot cross resolution");}
int main(){
 auto p=input();LLMEmbedder m;ConditioningCache c;auto first=c.get(m,1,p);need(m.calls==1,"initial miss");need(p.width==256&&p.height==256,"input not mutated");
 p.width=1024;p.height=768;auto hit=c.get(m,1,p);need(m.calls==1,"pure MOSS cross-resolution hit");need(first.c_crossattn.values()==hit.c_crossattn.values(),"reuse completed output");need(p.width==1024&&p.height==768,"hit input not mutated");
 need(m.seen[0].width==256&&m.seen[0].height==256,"encoder received actual original dimensions");need(p.ref_image_params.pass_to_dit,"normal pass_to_dit true works");
 c.clear();c.get(m,1,p);need(m.calls==2&&m.seen.back().width==1024&&m.seen.back().height==768,"clear invalidation and actual dims recompute");
 c.set_capacity(0);c.get(m,1,p);c.get(m,1,p);need(m.calls==4,"capacity0 unchanged");
 for(auto version:{VERSION_SD1,VERSION_SD2,VERSION_SDXL,VERSION_SD3,VERSION_FLUX,VERSION_QWEN_IMAGE,VERSION_ANIMA}){
  LLMEmbedder other;other.version=version;ConditioningCache cc;auto q=input();cc.get(other,1,q);cc.get(other,1,q);need(other.calls==1,"other model same-size cache");
  q.width=1024;cc.get(other,1,q);need(other.calls==2,"other model dimensions retained");
 }
 {OtherConditioner other;ConditioningCache cc;auto q=input();cc.get(other,1,q);q.height=768;cc.get(other,1,q);need(other.calls==2,"non-LLM conditioner unchanged");}
 {LLMEmbedder other;other.llm->enable_vision=true;ConditioningCache cc;auto q=input();cc.get(other,1,q);q.width=1024;cc.get(other,1,q);need(other.calls==2,"vision excludes cross-resolution");}
 {LLMEmbedder other;other.llm.reset();ConditioningCache cc;auto q=input();cc.get(other,1,q);q.width=1024;cc.get(other,1,q);need(other.calls==2,"missing runner excludes cross-resolution");}
 {LLMEmbedder a,b;ConditioningCache cc;auto q=input();cc.get(a,1,q);q.width=1024;cc.get(b,1,q);need(a.calls==1&&b.calls==1,"new conditioner identity cannot inherit resolution policy");}
 key_miss([](ConditionerParams&p){p.text+=" different";});
 key_miss([](ConditionerParams&p){p.clip_skip=2;});
 key_miss([](ConditionerParams&p){p.zero_out_masked=true;});
 key_miss([](ConditionerParams&p){p.ref_image_params.pass_to_vlm=true;});
 key_miss([](ConditionerParams&p){p.ref_image_params.pass_to_dit=false;});
 key_miss([](ConditionerParams&p){p.ref_image_params.ref_index_mode=Rope::RefIndexMode::INCREASE;});
 key_miss([](ConditionerParams&p){p.ref_image_params.force_ref_timestep_zero=true;});
 key_miss([](ConditionerParams&p){p.ref_image_params.resize_before_vae=false;});
 key_miss([](ConditionerParams&p){p.ref_image_params.vae_input_max_pixels=2048;});
 key_miss([](ConditionerParams&p){p.ref_image_params.vlm_resize_mode=RefImageResizeMode::LONGEST_SIDE;});
 key_miss([](ConditionerParams&p){p.ref_image_params.vlm_min_size=384;});
 key_miss([](ConditionerParams&p){p.ref_image_params.vlm_max_size=768;});
 key_miss([](ConditionerParams&p){p.ref_image_params.resize_vae_to_target=true;});

 {std::vector<sd::Tensor<float>> images;ConditioningCache cc;LLMEmbedder other;auto q=input();q.ref_images=&images;cc.get(other,1,q);q.width=1024;cc.get(other,1,q);need(other.calls==2,"even empty nonnull image ptr excludes");}
 {std::vector<MiniMaxH3PresentationItem> refs;ConditioningCache cc;LLMEmbedder other;auto q=input();q.minimax_h3_references=&refs;cc.get(other,1,q);q.width=1024;cc.get(other,1,q);need(other.calls==2,"even empty nonnull refs excludes");}
 {ConditioningCache cc;LLMEmbedder other;auto q=input();cc.get(other,1,q);std::vector<sd::Tensor<float>>images;q.ref_images=&images;cc.get(other,1,q);need(other.calls==2,"null versus empty image pointer remains key");}
 {ConditioningCache cc;LLMEmbedder other;auto q=input();cc.get(other,1,q);std::vector<MiniMaxH3PresentationItem>refs;q.minimax_h3_references=&refs;cc.get(other,1,q);need(other.calls==2,"null versus empty reference pointer remains key");}
 {ConditioningCache cc;LLMEmbedder other;auto q=input();std::vector<sd::Tensor<float>>images{{{1},{1.f}}};q.ref_images=&images;cc.get(other,1,q);images[0].values_[0]=2.f;cc.get(other,1,q);need(other.calls==2,"reference values compared after owned copy");images[0].shape_={2};cc.get(other,1,q);need(other.calls==3,"reference shape compared");}
 {ConditioningCache cc;LLMEmbedder other;auto q=input();std::vector<MiniMaxH3PresentationItem>refs(1);refs[0].kind=MiniMaxH3PresentationKind::IMAGE;refs[0].timestamps={0.f};refs[0].frames.push_back({{1},{1.f}});q.minimax_h3_references=&refs;cc.get(other,1,q);
 refs[0].kind=MiniMaxH3PresentationKind::VIDEO;cc.get(other,1,q);need(other.calls==2,"presentation kind key");refs[0].timestamps={1.f};cc.get(other,1,q);need(other.calls==3,"presentation times key");refs[0].frames[0].values_[0]=2.f;cc.get(other,1,q);need(other.calls==4,"presentation frame key");}
 foreign_output([](SDCondition&c){c.c_vector=sd::Tensor<float>({1},{1.f});});
 foreign_output([](SDCondition&c){c.c_concat=sd::Tensor<float>({1},{1.f});});
 foreign_output([](SDCondition&c){c.c_t5_ids=sd::Tensor<int32_t>({1},{1});});
 foreign_output([](SDCondition&c){c.c_t5_weights=sd::Tensor<float>({1},{1.f});});
 foreign_output([](SDCondition&c){c.c_input_ids=sd::Tensor<int32_t>({1},{1});});
 foreign_output([](SDCondition&c){c.c_position_ids=sd::Tensor<int32_t>({1},{1});});
 foreign_output([](SDCondition&c){c.c_token_types=sd::Tensor<int32_t>({1},{1});});
 foreign_output([](SDCondition&c){c.c_vinput_mask=sd::Tensor<int32_t>({1},{1});});
 foreign_output([](SDCondition&c){c.c_image_embeds.push_back({0,sd::Tensor<float>({1},{1.f})});});
 foreign_output([](SDCondition&c){c.c_ref_images.push_back(sd::Tensor<float>({1},{1.f}));});
 foreign_output([](SDCondition&c){c.c_ref_audios.push_back(sd::Tensor<float>({1},{1.f}));});
 foreign_output([](SDCondition&c){c.c_reference_blocks.push_back({});});
 foreign_output([](SDCondition&c){c.extra_c_crossattns.push_back(sd::Tensor<float>({1},{1.f}));});

 {LLMEmbedder other;other.mutate=[](SDCondition&c){c.c_crossattn={};};ConditioningCache cc;auto q=input();cc.get(other,1,q);q.width=1024;cc.get(other,1,q);need(other.calls==2,"empty condition not cached");}
 {ConditioningCache cc;cc.set_capacity(2);LLMEmbedder other;auto q=input();q.text="positive";cc.get(other,1,q);q.text="negative";cc.get(other,1,q);need(other.calls==2,"two LRU entries");
 q.text="positive";q.width=1024;cc.get(other,1,q);q.text="negative";cc.get(other,1,q);need(other.calls==2,"positive+negative reuse across sizes");
 q.text="third";cc.get(other,1,q);need(other.calls==3,"third miss");q.text="positive";cc.get(other,1,q);need(other.calls==4,"original LRU eviction unchanged");cc.set_capacity(1);q.text="negative";cc.get(other,1,q);need(other.calls==5,"capacity shrink respected");}
 std::cout<<"PASS "<<checks<<" actual-cache behavior checks; no model/GPU/PS5\n";
}
