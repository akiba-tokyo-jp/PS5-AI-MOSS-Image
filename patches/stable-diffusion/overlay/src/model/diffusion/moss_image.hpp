#ifndef __SD_MODEL_DIFFUSION_MOSS_IMAGE_HPP__
#define __SD_MODEL_DIFFUSION_MOSS_IMAGE_HPP__

#include <algorithm>
#include <cinttypes>

#include "model/common/ggml_block.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

#ifndef PS5_AI_MOSS_F32_ACCUMULATION
#define PS5_AI_MOSS_F32_ACCUMULATION 0
#endif
#if PS5_AI_MOSS_F32_ACCUMULATION != 0 && PS5_AI_MOSS_F32_ACCUMULATION != 1
#error "PS5_AI_MOSS_F32_ACCUMULATION must be 0 or 1"
#endif

#include "model/diffusion/moss_ffn_token_chunk.hpp"
#include "model/diffusion/moss_dit_ops.hpp"

namespace MOSSImage {
    // This MOSS-only policy changes accumulator precision, not weight dtype or
    // Vulkan's FP16 staging. Linear propagates it into runtime LoRA projections.
    inline std::shared_ptr<Linear> make_linear(int64_t in_features, int64_t out_features, bool bias) {
        return std::make_shared<Linear>(in_features, out_features, bias, false,
                                        PS5_AI_MOSS_F32_ACCUMULATION == 1);
    }

    struct MOSSImageConfig {
        int64_t dim      = 3072;
        int64_t ffn_dim  = 8192;
        int64_t text_dim = 4096;
        int64_t channels = 128;
        int64_t head_dim = 128;
        int64_t depth    = 42;
        int64_t freq_dim = 256;

        static MOSSImageConfig detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            MOSSImageConfig config;
            int64_t depth = 0;
            for (const auto& [name, tensor] : weights) {
                if (!starts_with(name, prefix + ".")) {
                    continue;
                }
                auto key = name.substr(prefix.size() + 1);
                if (key == "video_embedder.weight") {
                    config.channels = tensor.ne[0];
                    config.dim      = tensor.ne[1];
                } else if (key == "text_embedder.proj.weight") {
                    config.text_dim = tensor.ne[0];
                } else if (key == "layers.0.proj.ffn_down.weight") {
                    config.ffn_dim = tensor.ne[0];
                } else if (key == "layers.0.q_norm.weight") {
                    config.head_dim = tensor.ne[0];
                } else if (key == "time_embedding.0.weight") {
                    config.freq_dim = tensor.ne[0];
                }
                if (starts_with(key, "layers.")) {
                    depth = std::max<int64_t>(depth, std::stoi(key.substr(7)) + 1);
                }
            }
            if (depth) {
                config.depth = depth;
                LOG_VERBOSE("moss_image: dim = %" PRId64 ", ffn_dim = %" PRId64 ", depth = %" PRId64 ", head_dim = %" PRId64,
                            config.dim, config.ffn_dim, config.depth, config.head_dim);
            }
            return config;
        }
    };

    inline ggml_tensor* situ(ggml_context* c, ggml_tensor* hidden, ggml_tensor* gate, ggml_backend_t backend=nullptr) {
        if(moss_dit_opt::enabled(2))return moss_dit_opt::make(c,moss_dit_opt::SITU,hidden,gate,gate,nullptr,backend);
        auto a = ggml_scale(c, ggml_tanh(c, ggml_scale(c, gate, 0.25f)), 4.f);
        a      = ggml_mul(c, a, ggml_sigmoid(c, gate));
        auto b = ggml_scale(c, ggml_tanh(c, ggml_scale(c, hidden, 1.f / 25.f)), 25.f);
        return ggml_mul(c, a, b);
    }

    inline ggml_tensor* fused_norm_mod(GGMLRunnerContext* ctx,ggml_tensor* weighted,ggml_tensor* mod,unsigned offset) {
        if(weighted->op!=GGML_OP_MUL || weighted->src[0]->op!=GGML_OP_RMS_NORM)throw std::runtime_error("MOSS norm decomposition differs");
        return moss_dit_opt::make(ctx->ggml_ctx,offset?moss_dit_opt::NORM_MOD3:moss_dit_opt::NORM_MOD0,weighted->src[0],weighted->src[1],mod,nullptr,ctx->backend);
    }
    inline ggml_tensor* attention(GGMLRunnerContext* ctx,ggml_tensor* q,ggml_tensor* k,ggml_tensor* v,ggml_tensor* pe) {
        if(!moss_dit_opt::enabled(4))return Rope::attention(ctx,q,k,v,pe,nullptr,1.f,false);
        const auto heads=q->ne[1];
        auto fused=[&](ggml_tensor* t){
            if(t->op!=GGML_OP_MUL || t->src[0]->op!=GGML_OP_RMS_NORM)throw std::runtime_error("MOSS QK norm decomposition differs");
            return moss_dit_opt::make(ctx->ggml_ctx,moss_dit_opt::QK_WEIGHT_ROPE,t->src[0],t->src[1],pe,nullptr,ctx->backend);
        };
        return ggml_ext_attention_ext(ctx,fused(q),fused(k),v,heads,nullptr,true,ctx->flash_attn_enabled,1.f);
    }

    struct TextBlock : public GGMLBlock {
        MOSSImageConfig config;

        TextBlock(const MOSSImageConfig& config)
            : config(config) {
            blocks["norm1"]    = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["norm2"]    = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["q_norm"]   = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
            blocks["k_norm"]   = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
            blocks["qkv"]      = make_linear(config.dim, config.dim * 3, true);
            blocks["proj"]     = make_linear(config.dim, config.dim, true);
            blocks["ffn_up"]   = make_linear(config.dim, config.ffn_dim, true);
            blocks["ffn_gate"] = make_linear(config.dim, config.ffn_dim, true);
            blocks["ffn_down"] = make_linear(config.ffn_dim, config.dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe) {
            auto c      = ctx->ggml_ctx;
            auto linear = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            auto norm   = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, h); };
            auto y      = norm("norm1", x);
            auto qkv    = linear("qkv", y);
            auto split  = [&](int i) {
                return ggml_reshape_4d(c, ggml_ext_slice(c, qkv, 0, i * config.dim, (i + 1) * config.dim), config.head_dim, config.dim / config.head_dim, x->ne[1], x->ne[2]);
            };
            auto q = norm("q_norm", split(0));
            auto k = norm("k_norm", split(1));
            auto v = split(2);
            y      = attention(ctx, q, k, v, pe);
            y      = linear("proj", y);
            x      = ggml_add(c, x, y);
            y      = norm("norm2", x);
            y      = situ(c, linear("ffn_up", y), linear("ffn_gate", y),ctx->backend);
            y      = linear("ffn_down", y);
            return ggml_add(c, x, y);
        }
    };

    struct Block : public GGMLBlock {
        MOSSImageConfig config;

        Block(const MOSSImageConfig& config)
            : config(config) {
            blocks["norm1"]            = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["norm2"]            = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["q_norm"]           = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
            blocks["k_norm"]           = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
            blocks["proj.qkv"]         = make_linear(config.dim, config.dim * 3, false);
            blocks["proj.o"]           = make_linear(config.dim, config.dim, false);
            blocks["proj.ffn_up_gate"] = make_linear(config.dim, config.ffn_dim * 2, false);
            blocks["proj.ffn_down"]    = make_linear(config.ffn_dim, config.dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe, ggml_tensor* mod) {
            auto c      = ctx->ggml_ctx;
            auto linear = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            auto norm   = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, h); };
            auto part   = [&](int i) { return ggml_ext_slice(c, mod, 0, i * config.dim, (i + 1) * config.dim); };
            auto y      = norm("norm1", x);
            y           = moss_dit_opt::enabled(4)?fused_norm_mod(ctx,y,mod,0):ggml_add(c, ggml_add(c, y, ggml_mul(c, y, part(1))), part(0));
            auto qkv    = linear("proj.qkv", y);
            auto split  = [&](int i) {
                return ggml_reshape_4d(c, ggml_ext_slice(c, qkv, 0, i * config.dim, (i + 1) * config.dim), config.head_dim, config.dim / config.head_dim, x->ne[1], x->ne[2]);
            };
            auto q       = norm("q_norm", split(0));
            auto k       = norm("k_norm", split(1));
            auto v       = split(2);
            y            = attention(ctx, q, k, v, pe);
            y            = linear("proj.o", y);
            x            = moss_dit_opt::enabled(4)?moss_dit_opt::make(c,moss_dit_opt::GATED_RESIDUAL,x,y,part(2),nullptr,ctx->backend):ggml_add(c,x,ggml_mul(c,y,part(2)));
            y            = norm("norm2", x);
            y            = moss_dit_opt::enabled(4)?fused_norm_mod(ctx,y,mod,3):ggml_add(c, ggml_add(c, y, ggml_mul(c, y, part(4))), part(3));
            auto full_ffn = [&](ggml_tensor* tokens) {
                auto up_gate = linear("proj.ffn_up_gate", tokens);
                auto hidden = situ(c, ggml_ext_slice(c, up_gate, 0, 0, config.ffn_dim), ggml_ext_slice(c, up_gate, 0, config.ffn_dim, 2 * config.ffn_dim),ctx->backend);
                return linear("proj.ffn_down", hidden);
            };
            y = moss_ffn_token_chunk::forward(c, y, full_ffn);
            return moss_dit_opt::enabled(4)?moss_dit_opt::make(c,moss_dit_opt::GATED_RESIDUAL,x,y,part(5),nullptr,ctx->backend):ggml_add(c,x,ggml_mul(c,y,part(5)));
        }
    };

    struct Model : public GGMLBlock {
        MOSSImageConfig config;

        Model(const MOSSImageConfig& config)
            : config(config) {
            blocks["video_embedder"]     = make_linear(config.channels, config.dim, true);
            blocks["text_embedder.proj"] = make_linear(config.text_dim, config.dim, true);
            for (int i = 0; i < 2; ++i) {
                blocks["text_embedder.blocks." + std::to_string(i)] = std::make_shared<TextBlock>(config);
            }
            blocks["time_embedding.0"]        = make_linear(config.freq_dim, config.dim, true);
            blocks["time_embedding.2"]        = make_linear(config.dim, config.dim, true);
            blocks["shared_block_modulation"] = make_linear(config.dim, 6 * config.dim, true);
            blocks["video_head_modulation"]   = make_linear(config.dim, 2 * config.dim, true);
            for (int64_t i = 0; i < config.depth; ++i) {
                blocks["layers." + std::to_string(i)] = std::make_shared<Block>(config);
            }
            blocks["final_norm_video"]   = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["final_linear_video"] = make_linear(config.dim, config.channels, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* image, ggml_tensor* time, ggml_tensor* text, ggml_tensor* pe, const std::vector<int>& skip, bool refined_text=false) {
            auto c            = ctx->ggml_ctx;
            auto linear       = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            int64_t image_len = image->ne[0] * image->ne[1];
            auto x            = linear("video_embedder", DiT::patchify(c, image, 1, 1));
            if(!refined_text){
                text = linear("text_embedder.proj",text);
                auto text_pe=ggml_ext_slice(c,pe,3,image_len,image_len+text->ne[1]);
                for(int i=0;i<2;++i)text=std::dynamic_pointer_cast<TextBlock>(blocks["text_embedder.blocks."+std::to_string(i)])->forward(ctx,text,text_pe);
                if(moss_dit_opt::enabled(1))ctx->persist_cache_tensor("moss_internal_refined_text",text);
            }
            time      = linear("time_embedding.2", ggml_silu(c, linear("time_embedding.0", time)));
            auto mod  = linear("shared_block_modulation", time);
            auto head = linear("video_head_modulation", time);
            x         = ggml_concat(c, x, text, 1);
            for (int64_t i = 0; i < config.depth; ++i) {
                if (std::find(skip.begin(), skip.end(), i) == skip.end()) {
                    x = std::dynamic_pointer_cast<Block>(blocks["layers." + std::to_string(i)])->forward(ctx, x, pe, mod);
                    sd::ggml_graph_cut::mark_graph_cut(x, "moss_image.layers." + std::to_string(i), "x");
                }
            }
            x          = ggml_ext_slice(c, x, 1, 0, image_len);
            x          = std::dynamic_pointer_cast<RMSNorm>(blocks["final_norm_video"])->forward(ctx, x);
            auto shift = ggml_ext_slice(c, head, 0, 0, config.dim);
            auto scale = ggml_ext_slice(c, head, 0, config.dim, 2 * config.dim);
            x          = ggml_add(c, ggml_add(c, x, ggml_mul(c, x, scale)), shift);
            x          = linear("final_linear_video", x);
            return DiT::unpatchify(c, x, image->ne[1], image->ne[0], 1, 1);
        }
    };

    struct MOSSImageRunner : public DiffusionModelRunner {
        MOSSImageConfig config;
        Model model;
        std::vector<float> pe_data, time_data;
        std::vector<float> cache_text_key;
        int64_t cache_h=0,cache_w=0,cache_nt=0;
        void clear_dit_cache(){free_cache_ctx_and_buffer();cache_text_key.clear();cache_h=cache_w=cache_nt=0;}
        ~MOSSImageRunner(){moss_dit_opt::unregister_cache(this);}

        MOSSImageRunner(ggml_backend_t backend, const String2TensorStorage& weights, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager), config(MOSSImageConfig::detect_from_weights(weights, prefix)), model(config) {
            model.init(params_ctx, weights, prefix);
#if PS5_AI_MOSS_DIT_OPT_TRIAL
            moss_dit_opt::register_cache(this,[](void* p){static_cast<MOSSImageRunner*>(p)->clear_dit_cache();});
#endif
        }

        std::string get_desc() override { return "MOSS-Image"; }
        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& inputs) override {
            if (!inputs.x || !inputs.context || !inputs.timesteps || inputs.context->empty()) {
                LOG_ERROR("MOSS-Image requires latents, timesteps and text features");
                return {};
            }
            std::vector<int> skip;
            if (auto extra = std::get_if<SkipLayerDiffusionExtra>(&inputs.extra)) {
                if (extra->skip_layers) {
                    skip = *extra->skip_layers;
                }
            }
            auto graph = [&]() {
                auto gf   = new_graph_custom(65536);
                auto x    = make_input(*inputs.x);
                auto text = make_input(*inputs.context);
                GGML_ASSERT(x->ne[3] == 1 && x->ne[2] == config.channels && text->ne[0] == config.text_dim);
                int64_t h = x->ne[1], w = x->ne[0], nt = text->ne[1];
                auto cached_pe=moss_dit_opt::enabled(1)?get_cache_tensor_by_name("moss_internal_pe"):nullptr;
                auto cached_text=moss_dit_opt::enabled(1)?get_cache_tensor_by_name("moss_internal_refined_text"):nullptr;
                const bool hit=cached_pe&&cached_text;
                if(moss_dit_opt::enabled(1)){
                    const auto& key=inputs.context->values();
                    if(hit){
                        if(cache_h!=h||cache_w!=w||cache_nt!=nt||cache_text_key.size()!=key.size()||std::memcmp(cache_text_key.data(),key.data(),key.size()*sizeof(float)))throw std::runtime_error("MOSS per-generation invariant cache key changed");
                        ++moss_dit_opt::cache_hits;
                    }else{
                        if(cached_pe||cached_text)throw std::runtime_error("Partial MOSS invariant cache");
                        cache_h=h;cache_w=w;cache_nt=nt;cache_text_key.assign(key.begin(),key.end());++moss_dit_opt::cache_misses;
                    }
                }
                if(!hit){
                ++moss_dit_opt::rope_builds;
                Rope::Embedding embedding;
                embedding.positions.append_image(static_cast<int>(h), static_cast<int>(w));
                embedding.positions.images.back().height_axis = 2;
                embedding.positions.images.back().width_axis  = 3;
                embedding.positions.append_tokens(nt);
                for (int64_t token = 0; token < h * w + nt; ++token) {
                    float pos[4] = {0.f, float(token - h * w), 0.f, 0.f};
                    if (token < h * w) {
                        pos[1] = float(nt);
                        pos[2] = float(token / w) - float(h - 1) / 2.f;
                        pos[3] = float(token % w) - float(w - 1) / 2.f;
                    }
                    embedding.ids.push_back({pos[0], pos[1], pos[2], pos[3]});
                }
                embedding.values = Rope::embed_nd(embedding.ids, 1, 10000.f, std::vector<int>(4, static_cast<int>(config.head_dim / 4)), embedding.layout, &embedding.frequencies);
                pe_data          = finish_rope_pe(std::move(embedding));
                }
                time_data.resize(config.freq_dim);
                for (int64_t i = 0; i < config.freq_dim / 2; ++i) {
                    double angle                       = inputs.timesteps->values()[0] * std::pow(10000.0, -double(i) / double(config.freq_dim / 2));
                    time_data[i]                       = float(std::cos(angle));
                    time_data[i + config.freq_dim / 2] = float(std::sin(angle));
                }
                auto pe   = hit?cached_pe:ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.head_dim / 2, h * w + nt);
                auto time = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_F32, config.freq_dim, 1, 1);
                if(!hit)set_backend_tensor_data(pe, pe_data.data());
                set_backend_tensor_data(time, time_data.data());
                auto ctx = get_context(gf);
                if(moss_dit_opt::enabled(1)&&!hit)ctx.persist_cache_tensor("moss_internal_pe",pe);
                ggml_build_forward_expand(gf, model.forward(&ctx, x, time, hit?cached_text:text, pe, skip,hit));
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(graph, n_threads, false), inputs.x->dim());
        }
    };
}  // namespace MOSSImage

#endif  // __SD_MODEL_DIFFUSION_MOSS_IMAGE_HPP__
