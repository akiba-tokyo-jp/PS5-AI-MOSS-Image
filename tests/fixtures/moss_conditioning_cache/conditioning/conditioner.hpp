#pragma once
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <utility>
#include <stdexcept>
namespace sd { template<class T>struct Tensor {
 std::vector<int64_t> shape_;std::vector<T> values_;
 Tensor()=default;Tensor(std::initializer_list<int64_t>s,std::initializer_list<T>v):shape_(s),values_(v){}
 bool empty()const{return values_.empty();}const auto&shape()const{return shape_;}const auto&values()const{return values_;}
};}
namespace Rope {enum class RefIndexMode {FIXED,INCREASE,DECREASE};}
enum class RefImageResizeMode {NONE,AREA,LONGEST_SIDE};
struct MiniMaxH3ReferenceBlock {};
#define LOG_INFO(...) do {} while(0)
#define LOG_VERBOSE(...) do {} while(0)
enum SDVersion {
    VERSION_SD1,
    VERSION_SD1_INPAINT,
    VERSION_SD1_PIX2PIX,
    VERSION_SD1_TINY_UNET,
    VERSION_SD2,
    VERSION_SD2_INPAINT,
    VERSION_SD2_TINY_UNET,
    VERSION_SDXS_512_DS,
    VERSION_SDXS_09,
    VERSION_SDXL,
    VERSION_SDXL_INPAINT,
    VERSION_SDXL_PIX2PIX,
    VERSION_SDXL_VEGA,
    VERSION_SDXL_SSD1B,
    VERSION_SVD,
    VERSION_SD3,
    VERSION_FLUX,
    VERSION_FLUX_FILL,
    VERSION_FLUX_CONTROLS,
    VERSION_FLEX_2,
    VERSION_CHROMA_RADIANCE,
    VERSION_WAN2,
    VERSION_WAN2_2_I2V,
    VERSION_WAN2_2_TI2V,
    VERSION_WAN2_2_S2V,
    VERSION_LINGBOT_VIDEO,
    VERSION_QWEN_IMAGE,
    VERSION_QWEN_IMAGE_LAYERED,
    VERSION_QWEN_IMAGE_2_1,
    VERSION_HUNYUAN_VIDEO,
    VERSION_ANIMA,
    VERSION_FLUX2,
    VERSION_FLUX2_KLEIN,
    VERSION_LTXAV,
    VERSION_MINIMAX_H3,
    VERSION_HIDREAM_O1,
    VERSION_Z_IMAGE,
    VERSION_BOOGU_IMAGE,
    VERSION_OVIS_IMAGE,
    VERSION_ERNIE_IMAGE,
    VERSION_LENS,
    VERSION_MINIT2I,
    VERSION_LONGCAT,
    VERSION_PID,
    VERSION_IDEOGRAM4,
    VERSION_SEFI_IMAGE,
    VERSION_KREA2,
    VERSION_MAGE_FLOW,
    VERSION_SENSENOVA_U1_5,
    VERSION_LLADA_IMAGE,
    VERSION_ESRGAN,
    VERSION_PIXART,
    VERSION_MING_IMAGE,
    VERSION_Z_IMAGE_L2P,
    VERSION_IRIS,
    VERSION_FLUX3,
    VERSION_MOSS_IMAGE,
    VERSION_COUNT,
};
struct RefImageParams {
    bool pass_to_vlm                   = false;
    bool pass_to_dit                   = true;
    Rope::RefIndexMode ref_index_mode  = Rope::RefIndexMode::FIXED;
    bool force_ref_timestep_zero       = false;
    bool resize_before_vae             = true;
    int vae_input_max_pixels           = -1;
    RefImageResizeMode vlm_resize_mode = RefImageResizeMode::AREA;
    int vlm_min_size                   = -1;
    int vlm_max_size                   = -1;
    bool resize_vae_to_target          = false;
};
enum class MiniMaxH3PresentationKind {
    IMAGE,
    VIDEO,
    AUDIO,
};
struct MiniMaxH3PresentationItem {
    MiniMaxH3PresentationKind kind = MiniMaxH3PresentationKind::IMAGE;
    std::vector<sd::Tensor<float>> frames;
    std::vector<float> timestamps;
};
struct ConditionerParams {
    std::string text;
    int clip_skip                                                       = -1;
    int width                                                           = -1;
    int height                                                          = -1;
    bool zero_out_masked                                                = false;
    const std::vector<sd::Tensor<float>>* ref_images                    = nullptr;  // for qwen image edit
    const std::vector<MiniMaxH3PresentationItem>* minimax_h3_references = nullptr;
    RefImageParams ref_image_params;
};
struct SDCondition {
    sd::Tensor<float> c_crossattn;
    sd::Tensor<float> c_vector;
    sd::Tensor<float> c_concat;
    sd::Tensor<int32_t> c_t5_ids;
    sd::Tensor<float> c_t5_weights;
    sd::Tensor<int32_t> c_input_ids;
    sd::Tensor<int32_t> c_position_ids;
    sd::Tensor<int32_t> c_token_types;
    sd::Tensor<int32_t> c_vinput_mask;
    std::vector<std::pair<int, sd::Tensor<float>>> c_image_embeds;
    std::vector<sd::Tensor<float>> c_ref_images;
    std::vector<sd::Tensor<float>> c_ref_audios;
    std::vector<MiniMaxH3ReferenceBlock> c_reference_blocks;

    std::vector<sd::Tensor<float>> extra_c_crossattns;

    SDCondition() = default;

    SDCondition(sd::Tensor<float> c_crossattn,
                sd::Tensor<float> c_vector,
                sd::Tensor<float> c_concat)
        : c_crossattn(std::move(c_crossattn)), c_vector(std::move(c_vector)), c_concat(std::move(c_concat)) {}

    bool empty() const {
        if (!c_crossattn.empty() || !c_vector.empty() || !c_concat.empty() ||
            !c_t5_ids.empty() || !c_t5_weights.empty() ||
            !c_input_ids.empty() || !c_position_ids.empty() ||
            !c_token_types.empty() || !c_vinput_mask.empty()) {
            return false;
        }

        for (const auto& image_embed : c_image_embeds) {
            if (!image_embed.second.empty()) {
                return false;
            }
        }

        for (const auto& tensor : c_ref_images) {
            if (!tensor.empty()) {
                return false;
            }
        }

        for (const auto& tensor : c_ref_audios) {
            if (!tensor.empty()) {
                return false;
            }
        }

        for (const auto& tensor : extra_c_crossattns) {
            if (!tensor.empty()) {
                return false;
            }
        }

        return true;
    }
};
struct Conditioner {virtual ~Conditioner()=default;virtual SDCondition get_learned_condition(int,const ConditionerParams&)=0;};
namespace LLM {struct LLMRunner {bool enable_vision=false;};}
struct LLMEmbedder:Conditioner {
 SDVersion version=VERSION_MOSS_IMAGE;std::shared_ptr<LLM::LLMRunner>llm=std::make_shared<LLM::LLMRunner>();
 int calls=0;std::vector<ConditionerParams>seen;std::function<void(SDCondition&)>mutate;
 SDCondition get_learned_condition(int,const ConditionerParams&p)override {
  ++calls;seen.push_back(p);SDCondition c;c.c_crossattn=sd::Tensor<float>({1},{float(calls)});
  if(mutate)mutate(c);return c;
 }
};
struct OtherConditioner:Conditioner {
 int calls=0;SDCondition get_learned_condition(int,const ConditionerParams&)override {++calls;SDCondition c;c.c_crossattn=sd::Tensor<float>({1},{float(calls)});return c;}
};
