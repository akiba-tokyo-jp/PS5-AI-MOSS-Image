#pragma once
/* Pure metadata bound for frozen 8003 PS5 scalar FP32 OR FP16 math.
 * Both use <=4 bytes/element staging, the same scalar Br<=16/Bc32|64,
 * tile denoms32|64|128, and F32 split accumulation. Coopmat stays disabled.
 * No backend, pipeline, allocator or model-data access is performed here.
 * Requested-buffer maxima exclude previous context high-water marks and do not
 * establish aggregate/fragmentation/driver/whole-1024 readiness. */
#include "ggml.h"
#if PS5_AI_MOSS_DIT_OPT_TRIAL
#include "model/diffusion/moss_dit_ops.hpp"
#endif
#include "ggml-impl.h"
#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace native_scratch_bounds {
struct Profile {
    bool ps5_scalar_no_coopmat = false;
    bool f16_math_disabled = false;
    bool no_async = false;
    bool no_fusion = false;
    uint32_t shader_cores_upper = 0; // Frozen winsys 36, or provider count0 fallback.
    // 0: original no-fusion; 1: exact PS5 F32 same-shape SIGMOID/SILU->MUL.
    // Mode1 reuses unary_mul, which requests none of x/y/split/add-RMS staging.
    // General fusion must remain disabled in both modes; unknown modes refuse.
    uint32_t ps5_elementwise_fusion_mode = 0;
};
struct Bounds {
    bool known = true;
    const char * reason = "bounded_current_graph_prealloc_only";
    uint64_t x = 0, y = 0, split_k = 0;
    uint64_t add_rms_baseline = 16384; // Fixed1024 requested bytes, imported16KiB.
    uint32_t inspected_nodes = 0, unknown_nodes = 0;
    ggml_op first_unknown = GGML_OP_NONE;
    bool aggregate_capacity_proven = false;
};
static constexpr uint64_t page = 16384;
static bool mul(uint64_t a, uint64_t b, uint64_t & out) {
    if (b && a > UINT64_MAX / b) return false;
    out = a * b; return true;
}
static bool rounded(uint64_t n, uint64_t alignment, uint64_t & out) {
    if (!alignment || (alignment & (alignment-1)) || n > UINT64_MAX-(alignment-1)) return false;
    out = (n+alignment-1)&~(alignment-1); return true;
}
static uint64_t ceil_div(uint64_t n, uint64_t d) { return n/d+(n%d!=0); }
static uint64_t maximum(uint64_t a, uint64_t b) { return a>b?a:b; }
static bool float_tensor(const ggml_tensor * t, uint64_t & elements) {
    if (!t || (t->type!=GGML_TYPE_F16 && t->type!=GGML_TYPE_F32)) return false;
    elements=1;
    for (unsigned i=0;i<4;++i) {
        if (t->ne[i]<=0 || uint64_t(t->ne[i])>UINT32_MAX || !mul(elements,uint64_t(t->ne[i]),elements)) return false;
    }
    return true;
}
/* Exact source5689–5728 rule, enumerated over source1759–1761 tile pairs.
 * The helper accepts no computed pipeline object or device operation. */
static uint32_t matmul_split(uint32_t m,uint32_t n,uint32_t k,uint32_t cores,uint32_t tile) {
    uint32_t split=1;
    const uint64_t tiles=ceil_div(m,tile)*ceil_div(n,tile);
    if (cores && n>=tile && k>=2048) {
        if (tiles<=cores/2) split=uint32_t(cores/tiles);
        else if (tiles<=cores*2/3) split=3;
        if (split>8) split=8;
        while (split>1) {
            const uint64_t part=ceil_div(ceil_div(k,split),256)*256;
            if (part*(split-1)<k) break;
            --split;
        }
    }
    return split;
}
/* MOSS Q4_K_M weights use Q4_K/Q6_K plus F16/F32 tensors. The frozen
 * scalar Vulkan matrix branch can dequantize X to F16 (6218-6222), and
 * quantize/pad Y to Q8_1 (6214-6223). Four bytes per logical element is an
 * upper bound for those branches. Quantized X must be exactly contiguous;
 * this excludes the vector handler's unusual non-contiguous X staging rule.
 * BF16 has a separate matrix-left-only admission below. No IQ/MoE/kernel
 * addition is admitted here. */
static bool moss_quantized_weight(const ggml_tensor * t, uint64_t & elements) {
    if (!t) return false;
    uint64_t block=0, storage=0;
    switch (t->type) {
    case GGML_TYPE_Q4_0: block=32; storage=18; break;
    case GGML_TYPE_Q4_1: block=32; storage=20; break;
    case GGML_TYPE_Q4_K: block=256; storage=144; break;
    case GGML_TYPE_Q5_K: block=256; storage=176; break;
    case GGML_TYPE_Q6_K: block=256; storage=210; break;
    case GGML_TYPE_Q8_0: block=32; storage=34; break;
    default: return false;
    }
    elements=1;
    for (unsigned i=0;i<4;++i)
        if(t->ne[i]<=0 || uint64_t(t->ne[i])>UINT32_MAX || !mul(elements,uint64_t(t->ne[i]),elements))return false;
    if(elements>UINT32_MAX || uint64_t(t->ne[0])%block || t->nb[0]!=storage)return false;
    uint64_t stride=0;
    if(!mul(uint64_t(t->ne[0])/block,storage,stride))return false;
    for(unsigned i=1;i<4;++i) {
        if(t->nb[i]!=stride || !mul(stride,uint64_t(t->ne[i]),stride) || stride>UINT32_MAX)return false;
    }
    return true;
}
/* Frozen Vulkan6119/6475 explicitly handles BF16 matrix-left operands. The
 * matrix branch selects BF16 Y and copies/converts it when necessary
 * (6171-6195); x/y staging remains <=4 bytes per logical element (6223-6224).
 * Admit only contiguous, 32-bit-addressable BF16 weights, leaving activation,
 * FLASH, normalization and RoPE type admission unchanged. This establishes
 * the four-prealloc upper bound, not shader availability or numeric accuracy. */
static bool moss_bfloat_weight(const ggml_tensor * t, uint64_t & elements) {
    if(!t || t->type!=GGML_TYPE_BF16 || t->nb[0]!=2)return false;
    elements=1;uint64_t stride=2;
    for(unsigned i=0;i<4;++i) {
        if(t->ne[i]<=0 || uint64_t(t->ne[i])>UINT32_MAX || t->nb[i]!=stride ||
           !mul(elements,uint64_t(t->ne[i]),elements) || elements>UINT32_MAX ||
           !mul(stride,uint64_t(t->ne[i]),stride) || stride>UINT32_MAX)return false;
    }
    return true;
}
static bool matmul(const ggml_tensor * dst,const Profile & profile,Bounds & out) {
    const auto * a=dst->src[0]; const auto * b=dst->src[1];
    uint64_t ae=0,be=0,de=0;
    if (!(float_tensor(a,ae)||moss_quantized_weight(a,ae)||moss_bfloat_weight(a,ae))||!float_tensor(b,be)||!float_tensor(dst,de)||dst->type!=GGML_TYPE_F32 ||
        a->ne[0]!=b->ne[0] || dst->ne[0]!=a->ne[1] || dst->ne[1]!=b->ne[1] ||
        dst->ne[2]!=b->ne[2] || dst->ne[3]!=b->ne[3] || de>UINT32_MAX) return false;
    /* Staging covers matrix/vector/swap branches and F32 or F16 conversion.
     * qx staging<=4 bytes/logical element; qy padding denom<=128. A possible
     * temporaryQ8_1 Y is smaller than this F32 upper bound. */
    /* Frozen Vulkan6119/6475: with both operands F16, matrix dequant/copy
     * and vector copy staging are two bytes/element, including swapped inputs.
     * Neither branch can choose F32 Y or Q8 quantization for F16 input.
     * Mixed/F32 operands retain the previous four-byte conservative bound.
     * Padding, split accumulation, actual256MiB claim and pool checks stay.
     * Enable this narrower profile only in the physically verified F16 mode. */
    const uint64_t staging_bytes=!profile.f16_math_disabled &&
        a->type==GGML_TYPE_F16 && b->type==GGML_TYPE_F16 ? 2 : 4;
    uint64_t padded=0,ye=0,xbytes=0,ybytes=0;
    if (!rounded(uint64_t(b->ne[1]),128,padded)||padded>UINT32_MAX||!mul(padded,uint64_t(b->ne[0]),ye)||
        !mul(ye,uint64_t(b->ne[2]),ye)||!mul(ye,uint64_t(b->ne[3]),ye)||
        !mul(maximum(ae,be),staging_bytes,xbytes)||!mul(maximum(maximum(ae,be),ye),staging_bytes,ybytes)) return false;
    uint32_t split=1;
    for (uint32_t cores=0;cores<=profile.shader_cores_upper;++cores) {
        for (uint32_t tile: {32u,64u,128u}) {
            const auto s=matmul_split(uint32_t(a->ne[1]),uint32_t(b->ne[1]),uint32_t(b->ne[0]),cores,tile);
            if (s>split) split=s;
        }
    }
    uint64_t split_bytes=0;
    if (split>1 && (!mul(de,4,split_bytes)||!mul(split_bytes,split,split_bytes))) return false;
    if (!rounded(xbytes,page,xbytes)||!rounded(ybytes,page,ybytes)||!rounded(split_bytes,page,split_bytes)) return false;
    out.x=maximum(out.x,xbytes);out.y=maximum(out.y,ybytes);out.split_k=maximum(out.split_k,split_bytes);
    return true;
}
static bool dense_flash(const ggml_tensor * dst,const Profile & profile,Bounds & out) {
    const auto * q=dst->src[0];const auto * k=dst->src[1];const auto * v=dst->src[2];
    uint64_t qe=0,ke=0,ve=0,de=0;
    if (!float_tensor(q,qe)||!float_tensor(k,ke)||!float_tensor(v,ve)||!float_tensor(dst,de) ||
        q->type!=GGML_TYPE_F32 || dst->type!=GGML_TYPE_F32 || dst->src[3] || dst->src[4] ||
        q->ne[0]!=k->ne[0] || k->ne[1]!=v->ne[1] ||
        q->ne[2]!=k->ne[2] || q->ne[2]!=v->ne[2] || q->ne[3]!=k->ne[3] || q->ne[3]!=v->ne[3] ||
        dst->ne[0]!=v->ne[0] || dst->ne[1]!=q->ne[2] || dst->ne[2]!=q->ne[1] || dst->ne[3]!=q->ne[3]) return false;
    /* Scalar source1189–1261 gives Br<=16, Bc32/64. No mask=>no sparse/mask
     * scratch; nonquantizedK/V=>no dequant scratch; PS5=>no Intel temporary.
     * FLASH has no8x cap. Enumerate its own roundedKV rule8193–8234. */
    uint64_t upper=0;
    for (uint32_t cores=0;cores<=profile.shader_cores_upper;++cores) {
        const uint64_t actual_cores=cores?cores:16;
        for (uint32_t br: {1u,2u,4u,8u,16u}) for (uint32_t bc: {32u,64u}) {
            uint64_t groups=0;
            if (!mul(ceil_div(uint64_t(q->ne[1]),br),uint64_t(q->ne[2]),groups)||
                !mul(groups,uint64_t(q->ne[3]),groups)||!groups||groups>UINT32_MAX) return false;
            uint64_t split=groups<actual_cores*2 ? actual_cores*2/groups : 1;
            if (split>1) {
                uint64_t split_kv=uint64_t(k->ne[1])/split;
                if (!split_kv) split_kv=1;
                if (!rounded(split_kv,bc,split_kv)) return false;
                split=ceil_div(uint64_t(k->ne[1]),split_kv);
            }
            if (split>1) {
                uint64_t bytes=0;
                if (!mul(uint64_t(dst->ne[0])+2,uint64_t(dst->ne[1]),bytes)||!mul(bytes,4,bytes)||
                    !mul(bytes,split,bytes)||!mul(bytes,uint64_t(dst->ne[2]),bytes)||!mul(bytes,uint64_t(dst->ne[3]),bytes)) return false;
                upper=maximum(upper,bytes);
            }
        }
    }
    if (!rounded(upper,page,upper)) return false;
    out.split_k=maximum(out.split_k,upper);return true;
}
static bool small_plain_softmax(const ggml_tensor * t) {
    /* Frozen ggml-flex ggml_vk_soft_max: ncols<=16384 dispatches op_f32
     * without writing any of the four prealloc sizes. Larger rows request
     * x/y scratch and remain refused. Only unmasked, sink-free F32 is audited.
     * The caller separately requires the verified no-fusion profile. */
    if (!t || t->op!=GGML_OP_SOFT_MAX || t->type!=GGML_TYPE_F32 ||
        !t->src[0] || t->src[0]->type!=GGML_TYPE_F32 || t->src[1] || t->src[2]) return false;
    uint64_t source=0,destination=0;
    if (!float_tensor(t->src[0],source) || !float_tensor(t,destination) ||
        source!=destination || source>UINT32_MAX || t->ne[0]>16384) return false;
    for (unsigned i=0;i<4;++i) if (t->ne[i]!=t->src[0]->ne[i]) return false;
    return true;
}

/* Anima/Qwen-Image paths in the same frozen Vulkan provider. These handlers
 * dispatch directly from tensor buffers and do not grow any of the four
 * prealloc buffers under the required no-fusion profile. Keep their shader
 * integer addressing/type/shape bounds explicit instead of allowing new ops
 * solely by enum. This is not a claim of aggregate memory or GPU correctness. */
static bool u32_float_storage(const ggml_tensor * t, bool contiguous=false) {
    uint64_t elements=0;
    if (!float_tensor(t,elements)) return false;
    const uint64_t element_bytes=t->type==GGML_TYPE_F16?2:4;
    uint64_t bytes=0,span=element_bytes,expected=element_bytes;
    if (!mul(elements,element_bytes,bytes) || bytes>UINT32_MAX || t->nb[0]!=element_bytes) return false;
    for(unsigned i=0;i<4;++i) {
        if (t->nb[i]>UINT32_MAX || t->nb[i]%element_bytes ||
            (contiguous && t->ne[i]>1 && t->nb[i]!=expected)) return false;
        uint64_t offset=0;
        if (!mul(uint64_t(t->ne[i]-1),uint64_t(t->nb[i]),offset) || offset>UINT32_MAX-span) return false;
        span+=offset;
        if (!mul(expected,uint64_t(t->ne[i]),expected)) return false;
    }
    return true;
}
static bool same_shape(const ggml_tensor * a,const ggml_tensor * b) {
    if(!a || !b)return false;
    for(unsigned i=0;i<4;++i)if(a->ne[i]!=b->ne[i])return false;
    return true;
}
static bool plain_rms_norm(const ggml_tensor * t) {
    if (!t || t->op!=GGML_OP_RMS_NORM || t->type!=GGML_TYPE_F32 ||
        !t->src[0] || t->src[0]->type!=GGML_TYPE_F32 || !same_shape(t,t->src[0]) ||
        !u32_float_storage(t) || !u32_float_storage(t->src[0])) return false;
    // The plain handler dispatches one workgroup per ne1/ne2/ne3 row.
    // Use Vulkan's guaranteed workgroup-count floor, not a guessed PS5 limit.
    for(unsigned i=1;i<4;++i)if(t->ne[i]>65535)return false;
    return true;
}
static bool plain_rope_apply(const ggml_tensor * t) {
    if (!t || t->op!=GGML_OP_ROPE_APPLY || t->type!=GGML_TYPE_F32 ||
        !t->src[0] || !t->src[1] || t->src[0]->type!=GGML_TYPE_F32 || t->src[1]->type!=GGML_TYPE_F32 ||
        !u32_float_storage(t,true) ||
        !u32_float_storage(t->src[0]) || !u32_float_storage(t->src[1])) return false;
    const auto * x=t->src[0];const auto * pe=t->src[1];
    // ggml_rope_apply converts [D,H,L,N] to [D,L,H*N,1].
    uint64_t heads_batch=0;
    return mul(uint64_t(x->ne[1]),uint64_t(x->ne[3]),heads_batch) && heads_batch<=UINT32_MAX &&
        t->ne[0]==x->ne[0] && t->ne[1]==x->ne[2] && uint64_t(t->ne[2])==heads_batch && t->ne[3]==1 &&
        x->ne[0]%2==0 && pe->ne[0]==2 && pe->ne[1]==2 &&
        pe->ne[2]==x->ne[0]/2 && pe->ne[3]==x->ne[2];
}
static int32_t int_param(const ggml_tensor * t,unsigned index) {
    int32_t value=0;std::memcpy(&value,reinterpret_cast<const char *>(t->op_params)+index*sizeof(value),sizeof(value));return value;
}
static bool convolution_axis(uint64_t input,uint64_t kernel,int32_t stride,int32_t padding,int32_t dilation,uint64_t output) {
    if(!input || !kernel || stride<=0 || padding<0 || dilation<=0)return false;
    uint64_t extent=0;
    if(!mul(kernel-1,uint64_t(dilation),extent) || extent==UINT64_MAX)return false;
    ++extent;
    const uint64_t padded=input+2*uint64_t(padding);
    return padded>=extent && (padded-extent)/uint64_t(stride)+1==output;
}
static bool plain_conv3d(const ggml_tensor * t) {
    if (!t || t->op!=GGML_OP_CONV_3D || t->type!=GGML_TYPE_F32 || !t->src[0] || !t->src[1] ||
        t->src[1]->type!=GGML_TYPE_F32 || !u32_float_storage(t,true) ||
        !u32_float_storage(t->src[0],true) || !u32_float_storage(t->src[1],true)) return false;
    const auto * w=t->src[0];const auto * x=t->src[1];
    const int32_t ic=int_param(t,9),batch=int_param(t,10),oc=int_param(t,11);
    if(ic<=0 || batch<=0 || oc<=0 || uint64_t(ic)*uint64_t(oc)!=uint64_t(w->ne[3]) ||
       uint64_t(ic)*uint64_t(batch)!=uint64_t(x->ne[3]) || uint64_t(oc)*uint64_t(batch)!=uint64_t(t->ne[3]))return false;
    for(unsigned i=0;i<3;++i)if(!convolution_axis(uint64_t(x->ne[i]),uint64_t(w->ne[i]),
        int_param(t,i),int_param(t,i+3),int_param(t,i+6),uint64_t(t->ne[i])))return false;
    return true;
}
static bool plain_im2col3d(const ggml_tensor * t) {
    if (!t || t->op!=GGML_OP_IM2COL_3D || !t->src[0] || !t->src[1] ||
        t->src[1]->type!=GGML_TYPE_F32 || !u32_float_storage(t,true) ||
        !u32_float_storage(t->src[0],true) || !u32_float_storage(t->src[1])) return false;
    const auto * w=t->src[0];const auto * x=t->src[1];
    const int32_t ic=int_param(t,9);
    if(ic<=0 || x->ne[3]%ic || w->ne[3]%ic)return false;
    const uint64_t batch=uint64_t(x->ne[3])/uint64_t(ic);
    uint64_t kernel=uint64_t(ic);
    for(unsigned i=0;i<3;++i)if(!mul(kernel,uint64_t(w->ne[i]),kernel) || kernel>UINT32_MAX)return false;
    if(uint64_t(t->ne[0])!=kernel || !batch || uint64_t(t->ne[3])%batch)return false;
    const uint64_t output[]={uint64_t(t->ne[1]),uint64_t(t->ne[2]),uint64_t(t->ne[3])/batch};
    for(unsigned i=0;i<3;++i)if(!convolution_axis(uint64_t(x->ne[i]),uint64_t(w->ne[i]),
        int_param(t,i),int_param(t,i+3),int_param(t,i+6),output[i]))return false;
    return true;
}
static bool zero_prealloc_op(ggml_op op) {
    /* Whitelist only handlers without prealloc_size writes in frozen8003.
     * This concerns the four prealloc buffers, not descriptor/staging/driver BOs. */
    switch(op) {
    case GGML_OP_NONE:case GGML_OP_DUP:case GGML_OP_ADD:case GGML_OP_ADD1:case GGML_OP_SUB:
    case GGML_OP_MUL:case GGML_OP_DIV:case GGML_OP_SQR:case GGML_OP_SQRT:case GGML_OP_LOG:
    case GGML_OP_NORM:case GGML_OP_GROUP_NORM:case GGML_OP_SCALE:case GGML_OP_REPEAT:
    case GGML_OP_CONCAT:case GGML_OP_CONT:case GGML_OP_CPY:case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:case GGML_OP_PERMUTE:case GGML_OP_TRANSPOSE:case GGML_OP_GET_ROWS:
    case GGML_OP_IM2COL:case GGML_OP_CONV_2D:case GGML_OP_CONV_TRANSPOSE_2D:
    case GGML_OP_CONV_2D_DW:case GGML_OP_POOL_2D:case GGML_OP_UPSCALE:case GGML_OP_PAD:
    /* Frozen8003:11758,12644 timestep uses ordinary op_f32, no four-prealloc writes. */
    case GGML_OP_TIMESTEP_EMBEDDING:case GGML_OP_UNARY: return true;
    default:return false;
    }
}
static Bounds inspect(const ggml_cgraph * graph,const Profile & profile) {
    Bounds result;
    if (!graph || graph->n_nodes<0 || graph->n_nodes>327680 ||
        !profile.ps5_scalar_no_coopmat || !profile.no_async || !profile.no_fusion ||
        profile.shader_cores_upper!=36 || profile.ps5_elementwise_fusion_mode>1 ||
        (profile.ps5_elementwise_fusion_mode==1 && !profile.f16_math_disabled)) {
        result.known=false;result.reason="unsupported_or_unverified_profile";return result;
    }
    for(int i=0;i<graph->n_nodes;++i) {
        const auto * t=graph->nodes[i];++result.inspected_nodes;
        bool known=t && (t->op==GGML_OP_MUL_MAT ? matmul(t,profile,result) :
#if PS5_AI_MOSS_DIT_OPT_TRIAL
                         // The dedicated five-binding F32 shader reads tensor
                         // buffers directly and writes only its GGML output.
                         // Admit the exact callback/kind/shapes, never MAP_CUSTOM3
                         // as a blanket enum. No x/y/split/add-RMS request exists.
                         t->op==GGML_OP_MAP_CUSTOM3 ? moss_dit_opt::valid(t) :
#endif
                         t->op==GGML_OP_FLASH_ATTN_EXT ? dense_flash(t,profile,result) :
                         t->op==GGML_OP_SOFT_MAX ? small_plain_softmax(t) :
                         t->op==GGML_OP_RMS_NORM ? plain_rms_norm(t) :
                         t->op==GGML_OP_ROPE_APPLY ? plain_rope_apply(t) :
                         t->op==GGML_OP_CONV_3D ? plain_conv3d(t) :
                         t->op==GGML_OP_IM2COL_3D ? plain_im2col3d(t) : zero_prealloc_op(t->op));
        if (!known) {
            result.known=false;++result.unknown_nodes;
            if(result.unknown_nodes==1)result.first_unknown=t?t->op:GGML_OP_NONE;
        }
    }
    if(!result.known)result.reason="unsupported_node_prealloc_unknown";
    return result;
}
} // namespace native_scratch_bounds
