#pragma once
/* Pure metadata bound for the frozen 8003 GGML Vulkan / PS5 scalar profile.
 * No backend, pipeline, allocator or model-data access is performed here.
 * Requested-buffer maxima exclude previous context high-water marks and do not
 * establish aggregate/fragmentation/driver/whole-1024 readiness. */
#include "ggml.h"
#include "ggml-impl.h"
#include <cstdint>
#include <initializer_list>

namespace native_scratch_bounds {
struct Profile {
    bool ps5_scalar_no_coopmat = false;
    bool f16_math_disabled = false;
    bool no_async = false;
    bool no_fusion = false;
    uint32_t shader_cores_upper = 0; // Frozen winsys 36, or provider count0 fallback.
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
static bool matmul(const ggml_tensor * dst,const Profile & profile,Bounds & out) {
    const auto * a=dst->src[0]; const auto * b=dst->src[1];
    uint64_t ae=0,be=0,de=0;
    if (!float_tensor(a,ae)||!float_tensor(b,be)||!float_tensor(dst,de)||dst->type!=GGML_TYPE_F32 ||
        a->ne[0]!=b->ne[0] || dst->ne[0]!=a->ne[1] || dst->ne[1]!=b->ne[1] ||
        dst->ne[2]!=b->ne[2] || dst->ne[3]!=b->ne[3] || de>UINT32_MAX) return false;
    /* Staging covers matrix/vector/swap branches and F32 or F16 conversion.
     * qx staging<=4 bytes/logical element; qy padding denom<=128. A possible
     * temporaryQ8_1 Y is smaller than this F32 upper bound. */
    uint64_t padded=0,ye=0,xbytes=0,ybytes=0;
    if (!rounded(uint64_t(b->ne[1]),128,padded)||padded>UINT32_MAX||!mul(padded,uint64_t(b->ne[0]),ye)||
        !mul(ye,uint64_t(b->ne[2]),ye)||!mul(ye,uint64_t(b->ne[3]),ye)||
        !mul(maximum(ae,be),4,xbytes)||!mul(maximum(maximum(ae,be),ye),4,ybytes)) return false;
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
        !profile.ps5_scalar_no_coopmat || !profile.f16_math_disabled || !profile.no_async || !profile.no_fusion ||
        profile.shader_cores_upper!=36) {
        result.known=false;result.reason="unsupported_or_unverified_profile";return result;
    }
    for(int i=0;i<graph->n_nodes;++i) {
        const auto * t=graph->nodes[i];++result.inspected_nodes;
        bool known=t && (t->op==GGML_OP_MUL_MAT ? matmul(t,profile,result) :
                         t->op==GGML_OP_FLASH_ATTN_EXT ? dense_flash(t,profile,result) : zero_prealloc_op(t->op));
        if (!known) {
            result.known=false;++result.unknown_nodes;
            if(result.unknown_nodes==1)result.first_unknown=t?t->op:GGML_OP_NONE;
        }
    }
    if(!result.known)result.reason="unsupported_node_prealloc_unknown";
    return result;
}
} // namespace native_scratch_bounds
