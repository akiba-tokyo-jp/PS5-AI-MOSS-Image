/* SPDX-License-Identifier: GPL-3.0-or-later
 * Private scalar pipelines only. This include does not alter the device flag,
 * public matmul map, quantized kernels, VAE, or general F16 operations. */
#if PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL
namespace {
struct MossLoRAPrecisionPairs { vk_matmul_pipeline_pair f32,f16; };
static MossLoRAPrecisionPairs *moss_lora_precision_pairs{};
static VkDevice moss_lora_precision_owner{};
static unsigned moss_lora_precision_depth{};
static bool moss_lora_precision_uncertain{};
static bool moss_lora_precision_hardware(const vk_device& d) {
    return d->vendor_id==VK_VENDOR_ID_AMD && d->architecture==AMD_RDNA2 &&
        std::strstr(d->properties.deviceName.data(),"NAVI21") && d->fp16 &&
        !d->coopmat_support && !d->coopmat2 && !d->integer_dot_product && !d->dot2_f16 &&
        d->subgroup_size_control && d->subgroup_require_full_support &&
        (d->subgroup_size==32 || d->subgroup_size==64) &&
        d->subgroup_min_size<=d->subgroup_size && d->subgroup_max_size>=d->subgroup_size &&
        d->properties.limits.maxComputeWorkGroupInvocations>=128 &&
        d->properties.limits.maxComputeWorkGroupSize[0]>=128 &&
        // Two FLOAT_TYPE=vec2 F32 arrays: (64+64)*(32/2+1)*8.
        d->properties.limits.maxComputeSharedMemorySize>=17408;
}
static bool moss_lora_precision_runtime(const vk_device& d) {
    return moss_lora_precision_hardware(d) && d->serialize_submissions && !d->async_use_transfer_queue;
}
static MossLoRAPrecisionPairs& moss_lora_precision_metadata(const vk_device& d) {
    // Called only under existing load_shaders compile_mutex; queue policy is
    // initialized later, so this predicate uses immutable physical caps only.
    GGML_ASSERT(moss_lora_precision_hardware(d));
    if(!moss_lora_precision_pairs) {
        moss_lora_precision_pairs=new MossLoRAPrecisionPairs{};
        moss_lora_precision_owner=VkDevice(d->device);
    }
    GGML_ASSERT(moss_lora_precision_owner==VkDevice(d->device));
    return *moss_lora_precision_pairs;
}
static int moss_lora_precision_slot(const vk_device& d,const vk_pipeline& p) {
    if(!moss_lora_precision_pairs || moss_lora_precision_owner!=VkDevice(d->device) || !p)return -1;
    if(p==moss_lora_precision_pairs->f32.unaligned)return 0;
    if(p==moss_lora_precision_pairs->f32.aligned)return 1;
    if(p==moss_lora_precision_pairs->f16.unaligned)return 2;
    if(p==moss_lora_precision_pairs->f16.aligned)return 3;
    return -1;
}
static bool moss_lora_precision_node(const ggml_tensor* dst) {
    if(!dst || dst->op!=GGML_OP_MUL_MAT || dst->type!=GGML_TYPE_F32 ||
       (ggml_prec)dst->op_params[0]!=GGML_PREC_F32)return false;
    const auto *a=dst->src[0],*b=dst->src[1];
    if(!a || !b || (a->type!=GGML_TYPE_F32 && a->type!=GGML_TYPE_F16) || b->type!=GGML_TYPE_F32 ||
       a->ne[2]!=1 || a->ne[3]!=1 || b->ne[2]!=1 || b->ne[3]!=1 || dst->ne[2]!=1 || dst->ne[3]!=1 ||
       b->ne[0]!=a->ne[0] || dst->ne[0]!=a->ne[1] || dst->ne[1]!=b->ne[1] ||
       !ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(dst))return false;
    const bool geometry=(a->ne[1]==144 && (a->ne[0]==3072 || a->ne[0]==8192)) ||
        (a->ne[0]==144 && (a->ne[1]==3072 || a->ne[1]==16384));
    const bool canonical=ps5_ai_moss_lora_storage_index(a->name,a->ne[0],a->ne[1])>=0;
    const bool fixture=!std::strncmp(a->name,"moss_f16_storage_fixture.",std::strlen("moss_f16_storage_fixture."));
    return geometry && ((canonical && a->type==(PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL?GGML_TYPE_F16:GGML_TYPE_F32) &&
        (b->ne[1]==362 || b->ne[1]==4202)) || (fixture && b->ne[1]==9));
}
struct MossLoRAPrecisionGraphScope {
    MossLoRAPrecisionGraphScope(){++moss_lora_precision_depth;moss_lora_precision_uncertain=true;}
    ~MossLoRAPrecisionGraphScope(){--moss_lora_precision_depth;}
    void completed(ggml_backend_vk_context* ctx) {
        ps5_ai_ggml_dispatch_stats owned{};ps5_ai_ggml_dispatch_snapshot(&owned);
        if(moss_lora_precision_depth==1 && owned.hardware_verified && !owned.work_pending &&
           !ctx->submit_pending && ctx->compute_ctx.expired())moss_lora_precision_uncertain=false;
    }
};
}
extern "C" int ps5_ai_moss_lora_precision_backend_admission(void *existing_backend) {
    auto backend=static_cast<ggml_backend_t>(existing_backend);
    if(!backend || !ggml_backend_is_vk(backend) || !backend->context)return 1;
    auto *ctx=static_cast<ggml_backend_vk_context*>(backend->context);
    if(!ctx->device || !moss_lora_precision_hardware(ctx->device))return 2;
    if(!moss_lora_precision_runtime(ctx->device))return 3;
    if(!moss_lora_precision_pairs || moss_lora_precision_owner!=VkDevice(ctx->device->device))return 4;
    const auto& p=*moss_lora_precision_pairs;
    if(!p.f32.unaligned || !p.f32.aligned || !p.f16.unaligned || !p.f16.aligned ||
       !p.f32.unaligned->initialized || !p.f32.aligned->initialized ||
       !p.f16.unaligned->initialized || !p.f16.aligned->initialized)return 4;
    ps5_ai_ggml_dispatch_stats owned{};ps5_ai_ggml_dispatch_snapshot(&owned);
    if(moss_lora_precision_depth || moss_lora_precision_uncertain || !owned.hardware_verified || owned.work_pending ||
       ctx->submit_pending || !ctx->compute_ctx.expired())return 5;
    return 0;
}
#endif
