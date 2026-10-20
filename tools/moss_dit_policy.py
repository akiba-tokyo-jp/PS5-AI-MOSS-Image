# SPDX-License-Identifier: GPL-3.0-or-later
"""Default-off, bounded MOSS F32 pointwise/cache trial. No production admission."""
POLICY='per_generation_text_rope_cache_SiTU_normmod_residual_QKrope_LoRA_packed_add_F32_v1'
MACRO='PS5_AI_MOSS_DIT_OPT_TRIAL'
def flags(enabled):
    if type(enabled)is not bool:raise ValueError('DiT capability must be a boolean')
    return ['-D'+MACRO+'='+str(int(enabled))]
def fields(enabled):
    return {'moss_dit_optimizations':enabled,'moss_dit_policy':POLICY if enabled else 'disabled','moss_dit_compile_flags':flags(enabled)}
def validate(enabled,*,accumulation,turbo,upload,spectrum=False,ffn=False,lora=None):
    flags(enabled)
    if enabled and (not accumulation or not turbo or not upload or spectrum or ffn or lora is not None):
        raise ValueError('DiT trial requires unchanged Turbo/upload/F32 parent, FFN/private precision/Spectrum off')
def validate_core(manifest,enabled):
    expected=fields(enabled)
    if not enabled and not set(expected).intersection(manifest):return {'historical_off':True}
    for k,v in expected.items():
        if type(manifest.get(k))is not type(v)or manifest[k]!=v:raise ValueError('DiT core/native mismatch: '+k)
    for row in manifest.get('commands',[]):
        if sum(MACRO in arg for arg in row)!=1 or flags(enabled)[0]not in row:raise ValueError('DiT flag missing/overridden in actual core commands')
    if len(manifest.get('commands',[]))!=75:raise ValueError('All75 commands required')
    return expected
