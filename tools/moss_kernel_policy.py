# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, default-off DiT kernel experiments; no admission or budget changes."""
from pathlib import Path
import hashlib
MACRO='PS5_AI_MOSS_DIT_KERNEL_TRIAL'
POLICY='Q4_FFN_QKV_predecode_FA_shared_KV_LoRA_vec4_RMS_epilogue_F32_v1'
def flags(enabled):
    if type(enabled)is not bool:raise ValueError('Typed kernel capability required')
    return ['-D'+MACRO+'='+str(int(enabled))]
def fields(enabled):
    return {'moss_dit_kernels':enabled,'moss_dit_kernel_policy':POLICY if enabled else 'disabled','moss_dit_kernel_compile_flags':flags(enabled)}
def helper_input():
    path=Path(__file__).resolve();return dict(path=path.as_posix(),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def fa_flags(no_limit,mask):
    if type(no_limit)is not bool or (mask is not None and(type(mask)is not int or not 0<=mask<=15)):raise ValueError('Typed FA occupancy selector required')
    if no_limit and (mask is None or not mask&2):raise ValueError('FA occupancy trial requires selected shared-KV bit2')
    return ['-DPS5_AI_MOSS_DIT_FA_NO_OCCUPANCY_LIMIT='+str(int(no_limit))]
def validate(enabled,*,dit,accumulation,turbo,upload,mask=None):
    flags(enabled)
    if enabled and not (dit and accumulation and turbo and upload):raise ValueError('Kernel trial requires mask15 Turbo4/upload/F32 parent')
    if mask is not None and (type(mask)is not int or not 0<=mask<=15 or not enabled):raise ValueError('Invalid kernel selection')
def validate_core(manifest,enabled):
    if not enabled and not any(k in manifest for k in fields(False)):return
    if manifest.get('kernel_policy_input')!=helper_input():raise ValueError('Exact kernel-policy helper required')
    for k,v in fields(enabled).items():
        if type(manifest.get(k))is not type(v)or manifest[k]!=v:raise ValueError('Core/kernel identity mismatch: '+k)
    rows=manifest.get('commands',[])
    if len(rows)!=75 or any(sum(MACRO in a for a in row)!=1 or flags(enabled)[0]not in row for row in rows):raise ValueError('All75 actual kernel flags required')
