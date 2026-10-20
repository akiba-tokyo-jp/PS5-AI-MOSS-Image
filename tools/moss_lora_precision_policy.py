# SPDX-License-Identifier: GPL-3.0-or-later
"""One explicit precision experiment on the combined parent; defaults remain OFF."""
import hashlib
from pathlib import Path
SHARED='PS5_AI_MOSS_LORA_FP32_SHARED_TRIAL'
STORAGE='PS5_AI_MOSS_LORA_F16_STORAGE_TRIAL'
TRANSPORT='PS5_AI_MOSS_LORA_PRECISION_TRANSPORT_ONLY'
MACROS=(SHARED,STORAGE,TRANSPORT)
POLICIES={x:f'combined_Turbo_{x.upper()}_LoRA_storage_FP32_shared_math_r144_v1' for x in ('f32','f16')}
IDENTITIES={
 'f32':('moss-nsfw07-turbo4-r144-f32.safetensors',1337791552,'126fed9dd5268fe20ab0085525a4b7b7f8ff767a28aea907df2089c33360d4d3'),
 'f16':('moss-nsfw07-turbo4-r144-f16.safetensors',668930824,'7904da4b4437dd142b37d1d3b459f558eac1556b8f7e5b46941f6fd39fca735a')}

def flags(mode=None,*,transport=False,core=False):
    if mode is not None and (type(mode)is not str or mode not in POLICIES):raise RuntimeError('LoRA storage must be explicitly f32 or f16')
    if type(transport)is not bool or type(core)is not bool or (transport and core):raise RuntimeError('Invalid precision compiler scope')
    result=[f'-D{SHARED}={int(mode is not None)}',f'-D{STORAGE}={int(mode=="f16")}',f'-D{TRANSPORT}={int(transport and mode is not None)}']
    if mode is not None and core:
        # Core's original recipes already pin runtime admission0. The remaining
        # strict trial macros must accompany the shared contract on all75 TUs.
        result += ['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_TRIAL_10G=1',
            '-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1','-DPS5_AI_MOSS_PERF_VARIANT=1',
            '-DPS5_AI_MOSS_GPU_PROFILE=0','-DPS5_AI_MOSS_Q4_BK64_RAW64=0','-DPS5_AI_MOSS_SAMPLER_TRIAL=0']
    return result

def validate(mode=None,*,turbo=False,upload=False,spectrum=False,ffn=False,accumulation=False,
             gpu_profile=False,q4=False,sampler=False,variant=None,precision='fp32',experimental=False,
             trial_10g=False,operating=False,transport=False):
    flags(mode,transport=transport)
    values=(turbo,upload,spectrum,ffn,accumulation,gpu_profile,q4,sampler,experimental,trial_10g,operating,transport)
    if any(type(v)is not bool for v in values):raise RuntimeError('LoRA precision inputs must be explicit booleans')
    if mode is not None and (not turbo or upload==transport or spectrum or ffn or not accumulation or
       gpu_profile or q4 or sampler or type(variant)is not int or variant!=1 or precision!='fp16' or
       not experimental or not trial_10g or operating):
        raise RuntimeError('LoRA precision requires combined U1/T1/F32acc small10 variant1, FFN/operating/other trials off; gateway U0 transport only')

def fields(mode=None,*,transport=False,core=False):
    result=dict(moss_lora_storage=mode,moss_lora_fp32_shared_trial=mode is not None,
        moss_lora_f16_storage_trial=mode=='f16',moss_lora_precision_policy=POLICIES.get(mode,'disabled'),
        moss_lora_precision_compile_flags=flags(mode,transport=transport,core=core))
    if mode is not None:
        basename,size,sha=IDENTITIES[mode]
        result.update(moss_lora_storage_dtype=mode.upper(),moss_lora_selected_basename=basename,
            moss_lora_selected_bytes=size,moss_lora_selected_sha256=sha,
            moss_lora_storage_lossy=mode=='f16',moss_lora_precision_scope='CPU_transport_only_no_model_no_GPU' if transport else 'private_Vulkan_FP32_shared_LoRA_only')
    return result

def contract(mode=None):
    if mode is None:return {}
    result=fields(mode)
    result.update(schema='ps5-ai-moss-experimental-combined-lora-precision-trial-v1',
        private_LoRA_precision_GPU_readbacks_required=8,private_LoRA_pipeline_receipts_required=4,
        expected_private_LoRA_matrix_recordings=2016,expected_each_504_LoRA_matrices=4,
        new_core_compilation_required=True,old_core_reuse_allowed=False,image_quality_parity=False)
    if mode=='f16':result['moss_lora_composition_schema']='ps5-ai-moss-nsfw07-turbo1-r144-f16-storage-v1'
    return result

def validate_core(manifest,mode=None):
    expected=fields(mode,core=True);declared=any(k in manifest for k in expected)
    if declared:
        if any(type(manifest.get(k))is not type(v) or manifest[k]!=v for k,v in expected.items()):raise RuntimeError('Core LoRA precision identity mismatch')
        own=Path(__file__).resolve();rows=manifest.get('lora_precision_policy_inputs')
        actual={'path':own.as_posix(),'bytes':own.stat().st_size,'sha256':hashlib.sha256(own.read_bytes()).hexdigest()}
        if type(rows)is not list or len(rows)!=1 or type(rows[0])is not dict or set(rows[0])!=set(actual):raise RuntimeError('Missing exact precision policy input')
        row=dict(rows[0]);row['path']=Path(row['path']).resolve().as_posix()
        if row!=actual:raise RuntimeError('Changed precision policy helper')
    elif mode is not None:raise RuntimeError('Precision experiment requires freshly declared core identity')
    commands=manifest.get('commands',[]);members=manifest.get('members',[])
    if len(commands)!=75 or len(members)!=75:raise RuntimeError('Precision proof needs actual75 recipes')
    indices=set()
    for row in members:
        i=row.get('command_index')
        if type(i)is not int or i not in range(75) or i in indices:raise RuntimeError('Precision recipe index mismatch')
        indices.add(i);cmd=commands[i]
        if type(cmd)is not list or any(type(a)is not str for a in cmd) or cmd.count('-c')!=1 or Path(cmd[cmd.index('-c')+1]).resolve()!=Path(row['source']).resolve():raise RuntimeError('Precision proof needs actual compiler source')
        actual=[a for a in cmd if any(m in a for m in MACROS)]
        if actual!=(flags(mode) if declared else []):raise RuntimeError('Missing/conflicting precision compiler macros')
        if mode is not None:
            for flag in flags(mode,core=True)[3:]:
                macro=flag.split('=')[0][2:]
                if [a for a in cmd if a.startswith('-D'+macro+'=')]!=[flag]:raise RuntimeError('Missing/conflicting actual core trial guard')
    if mode is not None and (manifest.get('freshly_compiled_members')!=75 or manifest.get('old_archive_members_reused')!=0):raise RuntimeError('Precision experiment requires actual fresh75, never carry-forward')
    return dict(expected,actual_compile_commands_verified=75,historical_default_off_without_identity=not declared)
