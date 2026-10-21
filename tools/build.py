from moss_kernel_policy import flags as kernel_flags, fields as kernel_fields, validate as validate_kernel, validate_core as validate_kernel_core, fa_flags as kernel_fa_flags
from moss_dit_policy import flags as dit_flags, fields as dit_fields, validate as validate_dit, validate_core as validate_dit_core, validate_profile as validate_dit_profile
from moss_lora_precision_policy import flags as lora_precision_flags, fields as lora_precision_fields, contract as lora_precision_contract, validate as validate_lora_precision, validate_core as validate_core_lora_precision
# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows local native rebuild from exported sources + reviewed dependencies.

Does not connect to a PS5, install a title, patch firmware, or run inference.
See docs/build.md for the explicit dependency-build limitation.
"""
import argparse, hashlib, json, os, re, shutil, struct, subprocess
from pathlib import Path
from moss_feature_policy import (feature_compile_flags, validate_feature_modes,
    feature_manifest_fields, feature_contract_updates, validate_core_features,
    ffn_compile_flags,ffn_manifest_fields,validate_ffn_parent,ffn_contract_updates,validate_core_ffn)
from build_input_snapshot import collect as collect_inputs, require_unchanged

ROOT=Path(__file__).resolve().parents[1]

MOSS_SMALL_TRIAL_CONTRACT={
    'schema':'ps5-ai-moss-experimental-small-trial-v1',
    'experimental_small_trial':True,'production_runtime_admitted':False,
    'trial_10g':True,'gpu_pool_bytes':10737418240,'working_reserve_bytes':1073741824,
    'max_vram_gib':'Vulkan0=9.5','width':256,'height':256,'steps':4,
    'cfg_scale':3.0,'sampler':'euler','flow_shift':5.0,'max_generation_attempts':1,
    'selected_moss_lora_required':True,'startup_readback_gates_required':True}
MOSS_QUALITY_COMPARISON_CONTRACT={key:value for key,value in MOSS_SMALL_TRIAL_CONTRACT.items() if key!='steps'}
MOSS_QUALITY_COMPARISON_CONTRACT.update({
    'schema':'ps5-ai-moss-experimental-quality-step-comparison-v1','allowed_steps':[4,50],
    'seed':20261019,'moss_lora_strength':0.7})
MOSS_BENCHMARK_1024_CONTRACT=dict(MOSS_SMALL_TRIAL_CONTRACT)
MOSS_BENCHMARK_1024_CONTRACT.update({
    'schema':'ps5-ai-moss-experimental-1024-cold-warm-v1','width':1024,'height':1024,
    'steps':50,'seed':20261020,'moss_lora_strength':0.7,'max_generation_attempts':2})

MOSS_PERFORMANCE_VARIANTS={0:('baseline-fp32','fp32'),1:('fp16','fp16'),2:('submit-fusion-fp32','fp32')}

def moss_sampler_compile_flags(enabled):
    if type(enabled) is not bool:raise RuntimeError('Sampler trial must be an explicit bool')
    return ['-DPS5_AI_MOSS_SAMPLER_TRIAL='+str(int(enabled))]

def validate_moss_sampler_mode(enabled,accumulation,gpu_profile,q4_candidate,variant,precision,experimental,trial_10g):
    moss_sampler_compile_flags(enabled)
    if enabled and (accumulation is not True or gpu_profile is not False or q4_candidate is not False or
                    type(variant) is not int or variant!=1 or precision!='fp16' or experimental is not True or trial_10g is not True):
        raise RuntimeError('Sampler trial requires the distinct FP16/F32 small10GiB performance build with GPU profiling/Q4 candidates disabled')

def moss_performance_contract(variant,moss_f32_accumulation=False,moss_gpu_profile=False,moss_q4_bk64_raw64=False,moss_sampler_trial=False,moss_spectrum_trial=False,moss_bounded_weight_upload_trial=False,moss_turbo_trial=False,moss_ffn_token_chunk_trial=False,moss_ffn_chunk_tokens=0,moss_lora_storage=None):
    if type(variant) is not int or variant not in MOSS_PERFORMANCE_VARIANTS:
        raise RuntimeError('Performance variant must be explicit integer0,1,or2')
    if type(moss_f32_accumulation) is not bool or (moss_f32_accumulation and variant!=1):
        raise RuntimeError('F32 accumulation requires the explicit FP16 performance variant')
    if type(moss_gpu_profile) is not bool or (moss_gpu_profile and not moss_f32_accumulation):
        raise RuntimeError('GPU operator profiling requires the explicit FP16/F32 accumulation variant')
    moss_q4_bk64_compile_flags(moss_q4_bk64_raw64)
    if moss_q4_bk64_raw64 and (not moss_f32_accumulation or moss_gpu_profile):
        raise RuntimeError('Q4 BK64/raw64 candidate B requires FP16/F32 accumulation with GPU profiling disabled')
    name,precision=MOSS_PERFORMANCE_VARIANTS[variant]
    validate_moss_sampler_mode(moss_sampler_trial,moss_f32_accumulation,moss_gpu_profile,moss_q4_bk64_raw64,variant,precision,True,True)
    contract={key:value for key,value in MOSS_BENCHMARK_1024_CONTRACT.items() if key not in ('width','height')}
    contract.update({'schema':'ps5-ai-moss-experimental-performance-v1',
        'allowed_sizes':[[256,256],[1024,1024]],'max_generation_attempts':4,
        'performance_variant':variant,'performance_variant_name':name,'requested_precision':precision,
        'small256_success_and_visual_review_before1024':True,'automatic_POST_retries':False})
    if moss_f32_accumulation:contract['moss_f32_accumulation']=True
    if moss_gpu_profile:contract['moss_gpu_profile']=True
    if moss_q4_bk64_raw64:contract['moss_q4_bk64_raw64']=True
    validate_feature_modes(moss_spectrum_trial,moss_bounded_weight_upload_trial,moss_turbo_trial,accumulation=moss_f32_accumulation,
        gpu_profile=moss_gpu_profile,q4=moss_q4_bk64_raw64,sampler=moss_sampler_trial,variant=variant,
        precision=precision,experimental=True,trial_10g=True)
    contract.update(feature_contract_updates(moss_spectrum_trial,moss_bounded_weight_upload_trial,moss_turbo_trial))
    if moss_sampler_trial:contract.update({'schema':'ps5-ai-moss-experimental-sampler-trial-v1',
        'moss_sampler_trial':True,'moss_sampler_policy':'res_multistep_simple_30_cfg4_v1',
        'steps':30,'cfg_scale':4.0,'sampler':'res_multistep','scheduler':'simple',
        'expected_dit_forwards':60,'expected_q4_f32acc_recordings':9960,
        'negative_cfg_skip_scope':'all_negative_forwards_existing_MOSS_policy'})
    validate_ffn_parent(moss_ffn_token_chunk_trial,moss_ffn_chunk_tokens,spectrum=moss_spectrum_trial,bounded_upload=moss_bounded_weight_upload_trial,turbo=moss_turbo_trial,accumulation=moss_f32_accumulation)
    if moss_ffn_token_chunk_trial:
        contract.pop('expected_q4_f32acc_recordings',None)
        contract.update(ffn_contract_updates(moss_ffn_token_chunk_trial,moss_ffn_chunk_tokens))
    validate_lora_precision(moss_lora_storage,turbo=moss_turbo_trial,upload=moss_bounded_weight_upload_trial,
        spectrum=moss_spectrum_trial,ffn=moss_ffn_token_chunk_trial,accumulation=moss_f32_accumulation,
        gpu_profile=moss_gpu_profile,q4=moss_q4_bk64_raw64,sampler=moss_sampler_trial,
        variant=variant,precision=precision,experimental=True,trial_10g=True)
    contract.update(lora_precision_contract(moss_lora_storage))
    return contract

def validate_moss_performance_precision(variant,precision):
    if variant is not None:
        expected=moss_performance_contract(variant)['requested_precision']
        if precision!=expected:raise RuntimeError('Performance variant/precision mismatch')

def moss_accumulation_compile_flags(enabled):
    if type(enabled) is not bool:raise RuntimeError('F32 accumulation policy must be an explicit bool')
    return ['-DPS5_AI_MOSS_F32_ACCUMULATION='+str(int(enabled))]

def moss_accumulation_policy(enabled):
    moss_accumulation_compile_flags(enabled)
    return 'force_f32' if enabled else 'model_default'

def validate_moss_accumulation_mode(enabled,performance_variant,precision,experimental,trial_10g):
    moss_accumulation_compile_flags(enabled)
    if enabled and (performance_variant!=1 or type(performance_variant) is not int or precision!='fp16' or experimental is not True or trial_10g is not True):
        raise RuntimeError('F32 accumulation requires explicit small10GiB performance variant1 and fp16')

def moss_gpu_profile_compile_flags(enabled):
    if type(enabled) is not bool:raise RuntimeError('GPU operator profiling must be an explicit bool')
    return ['-DPS5_AI_MOSS_GPU_PROFILE='+str(int(enabled))]

def validate_moss_gpu_profile_mode(enabled,accumulation,performance_variant,precision,experimental,trial_10g):
    moss_gpu_profile_compile_flags(enabled)
    if enabled:
        if accumulation is not True:raise RuntimeError('GPU operator profiling requires explicit F32 accumulation')
        validate_moss_accumulation_mode(True,performance_variant,precision,experimental,trial_10g)

def validate_moss_gpu_profile_recipes(profile):
    ggml=profile.get('ggml_compile')
    if not isinstance(ggml,list) or not ggml:
        raise RuntimeError('Expected a nonempty native GGML compiler recipe list')
    recipes=[profile.get('platform_compile'),*ggml]
    if any(not isinstance(recipe,list) or not recipe or any(not isinstance(arg,str) for arg in recipe) for recipe in recipes):
        raise RuntimeError('Expected complete native platform/GGML compiler recipes')
    if any('PS5_AI_MOSS_GPU_PROFILE' in arg or 'PS5_AI_MOSS_DIT_GPU_PROFILE' in arg for recipe in recipes for arg in recipe):
        raise RuntimeError('Native recipes cannot override the GPU operator profiling policy')

def moss_q4_bk64_compile_flags(enabled):
    if type(enabled) is not bool:raise RuntimeError('Q4 BK64/raw64 candidate B must be an explicit bool')
    return ['-DPS5_AI_MOSS_Q4_BK64_RAW64='+str(int(enabled))]

def validate_moss_q4_bk64_mode(enabled,accumulation,gpu_profile,performance_variant,precision,experimental,trial_10g):
    moss_q4_bk64_compile_flags(enabled)
    if type(gpu_profile) is not bool:raise RuntimeError('GPU profiling policy must be an explicit bool')
    if enabled:
        if accumulation is not True or gpu_profile is not False:
            raise RuntimeError('Q4 BK64/raw64 candidate B requires explicit F32 accumulation and GPU profiling disabled')
        validate_moss_accumulation_mode(True,performance_variant,precision,experimental,trial_10g)

def validate_moss_q4_bk64_recipes(profile):
    # The existing validator also proves every raw recipe and argument is typed.
    validate_moss_gpu_profile_recipes(profile)
    recipes=[profile['platform_compile'],*profile['ggml_compile']]
    if any('PS5_AI_MOSS_Q4_BK64_RAW64' in arg for recipe in recipes for arg in recipe):
        raise RuntimeError('Native recipes cannot override the Q4 BK64/raw64 candidate B policy')

def validate_moss_core_accumulation(manifest,required):
    """Bind declared policy to every actual member/compiler command, not labels."""
    expected=moss_accumulation_compile_flags(required)[0]
    declared='moss_f32_accumulation' in manifest
    actual=manifest.get('moss_f32_accumulation',False)
    if type(actual) is not bool or actual!=required:
        raise RuntimeError('Core/native MOSS F32 accumulation policy mismatch')
    if declared:
        if manifest.get('moss_accumulation_policy')!=moss_accumulation_policy(required) or manifest.get('moss_accumulation_compile_flag')!=expected:
            raise RuntimeError('Core accumulation policy/flag declaration mismatch')
    elif any(key in manifest for key in ('moss_accumulation_policy','moss_accumulation_compile_flag')):
        raise RuntimeError('Partial historical core precision declaration')
    commands=manifest.get('commands',[]);members=manifest.get('members',[])
    if len(commands)!=75 or len(members)!=75:
        raise RuntimeError('Core precision proof needs all75 actual compile commands/members')
    indices=set()
    for row in members:
        index=row.get('command_index')
        if type(index) is not int or index not in range(75) or index in indices:
            raise RuntimeError('Core precision proof has duplicate/invalid command index')
        indices.add(index);command=commands[index]
        if not isinstance(command,list) or not command or any(not isinstance(arg,str) for arg in command) or command.count('-c')!=1:
            raise RuntimeError('Core precision proof is not a compiler command')
        offset=command.index('-c')
        if offset+1>=len(command) or not isinstance(row.get('source'),str) or Path(command[offset+1]).resolve()!=Path(row['source']).resolve():
            raise RuntimeError('Core precision command/source binding mismatch')
        definitions=[arg for arg in command if 'PS5_AI_MOSS_F32_ACCUMULATION' in arg]
        if definitions!=([expected] if declared else []):
            raise RuntimeError('Core precision policy lacks a single exact macro on every compile')
    return {'moss_f32_accumulation':required,'moss_accumulation_policy':moss_accumulation_policy(required),
            'historical_default0_without_macro':not declared,'actual_compile_commands_verified':75}

def same_typed_contract_value(actual,expected):
    return type(actual) is type(expected) and (
        len(actual)==len(expected) and all(same_typed_contract_value(a,e) for a,e in zip(actual,expected))
        if isinstance(expected,list) else actual==expected)

def moss_trial_flags(enabled,trial_10g,contract=None,quality_comparison=False,benchmark_1024=False,performance_variant=None,moss_f32_accumulation=False,moss_gpu_profile=False,moss_q4_bk64_raw64=False,moss_sampler_trial=False,moss_spectrum_trial=False,moss_bounded_weight_upload_trial=False,moss_turbo_trial=False,moss_ffn_token_chunk_trial=False,moss_ffn_chunk_tokens=0,moss_lora_storage=None):
    """A trial is permission to RUN verification, never prior GPU admission."""
    if any(type(value) is not bool for value in (enabled,trial_10g,quality_comparison,benchmark_1024,moss_sampler_trial)):
        raise RuntimeError('MOSS trial switches must be explicit booleans')
    validate_ffn_parent(moss_ffn_token_chunk_trial,moss_ffn_chunk_tokens,spectrum=moss_spectrum_trial,bounded_upload=moss_bounded_weight_upload_trial,turbo=moss_turbo_trial,accumulation=moss_f32_accumulation)
    moss_accumulation_compile_flags(moss_f32_accumulation)
    moss_gpu_profile_compile_flags(moss_gpu_profile)
    moss_q4_bk64_compile_flags(moss_q4_bk64_raw64)
    if moss_q4_bk64_raw64 and (moss_f32_accumulation is not True or moss_gpu_profile is not False or enabled is not True or trial_10g is not True or type(performance_variant) is not int or performance_variant!=1):
        raise RuntimeError('Q4 BK64/raw64 candidate B requires the explicit FP16/F32 trial with profiling disabled')
    if moss_gpu_profile and (moss_f32_accumulation is not True or enabled is not True or trial_10g is not True or type(performance_variant) is not int or performance_variant!=1):
        raise RuntimeError('GPU operator profiling requires the explicit FP16/F32 performance trial')
    if moss_f32_accumulation and (enabled is not True or trial_10g is not True or type(performance_variant) is not int or performance_variant!=1):
        raise RuntimeError('F32 accumulation requires the explicit FP16 performance trial')
    if moss_sampler_trial and (moss_f32_accumulation is not True or moss_gpu_profile is not False or moss_q4_bk64_raw64 is not False or enabled is not True or trial_10g is not True or type(performance_variant) is not int or performance_variant!=1):
        raise RuntimeError('Sampler trial requires the explicit FP16/F32 small10GiB performance build')
    validate_feature_modes(moss_spectrum_trial,moss_bounded_weight_upload_trial,moss_turbo_trial,accumulation=moss_f32_accumulation,
        gpu_profile=moss_gpu_profile,q4=moss_q4_bk64_raw64,sampler=moss_sampler_trial,variant=performance_variant,
        precision=MOSS_PERFORMANCE_VARIANTS.get(performance_variant,('', 'fp32'))[1],experimental=enabled,trial_10g=trial_10g)
    if performance_variant is not None:moss_performance_contract(performance_variant,moss_f32_accumulation,moss_gpu_profile,moss_q4_bk64_raw64,moss_sampler_trial,moss_spectrum_trial,moss_bounded_weight_upload_trial,moss_turbo_trial,moss_ffn_token_chunk_trial,moss_ffn_chunk_tokens,moss_lora_storage)
    if sum((quality_comparison,benchmark_1024,performance_variant is not None))>1:raise RuntimeError('MOSS trial modes are mutually exclusive')
    if not enabled:
        if trial_10g or quality_comparison or benchmark_1024 or performance_variant is not None:raise RuntimeError('Trial switches require --experimental-small-trial')
        return ['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=0','-DPS5_AI_MOSS_TRIAL_10G=0']
    expected_contract=moss_performance_contract(performance_variant,moss_f32_accumulation,moss_gpu_profile,moss_q4_bk64_raw64,moss_sampler_trial,moss_spectrum_trial,moss_bounded_weight_upload_trial,moss_turbo_trial,moss_ffn_token_chunk_trial,moss_ffn_chunk_tokens,moss_lora_storage) if performance_variant is not None else MOSS_BENCHMARK_1024_CONTRACT if benchmark_1024 else MOSS_QUALITY_COMPARISON_CONTRACT if quality_comparison else MOSS_SMALL_TRIAL_CONTRACT
    valid_contract=isinstance(contract,dict) and set(contract)==set(expected_contract) and all(
        same_typed_contract_value(contract[key],expected)
        for key,expected in expected_contract.items())
    if not trial_10g or not valid_contract:
        raise RuntimeError('Experimental MOSS requires the exact reviewed small10GiB trial profile and both switches')
    return ['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_TRIAL_10G=1']+(
        ['-DPS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL=1'] if quality_comparison else [])+(
        ['-DPS5_AI_MOSS_1024_COLD_WARM_TRIAL=1'] if benchmark_1024 else [])+(
        ['-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1','-DPS5_AI_MOSS_PERF_VARIANT='+str(performance_variant)] if performance_variant is not None else [])
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def archive_members(path):
    result={}
    with Path(path).open('rb') as stream:
        if stream.read(8)!=b'!<arch>\n':raise RuntimeError('Expected a regular archive')
        names=b''
        while header:=stream.read(60):
            if len(header)!=60 or header[58:]!=b'`\n':raise RuntimeError('Malformed archive header')
            name=header[:16].decode('ascii').strip();size=int(header[48:58]);body=stream.read(size)
            if len(body)!=size:raise RuntimeError('Truncated archive member')
            if size&1 and stream.read(1)!=b'\n':raise RuntimeError('Malformed archive padding')
            if name=='//':names=body;continue
            if name in ('/','/SYM64/'):continue
            if name.startswith('/') and name[1:].isdigit():
                offset=int(name[1:]);name=names[offset:names.index(b'/\n',offset)].decode('utf-8')
            else:name=name.rstrip('/')
            if name in result:raise RuntimeError('Duplicate archive member: '+name)
            result[name]={'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}
    return result

def check_archive_update(before,after,allowed):
    if set(before)!=set(after):raise RuntimeError('Archive member set changed')
    changed=sorted(name for name in before if before[name]!=after[name])
    if not set(changed).issubset(set(allowed)):raise RuntimeError('Unreviewed archive members changed: '+str(changed))
    return changed
def validate_moss_benchmark_gateway_wait_change(source,original):
    """Only the reviewed1800000->7200000 literal may differ from core history."""
    data=Path(source).read_bytes()
    new=b'#define OPENAI_WAIT_MS UINT64_C(7200000)'
    old=b'#define OPENAI_WAIT_MS UINT64_C(1800000)'
    if data.count(new)!=1:raise RuntimeError('Expected exactly one reviewed2h gateway wait literal')
    restored=data.replace(new,old,1)
    if len(restored)!=original['bytes'] or hashlib.sha256(restored).hexdigest()!=original['sha256']:
        raise RuntimeError('Gateway change exceeds the reviewed wait-only literal')
    return {'old_wait_ms':1800000,'new_wait_ms':7200000,'all_other_bytes_identical':True,
            'restored_original_sha256':original['sha256']}
def validate_moss_core(manifest_path, expected_members, build_config, quality_native_only=False, carryforward=None, benchmark_gateway_only=False,moss_f32_accumulation=False,moss_q4_bk64_raw64=False,moss_sampler_trial=False,moss_spectrum_trial=False,moss_bounded_weight_upload_trial=False,moss_turbo_trial=False,moss_ffn_token_chunk_trial=False,moss_ffn_chunk_tokens=0,moss_lora_storage=None):
    """Consume only a complete, immutable 75-member MOSS core rebuild."""
    moss_q4_bk64_compile_flags(moss_q4_bk64_raw64)
    if moss_q4_bk64_raw64 and (quality_native_only is not True or moss_f32_accumulation is not True or benchmark_gateway_only is not False):
        raise RuntimeError('Q4 candidate core carryforward requires the exact native-only F32 trial')
    moss_sampler_compile_flags(moss_sampler_trial)
    if moss_sampler_trial and (quality_native_only is not True or moss_f32_accumulation is not True or moss_q4_bk64_raw64 is not False or benchmark_gateway_only is not False):
        raise RuntimeError('Sampler core carryforward requires the exact native-only F32 trial')
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if (manifest.get('schema')!='moss-core-build-v1' or
        manifest.get('local_build_only') is not True or
        manifest.get('console_executed') is not False or
        manifest.get('hash_inputs_stable') is not True or
        manifest.get('state')!='crossbuilt_not_PS5_executed' or
        manifest.get('freshly_compiled_members')!=75 or
        manifest.get('old_archive_members_reused')!=0 or
        manifest.get('MOSS_RUNTIME_ADMITTED')!=0):
        raise RuntimeError('Expected a completed local MOSS full-core manifest')
    validate_moss_core_accumulation(manifest,moss_f32_accumulation)
    validate_core_features(manifest,moss_spectrum_trial,moss_bounded_weight_upload_trial,moss_turbo_trial)
    validate_core_lora_precision(manifest,moss_lora_storage)
    validate_core_ffn(manifest,moss_ffn_token_chunk_trial,moss_ffn_chunk_tokens)
    if moss_ffn_token_chunk_trial and quality_native_only:raise RuntimeError('FFN requires a fresh75 compilation')
    if (moss_spectrum_trial or moss_bounded_weight_upload_trial or moss_turbo_trial) and quality_native_only:
        raise RuntimeError('Inference feature cores require fresh75; native-only carryforward is forbidden')
    pins=manifest.get('source_pins',{})
    if pins!={'stable_diffusion_base':'8cd731e7b4667eb8f40a153315b9a7125d09a58d',
              'ggml':'d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93',
              'moss_backport_reference':'68285df319457e2d79310b48513e4ba1bf8ebe9a'}:
        raise RuntimeError('MOSS core source revisions do not match the reviewed backport')
    for key in ('SD','SDK','LLVM'):
        if Path(manifest.get('config_paths',{}).get(key,'')).resolve()!=Path(build_config[key]).resolve():
            raise RuntimeError('MOSS core/native build checkout mismatch: '+key)
    if (Path(manifest['builder']['path']).resolve()!=ROOT/'tools/build_moss_core.py' or
        Path(manifest['profile']['path']).resolve()!=ROOT/'build-profiles/moss-core.json'):
        raise RuntimeError('MOSS core belongs to a different repository')
    members=manifest.get('member_names',[])
    if len(members)!=75 or len(set(members))!=75 or set(members)!=set(expected_members):
        raise RuntimeError('MOSS full-core member set does not match all75 reviewed members')
    archive=manifest['archive'];path=Path(archive['path'])
    if not path.is_file() or path.stat().st_size!=archive['bytes'] or digest(path)!=archive['sha256']:
        raise RuntimeError('MOSS core archive changed or is missing')
    if set(archive_members(path))!=set(members):
        raise RuntimeError('MOSS archive contents do not match manifest')
    inputs=manifest.get('sources',[])+manifest.get('header_inputs',[])+manifest.get('overlay_inputs',[])+manifest.get('policy_inputs',[])+manifest.get('lora_precision_policy_inputs',[])+[manifest['profile'],manifest['builder']]
    if 'kernel_policy_input'in manifest:inputs.append(manifest['kernel_policy_input'])
    if len(manifest.get('sources',[]))!=75 or not manifest.get('header_inputs'):
        raise RuntimeError('Missing full-core source/header provenance')
    changed=[]
    for record in inputs:
        source=Path(record['path'])
        if not source.is_file() or source.stat().st_size!=record['bytes'] or digest(source)!=record['sha256']:
            changed.append(record)
    if changed:
        # This exception is restricted to the three native-only inputs changed
        # for explicit native-only trials. Every original compiler .d is pinned
        # and checked; a change to ANY consumed core input remains a hard error.
        allowed={ROOT/'native/app/moss_trial_contract.hpp',ROOT/'native/app/engine_sdcpp.cpp',
                 ROOT/'native/app/moss_precision_gate.inc'}
        # The separately reviewed1024 gateway change only extends its wait.
        # It still must be absent from all75 actual core dependency files.
        if benchmark_gateway_only is True:allowed.add(ROOT/'native/gateway/openai.inc')
        if moss_q4_bk64_raw64:
            # These existing superset files must remain absent from every
            # actual pinned core .d closure. New native-only header/gate
            # files have no old rows; the new native freeze binds them.
            allowed={ROOT/'native/app/engine_sdcpp.cpp',
                ROOT/'patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan.cpp',
                Path(build_config['SD']).resolve()/'ggml/src/ggml-vulkan/ggml-vulkan.cpp'}
        if moss_sampler_trial:
            allowed={ROOT/path for path in ('native/app/engine_sdcpp.cpp','native/app/moss_trial_contract.hpp',
                'native/app/server.c','native/gateway/server.c','native/gateway/openai.inc',
                'tools/build.py','tools/build_gateway.py')}
        if quality_native_only is not True or any(Path(row['path']).resolve() not in allowed for row in changed):
            raise RuntimeError('MOSS core source/header changed since compile: '+str(changed[0]['path']))
        gateway_wait_proof={}
        for row in changed:
            if benchmark_gateway_only is True and not moss_sampler_trial and Path(row['path']).resolve()==ROOT/'native/gateway/openai.inc':
                gateway_wait_proof=validate_moss_benchmark_gateway_wait_change(row['path'],row)
        members_proof=manifest.get('members',[])
        if len(members_proof)!=75 or {row['member'] for row in members_proof}!=set(members):
            raise RuntimeError('Missing all75 actual dependency records for native-only core carryforward')
        consumed=set();dep_records=[]
        recorded={Path(row['path']).resolve() for row in inputs}
        for member in members_proof:
            dependency=member['dependencies'];dep=Path(dependency['path']).resolve()
            expected_dep=Path(manifest_path).resolve().parent/'deps'/(member['member']+'.d')
            if dep!=expected_dep or not dep.is_file() or dep.stat().st_size!=dependency['bytes'] or digest(dep)!=dependency['sha256']:
                raise RuntimeError('Actual core dependency file missing/changed: '+str(dep))
            text=dep.read_text(encoding='utf-8').replace('\\\n',' ')
            if ': ' not in text:raise RuntimeError('Unrecognized actual compiler dependency format')
            names=re.findall(r'(?:\\[^\r\n]|[^\s])+',text.split(': ',1)[1])
            paths={Path(name.replace('\\ ',' ')).resolve() for name in names}
            if Path(member['source']).resolve() not in paths or not paths:
                raise RuntimeError('Core dependency file omits its actual source')
            if not paths.issubset(recorded):
                raise RuntimeError('Actual core dependency has no original hash record: '+str(paths-recorded))
            consumed.update(paths);dep_records.append(dependency)
        if any(Path(row['path']).resolve() in consumed for row in changed):
            raise RuntimeError('Native-only carryforward denied: changed file is a consumed core input')
        if carryforward is not None:
            carryforward.update({'schema':'moss-sampler-native-only-core-carryforward-v1' if moss_sampler_trial else 'moss-q4-bk64-raw64-native-only-core-carryforward-v1' if moss_q4_bk64_raw64 else 'moss-quality-native-only-core-carryforward-v1',
                'original_manifest_sha256':digest(manifest_path),'original_manifest_unchanged':True,
                'archive_sha256':archive['sha256'],'new_core_compilation':False,
                'moss_q4_bk64_raw64':moss_q4_bk64_raw64,'moss_sampler_trial':moss_sampler_trial,
                'allowed_changed_superset_paths':sorted(str(path) for path in allowed),
                'reviewed_gateway_wait_only_change':gateway_wait_proof,
                'actual_dependency_files_verified':75,'actual_dependency_paths':len(consumed),
                'changed_superset_files_not_consumed':[{'path':row['path'],'before':row,
                    'after':{'bytes':Path(row['path']).stat().st_size,'sha256':digest(row['path'])}} for row in changed],
                'dependency_files':dep_records})
    return path,manifest

def mapped(data):
    offset=struct.unpack_from('<Q',data,32)[0];size,count=struct.unpack_from('<HH',data,54)
    return [(p,hashlib.sha256(data[p[2]:p[2]+p[5]]).hexdigest()) for i in range(count)
            if (p:=struct.unpack_from('<IIQQQQQQ',data,offset+i*size))[0]==1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--precision',choices=['fp32','fp16'],default='fp32');p.add_argument('--core-manifest',type=Path,required=True);p.add_argument('--experimental-small-trial',action='store_true');p.add_argument('--trial-10g',action='store_true');p.add_argument('--quality-step-comparison-trial',action='store_true');p.add_argument('--benchmark-1024-cold-warm-trial',action='store_true');p.add_argument('--performance-trial-variant',type=int,choices=(0,1,2));p.add_argument('--moss-f32-accumulation',action='store_true');p.add_argument('--moss-gpu-profile',action='store_true');p.add_argument('--moss-q4-bk64-raw64',action='store_true');p.add_argument('--moss-sampler-trial',action='store_true');p.add_argument('--moss-spectrum-trial',action='store_true');p.add_argument('--moss-bounded-weight-upload-trial',action='store_true');p.add_argument('--moss-turbo-trial',action='store_true');p.add_argument('--moss-ffn-token-chunk-trial',action='store_true');p.add_argument('--moss-ffn-chunk-tokens',type=int,choices=(0,2048),default=0);p.add_argument('--moss-lora-storage',choices=('f32','f16'));p.add_argument('--moss-dit-optimization-mask',type=int,choices=(0,1,3,7,15));p.add_argument('--moss-dit-gpu-profile',action='store_true');p.add_argument('--moss-dit-kernel-mask',type=int,choices=range(16));p.add_argument('--moss-fa-shmem-no-limit',action='store_true');p.add_argument('--moss-img2img-trial',action='store_true');a=p.parse_args()
    c=json.loads(a.config.read_text());out=a.out.resolve()
    if (out/'manifest.json').exists():raise RuntimeError('Completed output exists; choose a new directory.')
    c.update(ROOT=str(ROOT),OUT=str(out));profile=json.loads((ROOT/'build-profiles/native.json').read_text())
    validate_moss_gpu_profile_recipes(profile)
    validate_moss_q4_bk64_recipes(profile)
    if any('PS5_AI_MOSS_SAMPLER_TRIAL' in arg for recipe in [profile['platform_compile'],*profile['ggml_compile']] for arg in recipe):
        raise RuntimeError('Native recipes cannot override the sampler trial policy')
    validate_moss_performance_precision(a.performance_trial_variant,a.precision)
    validate_moss_accumulation_mode(a.moss_f32_accumulation,a.performance_trial_variant,a.precision,a.experimental_small_trial,a.trial_10g)
    accumulation_flags=moss_accumulation_compile_flags(a.moss_f32_accumulation)
    validate_moss_gpu_profile_mode(a.moss_gpu_profile,a.moss_f32_accumulation,a.performance_trial_variant,a.precision,a.experimental_small_trial,a.trial_10g)
    dit_profile_flags=validate_dit_profile(a.moss_dit_gpu_profile,mask=a.moss_dit_optimization_mask,legacy_gpu_profile=a.moss_gpu_profile,
        accumulation=a.moss_f32_accumulation,turbo=a.moss_turbo_trial,upload=a.moss_bounded_weight_upload_trial,
        spectrum=a.moss_spectrum_trial,ffn=a.moss_ffn_token_chunk_trial,lora=a.moss_lora_storage,q4=a.moss_q4_bk64_raw64,sampler=a.moss_sampler_trial)
    gpu_profile_flags=moss_gpu_profile_compile_flags(a.moss_gpu_profile or a.moss_dit_gpu_profile)+dit_profile_flags
    validate_moss_q4_bk64_mode(a.moss_q4_bk64_raw64,a.moss_f32_accumulation,a.moss_gpu_profile,a.performance_trial_variant,a.precision,a.experimental_small_trial,a.trial_10g)
    q4_candidate_flags=moss_q4_bk64_compile_flags(a.moss_q4_bk64_raw64)
    validate_moss_sampler_mode(a.moss_sampler_trial,a.moss_f32_accumulation,a.moss_gpu_profile,a.moss_q4_bk64_raw64,a.performance_trial_variant,a.precision,a.experimental_small_trial,a.trial_10g)
    sampler_flags=moss_sampler_compile_flags(a.moss_sampler_trial)
    validate_feature_modes(a.moss_spectrum_trial,a.moss_bounded_weight_upload_trial,a.moss_turbo_trial,accumulation=a.moss_f32_accumulation,
        gpu_profile=a.moss_gpu_profile,q4=a.moss_q4_bk64_raw64,sampler=a.moss_sampler_trial,
        variant=a.performance_trial_variant,precision=a.precision,experimental=a.experimental_small_trial,trial_10g=a.trial_10g)
    feature_flags=feature_compile_flags(a.moss_spectrum_trial,a.moss_bounded_weight_upload_trial,a.moss_turbo_trial)
    validate_lora_precision(a.moss_lora_storage,turbo=a.moss_turbo_trial,upload=a.moss_bounded_weight_upload_trial,
        spectrum=a.moss_spectrum_trial,ffn=a.moss_ffn_token_chunk_trial,accumulation=a.moss_f32_accumulation,
        gpu_profile=a.moss_gpu_profile,q4=a.moss_q4_bk64_raw64,sampler=a.moss_sampler_trial,
        variant=a.performance_trial_variant,precision=a.precision,experimental=a.experimental_small_trial,trial_10g=a.trial_10g)
    dit_enabled=a.moss_dit_optimization_mask is not None
    validate_dit(dit_enabled,accumulation=a.moss_f32_accumulation,turbo=a.moss_turbo_trial,upload=a.moss_bounded_weight_upload_trial,spectrum=a.moss_spectrum_trial,ffn=a.moss_ffn_token_chunk_trial,lora=a.moss_lora_storage)
    if dit_enabled and (a.moss_gpu_profile or a.moss_q4_bk64_raw64 or a.moss_sampler_trial):raise RuntimeError('DiT scope excludes extra math/profile candidates')
    validate_dit_core(json.loads(a.core_manifest.read_text(encoding='utf-8')),dit_enabled)
    kernel_enabled=a.moss_dit_kernel_mask is not None
    validate_kernel(kernel_enabled,dit=dit_enabled and a.moss_dit_optimization_mask==15,accumulation=a.moss_f32_accumulation,turbo=a.moss_turbo_trial,upload=a.moss_bounded_weight_upload_trial,mask=a.moss_dit_kernel_mask)
    if kernel_enabled and a.moss_dit_gpu_profile:raise RuntimeError('Kernel timing uses uninstrumented matched runs')
    validate_kernel_core(json.loads(a.core_manifest.read_text(encoding='utf-8')),kernel_enabled)
    lora_flags=lora_precision_flags(a.moss_lora_storage)
    for recipe in [profile['platform_compile'],*profile['ggml_compile']]:
        if any('PS5_AI_MOSS_LORA_' in arg for arg in recipe):raise RuntimeError('Native recipes cannot override private LoRA precision')
    validate_ffn_parent(a.moss_ffn_token_chunk_trial,a.moss_ffn_chunk_tokens,spectrum=a.moss_spectrum_trial,bounded_upload=a.moss_bounded_weight_upload_trial,turbo=a.moss_turbo_trial,accumulation=a.moss_f32_accumulation)
    ffn_flags=ffn_compile_flags(a.moss_ffn_token_chunk_trial,a.moss_ffn_chunk_tokens)
    if any(any('PS5_AI_MOSS_FFN_'in arg for arg in recipe)for recipe in[profile['platform_compile'],*profile['ggml_compile']]):raise RuntimeError('Recipes cannot override FFN policy')
    for recipe in [profile['platform_compile'],*profile['ggml_compile']]:
        if any('PS5_AI_MOSS_SPECTRUM_TRIAL' in arg or 'PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL' in arg or 'PS5_AI_MOSS_TURBO_TRIAL' in arg for arg in recipe):
            raise RuntimeError('Native recipes cannot override inference feature policy')
    trial_path=ROOT/(('build-profiles/moss-performance-combined-lora-'+a.moss_lora_storage+'-fp32shared.json') if a.moss_lora_storage is not None else 'build-profiles/moss-performance-turbo-bounded-upload-ffn2048-fp16-f32accum.json' if a.moss_ffn_token_chunk_trial else 'build-profiles/moss-performance-turbo-bounded-upload-fp16-f32accum.json' if a.moss_turbo_trial and a.moss_bounded_weight_upload_trial else 'build-profiles/moss-performance-turbo-fp16-f32accum.json' if a.moss_turbo_trial else 'build-profiles/moss-performance-spectrum-fp16-f32accum.json' if a.moss_spectrum_trial else 'build-profiles/moss-performance-bounded-weight-upload-fp16-f32accum.json' if a.moss_bounded_weight_upload_trial else 'build-profiles/moss-performance-resmultistep-fp16-f32accum.json' if a.moss_sampler_trial else 'build-profiles/moss-performance-fp16-f32accum-q4-bk64-raw64.json' if a.moss_q4_bk64_raw64 else 'build-profiles/moss-performance-fp16-f32accum-gpu-profile.json' if a.moss_gpu_profile else 'build-profiles/moss-performance-fp16-f32accum.json' if a.moss_f32_accumulation else 'build-profiles/moss-performance-'+MOSS_PERFORMANCE_VARIANTS[a.performance_trial_variant][0]+'.json' if a.performance_trial_variant is not None else 'build-profiles/moss-1024-cold-warm.json' if a.benchmark_1024_cold_warm_trial else 'build-profiles/moss-quality-step-comparison.json' if a.quality_step_comparison_trial else 'build-profiles/moss-small10g-trial.json')
    trial_contract=json.loads(trial_path.read_text()) if a.experimental_small_trial else None
    trial_flags=moss_trial_flags(a.experimental_small_trial,a.trial_10g,trial_contract,a.quality_step_comparison_trial,a.benchmark_1024_cold_warm_trial,a.performance_trial_variant,a.moss_f32_accumulation,a.moss_gpu_profile,a.moss_q4_bk64_raw64,a.moss_sampler_trial,a.moss_spectrum_trial,a.moss_bounded_weight_upload_trial,a.moss_turbo_trial,a.moss_ffn_token_chunk_trial,a.moss_ffn_chunk_tokens,a.moss_lora_storage)
    for recipe in [profile['platform_compile'],*profile['ggml_compile']]:
        if any('PS5_AI_MOSS_DIT_'in arg for arg in recipe):raise RuntimeError('Recipes cannot override DiT policy')
    trial_flags += dit_flags(dit_enabled)+['-DPS5_AI_MOSS_DIT_OPT_MASK='+str(a.moss_dit_optimization_mask or 0)]
    trial_flags += kernel_flags(kernel_enabled)+['-DPS5_AI_MOSS_DIT_KERNEL_MASK='+str(a.moss_dit_kernel_mask or 0)]+kernel_fa_flags(a.moss_fa_shmem_no_limit,a.moss_dit_kernel_mask)
    if a.moss_img2img_trial and (not kernel_enabled or a.moss_dit_kernel_mask!=15 or not a.moss_fa_shmem_no_limit or a.moss_dit_gpu_profile or a.moss_lora_storage is not None):raise RuntimeError('Img2img requires the validated kernel15/FA/Turbo4/upload/F32 parent')
    trial_flags += ['-DPS5_AI_MOSS_IMG2IMG_TRIAL='+str(int(a.moss_img2img_trial))]
    def expand(s):
        for k,v in c.items():s=s.replace('${'+k+'}',str(v).replace('\\','/'))
        if '${' in s:raise RuntimeError('Unresolved build parameter: '+s)
        return s
    for row in profile['dependencies']:
        path=Path(c['DEPS'])/row['name']
        if not path.is_file() or digest(path)!=row['sha256']:raise RuntimeError('Missing/replaced reviewed dependency: '+row['name'])
    if profile.get('model_family')!='moss-image-sft' or profile.get('runtime_admitted') is not False:
        raise RuntimeError('Expected the unadmitted MOSS preparation build profile')
    admission=(ROOT/'native/app/moss_runtime_admission.h').read_text()
    if not re.search(r'^#define PS5_AI_MOSS_RUNTIME_ADMITTED 0$',admission,re.M):
        raise RuntimeError('MOSS preparation build must retain runtime admission0')
    baseline_core=archive_members(Path(c['DEPS'])/'libstable-diffusion.graph-admission-1024-v3.a')
    core_carryforward={}
    native_only_trial=a.quality_step_comparison_trial or a.benchmark_1024_cold_warm_trial or a.moss_q4_bk64_raw64 or a.moss_sampler_trial
    validate_moss_core(a.core_manifest,baseline_core,c,native_only_trial,core_carryforward,a.benchmark_1024_cold_warm_trial,a.moss_f32_accumulation,a.moss_q4_bk64_raw64,a.moss_sampler_trial,a.moss_spectrum_trial,a.moss_bounded_weight_upload_trial,a.moss_turbo_trial,a.moss_ffn_token_chunk_trial,a.moss_ffn_chunk_tokens,a.moss_lora_storage)
    batch_pair=profile.get('controlnet_batch_pair',0)
    if type(batch_pair)!=int or batch_pair not in (0,1):raise RuntimeError('ControlNet batch-pair profile must be 0 or 1')
    shader_header=profile['generated_shader_header'];shader_path=Path(expand(shader_header['path']))
    if not shader_path.is_file() or digest(shader_path)!=shader_header['sha256']:
        raise RuntimeError('Missing/replaced reviewed generated GGML shader header')
    for header in profile.get('compiler_headers',[]):
        path=Path(expand(header['path']))
        if not path.is_file() or digest(path)!=header['sha256']:
            raise RuntimeError('Missing/replaced reviewed compiler header: '+header['path'])
    frozen_inputs=collect_inputs(ROOT,a.config,c,profile,expand)
    out.mkdir(parents=True,exist_ok=True);(out/'obj').mkdir(exist_ok=True)
    (out/'inputs-before.private.json').write_text(json.dumps(frozen_inputs,indent=2)+'\n',encoding='utf-8')
    (out/'temp').mkdir(exist_ok=True)
    env=dict(os.environ);env['PATH']=str(c['HOST_BIN'])+os.pathsep+str(c['LLVM'])+os.pathsep+env.get('PATH','')
    env['TMP']=env['TEMP']=str(out/'temp')
    commands=[]
    with (out/'build.log').open('w',encoding='utf-8') as log:
        def run(cmd):
            cmd=[expand(str(v)) for v in cmd];commands.append(cmd)
            result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
            log.write(subprocess.list2cmdline(cmd)+'\n'+result.stdout+result.stderr+'\n');log.flush()
            if result.returncode:raise RuntimeError(result.stderr[-8000:])
            return result.stdout
        common=['--no-default-config','-target','x86_64-sie-ps5','-D__PROSPERO__=1','-DPS5_AI_MOSS_RUNTIME_ADMITTED=0',*trial_flags,*accumulation_flags,*gpu_profile_flags,*q4_candidate_flags,*sampler_flags,*feature_flags,*ffn_flags,*lora_flags,
          '-fvisibility-nodllstorageclass=default','-isysroot','${SDK}',
          '-isystem','${SDK}/target/include/c++/v1','-isystem','${SDK}/target/include',
          '-I','${PLATFORM}/src','-I','${ROOT}/native/app','-I','${ROOT}/native/ui',
          '-I','${SD}/include','-I','${SD}/src','-I','${SD}/ggml/include','-I','${SD}/ggml/src','-isystem','${SD}/thirdparty',
          '-I','${ROOT}/native/sha256','-I','${VULKAN_HEADERS}/include','-fno-stack-protector','-fno-plt','-femulated-tls',
          '-fdenormal-fp-math=ieee','-O2','-ffunction-sections','-fdata-sections','-Wall','-Wextra','-Werror','-MMD']
        for name in profile['objects']:
            stem=Path(name).stem
            if stem=='reference_sha256':source='native/sha256/sha256.c'
            elif stem=='app_crt':source='native/link/app_crt.cpp'
            elif stem in ('format','display'):source='native/ui/'+stem+'.c'
            elif stem in ('cpp_allocation','cxa_owned','threads'):source='native/runtime/'+stem+('.cpp' if stem=='cpp_allocation' else '.c')
            elif stem in ('engine_sdcpp','vulkan_bridge','cpp_allocation'):source='native/app/'+stem+'.cpp'
            else:source='native/app/'+stem+'.c'
            cpp=source.endswith('.cpp');extra=['-std=c++20','-fexceptions','-fcxx-exceptions'] if cpp else ['-std=c11']
            if stem in ('app_crt','cpp_allocation'):extra+=['-fno-exceptions','-fno-rtti']
            if stem=='server':extra+=['-Dmain=ps5_ai_server_main']
            if stem=='reference_image':extra+=['-Wno-unused-function']
            if stem=='engine_sdcpp':extra+=['-DPS5_AI_NATIVE_RADV=1']
            if stem in ('engine_sdcpp','startup'):extra+=['-DPS5_AI_ENABLE_FP16='+str(int(a.precision=='fp16'))]
            print('Compiling '+source,flush=True)
            run(['${LLVM}/clang.exe',*common,*extra,'-c','${ROOT}/'+source,'-o','${OUT}/obj/'+name])
        print('Compiling the owned RADV platform layer',flush=True);run([profile['platform_compile'][0],*trial_flags,*accumulation_flags,*gpu_profile_flags,*q4_candidate_flags,*sampler_flags,*feature_flags,*ffn_flags,*lora_flags,*profile['platform_compile'][1:]])
        driver=out/'libvulkan_radeon.native-direct-data.a';shutil.copyfile(Path(c['DEPS'])/driver.name,driver)
        before=run(['${LLVM}/llvm-ar.exe','t',str(driver)]).splitlines()
        if len(before)!=704 or before.count('radv_ps5_platform_native.o')!=1:raise RuntimeError('Unexpected driver archive members')
        run(['${LLVM}/llvm-ar.exe','r',str(driver),'${OUT}/obj/radv_ps5_platform_native.o'])
        print('Verifying the complete75-member MOSS core rebuild',flush=True)
        core_name='libstable-diffusion.graph-admission-1024-v3.a'
        baseline=archive_members(Path(c['DEPS'])/core_name)
        current_carryforward={}
        core_source,core_manifest=validate_moss_core(a.core_manifest,baseline,c,native_only_trial,current_carryforward,a.benchmark_1024_cold_warm_trial,a.moss_f32_accumulation,a.moss_q4_bk64_raw64,a.moss_sampler_trial,a.moss_spectrum_trial,a.moss_bounded_weight_upload_trial,a.moss_turbo_trial,a.moss_ffn_token_chunk_trial,a.moss_ffn_chunk_tokens,a.moss_lora_storage)
        if current_carryforward!=core_carryforward:raise RuntimeError('Core native-only carryforward evidence changed during build')
        core=out/'libstable-diffusion.moss.a';shutil.copyfile(core_source,core)
        core_changed=sorted(name for name,record in archive_members(core).items() if record!=baseline[name])

        print('Compiling the owned GGML scratch API provider',flush=True)
        ggml_name='libggml-vulkan.flex.a';ggml=out/'libggml-vulkan.optimized.a'
        shutil.copyfile(Path(c['DEPS'])/ggml_name,ggml);ggml_before=archive_members(ggml);ggml_objects=[]
        ggml_recipes=profile['ggml_compile']
        ggml_names=[Path(recipe[recipe.index('-o')+1]).name for recipe in ggml_recipes]
        if ggml_names!=['ggml-vulkan.cpp.o'] or ggml_names[0] not in ggml_before:
            raise RuntimeError('Unexpected GGML provider recipe/archive member')
        for recipe in ggml_recipes:
            run([recipe[0],*trial_flags,*accumulation_flags,*gpu_profile_flags,*q4_candidate_flags,*sampler_flags,*feature_flags,*ffn_flags,*lora_flags,*recipe[1:]]);ggml_objects.append(recipe[recipe.index('-o')+1])
        run(['${LLVM}/llvm-ar.exe','r',str(ggml),*ggml_objects])
        ggml_changed=check_archive_update(ggml_before,archive_members(ggml),ggml_names)
        substitutions={driver.name:driver,core_name:core,ggml_name:ggml}
        link=list(profile['link'])
        for name,path in substitutions.items():link=[v.replace('${DEPS}/'+name,str(path).replace('\\','/')) for v in link]
        print('Linking native application',flush=True);run(link)
        raw=out/'llvm-pie.elf';deploy=out/'llvm-pie.deploy.elf'
        run(['${LLVM}/llvm-strip.exe','--strip-debug','-o',str(deploy),str(raw)])
        if mapped(raw.read_bytes())!=mapped(deploy.read_bytes()):raise RuntimeError('Stripping modified loaded bytes')
        dynamic=run(['${LLVM}/llvm-readelf.exe','--dynamic',str(raw)])
        needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]',dynamic)
        if len(needed)!=6:raise RuntimeError('Expected six native runtime providers')
        definitions=run(['${LLVM}/llvm-nm.exe','--defined-only',str(raw)])
        undefined=run(['${LLVM}/llvm-nm.exe','--undefined-only',str(raw)])
        def local_address(name):
            rows=re.findall(r'^([0-9a-fA-F]+)\s+[A-Za-z]\s+'+re.escape(name)+r'$',definitions,re.M)
            if len(rows)!=1:raise RuntimeError('Missing/duplicate native initializer boundary: '+name)
            return int(rows[0],16)
        initializer_bytes=local_address('__init_array_end')-local_address('__init_array_start')
        expected=int(re.search(r'PS5_AI_NATIVE_EXPECTED_INITIALIZER_SLOTS (\d+)u',(ROOT/'native/app/native_constructor_expect.h').read_text())[1])
        if initializer_bytes<0 or initializer_bytes%8 or initializer_bytes//8!=expected:
            raise RuntimeError('Native constructor slot count differs from the reviewed startup guard')

        required_local=['ps5_ai_cpu_clip_shared_direct_live_bytes','ps5_ai_ggml_release_idle_scratch',
                        'ps5_ai_flex_largest_free_span_bytes']
        if dit_enabled:required_local += ['ps5_ai_moss_dit_capabilities']
        if kernel_enabled:required_local += ['ps5_ai_moss_dit_kernel_capabilities']
        if a.moss_lora_storage is not None:required_local += ['ps5_ai_moss_lora_storage_snapshot','ps5_ai_vk_lora_storage_compile','ps5_ai_vk_lora_storage_record','ps5_ai_moss_lora_precision_backend_admission']
        for name in required_local:
            if len(re.findall(r'\b'+re.escape(name)+r'$',definitions,re.M))!=1 or re.search(r'\b'+re.escape(name)+r'$',undefined,re.M):
                raise RuntimeError('Missing/duplicate native optimization provider: '+name)
        native=out/'eboot.elf';stubs=[str(Path(c['DEPS'])/n) for n in ['libSceAgc.so','libSceAgcDriver.so']]+[str(Path(c['SDK'])/'target/lib'/n) for n in ['libSceVideoOut.so','libSceSystemService.so']]
        run(['${NATIVE_TOOL}','link','--in',str(deploy),'--out',str(native),'--stub-dir','${SDK}/target/lib',*sum((['--stub',s] for s in stubs),[]),'--module-sdk','0x02000009','--companion-sdk','0x08050001','--file-name','eboot.elf'])
        title=out/'title/PPSA99190';(title/'sce_sys').mkdir(parents=True,exist_ok=True)
        for name in ('param.json','icon0.png'):shutil.copyfile(ROOT/'assets/sce_sys'/name,title/'sce_sys'/name)
        (title/'licenses').mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'licenses/WAI-Dashboard-Glyphs-OFL.txt',title/'licenses/WAI-Dashboard-Glyphs-OFL.txt')
        for notice in (ROOT/'licenses').glob('MOSS-models-*.txt'):
            shutil.copyfile(notice,title/'licenses'/notice.name)
        shutil.copyfile(ROOT/'licenses/MOSS-requested-LoRA-Apache-2.0.txt',title/'licenses/MOSS-requested-LoRA-Apache-2.0.txt')
        run(['${NATIVE_TOOL}','self','--sign','--in',str(native),'--out',str(title/'eboot.bin'),'--magic','0x1D3D154F'])
        inspection=run(['${NATIVE_TOOL}','self','--inspect','--file',str(title/'eboot.bin')]);(out/'inspect.txt').write_text(inspection)
        if 'integrity: valid' not in inspection:raise RuntimeError('SELF integrity not confirmed')
        current_inputs=collect_inputs(ROOT,a.config,c,profile,expand)
        require_unchanged(frozen_inputs,current_inputs)
        sources={str(s.relative_to(ROOT)):digest(s) for folder in ['native/app','native/ui','native/runtime','native/link',
                 'patches/stable-diffusion/overlay'] for s in (ROOT/folder).rglob('*') if s.is_file()}
        for name in ['build-profiles/native.json','tools/build.py']+([str(trial_path.relative_to(ROOT))] if a.experimental_small_trial else []):sources[name]=digest(ROOT/name)
        (out/'manifest.json').write_text(json.dumps({'local_build_only':True,'console_executed':False,
            'model_family':'moss-image-sft','requested_direct_bytes':10737418240 if a.trial_10g else 12884901888,
            'user_requested_direct_target_bytes':12884901888,
            'experimental_small_trial':a.experimental_small_trial,'trial_10g':a.trial_10g,
            'quality_step_comparison_trial':a.quality_step_comparison_trial,
            'benchmark_1024_cold_warm_trial':a.benchmark_1024_cold_warm_trial,
            'performance_trial':a.performance_trial_variant is not None,'performance_variant':a.performance_trial_variant,
            'performance_variant_name':MOSS_PERFORMANCE_VARIANTS[a.performance_trial_variant][0] if a.performance_trial_variant is not None else None,
            'core_native_only_carryforward':core_carryforward,
            'trial_compile_flags':trial_flags,'trial_contract':trial_contract,
            **feature_manifest_fields(a.moss_spectrum_trial,a.moss_bounded_weight_upload_trial,a.moss_turbo_trial),
            **ffn_manifest_fields(a.moss_ffn_token_chunk_trial,a.moss_ffn_chunk_tokens),
            **dit_fields(dit_enabled),
            **kernel_fields(kernel_enabled),
            'moss_dit_kernel_mask':a.moss_dit_kernel_mask,
            'moss_fa_shmem_no_limit':a.moss_fa_shmem_no_limit,
            'moss_img2img_trial':a.moss_img2img_trial,
            'moss_fa_shmem_compile_flags':kernel_fa_flags(a.moss_fa_shmem_no_limit,a.moss_dit_kernel_mask),
            'moss_dit_optimization_mask':a.moss_dit_optimization_mask,
            'core_dit_policy_proof':validate_dit_core(core_manifest,dit_enabled),
            'core_ffn_policy_proof':validate_core_ffn(core_manifest,a.moss_ffn_token_chunk_trial,a.moss_ffn_chunk_tokens),
            **lora_precision_fields(a.moss_lora_storage),
            'core_lora_precision_proof':validate_core_lora_precision(core_manifest,a.moss_lora_storage),
            'core_inference_feature_proof':validate_core_features(core_manifest,a.moss_spectrum_trial,a.moss_bounded_weight_upload_trial,a.moss_turbo_trial),
            'moss_sampler_trial':a.moss_sampler_trial,
            'moss_sampler_policy':'res_multistep_simple_30_cfg4_v1' if a.moss_sampler_trial else 'disabled',
            'moss_sampler_compile_flags':sampler_flags,
            'moss_f32_accumulation':a.moss_f32_accumulation,
            'moss_accumulation_policy':moss_accumulation_policy(a.moss_f32_accumulation),
            'moss_accumulation_compile_flags':accumulation_flags,
            'moss_gpu_profile':a.moss_gpu_profile or a.moss_dit_gpu_profile,
            'moss_dit_gpu_profile':a.moss_dit_gpu_profile,
            'moss_gpu_profile_policy':'mask15_Turbo4_rank144_first_two_complete_cached_DiT_graphs_parent_and_shader_v1' if a.moss_dit_gpu_profile else 'first_two_complete_dit_graphs_per_generation' if a.moss_gpu_profile else 'disabled',
            'moss_gpu_profile_compile_flags':gpu_profile_flags,
            'moss_q4_bk64_raw64':a.moss_q4_bk64_raw64,
            'moss_q4_bk64_policy':'canonical_moss_ffn_up_q4k_f32acc_bk64_raw64_v1' if a.moss_q4_bk64_raw64 else 'disabled',
            'moss_q4_bk64_compile_flags':q4_candidate_flags,
            'core_accumulation_policy_proof':validate_moss_core_accumulation(core_manifest,a.moss_f32_accumulation),
            'trial_contract_sha256':digest(trial_path) if a.experimental_small_trial else None,
            'runtime_admitted':False,'requested_precision':a.precision,'controlnet_batch_pair':batch_pair,
            'input_hashes_stable':True,'inputs_before_after':frozen_inputs,
            'full_core_manifest_sha256':digest(a.core_manifest),'full_core_manifest':core_manifest,
            'runner_sha256':digest(ROOT/'patches/stable-diffusion/overlay/src/core/ggml_runner.cpp'),
            'scratch_bounds_sha256':digest(ROOT/'patches/stable-diffusion/overlay/src/core/vulkan_scratch_bounds.hpp'),
            'needed':needed,'sources':sources,'generated_shader_header_sha256':digest(shader_path),
            'changed_core_members':core_changed,'changed_ggml_members':ggml_changed,
            'core_archive_sha256':digest(core),'ggml_archive_sha256':digest(ggml),
            'native_optimization_providers_defined':required_local,'initializer_slots':expected,
            'eboot_sha256':digest(title/'eboot.bin'),'commands':commands},indent=2)+'\n')
        print('Built and inspected '+str(title/'eboot.bin')+'. No deployment or inference was performed.',flush=True)
if __name__=='__main__':main()
