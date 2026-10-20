# SPDX-License-Identifier: GPL-3.0-or-later
"""Strict default-off, isolated MOSS inference feature build identities."""
from pathlib import Path
import hashlib
SPECTRUM_MACRO='PS5_AI_MOSS_SPECTRUM_TRIAL'
UPLOAD_MACRO='PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL'
TURBO_MACRO='PS5_AI_MOSS_TURBO_TRIAL'
TURBO_POLICY='euler_custom1000_910_717_24_cfg1_nsfw07_turbo1_rank144_v1'
SPECTRUM_POLICY='post_cfg_latent_manual_50_8_2_alternating_degree3_ridge1_blend04_history6_v1'
UPLOAD_POLICY='raw_same_format_GPU_weights_8MiB_per_worker_v1'

def feature_compile_flags(spectrum=False,bounded_upload=False,turbo=False):
    if any(type(value)is not bool for value in (spectrum,bounded_upload,turbo)):
        raise RuntimeError('MOSS inference feature flags must be explicit booleans')
    if sum((spectrum,bounded_upload,turbo))>1:
        raise RuntimeError('Spectrum, bounded upload and Turbo require isolated experiments')
    return ['-D'+SPECTRUM_MACRO+'='+str(int(spectrum)),
            '-D'+UPLOAD_MACRO+'='+str(int(bounded_upload)),
            '-D'+TURBO_MACRO+'='+str(int(turbo))]

def validate_feature_modes(spectrum=False,bounded_upload=False,turbo=False,*,accumulation=False,
                           gpu_profile=False,q4=False,sampler=False,variant=None,
                           precision='fp32',experimental=False,trial_10g=False):
    feature_compile_flags(spectrum,bounded_upload,turbo)
    if any(type(v)is not bool for v in (accumulation,gpu_profile,q4,sampler,experimental,trial_10g)):
        raise RuntimeError('MOSS feature mode inputs must be explicit booleans')
    if spectrum or bounded_upload or turbo:
        if (not accumulation or gpu_profile or q4 or sampler or type(variant)is not int or
            variant!=1 or precision!='fp16' or not experimental or not trial_10g):
            raise RuntimeError('Isolated feature requires FP16/F32 small10GiB variant1 and all other trials off')

def feature_manifest_fields(spectrum=False,bounded_upload=False,turbo=False):
    flags=feature_compile_flags(spectrum,bounded_upload,turbo)
    return {'moss_spectrum_trial':spectrum,
            'moss_spectrum_policy':SPECTRUM_POLICY if spectrum else 'disabled',
            'moss_bounded_weight_upload_trial':bounded_upload,
            'moss_bounded_weight_upload_policy':UPLOAD_POLICY if bounded_upload else 'disabled',
            'moss_turbo_trial':turbo,'moss_turbo_policy':TURBO_POLICY if turbo else 'disabled',
            'moss_inference_feature_compile_flags':flags}

def feature_contract_updates(spectrum=False,bounded_upload=False,turbo=False):
    feature_compile_flags(spectrum,bounded_upload,turbo)
    if not (spectrum or bounded_upload or turbo):return {}
    result=feature_manifest_fields(spectrum,bounded_upload,turbo)
    result.update({'schema':'ps5-ai-moss-experimental-turbo-trial-v1' if turbo else 'ps5-ai-moss-experimental-spectrum-trial-v1' if spectrum else
                   'ps5-ai-moss-experimental-bounded-weight-upload-trial-v1',
                   'scheduler':'custom_sigmas_author_training_grid' if turbo else 'discrete',
                   'expected_dit_forwards':4 if turbo else 60 if spectrum else 100,
                   'expected_q4_f32acc_recordings':672 if turbo else 9960 if spectrum else 16600,
                   'new_core_compilation_required':True,'old_core_reuse_allowed':False})
    if turbo:result.update({'steps':4,'cfg_scale':1.0,'sampler':'euler','moss_lora_strength':1.0,
        'custom_sigmas':[1.0,.910,.717,.024,0.0],'custom_sigmas_shift_applied_again':False,
        'moss_lora_rank':144,'moss_lora_composition_schema':'ps5-ai-moss-nsfw07-turbo1-r144-f32-v1',
        'moss_lora_components':[{'rank':16,'strength':0.7},{'rank':128,'strength':1.0}],
        'rank144_GPU_numerical_gates_required':True})
    elif spectrum:result.update({'actual_cfg_steps':30,'forecast_cfg_steps':20,
        'history_F32_payload_max_bytes':12582912,
        'history_plus_prediction_F32_payload_max_bytes':14680064})
    else:result.update({'bounded_weight_upload_chunk_bytes':8388608,
        'bounded_weight_upload_limit_scope':'per_worker',
        'bounded_weight_upload_GPU_readback_gate_required':True})
    return result

def validate_core_features(manifest,spectrum=False,bounded_upload=False,turbo=False):
    expected=feature_manifest_fields(spectrum,bounded_upload,turbo)
    fields=set(expected)
    declared=fields.intersection(manifest)
    # Preserve only historical feature-off manifests. Normal input hashing still
    # rejects old consumed-source changes, including every new feature core.
    if not declared:
        if spectrum or bounded_upload or turbo:raise RuntimeError('Old core has no compiled feature identity')
        historical=True
    else:
        if declared!=fields or any(type(manifest[k])is not type(v) or manifest[k]!=v for k,v in expected.items()):
            raise RuntimeError('Core/native feature policy mismatch')
        historical=False
        policy_inputs=manifest.get('policy_inputs')
        own=Path(__file__).resolve()
        if (type(policy_inputs)is not list or len(policy_inputs)!=1 or type(policy_inputs[0])is not dict or
            set(policy_inputs[0])!={'path','bytes','sha256'} or type(policy_inputs[0].get('path'))is not str or
            Path(policy_inputs[0]['path']).resolve()!=own or type(policy_inputs[0].get('bytes'))is not int or
            policy_inputs[0]['bytes']!=own.stat().st_size or
            policy_inputs[0]['sha256']!=hashlib.sha256(own.read_bytes()).hexdigest()):
            raise RuntimeError('Core feature identity must pin the exact live feature-policy helper')
    commands=manifest.get('commands',[]);members=manifest.get('members',[])
    if len(commands)!=75 or len(members)!=75:raise RuntimeError('Feature identity requires actual75 commands/members')
    indices=set()
    for row in members:
        i=row.get('command_index')
        if type(i)is not int or i not in range(75) or i in indices:raise RuntimeError('Invalid core command index')
        indices.add(i);c=commands[i]
        if type(c)is not list or not c or any(type(a)is not str for a in c) or c.count('-c')!=1:
            raise RuntimeError('Core feature identity is not an actual compiler recipe')
        offset=c.index('-c')
        if offset+1>=len(c) or type(row.get('source'))is not str or Path(c[offset+1]).resolve()!=Path(row['source']).resolve():
            raise RuntimeError('Core feature compiler source mismatch')
        for macro,flag in zip((SPECTRUM_MACRO,UPLOAD_MACRO,TURBO_MACRO),expected['moss_inference_feature_compile_flags']):
            if [a for a in c if macro in a] != ([] if historical else [flag]):
                raise RuntimeError('Missing/conflicting actual core feature macro: '+macro)
    return {'actual_compile_commands_verified':75,'historical_default0_without_macros':historical,
            **expected}
