from moss_lora_precision_policy import flags as lora_precision_flags, fields as lora_precision_fields, contract as lora_precision_contract, validate as validate_lora_precision, validate_core as validate_core_lora_precision
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the isolated MOSS CPU gateway / Windows protocol fixture. Never deploy or infer."""
import argparse, hashlib, json, os, re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def moss_gateway_sampler_flags(enabled):
    if type(enabled)is not bool:raise RuntimeError('Gateway sampler trial must be an explicit bool')
    if not enabled:return ['-DPS5_AI_MOSS_SAMPLER_TRIAL=0']
    return ['-DPS5_AI_MOSS_RUNTIME_ADMITTED=0','-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1',
        '-DPS5_AI_MOSS_TRIAL_10G=1','-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1','-DPS5_AI_MOSS_PERF_VARIANT=1',
        '-DPS5_AI_MOSS_F32_ACCUMULATION=1','-DPS5_AI_MOSS_GPU_PROFILE=0',
        '-DPS5_AI_MOSS_Q4_BK64_RAW64=0','-DPS5_AI_MOSS_SAMPLER_TRIAL=1']
def moss_gateway_inference_feature_flags():
    # This is a CPU protocol transport, never a Spectrum/weight-upload runner.
    return ['-DPS5_AI_MOSS_SPECTRUM_TRIAL=0','-DPS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL=0']

def moss_gateway_trial_flags(sampler=False,turbo=False):
    if type(sampler)is not bool or type(turbo)is not bool or (sampler and turbo):
        raise RuntimeError('Gateway sampler and Turbo trials must be isolated explicit booleans')
    if not turbo:return moss_gateway_sampler_flags(sampler)+['-DPS5_AI_MOSS_TURBO_TRIAL=0']
    return ['-DPS5_AI_MOSS_RUNTIME_ADMITTED=0','-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1',
        '-DPS5_AI_MOSS_TRIAL_10G=1','-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1','-DPS5_AI_MOSS_PERF_VARIANT=1',
        '-DPS5_AI_MOSS_F32_ACCUMULATION=1','-DPS5_AI_MOSS_GPU_PROFILE=0',
        '-DPS5_AI_MOSS_Q4_BK64_RAW64=0','-DPS5_AI_MOSS_SAMPLER_TRIAL=0','-DPS5_AI_MOSS_TURBO_TRIAL=1']

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--native',action='store_true');p.add_argument('--moss-sampler-trial',action='store_true');p.add_argument('--moss-turbo-trial',action='store_true');p.add_argument('--moss-lora-storage',choices=('f32','f16'));p.add_argument('--moss-img2img-trial',action='store_true');a=p.parse_args()
    sampler_flags=moss_gateway_trial_flags(a.moss_sampler_trial,a.moss_turbo_trial)
    if a.moss_img2img_trial and (not a.moss_turbo_trial or a.moss_lora_storage is not None):raise RuntimeError('Img2img gateway requires the F32 Turbo parent')
    sampler_flags+=['-DPS5_AI_MOSS_IMG2IMG_TRIAL='+str(int(a.moss_img2img_trial))]
    inference_feature_flags=moss_gateway_inference_feature_flags()
    validate_lora_precision(a.moss_lora_storage,turbo=a.moss_turbo_trial,upload=False,accumulation=a.moss_turbo_trial,
        sampler=a.moss_sampler_trial,variant=1,precision='fp16',experimental=True,trial_10g=True,transport=True)
    lora_flags=lora_precision_flags(a.moss_lora_storage,transport=True)
    ffn_flags=['-DPS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL=0','-DPS5_AI_MOSS_FFN_CHUNK_TOKENS=0']
    c=json.loads(a.config.read_text());out=a.out.resolve()
    if out==ROOT or ROOT in out.parents or any(p.name.lower() in ('anima','ps5-ai-anima','ps5-ai-wai') for p in [out,*out.parents]):
        raise RuntimeError('Use a separate MOSS build directory outside source/WAI/Anima outputs')
    if out.exists():raise RuntimeError('Use a fresh MOSS build directory; existing artifacts are preserved')
    out.mkdir(parents=True)
    clang=str(Path(c['LLVM'])/'clang.exe');src=ROOT/'native/gateway';sources=[src/'server.c',src/'cpu_hooks.c',ROOT/'native/sha256/sha256.c']
    env=dict(os.environ);env['PATH']=str(c['LLVM'])+os.pathsep+env.get('PATH','');env['TEMP']=env['TMP']=str(out)
    commands=[];warnings=[]
    def record(path):
        p=Path(path).resolve()
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        return dict(path=str(p),bytes=p.stat().st_size,sha256=digest)
    def run(cmd):
        cmd=list(map(str,cmd));commands.append(cmd)
        r=subprocess.run(cmd,env=env,cwd=ROOT,capture_output=True,text=True,errors='replace')
        if r.stderr.strip():warnings.append(dict(command=cmd,stderr=r.stderr))
        if r.returncode:raise RuntimeError(r.stderr[-8000:])
        return r.stdout
    spec=json.loads((ROOT/'docs/openapi.json').read_text());packed=json.dumps(spec,ensure_ascii=True,separators=(',',':'))
    (src/'openapi_schema.inc').write_text('/* Generated from docs/openapi.json. */\nstatic const char openapi_schema[]=\n'+json.dumps(packed)+';\n',encoding='utf-8',newline='\n')
    include=['-I',src,'-I',ROOT/'native/sha256']
    for name,flags in [('protocol-test',['-DPS5_AI_GATEWAY_PROTOCOL_TEST']),('host-forwarder',[])]:
        run([clang,'--target=x86_64-w64-mingw32','-std=gnu11','-O2','-Wall','-Wextra','-Werror',*sampler_flags,*inference_feature_flags,*ffn_flags,*lora_flags,*sources,*include,*flags,'-lws2_32','-o',out/(name+'.exe')])
    if a.native:
        sdk=Path(c['PAYLOAD_SDK']);sdk_source=Path(c['SDK_SOURCE']);original=sdk/'target/lib/crt1.o'
        if hashlib.sha256(original.read_bytes()).hexdigest()!='e13ce68fcd8525df4b480a37b6239827e30998284781f4dd0cf37ec06717371a':raise RuntimeError('Expected SDK0.44 payload CRT')
        crt=out/'crt1-credential-preserving.o';run([Path(c['LLVM'])/'llvm-objcopy.exe','--weaken-symbol=__patch_init',original,crt])
        patch=out/'patch-syscall-only.o'
        common=['--no-default-config','--target=x86_64-sie-ps5','-isystem',sdk/'target/include','-fno-stack-protector','-fno-plt','-femulated-tls','-fPIC','-O2','-Wall','-Wextra']
        run([clang,*common,'-std=c11','-I',sdk_source/'crt','-c',ROOT/'native/payload_crt/credential_preserving_patch.c','-o',patch])
        objects=[]
        for i,s in enumerate(sources):
            obj=out/f'gateway-{i}.o';run([clang,*common,'-std=gnu11','-D_BSD_SOURCE','-Werror',*sampler_flags,*inference_feature_flags,*ffn_flags,*lora_flags,*include,'-c',s,'-o',obj]);objects.append(obj)
        elf=out/'ps5-ai-moss-openai-gateway.elf'
        run([Path(c['LLVM'])/'ld.lld.exe','-m','elf_x86_64','-pie','-T',sdk/'ldscripts/elf_x86_64.x','--eh-frame-hdr','-z','max-page-size=0x4000','--hash-style=gnu','-L',sdk/'target/lib','-L',sdk/'target/user/homebrew/lib','-o',elf,*objects,crt,patch,'-lc','-lkernel_web','-lSceLibcInternal','-lSceNet'])
        startup=run([Path(c['LLVM'])/'llvm-objdump.exe','-d','--disassemble-symbols=__crt_start',elf])
        actual_patch=run([Path(c['LLVM'])/'llvm-objdump.exe','-d','--disassemble-symbols=__patch_init',elf])
        if '<__patch_init>' not in startup or 'kernel_set_proc' in actual_patch:raise RuntimeError('Credential-preserving startup audit failed')
        dynamic=run([Path(c['LLVM'])/'llvm-readelf.exe','--dynamic','--wide',elf])
        (out/'startup-audit.txt').write_text(startup+'\n'+actual_patch,encoding='utf-8',newline='\n')
        (out/'dynamic-audit.txt').write_text(dynamic,encoding='utf-8',newline='\n')
        needed=re.findall(r'Shared library: \[([^\]]+)\]',dynamic)
        native_libraries=[record(sdk/('target/lib/'+name)) for name in ['libc.a','libkernel_web.so','libSceLibcInternal.so','libSceNet.so']]
    inputs=[record(p) for p in [Path(__file__),ROOT/'docs/openapi.json',ROOT/'native/app/moss_img2img_trial.h',ROOT/'native/app/moss_sampler_trial.h',ROOT/'native/app/moss_turbo_trial.h',ROOT/'native/app/moss_lora_f16_storage_trial.h',ROOT/'tools/moss_lora_precision_policy.py',ROOT/'native/payload_crt/credential_preserving_patch.c',*sources,*sorted(src.glob('*.h')),*sorted(src.glob('*.inc'))]]
    manifest=dict(schema='moss-image-gateway-build-v1',state='crossbuilt_not_PS5_executed' if a.native else 'host_fixture_built_no_inference',
                  model_id='moss-image-sft',native_port=8192,gateway_port=8193,gateway_root='/data/ai/moss-openai-gateway',
                  artifact=record(elf) if a.native else None,inputs=inputs,commands=commands,warnings=warnings,
                  ELF_NEEDED=needed if a.native else [],native_libraries=native_libraries if a.native else [],
                  credential_preserving_CRT=True if a.native else None,PS5_connections=0,PS5_deployments=0,
                  gateway_inference_feature_scope='transport_only_no_Spectrum_or_weight_upload',
                  moss_spectrum_trial=False,moss_bounded_weight_upload_trial=False,
                  moss_inference_feature_compile_flags=inference_feature_flags,moss_ffn_token_chunk_trial=False,moss_ffn_chunk_tokens=0,moss_ffn_chunk_policy='disabled',moss_ffn_compile_flags=ffn_flags,
                  generation_contract_policy='euler_custom1000_910_717_24_cfg1_nsfw07_turbo1_rank144_v1' if a.moss_turbo_trial else 'res_multistep_simple_30_cfg4_v1' if a.moss_sampler_trial else 'baseline_euler_discrete_50_cfg3_v1',
                  moss_sampler_trial=a.moss_sampler_trial,moss_sampler_policy='res_multistep_simple_30_cfg4_v1' if a.moss_sampler_trial else 'disabled',
                  moss_sampler_compile_flags=sampler_flags,production_runtime_admitted=False,
                  moss_img2img_trial=a.moss_img2img_trial,
                  moss_turbo_trial=a.moss_turbo_trial,moss_turbo_policy='euler_custom1000_910_717_24_cfg1_nsfw07_turbo1_rank144_v1' if a.moss_turbo_trial else 'disabled',
                  **lora_precision_fields(a.moss_lora_storage,transport=True),
                  moss_gateway_trial_compile_flags=sampler_flags+inference_feature_flags+ffn_flags+lora_flags,
                  inference_performed=False,model_weights_loaded=False,GPU_used=False,PNG_generated=False)
    (out/'manifest.private.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(state=manifest['state'],artifact=manifest['artifact'],ELF_NEEDED=manifest['ELF_NEEDED'],warnings=len(warnings),PS5_connections=0)))
if __name__=='__main__':main()
