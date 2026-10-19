# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows local native rebuild from exported sources + reviewed dependencies.

Does not connect to a PS5, install a title, patch firmware, or run inference.
See docs/build.md for the explicit dependency-build limitation.
"""
import argparse, hashlib, json, os, re, shutil, struct, subprocess
from pathlib import Path
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

def same_typed_contract_value(actual,expected):
    return type(actual) is type(expected) and (
        len(actual)==len(expected) and all(same_typed_contract_value(a,e) for a,e in zip(actual,expected))
        if isinstance(expected,list) else actual==expected)

def moss_trial_flags(enabled,trial_10g,contract=None,quality_comparison=False,benchmark_1024=False):
    """A trial is permission to RUN verification, never prior GPU admission."""
    if any(type(value) is not bool for value in (enabled,trial_10g,quality_comparison,benchmark_1024)):
        raise RuntimeError('MOSS trial switches must be explicit booleans')
    if quality_comparison and benchmark_1024:raise RuntimeError('MOSS trial modes are mutually exclusive')
    if not enabled:
        if trial_10g or quality_comparison or benchmark_1024:raise RuntimeError('Trial switches require --experimental-small-trial')
        return ['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=0','-DPS5_AI_MOSS_TRIAL_10G=0']
    expected_contract=MOSS_BENCHMARK_1024_CONTRACT if benchmark_1024 else MOSS_QUALITY_COMPARISON_CONTRACT if quality_comparison else MOSS_SMALL_TRIAL_CONTRACT
    valid_contract=isinstance(contract,dict) and set(contract)==set(expected_contract) and all(
        same_typed_contract_value(contract[key],expected)
        for key,expected in expected_contract.items())
    if not trial_10g or not valid_contract:
        raise RuntimeError('Experimental MOSS requires the exact reviewed small10GiB trial profile and both switches')
    return ['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_TRIAL_10G=1']+(
        ['-DPS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL=1'] if quality_comparison else [])+(
        ['-DPS5_AI_MOSS_1024_COLD_WARM_TRIAL=1'] if benchmark_1024 else [])
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
def validate_moss_core(manifest_path, expected_members, build_config, quality_native_only=False, carryforward=None, benchmark_gateway_only=False):
    """Consume only a complete, immutable 75-member MOSS core rebuild."""
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
    inputs=manifest.get('sources',[])+manifest.get('header_inputs',[])+manifest.get('overlay_inputs',[])+[manifest['profile'],manifest['builder']]
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
        if quality_native_only is not True or any(Path(row['path']).resolve() not in allowed for row in changed):
            raise RuntimeError('MOSS core source/header changed since compile: '+str(changed[0]['path']))
        gateway_wait_proof={}
        for row in changed:
            if Path(row['path']).resolve()==ROOT/'native/gateway/openai.inc':
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
            carryforward.update({'schema':'moss-quality-native-only-core-carryforward-v1',
                'original_manifest_sha256':digest(manifest_path),'original_manifest_unchanged':True,
                'archive_sha256':archive['sha256'],'new_core_compilation':False,
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--precision',choices=['fp32','fp16'],default='fp32');p.add_argument('--core-manifest',type=Path,required=True);p.add_argument('--experimental-small-trial',action='store_true');p.add_argument('--trial-10g',action='store_true');p.add_argument('--quality-step-comparison-trial',action='store_true');p.add_argument('--benchmark-1024-cold-warm-trial',action='store_true');a=p.parse_args()
    c=json.loads(a.config.read_text());out=a.out.resolve()
    if (out/'manifest.json').exists():raise RuntimeError('Completed output exists; choose a new directory.')
    c.update(ROOT=str(ROOT),OUT=str(out));profile=json.loads((ROOT/'build-profiles/native.json').read_text())
    trial_path=ROOT/('build-profiles/moss-1024-cold-warm.json' if a.benchmark_1024_cold_warm_trial else 'build-profiles/moss-quality-step-comparison.json' if a.quality_step_comparison_trial else 'build-profiles/moss-small10g-trial.json')
    trial_contract=json.loads(trial_path.read_text()) if a.experimental_small_trial else None
    trial_flags=moss_trial_flags(a.experimental_small_trial,a.trial_10g,trial_contract,a.quality_step_comparison_trial,a.benchmark_1024_cold_warm_trial)
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
    native_only_trial=a.quality_step_comparison_trial or a.benchmark_1024_cold_warm_trial
    validate_moss_core(a.core_manifest,baseline_core,c,native_only_trial,core_carryforward,a.benchmark_1024_cold_warm_trial)
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
        common=['--no-default-config','-target','x86_64-sie-ps5','-D__PROSPERO__=1','-DPS5_AI_MOSS_RUNTIME_ADMITTED=0',*trial_flags,
          '-fvisibility-nodllstorageclass=default','-isysroot','${SDK}',
          '-isystem','${SDK}/target/include/c++/v1','-isystem','${SDK}/target/include',
          '-I','${PLATFORM}/src','-I','${ROOT}/native/app','-I','${ROOT}/native/ui',
          '-I','${SD}/include','-I','${SD}/ggml/include','-I','${SD}/ggml/src','-isystem','${SD}/thirdparty',
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
        print('Compiling the owned RADV platform layer',flush=True);run([profile['platform_compile'][0],*trial_flags,*profile['platform_compile'][1:]])
        driver=out/'libvulkan_radeon.native-direct-data.a';shutil.copyfile(Path(c['DEPS'])/driver.name,driver)
        before=run(['${LLVM}/llvm-ar.exe','t',str(driver)]).splitlines()
        if len(before)!=704 or before.count('radv_ps5_platform_native.o')!=1:raise RuntimeError('Unexpected driver archive members')
        run(['${LLVM}/llvm-ar.exe','r',str(driver),'${OUT}/obj/radv_ps5_platform_native.o'])
        print('Verifying the complete75-member MOSS core rebuild',flush=True)
        core_name='libstable-diffusion.graph-admission-1024-v3.a'
        baseline=archive_members(Path(c['DEPS'])/core_name)
        current_carryforward={}
        core_source,core_manifest=validate_moss_core(a.core_manifest,baseline,c,native_only_trial,current_carryforward,a.benchmark_1024_cold_warm_trial)
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
            run([recipe[0],*trial_flags,*recipe[1:]]);ggml_objects.append(recipe[recipe.index('-o')+1])
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
            'benchmark_1024_cold_warm_trial':a.benchmark_1024_cold_warm_trial,'core_native_only_carryforward':core_carryforward,
            'trial_compile_flags':trial_flags,'trial_contract':trial_contract,
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
