# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows local native rebuild from exported sources + reviewed dependencies.

Does not connect to a PS5, install a title, patch firmware, or run inference.
See docs/build.md for the explicit dependency-build limitation.
"""
import argparse, hashlib, json, os, re, shutil, struct, subprocess
from pathlib import Path
from build_input_snapshot import collect as collect_inputs, require_unchanged

ROOT=Path(__file__).resolve().parents[1]
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
def validate_moss_core(manifest_path, expected_members, build_config):
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
    for record in inputs:
        source=Path(record['path'])
        if not source.is_file() or source.stat().st_size!=record['bytes'] or digest(source)!=record['sha256']:
            raise RuntimeError('MOSS core source/header changed since compile: '+str(source))
    return path,manifest

def mapped(data):
    offset=struct.unpack_from('<Q',data,32)[0];size,count=struct.unpack_from('<HH',data,54)
    return [(p,hashlib.sha256(data[p[2]:p[2]+p[5]]).hexdigest()) for i in range(count)
            if (p:=struct.unpack_from('<IIQQQQQQ',data,offset+i*size))[0]==1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--precision',choices=['fp32','fp16'],default='fp32');p.add_argument('--core-manifest',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.config.read_text());out=a.out.resolve()
    if (out/'manifest.json').exists():raise RuntimeError('Completed output exists; choose a new directory.')
    c.update(ROOT=str(ROOT),OUT=str(out));profile=json.loads((ROOT/'build-profiles/native.json').read_text())
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
    validate_moss_core(a.core_manifest,baseline_core,c)
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
        common=['--no-default-config','-target','x86_64-sie-ps5','-D__PROSPERO__=1','-DPS5_AI_MOSS_RUNTIME_ADMITTED=0',
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
        print('Compiling the owned RADV platform layer',flush=True);run(profile['platform_compile'])
        driver=out/'libvulkan_radeon.native-direct-data.a';shutil.copyfile(Path(c['DEPS'])/driver.name,driver)
        before=run(['${LLVM}/llvm-ar.exe','t',str(driver)]).splitlines()
        if len(before)!=704 or before.count('radv_ps5_platform_native.o')!=1:raise RuntimeError('Unexpected driver archive members')
        run(['${LLVM}/llvm-ar.exe','r',str(driver),'${OUT}/obj/radv_ps5_platform_native.o'])
        print('Verifying the complete75-member MOSS core rebuild',flush=True)
        core_name='libstable-diffusion.graph-admission-1024-v3.a'
        baseline=archive_members(Path(c['DEPS'])/core_name)
        core_source,core_manifest=validate_moss_core(a.core_manifest,baseline,c)
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
            run(recipe);ggml_objects.append(recipe[recipe.index('-o')+1])
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
        for name in ['build-profiles/native.json','tools/build.py']:sources[name]=digest(ROOT/name)
        (out/'manifest.json').write_text(json.dumps({'local_build_only':True,'console_executed':False,
            'model_family':'moss-image-sft','requested_direct_bytes':12884901888,
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
