# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows local native rebuild from exported sources + reviewed dependencies.

Does not connect to a PS5, install a title, patch firmware, or run inference.
See docs/build.md for the explicit dependency-build limitation.
"""
import argparse, hashlib, json, os, re, shutil, struct, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mapped(data):
    offset=struct.unpack_from('<Q',data,32)[0];size,count=struct.unpack_from('<HH',data,54)
    return [(p,hashlib.sha256(data[p[2]:p[2]+p[5]]).hexdigest()) for i in range(count)
            if (p:=struct.unpack_from('<IIQQQQQQ',data,offset+i*size))[0]==1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--precision',choices=['fp32','fp16'],default='fp32');a=p.parse_args()
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
    out.mkdir(parents=True,exist_ok=True);(out/'obj').mkdir(exist_ok=True)
    env=dict(os.environ);env['PATH']=str(c['HOST_BIN'])+os.pathsep+str(c['LLVM'])+os.pathsep+env.get('PATH','')
    commands=[]
    with (out/'build.log').open('w',encoding='utf-8') as log:
        def run(cmd):
            cmd=[expand(str(v)) for v in cmd];commands.append(cmd)
            result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
            log.write(subprocess.list2cmdline(cmd)+'\n'+result.stdout+result.stderr+'\n');log.flush()
            if result.returncode:raise RuntimeError(result.stderr[-8000:])
            return result.stdout
        common=['--no-default-config','-target','x86_64-sie-ps5','-D__PROSPERO__=1',
          '-fvisibility-nodllstorageclass=default','-isysroot','${SDK}',
          '-isystem','${SDK}/target/include/c++/v1','-isystem','${SDK}/target/include',
          '-I','${PLATFORM}/src','-I','${ROOT}/native/app','-I','${ROOT}/native/ui',
          '-I','${SD}/include','-I','${SD}/ggml/include','-isystem','${SD}/thirdparty',
          '-I','${VULKAN_HEADERS}/include','-fno-stack-protector','-fno-plt','-femulated-tls',
          '-fdenormal-fp-math=ieee','-O2','-ffunction-sections','-fdata-sections','-Wall','-Wextra','-Werror','-MMD']
        for name in profile['objects']:
            stem=Path(name).stem
            if stem=='app_crt':source='native/link/app_crt.cpp'
            elif stem in ('format','display'):source='native/ui/'+stem+'.c'
            elif stem in ('cpp_allocation','cxa_owned','threads'):source='native/runtime/'+stem+('.cpp' if stem=='cpp_allocation' else '.c')
            elif stem in ('engine_sdcpp','vulkan_bridge','cpp_allocation'):source='native/app/'+stem+'.cpp'
            else:source='native/app/'+stem+'.c'
            cpp=source.endswith('.cpp');extra=['-std=c++20','-fexceptions','-fcxx-exceptions'] if cpp else ['-std=c11']
            if stem in ('app_crt','cpp_allocation'):extra+=['-fno-exceptions','-fno-rtti']
            if stem=='server':extra+=['-Dmain=ps5_ai_server_main']
            if stem=='engine_sdcpp':extra+=['-DPS5_AI_NATIVE_RADV=1']
            if stem in ('engine_sdcpp','startup'):extra+=['-DPS5_AI_ENABLE_FP16='+str(int(a.precision=='fp16'))]
            print('Compiling '+source,flush=True)
            run(['${LLVM}/clang.exe',*common,*extra,'-c','${ROOT}/'+source,'-o','${OUT}/obj/'+name])
        print('Compiling the owned RADV platform layer',flush=True);run(profile['platform_compile'])
        driver=out/'libvulkan_radeon.native-direct-data.a';shutil.copyfile(Path(c['DEPS'])/driver.name,driver)
        before=run(['${LLVM}/llvm-ar.exe','t',str(driver)]).splitlines()
        if len(before)!=704 or before.count('radv_ps5_platform_native.o')!=1:raise RuntimeError('Unexpected driver archive members')
        run(['${LLVM}/llvm-ar.exe','r',str(driver),'${OUT}/obj/radv_ps5_platform_native.o'])
        print('Compiling graph admission and finite output checks',flush=True)
        run(profile['runner_compile'])
        core_name='libstable-diffusion.graph-admission-1024-v3.a'
        core=out/'libstable-diffusion.precision.a';shutil.copyfile(Path(c['DEPS'])/core_name,core)
        before=run(['${LLVM}/llvm-ar.exe','t',str(core)]).splitlines()
        if len(before)!=75 or before.count('ggml_runner.cpp.o')!=1:raise RuntimeError('Unexpected core archive members')
        run(['${LLVM}/llvm-ar.exe','r',str(core),'${OUT}/ggml_runner.cpp.o'])
        link=[v.replace('${DEPS}/'+driver.name,str(driver).replace('\\','/')).replace('${DEPS}/'+core_name,str(core).replace('\\','/')) for v in profile['link']]
        print('Linking native application',flush=True);run(link)
        raw=out/'llvm-pie.elf';deploy=out/'llvm-pie.deploy.elf'
        run(['${LLVM}/llvm-strip.exe','--strip-debug','-o',str(deploy),str(raw)])
        if mapped(raw.read_bytes())!=mapped(deploy.read_bytes()):raise RuntimeError('Stripping modified loaded bytes')
        dynamic=run(['${LLVM}/llvm-readelf.exe','--dynamic',str(raw)])
        needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]',dynamic)
        if len(needed)!=6:raise RuntimeError('Expected six native runtime providers')
        native=out/'eboot.elf';stubs=[str(Path(c['DEPS'])/n) for n in ['libSceAgc.so','libSceAgcDriver.so']]+[str(Path(c['SDK'])/'target/lib'/n) for n in ['libSceVideoOut.so','libSceSystemService.so']]
        run(['${NATIVE_TOOL}','link','--in',str(deploy),'--out',str(native),'--stub-dir','${SDK}/target/lib',*sum((['--stub',s] for s in stubs),[]),'--module-sdk','0x02000009','--companion-sdk','0x08050001','--file-name','eboot.elf'])
        title=out/'title/PPSA99188';(title/'sce_sys').mkdir(parents=True,exist_ok=True)
        for name in ('param.json','icon0.png'):shutil.copyfile(ROOT/'assets/sce_sys'/name,title/'sce_sys'/name)
        (title/'licenses').mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'licenses/WAI-Dashboard-Glyphs-OFL.txt',title/'licenses/WAI-Dashboard-Glyphs-OFL.txt')
        run(['${NATIVE_TOOL}','self','--sign','--in',str(native),'--out',str(title/'eboot.bin'),'--magic','0x1D3D154F'])
        inspection=run(['${NATIVE_TOOL}','self','--inspect','--file',str(title/'eboot.bin')]);(out/'inspect.txt').write_text(inspection)
        if 'integrity: valid' not in inspection:raise RuntimeError('SELF integrity not confirmed')
        (out/'manifest.json').write_text(json.dumps({'local_build_only':True,'console_executed':False,'requested_precision':a.precision,'runner_sha256':digest(ROOT/'patches/stable-diffusion/overlay/src/core/ggml_runner.cpp'),'scratch_bounds_sha256':digest(ROOT/'patches/stable-diffusion/overlay/src/core/vulkan_scratch_bounds.hpp'),'needed':needed,'sources':{str(s.relative_to(ROOT)):digest(s) for folder in ['native/app','native/ui','native/runtime','native/link'] for s in (ROOT/folder).iterdir() if s.is_file()},'eboot_sha256':digest(title/'eboot.bin'),'commands':commands},indent=2)+'\n')
        print('Built and inspected '+str(title/'eboot.bin')+'. No deployment or inference was performed.',flush=True)
if __name__=='__main__':main()
