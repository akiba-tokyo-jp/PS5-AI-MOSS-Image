# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the CPU gateway / Windows protocol fixture. Never deploy or infer."""
import argparse, json, os, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--native',action='store_true');a=p.parse_args()
    c=json.loads(a.config.read_text());out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    clang=str(Path(c['LLVM'])/'clang.exe');src=ROOT/'native/gateway';sources=[src/'server.c',src/'cpu_hooks.c',ROOT/'native/sha256/sha256.c']
    env=dict(os.environ);env['PATH']=str(c['LLVM'])+os.pathsep+env.get('PATH','')
    def run(cmd):
        r=subprocess.run(list(map(str,cmd)),env=env,cwd=ROOT,capture_output=True,text=True,errors='replace')
        if r.returncode:raise RuntimeError(r.stderr[-8000:])
    spec=json.loads((ROOT/'docs/openapi.json').read_text());packed=json.dumps(spec,ensure_ascii=True,separators=(',',':'))
    (src/'openapi_schema.inc').write_text('/* Generated from docs/openapi.json. */\nstatic const char openapi_schema[]=\n'+json.dumps(packed)+';\n',encoding='utf-8',newline='\n')
    include=['-I',src,'-I',ROOT/'native/sha256']
    for name,flags in [('protocol-test',['-DPS5_AI_GATEWAY_PROTOCOL_TEST']),('host-forwarder',[])]:
        run([clang,'--target=x86_64-w64-mingw32','-std=gnu11','-O2','-Wall','-Wextra','-Werror',*sources,*include,*flags,'-lws2_32','-o',out/(name+'.exe')])
    if a.native:
        sdk=Path(c['PAYLOAD_SDK']);sdk_source=Path(c['SDK_SOURCE']);original=sdk/'target/lib/crt1.o'
        import hashlib
        if hashlib.sha256(original.read_bytes()).hexdigest()!='e13ce68fcd8525df4b480a37b6239827e30998284781f4dd0cf37ec06717371a':raise RuntimeError('Expected SDK0.44 payload CRT')
        crt=out/'crt1-credential-preserving.o';run([Path(c['LLVM'])/'llvm-objcopy.exe','--weaken-symbol=__patch_init',original,crt])
        patch=out/'patch-syscall-only.o'
        common=['--no-default-config','--target=x86_64-sie-ps5','-isystem',sdk/'target/include','-fno-stack-protector','-fno-plt','-femulated-tls','-fPIC','-O2','-Wall','-Wextra']
        run([clang,*common,'-std=c11','-I',sdk_source/'crt','-c',ROOT/'native/payload_crt/credential_preserving_patch.c','-o',patch])
        objects=[]
        for i,s in enumerate(sources):
            obj=out/f'gateway-{i}.o';run([clang,*common,'-std=gnu11','-D_BSD_SOURCE','-Werror',*include,'-c',s,'-o',obj]);objects.append(obj)
        run([Path(c['LLVM'])/'ld.lld.exe','-m','elf_x86_64','-pie','-T',sdk/'ldscripts/elf_x86_64.x','--eh-frame-hdr','-z','max-page-size=0x4000','--hash-style=gnu','-L',sdk/'target/lib','-L',sdk/'target/user/homebrew/lib','-o',out/'ps5-ai-openai-gateway.elf',*objects,crt,patch,'-lc','-lkernel_web','-lSceLibcInternal','-lSceNet'])
    print('Gateway build complete. No deployment, model loading or inference performed.')
if __name__=='__main__':main()
