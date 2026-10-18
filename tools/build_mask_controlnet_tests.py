# SPDX-License-Identifier: GPL-3.0-or-later
"""Build intentional-failure Windows fixtures; never PS5 or model inference."""
import argparse,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 c=json.loads(a.config.read_text());out=a.out.resolve();host=out/'host';host.mkdir(parents=True,exist_ok=True)
 assert not (host/'build-manifest.json').exists(),'Preserve completed host fixture build'
 env=dict(os.environ);env['PATH']=str(c['LLVM'])+os.pathsep+env.get('PATH','')
 native,gateway,sha=R/'native/app',R/'native/gateway',R/'native/sha256';commands=[]
 common=[Path(c['LLVM'])/'clang.exe','--target=x86_64-w64-mingw32','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-unused-function']
 def run(cmd):
  cmd=list(map(str,cmd));r=subprocess.run(cmd,cwd=R,env=env,capture_output=True,text=True,errors='replace');commands.append(cmd)
  with (host/'build.log').open('a',encoding='utf-8') as f:f.write(subprocess.list2cmdline(cmd)+'\n'+r.stdout+r.stderr+'\n')
  if r.returncode:raise RuntimeError(r.stderr[-8000:])
 deps=['-I',native,'-I',sha,'-isystem',Path(c['SD'])/'thirdparty']
 for source,name in [(native/'server.c','native-failure'),(R/'tests/mask_native_owner.c','mask-native-owner')]:
  run([*common,*deps,source,native/'reference_image.c',sha/'sha256.c',R/'tests/reference_engine_failure_v1.c','-lws2_32','-o',host/(name+'.exe')])
 run([*common,*deps,R/'tests/mask_composite.c',native/'reference_image.c',sha/'sha256.c','-o',host/'mask-composite.exe'])
 for name,flag in [('gateway-protocol','-DPS5_AI_GATEWAY_PROTOCOL_TEST'),('gateway-forwarder','-DPS5_AI_UPSTREAM_PORT=18188')]:
  run([*common,'-I',gateway,'-I',sha,gateway/'server.c',gateway/'cpu_hooks.c',sha/'sha256.c',flag,'-lws2_32','-o',host/(name+'.exe')])
 run([*common,'-include','assert.h','-I',gateway,'-I',sha,R/'tests/mask_completion_contract.c',gateway/'cpu_hooks.c',sha/'sha256.c','-lws2_32','-o',host/'mask-completion-contract.exe'])
 (host/'build-manifest.json').write_text(json.dumps(dict(commands=commands,Windows_inference=False,PS5_connections=0),indent=2)+'\n')
 print('Six actual-source host fixtures built. No model inference or PS5 connection.')
if __name__=='__main__':main()
