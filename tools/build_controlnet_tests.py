# SPDX-License-Identifier: GPL-3.0-or-later
"""Build Windows HTTP fixtures. The fake engine always fails; no inference."""
import argparse,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.config.read_text());out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ);env['PATH']=str(c['LLVM'])+os.pathsep+env.get('PATH','')
    common=[Path(c['LLVM'])/'clang.exe','--target=x86_64-w64-mingw32','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-unused-function']
    native=R/'native/app';gateway=R/'native/gateway';sha=R/'native/sha256';commands=[]
    def run(args):
        args=list(map(str,args));r=subprocess.run(args,cwd=R,env=env,capture_output=True,text=True,errors='replace');commands.append(args)
        if r.returncode:raise RuntimeError(r.stderr[-8000:])
    run([*common,'-I',native,'-I',sha,'-isystem',Path(c['SD'])/'thirdparty',native/'server.c',native/'reference_image.c',sha/'sha256.c',R/'tests/reference_engine_failure_v1.c','-lws2_32','-o',out/'native-failure.exe'])
    for name,flag in [('gateway-protocol','-DPS5_AI_GATEWAY_PROTOCOL_TEST'),('gateway-forwarder','-DPS5_AI_UPSTREAM_PORT=18188')]:
        run([*common,'-I',gateway,'-I',sha,gateway/'server.c',gateway/'cpu_hooks.c',sha/'sha256.c',flag,'-lws2_32','-o',out/(name+'.exe')])
    (out/'manifest.json').write_text(json.dumps({'commands':commands,'Windows_inference':False,'PS5_connections':0},indent=2)+'\n')
    print('Built three host-only protocol fixtures. No inference or PS5 connection.')

if __name__=='__main__':main()
