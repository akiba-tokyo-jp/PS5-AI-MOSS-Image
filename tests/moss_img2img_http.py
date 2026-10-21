# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile and exercise actual native/gateway request parsers. No inference/network."""
from pathlib import Path
import argparse,json,subprocess,os,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from build_gateway import moss_gateway_trial_flags
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir()
 cfg=json.loads(a.config.read_text());cc=Path(cfg['LLVM'])/'clang.exe';env=dict(os.environ,TMP=str(a.out),TEMP=str(a.out));checks=0
 def run(cmd):
  v=subprocess.run(list(map(str,cmd)),cwd=R,env=env,capture_output=True,text=True);assert not v.returncode,v.stderr[-6000:];return v.stdout
 for gateway in [False,True]:
  directory=R/('native/gateway'if gateway else'native/app');c=a.out/('gateway.c'if gateway else'native.c')
  fn='openai_validate(argv[1],&r,&url,error,sizeof(error),param)'if gateway else'validate_request(argv[1],&r,error,sizeof(error))'
  c.write_text('#define main original_server_main\n#include "server.c"\n#undef main\nint main(int argc,char**argv){if(argc!=2)return 2;ps5_ai_request r;char error[768],param[64];int url;int ok='+fn+';printf("{\\"accepted\\":%s,\\"steps\\":%d,\\"strength\\":%.9f}",ok?"true":"false",ok?r.steps:0,ok?r.denoise_strength:0.0);return 0;}\n',encoding='utf-8',newline='\n')
  for enabled in [0,1]:
   exe=a.out/(('gateway'if gateway else'native')+str(enabled)+'.exe');sha=R/'native/sha256'
   deps=[directory/'cpu_hooks.c']if gateway else[directory/'reference_image.c',R/'tests/reference_engine_failure_v1.c']
   flags=moss_gateway_trial_flags(False,True)+['-DPS5_AI_MOSS_IMG2IMG_TRIAL='+str(enabled)]
   run([cc,'--target=x86_64-w64-mingw32','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-unused-function','-Wno-unused-variable',*flags,'-DPS5_AI_GATEWAY_PROTOCOL_TEST','-I',directory,'-I',sha,'-isystem',Path(cfg['SD'])/'thirdparty',c,*deps,sha/'sha256.c','-lws2_32','-o',exe])
   def test(body,expected):
    nonlocal checks
    result=json.loads(run([exe,json.dumps(body,allow_nan=True)]));assert result['accepted']==expected,(gateway,enabled,body,result);checks+=1
    if expected:assert result['steps']==4
    return result
   test(dict(prompt='fixture'),True)
   image=dict(prompt='fixture',init_image_id='a'*32)
   v=test(image,bool(enabled))
   if enabled:assert v['strength']==.5
   for strength in [.01,.1,.249999,.25,.499999,.5,.749,.249999999,.499999999]:test(dict(image,denoise_strength=strength),bool(enabled))
   for strength in [-1,0,.009,.75,.749999999,1,1e300,float('nan'),float('inf'),'0.5',True]:test(dict(image,denoise_strength=strength),False)
   test(dict(prompt='fixture',denoise_strength=.5),False)
   for bad in ['','a'*31,'g'*32,'../input','A'*32]:test(dict(image,init_image_id=bad),False)
   for field in ['reference_image_id','mask_image_id','ip_adapter_strength','controlnet','control_image_id','control_strength','init_image_path']:
    test(dict(image,**{field:None}),False)
   for payload in ['{"prompt":"x","init_image_id":"'+('a'*32)+'","denoise_strength":.5}','{"prompt":"x","prompt":"y"}']:
    assert not json.loads(run([exe,payload]))['accepted'];checks+=1
 result=dict(passed=checks,actual_native_and_gateway_parsers=True,feature_off_and_on_tested=True,model_inference=False,PS5_connections=0)
 (a.out/'receipt.private.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
