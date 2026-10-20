"""Actual sampler contracts and native/gateway parsers; host only, no network."""
import argparse,copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 out=a.out.resolve()
 if out.drive.upper()!='D:'or out.exists()or out.is_relative_to(R):raise RuntimeError('Fresh D output outside repository required')
 out.mkdir(parents=True);c=json.loads(a.config.read_text());llvm=Path(c['LLVM']);sys.path.insert(0,str(R/'tools'))
 def module(name,file):
  spec=importlib.util.spec_from_file_location(name,R/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 b=module('actual_native_build','tools/build.py');g=module('actual_gateway_build','tools/build_gateway.py')
 checks=[];commands=[];env=dict(os.environ,TMP=str(out),TEMP=str(out),PATH=str(llvm)+os.pathsep+os.environ.get('PATH',''))
 def need(ok,name):
  if not ok:raise AssertionError(name)
  checks.append(name)
 def refuse(fn,name):
  try:fn()
  except RuntimeError:checks.append(name);return
  raise AssertionError(name)
 def run(cmd,label,good=True):
  cmd=list(map(str,cmd));commands.append(cmd);r=subprocess.run(cmd,capture_output=True,text=True,errors='replace',env=env,cwd=R)
  (out/(label+'.log')).write_text(r.stdout+r.stderr)
  need((r.returncode==0)==good,label);return r
 profile=json.loads((R/'build-profiles/moss-performance-resmultistep-fp16-f32accum.json').read_text())
 need(profile==b.moss_performance_contract(1,True,False,False,True),'exact new typed profile')
 flags=b.moss_trial_flags(True,True,profile,False,False,1,True,False,False,True)+b.moss_accumulation_compile_flags(True)+b.moss_gpu_profile_compile_flags(False)+b.moss_q4_bk64_compile_flags(False)+b.moss_sampler_compile_flags(True)
 need('-DPS5_AI_MOSS_SAMPLER_TRIAL=1'in flags and len(flags)==len(set(flags)),'single explicit sampler policy flag')
 for key in profile:
  bad=copy.deepcopy(profile);del bad[key]
  refuse(lambda bad=bad:b.moss_trial_flags(True,True,bad,False,False,1,True,False,False,True),'missing '+key)
 for key,value in [('steps',50),('cfg_scale',3.0),('scheduler','discrete'),('sampler','euler'),('max_generation_attempts',5),('moss_lora_strength',1.0),('expected_dit_forwards',100),('expected_q4_f32acc_recordings',16600),('production_runtime_admitted',True)]:
  bad={**profile,key:value};refuse(lambda bad=bad:b.moss_trial_flags(True,True,bad,False,False,1,True,False,False,True),'wrong '+key)
 for args in [(True,False,False,False,1,'fp16',True,True),(True,True,True,False,1,'fp16',True,True),(True,True,False,True,1,'fp16',True,True),(True,True,False,False,0,'fp32',True,True),(True,True,False,False,1,'fp16',False,True)]:
  refuse(lambda args=args:b.validate_moss_sampler_mode(*args),'incompatible '+str(args))
 old=json.loads((R/'build-profiles/moss-performance-fp16-f32accum.json').read_text())
 need(old==b.moss_performance_contract(1,True),'original50Euler contract unchanged')
 oldflags=b.moss_trial_flags(True,True,old,False,False,1,True)+b.moss_accumulation_compile_flags(True)
 common=[llvm/'clang++.exe','-std=c++20','-static','-O2','-Wall','-Wextra','-Werror','-I',R/'native/app']
 for name,defs in [('sampler',flags),('original',oldflags),('production',[])]:
  exe=out/(name+'.exe');run([*common,*defs,R/'tests/moss_sampler_trial_contract.cpp','-o',exe],name+'-compile')
  need('PASS'in run([exe],name+'-run').stdout,name+' actual bounded claims')
 syntax=out/'header.cpp';syntax.write_text('#include "moss_trial_contract.hpp"\n')
 for name,defs in [('missing-coupling',['-DPS5_AI_MOSS_SAMPLER_TRIAL=1']),('bad-flag',[*oldflags,'-DPS5_AI_MOSS_SAMPLER_TRIAL=2']),('profile-enabled',[x for x in flags if not x.startswith('-DPS5_AI_MOSS_GPU_PROFILE=')]+['-DPS5_AI_MOSS_GPU_PROFILE=1'])]:
  run([*common,*defs,'-fsyntax-only',syntax],name,False)
 # Compile the real parser functions in the full source translation units.
 for mode in (False,True):
  defs=g.moss_gateway_sampler_flags(mode);need(defs[-1]=='-DPS5_AI_MOSS_SAMPLER_TRIAL='+str(int(mode)),'gateway explicit flag '+str(mode))
  for family in ('app','gateway'):
   unit=out/(family+str(int(mode))+'.c');exe=unit.with_suffix('.exe');is_gateway=family=='gateway'
   unit.write_text('''#define main unused_server_main
#include "server.c"
#undef main
int main(void){ps5_ai_request r;char error[768];
 if(!validate_request("{\\"prompt\\":\\"host parser fixture\\"}",&r,error,sizeof(error)))return 1;
 if(r.steps!=PS5_AI_MOSS_DEFAULT_REQUEST_STEPS || r.cfg_scale!=PS5_AI_MOSS_DEFAULT_REQUEST_CFG || strcmp(r.sampler,PS5_AI_MOSS_DEFAULT_REQUEST_SAMPLER))return 2;
 if(validate_request("{\\"prompt\\":\\"x\\",\\"sampler\\":\\"res_multistep\\",\\"steps\\":30,\\"cfg_scale\\":4,\\"seed\\":20261020}",&r,error,sizeof(error))!=PS5_AI_MOSS_SAMPLER_TRIAL)return 3;
 if(validate_request("{\\"prompt\\":\\"x\\",\\"sampler\\":\\"euler\\"}",&r,error,sizeof(error))==PS5_AI_MOSS_SAMPLER_TRIAL)return 4;
 if(validate_request("{\\"prompt\\":\\"x\\",\\"scheduler\\":\\"simple\\"}",&r,error,sizeof(error)))return 5;
'''+(''' int url;char param[64];
 if(!openai_validate("{\\"prompt\\":\\"host OpenAI parser fixture\\"}",&r,&url,error,sizeof(error),param))return 6;
 if(r.steps!=PS5_AI_MOSS_DEFAULT_REQUEST_STEPS || r.cfg_scale!=PS5_AI_MOSS_DEFAULT_REQUEST_CFG || strcmp(r.sampler,PS5_AI_MOSS_DEFAULT_REQUEST_SAMPLER))return 7;
 if(openai_validate("{\\"prompt\\":\\"x\\",\\"sampler\\":\\"res_multistep\\",\\"steps\\":30,\\"cfg_scale\\":4,\\"seed\\":20261020}",&r,&url,error,sizeof(error),param)!=PS5_AI_MOSS_SAMPLER_TRIAL)return 8;
'''if is_gateway else'')+''' puts("PASS actual host parser profile; no network/model/GPU");return 0;}
''')
   source=R/('native/'+family);deps=[R/'native/sha256/sha256.c']
   if is_gateway:deps+=[source/'cpu_hooks.c'];defs+=['-DPS5_AI_GATEWAY_PROTOCOL_TEST']
   else:deps+=[source/'reference_image.c',R/'tests/reference_engine_failure_v1.c']
   cmd=[llvm/'clang.exe','--target=x86_64-w64-mingw32','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-unused-function',*defs,'-I',source,'-I',R/'native/app','-I',R/'native/sha256','-isystem',Path(c['SD'])/'thirdparty',unit,*deps,'-lws2_32','-o',exe]
   run(cmd,family+str(int(mode))+'-compile');need('PASS'in run([exe],family+str(int(mode))+'-run').stdout,'actual '+family+' parser mode '+str(mode))
 sources=['native/app/moss_sampler_trial.h','native/app/moss_trial_contract.hpp','native/app/engine_sdcpp.cpp','native/app/server.c','native/gateway/server.c','native/gateway/openai.inc','tools/build.py','tools/build_gateway.py','build-profiles/moss-performance-resmultistep-fp16-f32accum.json','tests/moss_sampler_trial_contract.cpp','tests/moss_sampler_trial_build.py']
 receipt=dict(success=True,count=len(checks),checks=checks,commands=commands,source_sha256={x:hashlib.sha256((R/x).read_bytes()).hexdigest()for x in sources},PS5_connections=0,GPU_calls=0,model_inference=False,network_connections=0)
 (out/'receipt.private.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k]for k in ('success','count','PS5_connections','GPU_calls','model_inference','network_connections')}))

if __name__=='__main__':main()
