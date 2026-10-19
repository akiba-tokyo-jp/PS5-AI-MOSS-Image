# SPDX-License-Identifier: GPL-3.0-or-later
"""Strict1024 benchmark profile and real two-claim helper; no native build."""
import argparse,copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--llvm',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out.resolve()
 if o.drive.upper()!='D:'or o.exists()or o.is_relative_to(R):raise RuntimeError('Fresh D directory outside repository required')
 o.mkdir();sys.path.insert(0,str(R/'tools'));s=importlib.util.spec_from_file_location('mossbuild',R/'tools/build.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 checks=[]
 def need(ok,name):
  if not ok:raise AssertionError(name)
  checks.append(name)
 def refuse(f,name):
  try:f()
  except RuntimeError:checks.append(name);return
  raise AssertionError(name)
 c=json.loads((R/'build-profiles/moss-1024-cold-warm.json').read_text())
 need(m.moss_trial_flags(True,True,c,False,True)==['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_TRIAL_10G=1','-DPS5_AI_MOSS_1024_COLD_WARM_TRIAL=1'],'distinct explicit1024 compile flag')
 for e,t,q,b in [(False,False,False,True),(True,False,False,True),(True,True,True,True),(1,True,False,True),(True,True,False,1)]:
  refuse(lambda:m.moss_trial_flags(e,t,c,q,b),'invalid switch tuple '+str((e,t,q,b)))
 refuse(lambda:m.moss_trial_flags(True,True,c),'benchmark cannot impersonate legacy256 profile')
 refuse(lambda:m.moss_trial_flags(True,True,c,True),'benchmark cannot impersonate quality256 profile')
 for key in c:
  bad=copy.deepcopy(c);del bad[key];refuse(lambda:m.moss_trial_flags(True,True,bad,False,True),'missing benchmark field '+key)
 for key,value in [('max_generation_attempts',3),('max_generation_attempts',2.0),('width',256),('steps',4),('seed',20261019),('moss_lora_strength',0),('moss_lora_strength',.8),('gpu_pool_bytes',12884901888),('production_runtime_admitted',True),('experimental_small_trial',1),('unknown',True)]:
  bad=copy.deepcopy(c);bad[key]=value;refuse(lambda:m.moss_trial_flags(True,True,bad,False,True),'changed benchmark field '+key+' '+str(value))
 common=[str(a.llvm/'clang++.exe'),'-std=c++20','-static','-O2','-Wall','-Wextra','-Werror','-I',str(R/'native/app')]
 flags=m.moss_trial_flags(True,True,c,False,True);exe=o/'benchmark-contract.exe';env=dict(os.environ,TMP=str(o),TEMP=str(o))
 command=[*common,*flags,str(R/'tests/moss_benchmark_contract.cpp'),'-o',str(exe)];r=subprocess.run(command,capture_output=True,text=True,env=env);(o/'compile.log').write_text(r.stdout+r.stderr);need(r.returncode==0,'actual benchmark helper compiled')
 r=subprocess.run([str(exe)],capture_output=True,text=True,env=env);(o/'run.log').write_text(r.stdout+r.stderr);need(r.returncode==0 and'PASS 20'in r.stdout,'actual request/2claims/thirdrefusal/12concurrent2winners verified')
 for extra in [['-DPS5_AI_MOSS_1024_COLD_WARM_TRIAL=1'],['-DPS5_AI_MOSS_1024_COLD_WARM_TRIAL=2'],['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_1024_COLD_WARM_TRIAL=1','-DPS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL=1']]:
  r=subprocess.run([*common,'-fsyntax-only',*extra,str(R/'tests/moss_benchmark_contract.cpp')],capture_output=True,text=True,env=env);need(r.returncode!=0,'incompatible mode compile refusal '+str(extra))
 files=['native/app/moss_trial_contract.hpp','native/app/engine_sdcpp.cpp','tools/build.py','build-profiles/moss-1024-cold-warm.json','tests/moss_benchmark_contract.cpp','tests/moss_benchmark_build.py']
 result={'success':True,'count':len(checks),'checks':checks,'compile_command':command,'GPU_calls':0,'PS5_connections':0,'model_inference':False,'source_sha256':{x:hashlib.sha256((R/x).read_bytes()).hexdigest()for x in files}}
 (o/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['success','count','GPU_calls','PS5_connections','model_inference']}))
if __name__=='__main__':main()
