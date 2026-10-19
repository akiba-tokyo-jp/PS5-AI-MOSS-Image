# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact performance profiles and actual atomic helper; no PS5/model/GPU."""
import argparse,copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--llvm',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out.drive.upper()!='D:' or out.exists() or out.is_relative_to(R):raise RuntimeError('Fresh D directory outside repository required')
 out.mkdir(parents=True);sys.path.insert(0,str(R/'tools'));spec=importlib.util.spec_from_file_location('mossbuild',R/'tools/build.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
 checks=[];commands=[]
 def need(ok,name):
  if not ok:raise AssertionError(name)
  checks.append(name)
 def refuse(f,name):
  try:f()
  except RuntimeError:checks.append(name);return
  raise AssertionError(name)
 env=dict(os.environ,TMP=str(out),TEMP=str(out))
 common=[str(a.llvm/'clang++.exe'),'-std=c++20','-static','-O2','-Wall','-Wextra','-Werror','-I',str(R/'native/app')]
 def run(args,log):
  commands.append(args);result=subprocess.run(args,capture_output=True,text=True,env=env);(out/log).write_text(result.stdout+result.stderr);return result
 for variant,(name,precision) in b.MOSS_PERFORMANCE_VARIANTS.items():
  profile=json.loads((R/('build-profiles/moss-performance-'+name+'.json')).read_text())
  flags=b.moss_trial_flags(True,True,profile,False,False,variant)
  need(flags==['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_TRIAL_10G=1','-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1','-DPS5_AI_MOSS_PERF_VARIANT='+str(variant)],'distinct flags variant '+str(variant))
  b.validate_moss_performance_precision(variant,precision);need(True,'precision bound '+name)
  refuse(lambda:b.validate_moss_performance_precision(variant,'fp16' if precision=='fp32' else 'fp32'),'wrong precision '+name)
  for key in profile:
   bad=copy.deepcopy(profile);del bad[key];refuse(lambda:b.moss_trial_flags(True,True,bad,False,False,variant),'missing '+name+' '+key)
  for key,value in [('max_generation_attempts',5),('max_generation_attempts',4.0),('allowed_sizes',[[256,256],[512,512]]),('steps',4),('seed',20261019),('moss_lora_strength',0.0),('gpu_pool_bytes',12884901888),('production_runtime_admitted',True),('performance_variant',True),('small256_success_and_visual_review_before1024',False),('unknown',True)]:
   bad=copy.deepcopy(profile);bad[key]=value;refuse(lambda:b.moss_trial_flags(True,True,bad,False,False,variant),'wrong '+name+' '+key)
  for enabled,trial,q,bench in [(False,True,False,False),(True,False,False,False),(True,True,True,False),(True,True,False,True)]:
   refuse(lambda:b.moss_trial_flags(enabled,trial,profile,q,bench,variant),'incompatible flags '+name+str((enabled,trial,q,bench)))
  exe=out/(name+'.exe');result=run([*common,*flags,str(R/'tests/moss_performance_contract.cpp'),'-o',str(exe)],name+'-compile.log');need(result.returncode==0,'actual helper compiled '+name)
  result=run([str(exe)],name+'-run.log');need(result.returncode==0 and 'PASS 30 ' in result.stdout,'actual invalid0/failureconsumed/max4/concurrency '+name)
 for variant in [-1,3,True,1.0,'1']:
  refuse(lambda:b.moss_trial_flags(True,True,{},False,False,variant),'invalid variant '+repr(variant))
 exe=out/'production-closed.exe';result=run([*common,str(R/'tests/moss_performance_contract.cpp'),'-o',str(exe)],'production-compile.log');need(result.returncode==0,'production0 compiled')
 result=run([str(exe)],'production-run.log');need(result.returncode==0 and 'PASS 2 ' in result.stdout,'production0 closed')
 syntax=out/'include-only.cpp';syntax.write_text('#include "moss_trial_contract.hpp"\n')
 for flags in [
  ['-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1'],
  ['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_PERFORMANCE_TRIAL=2'],
  ['-DPS5_AI_MOSS_PERF_VARIANT=1'],
  ['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1','-DPS5_AI_MOSS_PERF_VARIANT=3'],
  ['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1','-DPS5_AI_MOSS_1024_COLD_WARM_TRIAL=1'],
  ['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1','-DPS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL=1']]:
  result=run([*common,*flags,'-fsyntax-only',str(syntax)],'negative-'+str(len(checks))+'.log');need(result.returncode!=0,'compile-time incompatible mode refused '+str(flags))
 sources=['native/app/moss_trial_contract.hpp','tools/build.py','tests/moss_performance_contract.cpp','tests/moss_performance_build.py']+['build-profiles/moss-performance-'+name+'.json' for name,_ in b.MOSS_PERFORMANCE_VARIANTS.values()]
 receipt={'success':True,'count':len(checks),'checks':checks,'commands':commands,'source_sha256':{x:hashlib.sha256((R/x).read_bytes()).hexdigest() for x in sources},'production_runtime_admitted':False,'PS5_connections':0,'GPU_calls':0,'model_inference':False}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ['success','count','PS5_connections','GPU_calls','model_inference']}))
if __name__=='__main__':main()
