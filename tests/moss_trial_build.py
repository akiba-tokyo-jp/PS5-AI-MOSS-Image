"""Only validate explicit trial build flags and real claim helper; no build/model."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,os,subprocess,sys
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--llvm',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out.drive.upper()!='D:' or out.exists() or out.is_relative_to(R):raise RuntimeError('Fresh D output outside repository required')
 out.mkdir();sys.path.insert(0,str(R/'tools'))
 spec=importlib.util.spec_from_file_location('moss_build',R/'tools/build.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 checks=[]
 def need(v,label):
  if not v:raise AssertionError(label)
  checks.append(label)
 def refuse(fn,label):
  try:fn()
  except RuntimeError:checks.append(label);return
  raise AssertionError('Accepted '+label)
 contract=json.loads((R/'build-profiles/moss-small10g-trial.json').read_text())
 need(m.moss_trial_flags(False,False)==['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=0','-DPS5_AI_MOSS_TRIAL_10G=0'],'default is closed12GiB profile')
 need(m.moss_trial_flags(True,True,contract)==['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_TRIAL_10G=1'],'explicit matching trial permits verification')
 refuse(lambda:m.moss_trial_flags(False,True,contract),'10GiB switch without explicit trial rejected')
 refuse(lambda:m.moss_trial_flags(True,False,contract),'trial without exact selected profile rejected')
 refuse(lambda:m.moss_trial_flags(1,True,contract),'nonboolean switch rejected')
 refuse(lambda:m.moss_trial_flags(True,True,{}),'absent trial contract rejected')
 for key in contract:
  bad=copy.deepcopy(contract);bad.pop(key);refuse(lambda:m.moss_trial_flags(True,True,bad),'missing contract field '+key)
 bad=copy.deepcopy(contract);bad['experimental_small_trial']=1;refuse(lambda:m.moss_trial_flags(True,True,bad),'integer must not impersonate trial boolean')
 bad=copy.deepcopy(contract);bad['width']=512;refuse(lambda:m.moss_trial_flags(True,True,bad),'unreviewed512 size rejected')
 bad=copy.deepcopy(contract);bad['production_runtime_admitted']=True;refuse(lambda:m.moss_trial_flags(True,True,bad),'production proof cannot be faked by trial profile')
 bad=copy.deepcopy(contract);bad['unknown']=True;refuse(lambda:m.moss_trial_flags(True,True,bad),'unknown trial field rejected')
 quality=json.loads((R/'build-profiles/moss-quality-step-comparison.json').read_text())
 need(m.moss_trial_flags(True,True,quality,True)==['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_TRIAL_10G=1','-DPS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL=1'],'explicit quality comparison profile has distinct compile flag')
 refuse(lambda:m.moss_trial_flags(True,True,quality),'quality profile cannot replace legacy4step profile')
 refuse(lambda:m.moss_trial_flags(True,True,contract,True),'legacy4step profile cannot enable quality comparison flag')
 refuse(lambda:m.moss_trial_flags(False,False,quality,True),'quality flag alone cannot run admission')
 refuse(lambda:m.moss_trial_flags(True,False,quality,True),'quality flag does not imply10GiB permission')
 refuse(lambda:m.moss_trial_flags(True,True,quality,1),'nonboolean quality flag rejected')
 for key in quality:
  bad=copy.deepcopy(quality);bad.pop(key);refuse(lambda:m.moss_trial_flags(True,True,bad,True),'quality missing field '+key)
 for key,value in [('steps',4),('allowed_steps',[4]),('allowed_steps',[50,4]),('allowed_steps',[4.0,50]),('allowed_steps',[4,51]),('seed',20261018),('moss_lora_strength',0.0),('moss_lora_strength',0.8),('width',512),('production_runtime_admitted',True),('unknown',True),('gpu_pool_bytes',12884901888)]:
  bad=copy.deepcopy(quality);bad[key]=value;refuse(lambda:m.moss_trial_flags(True,True,bad,True),'quality mutated field '+key+' '+str(value))
 env=dict(os.environ,TMP=str(out),TEMP=str(out));commands=[]
 for mode,quality_mode in [(0,0),(1,0),(1,1)]:
  exe=out/('claim%d-quality%d.exe'%(mode,quality_mode));cmd=[str(a.llvm/'clang++.exe'),'-std=c++20','-static','-O2','-Wall','-Wextra','-Werror','-I',str(R/'native/app'),'-DPS5_AI_MOSS_RUNTIME_ADMITTED=0','-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL='+str(mode),'-DPS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL='+str(quality_mode),str(R/'tests/moss_trial_contract.cpp'),'-o',str(exe)];commands.append(cmd)
  c=subprocess.run(cmd,capture_output=True,text=True,env=env);need(c.returncode==0,'actual claim compile mode%d: '%mode+c.stderr)
  c=subprocess.run([str(exe)],capture_output=True,text=True,env=env);need(c.returncode==0 and 'GPU_calls=0 model_inference=0' in c.stdout,'actual claim run mode%d '%mode+c.stdout)
 for definitions in [['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=2'],['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_RUNTIME_ADMITTED=1'],['-DPS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL=1'],['-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_QUALITY_STEP_COMPARISON_TRIAL=2']]:
  cmd=[str(a.llvm/'clang++.exe'),'-std=c++20','-fsyntax-only','-I',str(R/'native/app'),*definitions,str(R/'tests/moss_trial_contract.cpp')];commands.append(cmd)
  c=subprocess.run(cmd,capture_output=True,text=True,env=env);need(c.returncode!=0,'invalid/conflicting modes compile-refused')
 files=['native/app/moss_runtime_admission.h','native/app/moss_trial_contract.hpp','tests/moss_trial_contract.cpp','tools/build.py','tests/moss_trial_build.py','build-profiles/moss-small10g-trial.json','build-profiles/moss-quality-step-comparison.json']
 receipt={'success':True,'checks':checks,'count':len(checks),'commands':commands,'GPU_calls':0,'PS5_connections':0,'model_inference':False,'native_oracles_executed':False,'source_sha256':{s:hashlib.sha256((R/s).read_bytes()).hexdigest() for s in files}}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ['success','count','GPU_calls','PS5_connections','model_inference']}))
if __name__=='__main__':main()
