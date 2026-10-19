# SPDX-License-Identifier: GPL-3.0-or-later
"""Precision-policy admission and all75 real recipe command bindings; no compile/GPU/model."""
import argparse,copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out.drive.upper()!='D:' or out.exists() or out.is_relative_to(R):raise RuntimeError('Fresh D output outside repository required')
 out.mkdir(parents=True);sys.path.insert(0,str(R/'tools'));checks=[]
 def load(name,path):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 b=load('native_build',R/'tools/build.py');c=load('core_build',R/'tools/build_moss_core.py')
 def need(ok,name):
  if not ok:raise AssertionError(name)
  checks.append(name)
 def refuse(f,name):
  try:f()
  except (RuntimeError,ValueError):checks.append(name);return
  raise AssertionError(name)
 new=json.loads((R/'build-profiles/moss-performance-fp16-f32accum.json').read_text(encoding='utf-8'))
 need(new==b.moss_performance_contract(1,True),'new exact typed FP16/F32acc profile')
 flags=b.moss_trial_flags(True,True,new,False,False,1,True)
 need(flags==b.moss_trial_flags(True,True,b.moss_performance_contract(1),False,False,1),'trial flags stay distinct from accumulation policy flags')
 b.validate_moss_accumulation_mode(True,1,'fp16',True,True);need(True,'explicit performance1/fp16/small10GiB accepted')
 for mode in [(True,0,'fp32',True,True),(True,2,'fp32',True,True),(True,None,'fp16',True,True),(True,True,'fp16',True,True),(True,1,'fp32',True,True),(True,1,'fp16',False,True),(True,1,'fp16',True,False),(1,1,'fp16',True,True)]:
  refuse(lambda:b.validate_moss_accumulation_mode(*mode),'invalid mode '+repr(mode))
 for key in new:
  bad=copy.deepcopy(new);del bad[key];refuse(lambda:b.moss_trial_flags(True,True,bad,False,False,1,True),'missing exact field '+key)
 for key,value in [('moss_f32_accumulation',False),('moss_f32_accumulation',1),('moss_f32_accumulation','true'),('requested_precision','fp32'),('performance_variant',0),('performance_variant',True),('steps',4),('max_generation_attempts',5),('gpu_pool_bytes',12884901888),('unknown',True)]:
  bad=copy.deepcopy(new);bad[key]=value;refuse(lambda:b.moss_trial_flags(True,True,bad,False,False,1,True),'changed exact field '+key+' '+repr(value))
 refuse(lambda:b.moss_trial_flags(True,True,new,False,False,1),'new profile cannot impersonate old default0')
 for variant,(name,precision) in b.MOSS_PERFORMANCE_VARIANTS.items():
  old=json.loads((R/('build-profiles/moss-performance-'+name+'.json')).read_text(encoding='utf-8'))
  need(old==b.moss_performance_contract(variant) and 'moss_f32_accumulation' not in old,'old profile unchanged/default0 '+name)
  need(b.moss_accumulation_compile_flags(False)==['-DPS5_AI_MOSS_F32_ACCUMULATION=0'],'native explicit default0 '+name)
  refuse(lambda:b.moss_trial_flags(True,True,old,False,False,variant,True),'old profile cannot opt in without exact new field '+name)
 for bad in [0,1,None,0.0,'1']:
  refuse(lambda:b.moss_accumulation_compile_flags(bad),'native refuses non-bool policy '+repr(bad))
  refuse(lambda:c.moss_accumulation_compile_flag(bad),'core refuses non-bool policy '+repr(bad))
 rows=c.validate_profile(json.loads((R/'build-profiles/moss-core.json').read_text(encoding='utf-8')))
 substitutions={'ROOT':R.as_posix(),'SD':'D:/metadata-only/moss','SDK':'D:/metadata-only/sdk','LLVM':'D:/metadata-only/llvm','OUT':out.as_posix()}
 def expand(s):
  for key,value in substitutions.items():s=s.replace('${'+key+'}',value)
  return s
 def manifest(enabled):
  commands=c.materialize_compile_commands(rows,expand,enabled)
  return {'moss_f32_accumulation':enabled,'moss_accumulation_policy':b.moss_accumulation_policy(enabled),'moss_accumulation_compile_flag':c.moss_accumulation_compile_flag(enabled),
   'commands':commands,'members':[{'member':row['member'],'command_index':i,'source':commands[i][commands[i].index('-c')+1]} for i,row in enumerate(rows)]}
 defaults=manifest(False);forced=manifest(True)
 for enabled,m in [(False,defaults),(True,forced)]:
  proof=b.validate_moss_core_accumulation(m,enabled)
  need(proof['actual_compile_commands_verified']==75 and not proof['historical_default0_without_macro'],'all75 actual recipe source/index/single-macro proof '+str(enabled))
  need(all(command.count(c.moss_accumulation_compile_flag(enabled))==1 for command in m['commands']),'exactly one core macro per compile '+str(enabled))
  need(b.moss_accumulation_compile_flags(enabled)==[c.moss_accumulation_compile_flag(enabled)],'native/core flags identical '+str(enabled))
 refuse(lambda:b.validate_moss_core_accumulation(forced,False),'forced core rejected by default native')
 refuse(lambda:b.validate_moss_core_accumulation(defaults,True),'default core rejected by forced native')
 historical=copy.deepcopy(defaults)
 for key in ['moss_f32_accumulation','moss_accumulation_policy','moss_accumulation_compile_flag']:del historical[key]
 for command in historical['commands']:command.remove('-DPS5_AI_MOSS_F32_ACCUMULATION=0')
 need(b.validate_moss_core_accumulation(historical,False)['historical_default0_without_macro'],'historical commands without macro only default0')
 refuse(lambda:b.validate_moss_core_accumulation(historical,True),'historical core cannot satisfy forced native')
 negative=[]
 def mutated(label,change):
  value=copy.deepcopy(forced);change(value);negative.append((label,value))
 mutated('integer declaration',lambda m:m.update(moss_f32_accumulation=1))
 mutated('wrong policy declaration',lambda m:m.update(moss_accumulation_policy='model_default'))
 mutated('wrong macro declaration',lambda m:m.update(moss_accumulation_compile_flag='-DPS5_AI_MOSS_F32_ACCUMULATION=0'))
 mutated('missing command',lambda m:m['commands'].pop())
 mutated('missing member',lambda m:m['members'].pop())
 mutated('duplicate command index',lambda m:m['members'][1].update(command_index=0))
 mutated('boolean command index',lambda m:m['members'][0].update(command_index=False))
 mutated('wrong source binding',lambda m:m['members'][0].update(source='D:/metadata-only/wrong.c'))
 mutated('one missing macro',lambda m:m['commands'][37].remove('-DPS5_AI_MOSS_F32_ACCUMULATION=1'))
 mutated('duplicate macro',lambda m:m['commands'][37].append('-DPS5_AI_MOSS_F32_ACCUMULATION=1'))
 mutated('overriding macro',lambda m:m['commands'][37].append('-DPS5_AI_MOSS_F32_ACCUMULATION=0'))
 mutated('undefining macro',lambda m:m['commands'][37].append('-UPS5_AI_MOSS_F32_ACCUMULATION'))
 mutated('duplicate compile source token',lambda m:m['commands'][37].append('-c'))
 for label,m in negative:refuse(lambda:b.validate_moss_core_accumulation(m,True),'command proof rejects '+label)
 partial=copy.deepcopy(historical);partial['moss_accumulation_policy']='model_default';refuse(lambda:b.validate_moss_core_accumulation(partial,False),'partial historical policy rejected')
 for definition in ['-DPS5_AI_MOSS_F32_ACCUMULATION=1','-DPS5_AI_MOSS_F32_ACCUMULATION=0','-UPS5_AI_MOSS_F32_ACCUMULATION']:
  bad=copy.deepcopy(rows);bad[0]['argv'].append(definition);refuse(lambda:c.materialize_compile_commands(bad,expand,True),'recipe override rejected '+definition)
 for path in [R/'tools/build.py',R/'tools/build_moss_core.py']:
  result=subprocess.run([sys.executable,str(path),'--help'],capture_output=True,text=True);need(result.returncode==0 and '--moss-f32-accumulation' in result.stdout,'actual CLI exposes opt-in '+path.name)
 files=['tools/build.py','tools/build_moss_core.py','build-profiles/moss-performance-fp16-f32accum.json','tests/moss_f32_accumulation_build.py']
 receipt={'schema':'moss-f32-accumulation-build-host-v1','success':True,'count':len(checks),'checks':checks,'all75_real_recipes_materialized':True,'compiler_executed':False,'PS5_connections':0,'GPU_calls':0,'model_inference':False,'source_sha256':{name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in files}}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps({key:receipt[key] for key in ['success','count','compiler_executed','PS5_connections','GPU_calls','model_inference']}))
if __name__=='__main__':main()
