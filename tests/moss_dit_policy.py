# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline opt-in/parent/actual-command guards. No inference/network or model reads."""
from pathlib import Path
import argparse,copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import moss_dit_policy as policy
import build_moss_core as core
import build

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();a.out.mkdir(parents=True);checks=[]
 def yes(v,m):assert v,m;checks.append(m)
 def no(f,m):
  try:f()
  except(ValueError,RuntimeError,TypeError):checks.append('reject '+m);return
  raise AssertionError(m)
 for v in (0,1,None,'true'):no(lambda v=v:policy.flags(v),'untyped enable '+repr(v))
 parent=dict(accumulation=True,turbo=True,upload=True,spectrum=False,ffn=False,lora=None)
 policy.validate(True,**parent);yes(policy.fields(False)['moss_dit_policy']=='disabled','Default off')
 for k,v in dict(accumulation=False,turbo=False,upload=False,spectrum=True,ffn=True,lora='f16').items():
  wrong=dict(parent);wrong[k]=v;no(lambda wrong=wrong:policy.validate(True,**wrong),'noncanonical parent '+k)
 rows=core.validate_profile(json.loads(core.PROFILE.read_text()))
 for enabled in(False,True):
  cmd=core.materialize_compile_commands(rows,lambda x:x,True,False,True,True,False,0,None,enabled)
  manifest=dict(policy.fields(enabled),commands=cmd)
  yes(len(cmd)==75 and policy.validate_core(manifest,enabled)==policy.fields(enabled),'All75 capability '+str(enabled))
  for i in(0,74):
   for change in('remove','duplicate','opposite','undef'):
    bad=copy.deepcopy(manifest);flag=policy.flags(enabled)[0]
    if change=='remove':bad['commands'][i].remove(flag)
    elif change=='duplicate':bad['commands'][i].append(flag)
    elif change=='opposite':bad['commands'][i][bad['commands'][i].index(flag)]=policy.flags(not enabled)[0]
    else:bad['commands'][i].append('-U'+policy.MACRO)
    no(lambda bad=bad:policy.validate_core(bad,enabled),str(enabled)+change+str(i))
  for field in policy.fields(enabled):
   bad=copy.deepcopy(manifest);del bad[field];no(lambda bad=bad:policy.validate_core(bad,enabled),'missing '+field)
 for marker in('-DPS5_AI_MOSS_DIT_OPT_TRIAL=0','-UPS5_AI_MOSS_DIT_OPT_TRIAL'):
  bad=copy.deepcopy(rows);bad[5]['argv'].append(marker);no(lambda bad=bad:core.materialize_compile_commands(bad,lambda x:x,True,False,True,True,False,0,None,True),'recipe override '+marker)
 contract=build.moss_performance_contract(1,True,moss_turbo_trial=True,moss_bounded_weight_upload_trial=True)
 yes(contract['max_generation_attempts']==4 and contract['production_runtime_admitted']is False,'Bounded parent admission retained')
 yes(contract==json.loads((ROOT/'build-profiles/moss-performance-turbo-bounded-upload-fp16-f32accum.json').read_text()),'Parent config and sigmas unchanged')
 value=dict(checks=checks,passed=len(checks),PS5_connections=0,inference=False);(a.out/'receipt.private.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(passed=len(checks),PS5_connections=0)))
if __name__=='__main__':main()
