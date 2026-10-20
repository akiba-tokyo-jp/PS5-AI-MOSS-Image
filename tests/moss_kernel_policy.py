# SPDX-License-Identifier: GPL-3.0-or-later
"""Actual 75-recipe capability identity and refusal paths. No inference/network."""
import argparse,copy,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
import moss_kernel_policy as p
import build_moss_core as b
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir();checks=[]
 def rejected(fn,label):
  try:fn()
  except(ValueError,TypeError):checks.append(label);return
  raise AssertionError(label)
 for v in [0,1,None,'1']:rejected(lambda v=v:p.flags(v),'untyped capability '+repr(v))
 for missing in ['dit','accumulation','turbo','upload']:
  parent=dict(dit=True,accumulation=True,turbo=True,upload=True);parent[missing]=False
  rejected(lambda parent=parent:p.validate(True,**parent),'missing parent '+missing)
 rows=b.validate_profile(json.loads(b.PROFILE.read_text()))
 for enabled in [False,True]:
  commands=b.materialize_compile_commands(rows,lambda s:s,True,False,True,True,False,0,None,True,enabled)
  m=dict(p.fields(enabled),commands=commands,kernel_policy_input=p.helper_input());p.validate_core(m,enabled);checks.append('all75 '+str(enabled))
  bad=copy.deepcopy(m);del bad['kernel_policy_input'];rejected(lambda bad=bad:p.validate_core(bad,enabled),'missing helper '+str(enabled))
  for i in [0,74]:
   for change in ['remove','opposite','duplicate','undef']:
    bad=copy.deepcopy(m);flag=p.flags(enabled)[0]
    if change=='remove':bad['commands'][i].remove(flag)
    elif change=='opposite':bad['commands'][i][bad['commands'][i].index(flag)]=p.flags(not enabled)[0]
    elif change=='duplicate':bad['commands'][i].append(flag)
    else:bad['commands'][i].append('-U'+p.MACRO)
    rejected(lambda bad=bad:p.validate_core(bad,enabled),str(enabled)+change+str(i))
  for key in p.fields(enabled):
   bad=copy.deepcopy(m);del bad[key];rejected(lambda bad=bad:p.validate_core(bad,enabled),'missing '+key)
 for mask in [-1,16,True,'1']:
  rejected(lambda mask=mask:p.validate(True,dit=True,accumulation=True,turbo=True,upload=True,mask=mask),'invalid mask '+repr(mask))
 for mask in range(16):p.validate(True,dit=True,accumulation=True,turbo=True,upload=True,mask=mask);checks.append('mask '+str(mask))
 for v in [0,1,None,'true']:rejected(lambda v=v:p.fa_flags(v,3),'untyped FA '+repr(v))
 for mask in [None,0,1,4,5,8,9,12,13,True,'3',16]:rejected(lambda mask=mask:p.fa_flags(True,mask),'FA without valid shared-KV '+repr(mask))
 for mask in range(16):
  assert p.fa_flags(False,mask)==['-DPS5_AI_MOSS_DIT_FA_NO_OCCUPANCY_LIMIT=0'];checks.append('FA default off '+str(mask))
  if mask&2:assert p.fa_flags(True,mask)==['-DPS5_AI_MOSS_DIT_FA_NO_OCCUPANCY_LIMIT=1'];checks.append('FA no-limit opt-in '+str(mask))
 result=dict(passed=len(checks),checks=checks,PS5_connections=0,host_learned_inference=False)
 (a.out/'receipt.private.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=len(checks),PS5_connections=0)))
if __name__=='__main__':main()
