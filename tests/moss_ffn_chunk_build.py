"""Portable host-only FFN profile/recipe guards; never inference or cross-build."""
from pathlib import Path
import argparse,ast,copy,importlib.util,json,sys
def main():
 p=argparse.ArgumentParser();p.add_argument('--support-root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];support=(a.support_root or root).resolve()
 out=a.out.resolve();assert out.drive.upper()=='D:'and not out.exists();out.mkdir()
 sys.path.insert(0,str(support/'tools'));sys.path.insert(0,str(root/'tools'))
 def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 f=module('moss_feature_policy',root/'tools/moss_feature_policy.py');sys.modules['moss_feature_policy']=f
 b=module('new_ffn_build',root/'tools/build.py');core=module('new_ffn_core',root/'tools/build_moss_core.py');checks=[]
 def yes(ok,msg):assert ok,msg;checks.append(msg)
 def no(fn,msg):
  try:fn()
  except(RuntimeError,ValueError,TypeError):checks.append('reject '+msg);return
  raise AssertionError(msg)
 profile=json.loads((root/'build-profiles/moss-performance-turbo-bounded-upload-ffn2048-fp16-f32accum.json').read_text())
 contract=b.moss_performance_contract(1,True,False,False,False,False,True,True,True,2048)
 yes(profile==contract and type(contract['moss_ffn_token_chunk_trial'])is bool and contract['moss_ffn_token_chunk_trial']is True,'Typed opt-in FFN2048 contract matches guarded builder')
 yes('expected_q4_f32acc_recordings'not in profile and profile['expected_q4_f32acc_recordings_policy']=='4*42*(2+2*ceil(actual_main_tokens/chunk_tokens))','No fixed672 counter for chunked1024')
 for profile_name,args in [('moss-performance-turbo-bounded-upload-fp16-f32accum.json',(1,True,False,False,False,False,True,True)),('moss-performance-turbo-fp16-f32accum.json',(1,True,False,False,False,False,False,True)),('moss-performance-fp16-f32accum.json',(1,True))]:
  yes(b.moss_performance_contract(*args)==json.loads((support/'build-profiles'/profile_name).read_text()),'Default/off legacy contract unchanged '+profile_name)
 for enabled,tokens in[(1,2048),(True,True),(True,0),(False,2048),(True,1024),(True,512),(True,-1),(False,1),(True,2048.0)]:no(lambda e=enabled,t=tokens:f.ffn_compile_flags(e,t),'typed coupled flag '+repr((enabled,tokens)))
 for kwargs in[dict(spectrum=True,bounded_upload=True,turbo=True,accumulation=True),dict(spectrum=False,bounded_upload=False,turbo=True,accumulation=True),dict(spectrum=False,bounded_upload=True,turbo=False,accumulation=True),dict(spectrum=False,bounded_upload=True,turbo=True,accumulation=False)]:no(lambda k=kwargs:f.validate_ffn_parent(True,2048,**k),'wrong parent '+str(kwargs))
 recipe_profile=json.loads((support/'build-profiles/moss-core.json').read_text());rows=core.validate_profile(recipe_profile)
 commands=core.materialize_compile_commands(rows,lambda x:x,True,False,True,True,True,2048)
 flags=f.ffn_compile_flags(True,2048)
 yes(len(commands)==75 and all([arg for arg in c if 'PS5_AI_MOSS_FFN_'in arg]==flags for c in commands),'All75 actual portable recipes have exact FFN flag pair')
 members=[dict(command_index=i)for i in range(75)];manifest=dict(f.ffn_manifest_fields(True,2048),commands=commands,members=members,freshly_compiled_members=75,old_archive_members_reused=0)
 yes(f.validate_core_ffn(manifest,True,2048)['actual_compile_commands_verified']==75,'Actual75 recipe/index policy proof')
 for kind in('missing','duplicate','opposite','undef'):
  for index in(0,74):
   bad=copy.deepcopy(manifest);c=bad['commands'][index]
   if kind=='missing':c.remove(flags[0])
   elif kind=='duplicate':c.append(flags[1])
   elif kind=='opposite':c[c.index(flags[0])]='-DPS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL=0'
   else:c.append('-UPS5_AI_MOSS_FFN_CHUNK_TOKENS')
   no(lambda bad=bad:f.validate_core_ffn(bad,True,2048),'core recipe '+kind+' '+str(index))
 for key,value in[('moss_ffn_token_chunk_trial',1),('moss_ffn_chunk_tokens',True),('moss_ffn_chunk_tokens',1024),('old_archive_members_reused',1),('freshly_compiled_members',74)]:
  bad=copy.deepcopy(manifest);bad[key]=value;no(lambda bad=bad:f.validate_core_ffn(bad,True,2048),'core identity '+key)
 native=ast.parse((root/'tools/build.py').read_text());sites=[]
 for n in ast.walk(native):
  if isinstance(n,ast.List)and any(isinstance(e,ast.Starred)and isinstance(e.value,ast.Name)and e.value.id=='feature_flags'for e in n.elts):
   yes(sum(isinstance(e,ast.Starred)and isinstance(e.value,ast.Name)and e.value.id=='ffn_flags'for e in n.elts)==1,'Native actual emission carries FFN flags at line'+str(n.lineno));sites.append(n)
 yes(len(sites)==3,'Exact common13/ownedplatform/GGML-provider emission sites')
 for key in profile:
  bad=copy.deepcopy(profile);bad.pop(key)
  no(lambda bad=bad:b.moss_trial_flags(True,True,bad,False,False,1,True,False,False,False,False,True,True,True,2048),'missing typed profile field '+key)
 receipt=dict(schema='moss-ffn2048-build-host-policy-v1',passed=len(checks),checks=checks,synthetic_actual_artifacts_claimed=False,compiler_invocations=0,PS5_connections=0,credential_reads=0,model_inference=False)
 (out/'receipt.private.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(dict(passed=len(checks),PS5_connections=0)))
if __name__=='__main__':main()
