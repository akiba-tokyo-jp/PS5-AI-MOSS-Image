# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-only combined-feature contracts; no compilation, model or PS5 calls.

Normal checkout: python -B tests/moss_combined_build.py
A partial offline proposal may add --support-root for unchanged helper imports.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,itertools,json,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--support-root',type=Path);p.add_argument('--out',type=Path);args,rest=p.parse_known_args()
sys.path.insert(0,str(ROOT/'tools'))
if args.support_root:sys.path.insert(1,str(args.support_root.resolve()/'tools'))
import moss_feature_policy as f
spec=importlib.util.spec_from_file_location('combined_native_builder',ROOT/'tools/build.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
class CombinedBuild(unittest.TestCase):
 def test_feature_matrix_and_types(self):
  for s,u,t in itertools.product([False,True],repeat=3):
   if s and(u or t):
    with self.assertRaises(RuntimeError):f.feature_compile_flags(s,u,t)
   else:self.assertEqual(f.feature_compile_flags(s,u,t),['-D'+m+'='+str(int(v))for m,v in zip((f.SPECTRUM_MACRO,f.UPLOAD_MACRO,f.TURBO_MACRO),(s,u,t))])
  for v in [0,1,None,'1']:
   with self.assertRaises(RuntimeError):f.feature_compile_flags(False,v,True)
 def test_exact_new_and_preserved_legacy_contracts(self):
  got=b.moss_performance_contract(1,True,False,False,False,False,True,True)
  expected=json.loads((ROOT/'build-profiles/moss-performance-turbo-bounded-upload-fp16-f32accum.json').read_text(encoding='utf8'))
  self.assertEqual(got,expected);self.assertIs(got['moss_turbo_bounded_upload_trial'],True)
  self.assertEqual((got['steps'],got['cfg_scale'],got['expected_dit_forwards'],got['expected_q4_f32acc_recordings'],got['max_generation_attempts']),(4,1.,4,672,4))
  self.assertEqual(got['bounded_weight_upload_chunk_bytes'],8388608);self.assertEqual(got['bounded_weight_upload_limit_scope'],'per_worker')
  self.assertIs(got['new_core_compilation_required'],True);self.assertIs(got['old_core_reuse_allowed'],False)
  base=args.support_root.resolve()if args.support_root else ROOT
  for s,u,t,name in [(False,False,True,'turbo'),(True,False,False,'spectrum'),(False,True,False,'bounded-weight-upload')]:
   old=json.loads((base/('build-profiles/moss-performance-'+name+'-fp16-f32accum.json')).read_text(encoding='utf8'))
   self.assertEqual(b.moss_performance_contract(1,True,False,False,False,s,u,t),old)
 def test_unsafe_mode_refusals(self):
  good=dict(accumulation=True,gpu_profile=False,q4=False,sampler=False,variant=1,precision='fp16',experimental=True,trial_10g=True)
  f.validate_feature_modes(False,True,True,**good)
  for key,value in [('accumulation',False),('gpu_profile',True),('q4',True),('sampler',True),('variant',0),('variant',True),('precision','fp32'),('experimental',False),('trial_10g',False),('accumulation',1)]:
   bad=dict(good);bad[key]=value
   with self.assertRaises(RuntimeError):f.validate_feature_modes(False,True,True,**bad)
 def test_core_identity_not_an_actual_build(self):
  # HOST FIXTURE: only command-policy validation, not object/dependency proof.
  own=Path(f.__file__).resolve();meta=f.feature_manifest_fields(False,True,True)
  fixture=dict(meta,policy_inputs=[dict(path=str(own),bytes=own.stat().st_size,sha256=hashlib.sha256(own.read_bytes()).hexdigest())],commands=[],members=[])
  for i in range(75):
   source=str((ROOT/('HOST-FIXTURE/source'+str(i)+'.cpp')).resolve())
   fixture['commands'].append(['clang','-DPS5_AI_MOSS_F32_ACCUMULATION=1',*meta['moss_inference_feature_compile_flags'],'-c',source,'-o','HOST-FIXTURE/object'+str(i)+'.o'])
   fixture['members'].append(dict(command_index=i,source=source))
  self.assertEqual(f.validate_core_features(fixture,False,True,True)['actual_compile_commands_verified'],75)
  for mutate in [lambda x:x.update(moss_turbo_bounded_upload_trial=1),lambda x:x.update(moss_turbo_bounded_upload_policy='disabled'),lambda x:x['commands'][12].append('-UPS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL'),lambda x:x['commands'][24].remove('-DPS5_AI_MOSS_TURBO_TRIAL=1'),lambda x:x['policy_inputs'][0].update(sha256='0'*64),lambda x:x['members'][1].update(command_index=0)]:
   bad=copy.deepcopy(fixture);mutate(bad)
   with self.assertRaises(RuntimeError):f.validate_core_features(bad,False,True,True)
  with self.assertRaises(RuntimeError):f.validate_core_features(fixture,False,False,True)
 def test_contract_extra_missing_and_typed_fields(self):
  good=b.moss_performance_contract(1,True,False,False,False,False,True,True)
  def flags(v):return b.moss_trial_flags(True,True,v,False,False,1,True,False,False,False,False,True,True)
  self.assertIn('-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1',flags(good))
  for mode in ['missing','extra','integer_bool']:
   bad=dict(good)
   if mode=='missing':del bad['moss_turbo_bounded_upload_policy']
   elif mode=='extra':bad['arbitrary']=True
   else:bad['moss_turbo_bounded_upload_trial']=1
   with self.assertRaises(RuntimeError):flags(bad)
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CombinedBuild))
 if args.out:
  out=args.out.resolve();assert not out.exists()and out.drive.upper()=='D:';out.mkdir(parents=True)
  (out/'receipt.private.json').write_text(json.dumps(dict(schema='moss-combined-build-HOST-FIXTURE-tests-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,PS5_connections=0,model_reads=0,compilation=False,actual_core_build_claimed=False),indent=2)+'\n',encoding='utf8')
 raise SystemExit(0 if result.wasSuccessful()else 1)
