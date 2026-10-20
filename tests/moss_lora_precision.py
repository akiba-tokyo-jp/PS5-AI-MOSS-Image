# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused host contracts/receipt tests; no inference, model reads or GPU calls."""
import argparse,copy,hashlib,json,struct,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import moss_lora_precision_policy as policy
import build,build_moss_core

def record(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=p.as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def module_symbols(raw):
 if raw[:6]!=b'\x7fELF\x02\x01':raise RuntimeError('ELF64 little-endian shader object required')
 off=struct.unpack_from('<Q',raw,40)[0];width,count=struct.unpack_from('<HH',raw,58)
 if width!=64 or not 0<count<65535 or off+64*count>len(raw):raise RuntimeError('Bad ELF sections')
 sections=[struct.unpack_from('<IIQQQQIIQQ',raw,off+i*64)for i in range(count)];result={}
 for s in sections:
  if s[1]!=2:continue
  if s[9]!=24 or s[6]>=count:raise RuntimeError('Bad ELF symbols')
  t=sections[s[6]];names=raw[t[4]:t[4]+t[5]]
  for pos in range(s[4],s[4]+s[5],24):
   n,info,other,index,value,size=struct.unpack_from('<IBBHQQ',raw,pos)
   if n>=len(names):raise RuntimeError('Bad ELF symbol name')
   text=names[n:names.index(0,n)].decode('ascii')
   if text not in ('matmul_f32_f32_fp32_data','matmul_f16_f32_fp32_data'):continue
   if index>=count or text in result:raise RuntimeError('Bad selected shader definition')
   start=sections[index][4]+value
   if start+size>len(raw):raise RuntimeError('Bad shader span')
   result[text]=raw[start:start+size]
 return result
def patch_controls(raw,rte,denorm):
 if not(rte or denorm):return raw
 w=list(struct.unpack('<'+'I'*(len(raw)//4),raw));pos=5;cap=ext=exe=pos;entry=0
 while pos<len(w):
  n,op=w[pos]>>16,w[pos]&65535
  if not n or pos+n>len(w):raise RuntimeError('Bad SPIRV instruction')
  if op==17:cap=ext=pos+n
  elif op==10:ext=pos+n
  elif op==15:entry=w[pos+2];exe=pos+n
  elif op in(16,331):exe=pos+n
  elif entry:break
  pos+=n
 if not entry:raise RuntimeError('SPIRV entry point missing')
 if rte:w[exe:exe]=[(4<<16)|16,entry,4462,16]
 if denorm:w[exe:exe]=[(4<<16)|16,entry,4459,16]
 text=b'SPV_KHR_float_controls\0';text+=b'\0'*((-len(text))%4)
 w[ext:ext]=[((1+len(text)//4)<<16)|10,*struct.unpack('<'+'I'*(len(text)//4),text)]
 if rte:w[cap:cap]=[(2<<16)|17,4467]
 if denorm:w[cap:cap]=[(2<<16)|17,4464]
 return struct.pack('<'+'I'*len(w),*w)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 out=a.out.resolve()
 if out.exists()or out==ROOT or ROOT in out.parents:raise RuntimeError('Use fresh output outside repository')
 out.mkdir(parents=True);c=json.loads(a.config.read_text());llvm=Path(c['LLVM']);checks=0
 def test(value):
  nonlocal checks;checks+=1
  if not value:raise AssertionError('precision host check '+str(checks))
 def reject(call):
  try:call()
  except(RuntimeError,ValueError):test(True)
  else:test(False)
 valid=dict(turbo=True,upload=True,accumulation=True,variant=1,precision='fp16',experimental=True,trial_10g=True)
 for mode in(None,'f32','f16'):
  policy.validate(mode,**valid);test(True)
  if mode is not None:
   for key,value in dict(turbo=False,upload=False,spectrum=True,ffn=True,accumulation=False,gpu_profile=True,q4=True,sampler=True,variant=0,precision='fp32',experimental=False,trial_10g=False,operating=True).items():
    options=dict(valid);options[key]=value;reject(lambda:policy.validate(mode,**options))
   profile=build.moss_performance_contract(1,True,moss_turbo_trial=True,moss_bounded_weight_upload_trial=True,moss_lora_storage=mode)
   test(profile==json.loads((ROOT/('build-profiles/moss-performance-combined-lora-'+mode+'-fp32shared.json')).read_text()))
   test(profile['expected_private_LoRA_matrix_recordings']==2016 and profile['expected_q4_f32acc_recordings']==672 and profile['steps']==4)
   test(profile['max_generation_attempts']==4 and profile['old_core_reuse_allowed']is False)
 for mode in(True,False,'F16','',0):reject(lambda:policy.flags(mode))
 rows=[dict(argv=['clang','-c','synthetic.cpp','-o','synthetic.o'])for i in range(75)]
 for mode in('f32','f16'):
  commands=build_moss_core.materialize_compile_commands(rows,lambda x:x,True,False,True,True,False,0,mode)
  # This is a pure host synthetic manifest, never a real core-build claim.
  manifest=dict(policy.fields(mode,core=True),commands=commands,members=[dict(command_index=i,source='synthetic.cpp')for i in range(75)],
      freshly_compiled_members=75,old_archive_members_reused=0,lora_precision_policy_inputs=[record(ROOT/'tools/moss_lora_precision_policy.py')])
  test(policy.validate_core(manifest,mode)['actual_compile_commands_verified']==75)
  for mutate in(range(7)):
   m=copy.deepcopy(manifest)
   if mutate==0:m['commands'][0].append('-D'+policy.SHARED+'=0')
   if mutate==1:m['commands'][1].remove('-D'+policy.STORAGE+'='+str(int(mode=='f16')))
   if mutate==2:m['lora_precision_policy_inputs']=[]
   if mutate==3:m['lora_precision_policy_inputs'][0]['sha256']='0'*64
   if mutate==4:m['old_archive_members_reused']=1
   if mutate==5:m['members'][1]['command_index']=0
   if mutate==6:m['moss_lora_storage']='f16'if mode=='f32'else'f32'
   reject(lambda:policy.validate_core(m,mode))
 # Use the pinned existing shader archive from the same reviewed build profile.
 native=json.loads((ROOT/'build-profiles/native.json').read_text());dep=next(r for r in native['dependencies']if r['name']=='libggml-vulkan.flex.a')
 archive=Path(c['DEPS'])/dep['name'];test(record(archive)['sha256']==dep['sha256'])
 raw=subprocess.check_output([str(llvm/'llvm-ar.exe'),'p',str(archive),'mul_mm.comp.cpp.o'])
 symbols=module_symbols(raw);modules=out/'modules';modules.mkdir()
 for name,sha in [('matmul_f32_f32_fp32','9f875beba3c839333f3bcfdddf6c5d444e0c61f73a103f286146002bbb68d2f1'),('matmul_f16_f32_fp32','1fdb392b957fac72c5512aa1334cbe872dd7460e922811ccf9df1086b8ed3c2c')]:
  data=symbols[name+'_data'];test(hashlib.sha256(data).hexdigest()==sha)
  for mode in range(4):(modules/(name+'-'+str(mode)+'.spv')).write_bytes(patch_controls(data,mode&1,(mode>>1)&1))
 commands=[];outputs=[]
 shaobj=out/'sha256.o';cmd=[str(llvm/'clang.exe'),'--target=x86_64-w64-mingw32','-O2','-c',str(ROOT/'native/sha256/sha256.c'),'-o',str(shaobj)]
 commands.append(cmd);subprocess.run(cmd,check=True,capture_output=True)
 guards=['-DPS5_AI_MOSS_TURBO_TRIAL=1','-DPS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL=1','-DPS5_AI_MOSS_F32_ACCUMULATION=1','-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_TRIAL_10G=1','-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1','-DPS5_AI_MOSS_PERF_VARIANT=1']
 for mode in('f32','f16'):
  exe=out/('receipt-'+mode+'.exe');cmd=[str(llvm/'clang++.exe'),'--target=x86_64-w64-mingw32','-std=c++17','-O2','-static','-pthread',*guards,*policy.flags(mode),'-I'+str(ROOT/'native/app'),'-I'+str(ROOT/'native/sha256'),'-I'+str(Path(c['SD'])/'thirdparty'),str(ROOT/'tests/moss_lora_precision.cpp'),str(shaobj),'-o',str(exe)]
  commands.append(cmd);run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/('compile-'+mode+'.log')).write_text(run.stdout)
  if run.returncode:raise RuntimeError('Host receipt compilation failed')
  run=subprocess.run([str(exe),str(modules)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/('receipt-'+mode+'.log')).write_text(run.stdout)
  if run.returncode:raise RuntimeError('Host receipt test failed: '+run.stdout)
  outputs.append(json.loads(run.stdout));test(outputs[-1]['success']is True)
 body=dict(schema='moss-private-LoRA-precision-host-contracts-v1',success=True,policy_checks=checks,receipt_tests=outputs,
     archive=record(archive),commands=commands,inputs=[record(ROOT/'tools/moss_lora_precision_policy.py'),record(ROOT/'tests/moss_lora_precision.cpp'),record(ROOT/'tests/moss_lora_precision.py'),record(ROOT/'native/app/moss_lora_f16_storage_trial.h'),record(ROOT/'native/app/moss_lora_f16_storage_contract.hpp'),*[record(p)for p in(ROOT/'native/app').glob('moss_lora_precision*')]],
     synthetic_metadata_only=True,model_coefficients_read=False,GPU_execution=False,PS5_calls=0)
 (out/'receipt.private.json').write_text(json.dumps(body,indent=2)+'\n');print(json.dumps(dict(success=True,policy_checks=checks,receipt_tests=outputs)))
if __name__=='__main__':main()
