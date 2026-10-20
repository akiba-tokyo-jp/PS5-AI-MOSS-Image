# SPDX-License-Identifier: MIT
"""Actual rank144 classifier and diagnostic policy; no GPU/inference/network."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];out=a.out.resolve()
 if out.exists() or out.drive.upper()!='D:' or out.is_relative_to(root):raise ValueError('Fresh D output required')
 out.mkdir(parents=True);cfg=json.loads(a.config.read_text());sd=Path(cfg['SD']);llvm=Path(cfg['LLVM'])
 sys.path.insert(0,str(root/'tools'));import moss_dit_policy as policy
 parent=dict(mask=15,legacy_gpu_profile=False,accumulation=True,turbo=True,upload=True)
 assert policy.validate_profile(True,**parent)==['-DPS5_AI_MOSS_DIT_GPU_PROFILE=1']
 refusals=0
 for key,value in dict(mask=7,legacy_gpu_profile=True,accumulation=False,turbo=False,upload=False,spectrum=True,ffn=True,lora='f16',q4=True,sampler=True).items():
  bad=dict(parent);bad[key]=value
  try:policy.validate_profile(True,**bad)
  except ValueError:refusals+=1
  else:raise AssertionError(key)
 backend=root/'patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan.cpp'
 tracked=[Path(__file__),root/'tests/moss_dit_gpu_profile.cpp',root/'native/app/moss_gpu_profile.h',root/'native/app/moss_dit_gpu_profile_policy.h',root/'tools/moss_dit_policy.py',root/'tools/build.py',backend,backend.with_name('ggml-vulkan-common.h'),root/'native/app/radv_ps5_platform_native.c',root/'patches/stable-diffusion/overlay/src/model/diffusion/moss_dit_ops.hpp',sd/'ggml/src/ggml.c']
 def dig(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 before={str(x):dig(x)for x in tracked}
 def function(anchor,path=backend):
  s=path.read_text();start=s.index(anchor);end=s.index('{',start)+1;depth=1
  while depth:
   if s[end]=='{':depth+=1
   if s[end]=='}':depth-=1
   end+=1
  return s[start:end]
 classifier=out/'actual-classifier.cpp'
 classifier.write_text('#include "moss_dit_ops.hpp"\n#include "moss_gpu_profile.h"\n'+function('static unsigned ps5_ai_moss_gpu_actual_node_class(')+'\n'+function('static ps5_ai_moss_gpu_coverage moss_gpu_profile_actual_graph_coverage(')+'\nextern "C" unsigned ps5_ai_test_actual_node_class(const ggml_tensor*n){return ps5_ai_moss_gpu_actual_node_class(n);}\nextern "C" ps5_ai_moss_gpu_coverage ps5_ai_test_actual_graph_coverage(const ggml_cgraph*g){return moss_gpu_profile_actual_graph_coverage(g);}\n')
 (out/'ggml-version.h').write_text('#define GGML_VERSION "0.25.3"\n#define GGML_COMMIT "d25b121"\n')
 flags=['-DGGML_MAX_NAME=160','-DPS5_AI_MOSS_GPU_PROFILE=1','-DPS5_AI_MOSS_DIT_GPU_PROFILE=1','-DPS5_AI_MOSS_DIT_OPT_TRIAL=1','-DPS5_AI_MOSS_DIT_OPT_MASK=15','-DPS5_AI_MOSS_TURBO_TRIAL=1','-DPS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL=1','-DPS5_AI_MOSS_F32_ACCUMULATION=1','-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1','-DPS5_AI_MOSS_TRIAL_10G=1','-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1','-DPS5_AI_MOSS_PERF_VARIANT=1']
 includes=[]
 for path in [root/'native/app',root/'patches/stable-diffusion/overlay/src/model/diffusion',sd/'ggml/include',sd/'ggml/src',out]:includes+=['-I',str(path)]
 env=dict(os.environ,TMP=str(out),TEMP=str(out));env['PATH']=str(llvm)+os.pathsep+env.get('PATH','');commands=[]
 def run(label,cmd,expected=None):
  cmd=list(map(str,cmd));commands.append(cmd);r=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=120);text=r.stdout+r.stderr;(out/(label+'.log')).write_text(text,encoding='utf-8')
  if expected is None:
   if r.returncode:raise RuntimeError(label+': '+text[-6000:])
  elif r.returncode==0 or expected not in text:raise RuntimeError('Missing refusal '+label+': '+text[-3000:])
  return text
 obj=out/'exact-metadata.cpp'
 obj.write_text('#include \"ggml.h\"\n#include <cstdio>\n#include <cstdlib>\nextern \"C\" void ggml_abort(const char*,int,const char*,...){std::abort();}\n'+function('int64_t ggml_nelements(',sd/'ggml/src/ggml.c')+'\n'+function('struct ggml_tensor * ggml_set_name(',sd/'ggml/src/ggml.c')+'\n')
 exe=out/'rank144.exe'
 run('compile-rank144',[llvm/'clang++.exe','-std=c++20','-O1','-Wall','-Wextra','-Werror','-ffunction-sections','-fdata-sections','-Wl,--gc-sections','-static',*flags,*includes,root/'tests/moss_dit_gpu_profile.cpp',classifier,obj,'-o',exe])
 output=run('run-rank144',[exe]);print(output.strip())
 h=out/'header.c';h.write_text('#include "moss_gpu_profile.h"\n')
 for flag in [x for x in flags[1:] if not x.startswith('-DPS5_AI_MOSS_DIT_GPU_PROFILE=')]:
  wrong=[x for x in flags if x!=flag]
  run('refuse-'+flag.split('=')[0][2:],[llvm/'clang.exe',*wrong,*includes,'-fsyntax-only',h], 'MOSS_DIT_GPU_profile_requires_exact_mask15_Turbo_upload_F32_10GiB_diagnostic');refusals+=1
 for extra in ['-DPS5_AI_MOSS_RUNTIME_ADMITTED=1','-DPS5_AI_MOSS_SPECTRUM_TRIAL=1','-DPS5_AI_MOSS_Q4_BK64_RAW64=1','-DPS5_AI_MOSS_SAMPLER_TRIAL=1','-DPS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL=1','-DPS5_AI_MOSS_LORA_F16_STORAGE_TRIAL=1']:
  run('refuse-extra-'+extra.split('=')[0][2:],[llvm/'clang.exe',*flags,extra,*includes,'-fsyntax-only',h], 'MOSS_DIT_GPU_profile_requires_exact_mask15_Turbo_upload_F32_10GiB_diagnostic');refusals+=1
 assert before=={str(x):dig(x)for x in tracked},'Source changed during verification'
 (out/'receipt.private.json').write_text(json.dumps(dict(success=True,source_sha256=before,commands=commands,refusals=refusals,output=output,GPU_calls=0,inference=False,PS5_connections=0),indent=2)+'\n')
 print(json.dumps(dict(success=True,policy_refusals=refusals,GPU_calls=0,inference=False)))
if __name__=='__main__':main()
