#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure host GGML metadata/config tests. No backend, models, network or PS5."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'receipt.private.json').exists():raise RuntimeError('Select a fresh output directory')
 llvm=Path('C:/Users/User/ps5-ai/toolchain/llvm-mingw-20261006-ucrt-x86_64/bin')
 sd=Path('D:/PS5-AI/moss/vendor/stable-diffusion.cpp')
 objects=Path('D:/PS5-AI/moss/diagnostics/bf16-scratch-v2/actual-graph-host-v4')
 files=[ROOT/'tests/moss_submission_fusion.cpp',ROOT/'tests/moss_submission_fusion.py']
 files += [ROOT/'patches/stable-diffusion/overlay/ggml/src/ggml-vulkan'/n for n in ('ggml-vulkan.cpp','ggml-vulkan-types.h','ggml-vulkan-ps5-profile.h')]
 files += [ROOT/'patches/stable-diffusion/overlay/src/core'/n for n in ('vulkan_scratch_bounds.hpp','runner_admission.hpp')]
 cmd=[llvm/'clang++.exe','-std=c++20','-O1','-Wall','-Wextra','-Werror','-Wno-unused-function','-static','-ffunction-sections','-fdata-sections','-DGGML_MAX_NAME=160',
      '-I',ROOT/'patches/stable-diffusion/overlay/ggml/src/ggml-vulkan','-I',ROOT/'patches/stable-diffusion/overlay/src/core',
      '-I',sd/'ggml/include','-I',sd/'ggml/src',ROOT/'tests/moss_submission_fusion.cpp',
      objects/'ggml.c.o',objects/'ggml-quants.c.o',objects/'ggml-threading.cpp.o','-Wl,--gc-sections','-o',out/'checks.exe']
 env=dict(os.environ);env['PATH']=str(llvm)+os.pathsep+env.get('PATH','');env['TMP']=env['TEMP']=str(out)
 def run(command,name):
  r=subprocess.run(list(map(str,command)),env=env,capture_output=True,text=True,timeout=120)
  (out/(name+'.log')).write_text(r.stdout+r.stderr,encoding='utf-8')
  if r.returncode:raise RuntimeError(name+' failed: '+(r.stdout+r.stderr)[-3000:])
  return r.stdout
 run(cmd,'compile');result=run([out/'checks.exe'],'run')
 receipt={'success':True,'output':result.strip(),'GPU_calls':0,'model_inference':False,'PS5_contacted':False,
          'sources':[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files],
          'metadata_objects':[{'path':str(objects/n),'sha256':hashlib.sha256((objects/n).read_bytes()).hexdigest()} for n in ('ggml.c.o','ggml-quants.c.o','ggml-threading.cpp.o')]}
 (out/'command.private.json').write_text(json.dumps(list(map(str,cmd)),indent=2)+'\n')
 (out/'receipt.private.json').write_text(json.dumps(receipt,indent=2)+'\n');print(result.strip())
if __name__=='__main__':main()
