# SPDX-License-Identifier: GPL-3.0-or-later
"""Run real scratch admission regressions and finite checks without inference."""
import argparse,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 c=json.loads(a.config.read_text());out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'receipt.json').exists():raise RuntimeError('Completed output exists; select a new directory')
 compiler=Path(c['LLVM'])/'clang++.exe';env=dict(os.environ);env['PATH']=str(Path(c['LLVM']))+os.pathsep+env.get('PATH','')
 checks=[]
 def run(name,text,extra=()):
  src=out/(name+'.cpp');exe=out/(name+'.exe');src.write_text(text,encoding='utf-8')
  subprocess.run([str(compiler),'--target=x86_64-w64-mingw32','-std=c++20','-O2',*map(str,extra),str(src),'-o',str(exe)],env=env,check=True,capture_output=True)
  return subprocess.check_output([str(exe)],env=env,text=True)
 text=(ROOT/'tests/native_scratch.cpp').read_text();includes=['-DGGML_MAX_NAME=160','-I',ROOT/'patches/stable-diffusion/overlay/src/core','-I',Path(c['SD'])/'ggml/src','-I',Path(c['SD'])/'ggml/include']
 for name,source in [('fp32',text),('fp16',text.replace('static Profile profile{true,true,true,true,36};','static Profile profile{true,false,true,true,36};'))]:
  value=json.loads(run(name,source,includes));assert value['checks_passed']==24;checks.append({'profile':name,**value})
 source=(ROOT/'native/app/engine_sdcpp.cpp').read_text();start=source.index('extern "C" int ps5_ai_precision_check_output');end=source.index('static void verify_native_hardware()',start)
 function=source[start:end]
 run('finite', '#include <atomic>\n#include <cmath>\n#include <cstdint>\n#include <cstring>\n#include <cassert>\n#include <limits>\nstatic std::atomic<uint64_t> precision_outputs{0},precision_values{0},precision_nonfinite{0},precision_unet_outputs{0};\n'+function+'''
int main(){float a[]={-1,0,1,65504};assert(ps5_ai_precision_check_output(a,4,"unet"));
float b[]={0,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()};
assert(!ps5_ai_precision_check_output(b,4,"taesd"));assert(precision_nonfinite==3);
assert(!ps5_ai_precision_check_output(nullptr,4,"unet"));assert(precision_nonfinite==7);
assert(ps5_ai_precision_check_output(nullptr,0,"unet"));assert(precision_outputs==4&&precision_values==12&&precision_unet_outputs==3);}
''')
 receipt={'state':'PASS','scratch_checks':48,'profiles':checks,'finite_checks':'normal / NaN / +Inf / -Inf / null / counters','GPU_calls':0,'model_inference':False}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
