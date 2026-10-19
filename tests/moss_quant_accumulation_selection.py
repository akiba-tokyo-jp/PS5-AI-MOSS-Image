#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real selection functions, map metadata only. No Vulkan, model or PS5 calls."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'tests/fixtures/moss-quant-selection-before-observation.cpp.txt'
REL=Path('patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan.cpp')
def function(text,start,end):
 a=text.index(start);return text[a:text.index(end,a)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out.exists() or out.is_relative_to(ROOT):raise ValueError('Select fresh output outside repository')
 config=json.loads(a.config.read_text(encoding='utf-8-sig'));llvm=Path(config['LLVM']).resolve();sd=Path(config['SD']).resolve()
 tracked=[Path(__file__).resolve(),ROOT/REL,REFERENCE,ROOT/'native/app/phase_profile.h']
 before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
 out.mkdir(parents=True)
 current=(ROOT/REL).read_text(encoding='utf-8');previous=REFERENCE.read_text(encoding='utf-8')
 precision=function(current,'static bool ggml_vk_get_mul_mat_mat_f16acc(', 'static const std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(')
 assert precision==function(previous,'static bool ggml_vk_get_mul_mat_mat_f16acc(', 'static const std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(')
 new=function(current,'static const std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(', 'static vk_pipeline ggml_vk_guess_matmul_pipeline_map(')
 old=function(previous,'static const std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(', '// END_REFERENCE_FUNCTIONS').replace('ggml_vk_get_mul_mat_mat_pipeline_map(','old_pipeline_map(')
 prelude=r'''
#include "ggml.h"
#include <map>
#include <vector>
#include <tuple>
#include <cstdio>
#include <cstdlib>
using vk_matmul_pipeline_pair=int;
struct vk_matmul_pipeline_key{ggml_type a,b;bool id,f16acc;bool operator<(const vk_matmul_pipeline_key&o)const{return std::tie(a,b,id,f16acc)<std::tie(o.a,o.b,o.id,o.f16acc);}};
struct Device{bool fp16,coopmat2,coopmat_support,coopmat_acc_f16_support;std::map<vk_matmul_pipeline_key,std::vector<int>>pipeline_matmul;};
struct ggml_backend_vk_context{Device*device;};
static unsigned checks;static void check(bool v){++checks;if(!v)std::abort();}
'''
 tests=r'''
int main(){
 for(auto lhs:{GGML_TYPE_Q4_K,GGML_TYPE_Q6_K,GGML_TYPE_F16,GGML_TYPE_F32,GGML_TYPE_BF16})
 for(auto rhs:{GGML_TYPE_F32,GGML_TYPE_Q8_1})for(auto prec:{GGML_PREC_DEFAULT,GGML_PREC_F32})
 for(unsigned flags=0;flags<16;++flags)for(bool id:{false,true})for(unsigned maps=0;maps<16;++maps){
  Device d{};d.fp16=flags&1;d.coopmat2=flags&2;d.coopmat_support=flags&4;d.coopmat_acc_f16_support=flags&8;
  for(bool half:{false,true}){unsigned state=(maps>>(half?2:0))&3;if(state)d.pipeline_matmul[{lhs,rhs,id,half}]=state==1?std::vector<int>{}:std::vector<int>{half?222:111};}
  ggml_backend_vk_context ctx{&d};const auto*baseline=old_pipeline_map(&ctx,lhs,rhs,prec,id);
  bool actual=true;const auto*result=ggml_vk_get_mul_mat_mat_pipeline_map(&ctx,lhs,rhs,prec,id,&actual);
  check(result==baseline);check(ggml_vk_get_mul_mat_mat_pipeline_map(&ctx,lhs,rhs,prec,id)==baseline);
  bool expected=false;if(result){bool found=false;for(const auto&entry:d.pipeline_matmul)if(&entry.second==result){expected=entry.first.f16acc;found=true;}check(found);}
  check(actual==expected);
 }
 std::printf("PASS %u original-vs-instrumented matrix map/actual fallback-key checks; Vulkan=0/model=0\n",checks);
}
'''
 source=out/'selection.cpp';source.write_text(prelude+precision+old+new+tests,encoding='utf-8')
 cmd=[str(llvm/'clang++.exe'),'-std=c++17','-O1','-Wall','-Wextra','-Werror','-static','-I',str(sd/'ggml/include'),str(source),'-o',str(out/'selection.exe')]
 env=dict(os.environ);env['PATH']=str(llvm)+os.pathsep+env.get('PATH','');r=subprocess.run(cmd,env=env,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr);r.check_returncode()
 r=subprocess.run([str(out/'selection.exe')],env=env,capture_output=True,text=True);(out/'run.log').write_text(r.stdout+r.stderr);r.check_returncode()
 # Only the two actual matrix recorders contain callbacks; no vector path does.
 for start,end in [('static void ggml_vk_mul_mat_q_f16(', 'static bool ggml_vk_should_use_mmvq('),('static void ggml_vk_mul_mat_id_q_f16(', 'static void ggml_vk_mul_mat_vec_id_q_f16(')]:
  body=function(current,start,end);assert body.count('ps5_ai_vk_quant_accumulation_selected(unsigned(src0->type)')==1
  assert body.count('&selected_matrix_f16acc')==3
  call='    ggml_vk_matmul_id(' if '_id_' in start else '    ggml_vk_matmul('
  assert body.index(call)<body.index('ps5_ai_vk_quant_accumulation_selected(unsigned(src0->type)')
 assert current.count('ps5_ai_vk_quant_accumulation_selected(unsigned(src0->type)')==2
 assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
 receipt={'success':True,'reference_license':'MIT','inputs_before_after_stable':before,'output':r.stdout.strip(),'selection_behavior_identical_to_reference':True,'actual_map_key_not_requested_precision':True,'callback_after_matrix_command_recording':True,'vector_path_callback_count':0,'GPU_calls':0,'model_inference':False,'PS5_connections':0,
  'sources':[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}for f in [*tracked,source]],'command':cmd}
 (out/'receipt.private.json').write_text(json.dumps(receipt,indent=2)+'\n');print(r.stdout.strip())
if __name__=='__main__':main()
