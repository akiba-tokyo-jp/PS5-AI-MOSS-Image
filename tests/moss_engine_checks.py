#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-only MOSS configuration, allocator ownership and GGML-metadata checks.
No model data, inference, network connection or native GPU execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llvm',type=Path,required=True)
    p.add_argument('--sd',type=Path,required=True)
    p.add_argument('--platform',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.out.resolve()
    if (out/'receipt.json').exists():raise RuntimeError('Choose a fresh output directory')
    out.mkdir(parents=True,exist_ok=True);sd=a.sd.resolve();llvm=a.llvm.resolve()
    env=dict(os.environ);env['TEMP']=env['TMP']=str(out)
    env['PATH']=str(llvm)+os.pathsep+env.get('PATH','')
    commands=[];results=[]
    def run(cmd,label):
        commands.append(list(map(str,cmd)))
        result=subprocess.run(list(map(str,cmd)),cwd=root,env=env,capture_output=True,text=True,timeout=120)
        (out/(label+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError(label+' failed: '+(result.stdout+result.stderr)[-4000:])
        return result.stdout
    inc=['-I',root/'native/app','-I',sd/'include','-I',root/'patches/stable-diffusion/overlay/src/core',
         '-I',root/'patches/stable-diffusion/overlay/ggml/src','-I',sd/'ggml/include',
         '-isystem',sd/'ggml/src','-isystem',sd/'thirdparty']
    common=['-O2','-Wall','-Wextra','-Werror','-static']
    for stem in ('moss_engine_config','moss_cpu_text_pool'):
        exe=out/(stem+'.exe')
        run([llvm/'clang++.exe','-std=c++20',*common,*inc,root/'tests'/(stem+'.cpp'),'-o',exe],stem+'.compile')
        results.append({'suite':stem,'stdout':run([exe],stem+'.test').strip()})
    pool=out/'moss_direct_pool.exe'
    run([llvm/'clang.exe','-std=c17',*common,'-Wno-deprecated-pragma',*inc,'-I',a.platform.resolve()/'include',
         root/'tests/moss_direct_pool.c','-o',pool],'moss_direct_pool.compile')
    cases=('arithmetic unconfigured invalid_config refusal_rollback physical_zero changed_failure allocation_canary physical_overlap map_null map_failed VA_range VA_overlap hint_ignored protection_canary query_canary query_error map_log_failure short_write_EINTR cpu_headroom_refusal near_capacity_success partial_fresh_reserve_refusal partial_tail_allocation_refusal success claim_growth region_fragmentation partial_full_ownership partial_pending_gate partial_live_release_refused pending quiet_failure bad_free unmap_failure release_failure old_measured_capacity_refused 12GiB_capacity_refused').split()
    for case in cases:run([pool,case],'pool-'+case)
    results.append({'suite':'moss_direct_pool','cases':len(cases),'all_passed':True,'physical_memory_mock':True})
    (out/'ggml-version.h').write_text('#pragma once\n#define GGML_VERSION "0.25.3"\n#define GGML_COMMIT "d25b121"\n')
    objs=[]
    for name in ('ggml.c','ggml-quants.c','ggml-threading.cpp'):
        obj=out/(name+'.o');objs.append(obj)
        cc=llvm/('clang++.exe' if name.endswith('.cpp') else 'clang.exe')
        run([cc,'-O0','-ffunction-sections','-fdata-sections','-I',sd/'ggml/include','-I',sd/'ggml/src','-I',out,
             '-DGGML_SCHED_MAX_COPIES=4','-c',sd/'ggml/src'/name,'-o',obj],name+'.compile')
    op=out/'moss_operator_bounds.exe'
    run([llvm/'clang++.exe','-std=c++20',*common,'-Wno-unused-function','-ffunction-sections',*inc,
         root/'tests/moss_operator_bounds.cpp',*objs,'-Wl,--gc-sections','-o',op],'moss_operator_bounds.compile')
    results.append({'suite':'moss_operator_bounds','stdout':run([op],'moss_operator_bounds.test').strip(),
                    'actual_GGML_constructors':True,'graph_executed':False})
    files=[root/'native/app'/name for name in ('engine_config.h','moss_runtime_admission.h','placement.h','pool.c','startup.c','cpu_clip_pool_buffer.hpp')]
    files += [root/'tests'/name for name in ('moss_engine_config.cpp','moss_cpu_text_pool.cpp','moss_direct_pool.c','moss_operator_bounds.cpp','moss_engine_checks.py')]
    files += [root/'patches/stable-diffusion/overlay/src/core/vulkan_scratch_bounds.hpp']
    receipt={'success':True,'PS5_contacted':False,'model_inference_executed':False,'native_GPU_calls':0,
             'runtime_admission_enabled':False,'requested_pool_bytes':12884901888,'region_bytes':1073741824,
             'actual_console_capacity_proven':False,'results':results,
             'sources':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n',encoding='utf-8')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'success':True,'suites':results,'no_PS5_or_inference':True},indent=2))


if __name__=='__main__':main()
