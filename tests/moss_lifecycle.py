"""Actual native HTTP shutdown routing against CPU-only initialization fixtures.

All HTTP uses 127.0.0.1. There is no image POST, model, GPU, PS5 or OS title close.
An early-failure fixture aborts if any shutdown/cleanup hook is reached, proving
that the 409 handoff is produced before worker join/epoch-zero GPU retirement.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time

R=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve()
    if out==R or R in out.parents or (os.name=='nt' and out.drive.upper()!='D:'):
        raise RuntimeError('Use a fresh D: output directory outside the repository')
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());cc=Path(cfg['LLVM'])/'clang.exe'
    env=dict(os.environ);env['PATH']=str(cc.parent)+os.pathsep+env.get('PATH','')
    env['TEMP']=env['TMP']=str(out);checks=[];commands=[]
    def check(value,label):
        if not value:raise AssertionError(label)
        checks.append(label)
    def run(cmd):
        cmd=list(map(str,cmd));commands.append(cmd)
        c=subprocess.run(cmd,env=env,cwd=R,capture_output=True,text=True,errors='replace')
        with (out/'host.log').open('a',encoding='utf-8') as f:f.write(c.stdout+c.stderr)
        c.check_returncode();return c.stdout
    common=[cc,'--target=x86_64-w64-mingw32','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-unused-function','-I',R/'native/app']
    run([*common,R/'tests/moss_lifecycle.c','-o',out/'state.exe'])
    check('PASS 47 ' in run([out/'state.exe']),'actual packed state:47 boundary/ABI/first-failure checks')
    fixture='''#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine.h"
#include "lifecycle.h"
#include "lifecycle_impl.inc"
#include "moss_runtime_admission.h"
#if PS5_AI_MOSS_RUNTIME_ADMITTED != 0
#error This host fixture must exercise the repository's unchanged guard0
#endif
int ps5_ai_engine_init(const char *p,char *e,size_t n){(void)p;(void)e;(void)n;
 native_lifecycle_mark(PS5_AI_LIFE_CONFIG);
#if HEALTHY_FIXTURE
 native_lifecycle_mark(PS5_AI_LIFE_READY);return 0;
#else
 if(n)snprintf(e,n,"%s",PS5_AI_MOSS_UNADMITTED_REASON);
 native_lifecycle_failed(PS5_AI_LIFE_INIT_FAILED);return -1;
#endif
}
void ps5_ai_engine_get_status(ps5_ai_engine_status *s){memset(s,0,sizeof(*s));s->available=s->gpu_available=HEALTHY_FIXTURE;strcpy(s->model,"/app0/models/MOSS-Image-SFT-Q4_K.gguf");if(!HEALTHY_FIXTURE)snprintf(s->reason,sizeof(s->reason),"%s",PS5_AI_MOSS_UNADMITTED_REASON);}
int ps5_ai_engine_generate(const ps5_ai_request *r,const char *p,ps5_ai_stats *s,ps5_ai_progress_cb cb,void *u,char *e,size_t n){(void)r;(void)p;(void)s;(void)cb;(void)u;(void)e;(void)n;abort();}
static void shutdown_hook(void){if(!HEALTHY_FIXTURE)abort();}
void ps5_ai_engine_shutdown(void){shutdown_hook();}
int ps5_ai_engine_shutdown_checked(char *e,size_t n){(void)e;(void)n;shutdown_hook();return 0;}
void ps5_ai_v3_shutdown_request_accepted(int r,int q){(void)r;(void)q;shutdown_hook();}
void ps5_ai_v3_worker_join_receipt(int j,int r,int q,int c){(void)j;(void)r;(void)q;(void)c;shutdown_hook();}
void ps5_ai_native_ui_phase(unsigned p){(void)p;}
void ps5_ai_native_ui_job_begin(const char *p){(void)p;}
void ps5_ai_native_ui_log(const char *p){(void)p;}
'''
    f=out/'fixture.c';f.write_text(fixture,encoding='utf-8')
    token=secrets.token_hex(32)
    def request(port,method,path,auth=True):
        c=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
        headers={'Content-Length':'0'}
        if auth:headers['Authorization']='Bearer '+token
        c.request(method,path,headers=headers);r=c.getresponse();body=r.read();c.close()
        return r.status,json.loads(body)
    for healthy in (0,1):
        name='ready' if healthy else 'early-failure'
        exe=out/(name+'.exe')
        run([*common,'-DPS5_AI_LIFECYCLE_HOST_TEST',f'-DHEALTHY_FIXTURE={healthy}',
             '-I',R/'native/sha256','-isystem',Path(cfg['SD'])/'thirdparty',R/'native/app/server.c',
             R/'native/app/reference_image.c',R/'native/sha256/sha256.c',f,'-lws2_32','-o',exe])
        root=out/name;(root/'config').mkdir(parents=True)
        (root/'config/api-token').write_text(token,encoding='ascii')
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        with (out/(name+'.log')).open('wb') as log:
            proc=subprocess.Popen([str(exe),'--root',str(root),'--bind','127.0.0.1','--port',str(port)],
                stdout=log,stderr=log,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline=time.monotonic()+10
                while time.monotonic()<deadline:
                    if proc.poll() is not None:raise RuntimeError(name+' exited before health')
                    try:
                        code,health=request(port,'GET','/health')
                        if code==200:break
                    except OSError:pass
                    time.sleep(.05)
                else:raise TimeoutError(name+' startup')
                expected='checked_gpu_cleanup' if healthy else 'trusted_os_close_required'
                check(health['model_id']=='moss-image-sft',name+' MOSS model identity preserved')
                check(health['lifecycle']['title_id']=='PPSA99190',name+' separate MOSS title identity preserved')
                check(health['lifecycle']['in_app_shutdown']==expected,name+' truthful shutdown route')
                check(health['lifecycle']['constructor_effects']=='unknown',name+' constructor uncertainty preserved')
                check(health['lifecycle']['os_close_authorized'] is False,name+' never authorizes OS close')
                check(request(port,'POST','/api/admin/shutdown',auth=False)[0]==401,name+' handoff authenticated')
                code,body=request(port,'POST','/api/admin/shutdown')
                if healthy:
                    check(code==202 and body['exit_verified'] is False,'ready path retains202 request-only semantics')
                    proc.wait(timeout=8);check(proc.returncode==0,'ready mock listener/worker hooks complete unchanged')
                else:
                    check(health['status']=='degraded' and health['gpu_available'] is False,'guard0 has no available GPU engine')
                    check(health['reason'].startswith('MOSS runtime not admitted:'),'repository guard0 reason preserved')
                    check(code==409 and body['recovery']['failed_stage']=='configuration','early failure409 before cleanup/join hooks')
                    check(body['recovery']['fault_retention_entered'] is False,'early failure does not invent a retained GPU fault')
                    check(request(port,'GET','/health')[0]==200 and proc.poll() is None,'early failure leaves observable API instead of epoch0 retention')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait(timeout=5)
    sources=['native/app/lifecycle.h','native/app/lifecycle_impl.inc','native/app/server.c','native/app/startup.c','tests/moss_lifecycle.c','tests/moss_lifecycle.py']
    result={'success':True,'checks':checks,'host_unit_checks':47,'HTTP_checks':len(checks)-1,
        'PS5_connections':0,'GPU_calls':0,'image_POSTs':0,'OS_title_closes':0,
        'normal_PS5_exit_verified':False,'runtime_admitted_macro':0,'healthy_READY_is_host_mock_only':True,'commands':commands,
        'sources':{s:hashlib.sha256((R/s).read_bytes()).hexdigest() for s in sources}}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'success':True,'host_unit_checks':47,'HTTP_checks':len(checks)-1,'PS5_connections':0,'GPU_calls':0}))

if __name__=='__main__':main()
