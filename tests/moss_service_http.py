# SPDX-License-Identifier: GPL-3.0-or-later
"""Actual Windows HTTP/parser fixtures, no learned inference or synthetic PNGs."""
from pathlib import Path
import argparse,json,sys,subprocess,os,socket,http.client,time,secrets
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from build_gateway import moss_gateway_trial_flags
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir()
    cfg=json.loads(a.config.read_text());cc=Path(cfg['LLVM'])/'clang.exe';flags=moss_gateway_trial_flags(False,True)+['-DPS5_AI_MOSS_SERVICE_MODE=1','-DPS5_AI_MOSS_IMG2IMG_TRIAL=1'];env=dict(os.environ,TMP=str(a.out),TEMP=str(a.out))
    def run(cmd):
        v=subprocess.run(list(map(str,cmd)),cwd=R,env=env,capture_output=True,text=True);assert not v.returncode,v.stderr[-6000:];return v.stdout
    common=[cc,'--target=x86_64-w64-mingw32','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-unused-function','-Wno-unused-variable',*flags,'-I',R/'native/app','-I',R/'native/sha256','-isystem',Path(cfg['SD'])/'thirdparty']
    exe=a.out/'native-failure.exe';run([*common,R/'native/app/server.c',R/'native/app/reference_image.c',R/'native/sha256/sha256.c',R/'tests/reference_engine_failure_v1.c','-lws2_32','-o',exe])
    # Exercise the actual OpenAI parser too, including different aspect ratios/seeds.
    unit=a.out/'gateway-unit.c';unit.write_text('#define main original_main\n#include "'+(R/'native/gateway/server.c').as_posix()+'"\n#undef main\nint main(int n,char**v){if(n!=2)return 2;ps5_ai_request r;char e[768],p[64];int u;int ok=openai_validate(v[1],&r,&u,e,sizeof(e),p);printf("%d",ok);return 0;}\n')
    gateway=a.out/'gateway-unit.exe';run([*common,'-DPS5_AI_GATEWAY_PROTOCOL_TEST','-I',R/'native/gateway',unit,R/'native/gateway/cpu_hooks.c',R/'native/sha256/sha256.c','-lws2_32','-o',gateway])
    cases=0
    for body,expected in[(dict(prompt='my prompt',size='512x768',seed=17),'1'),(dict(prompt='Photorealistic',size='1024x1024',seed=-1),'1'),(dict(prompt='custom',seed=9223372036854775807),'1'),(dict(prompt='custom',size='65x512'),'0'),(dict(prompt='custom',steps=20),'0'),(dict(prompt='custom',cfg_scale=5),'0'),(dict(prompt='custom',init_image_id='a'*32,denoise_strength=1),'1'),(dict(prompt='custom',reference_image_id=None),'0')]:
        assert run([gateway,json.dumps(body)])==expected;cases+=1
    root=a.out/'root';(root/'config').mkdir(parents=True);token=secrets.token_hex(32);(root/'config/api-token').write_text(token)
    with socket.socket()as x:x.bind(('127.0.0.1',0));port=x.getsockname()[1]
    log=(a.out/'process.private.log').open('wb');proc=subprocess.Popen([str(exe),'--root',str(root),'--bind','127.0.0.1','--port',str(port),'--storage-limit-mib','64'],stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    def call(method,path,body=None,auth=True):
        c=http.client.HTTPConnection('127.0.0.1',port,timeout=10);h={'Content-Type':'application/json'}
        if auth:h['Authorization']='Bearer '+token
        c.request(method,path,json.dumps(body)if body is not None else None,headers=h);r=c.getresponse();code=r.status;raw=r.read();c.close();return code,json.loads(raw)
    try:
        for _ in range(100):
            try:code,h=call('GET','/health');break
            except OSError:time.sleep(.05)
        else:raise RuntimeError('Fixture did not start')
        assert code==200 and h['service_mode']and not h['requires_256_first']and h['max_generation_attempts']is None;cases+=1
        assert call('GET','/health',auth=False)[0]==401;cases+=1
        ids=[];random_seeds=[]
        for i in range(40):
            body=dict(prompt='Unrestricted caller text '+str(i),negative_prompt='',width=1024 if i==0 else 256,height=1024 if i==0 else 256,seed=-1 if i%2 else i)
            # First accepted request is1024. Fixture deliberately fails inference.
            if i==1:body.update(width=512,height=768)
            code,j=call('POST','/api/generate',body);assert code==202 and j['seed']>=0;ids.append(j['job_id'])
            if i%2:random_seeds.append(j['seed'])
            for _ in range(100):
                code,j=call('GET','/api/jobs/'+ids[-1]);assert code==200
                if j['status']=='failed':break
                time.sleep(.03)
            else:raise RuntimeError('Failure fixture did not finish')
            assert not j['gpu_used']and j['image_url']is None;cases+=1
        assert len(set(random_seeds))==len(random_seeds);assert call('GET','/api/jobs/'+ids[0])[0]==200;cases+=2
        quota=root/'output/quota-fixture.data'
        with quota.open('wb')as f:f.truncate(40<<20)
        code,j=call('POST','/api/generate',dict(prompt='quota check'));assert code==507;cases+=1;quota.unlink()
        assert call('POST','/api/generate',dict(prompt='bad dimension',width=65))[0]==400;cases+=1
        assert call('GET','/health')[1]['running_jobs']==0;cases+=1
        result=dict(passed=cases,serial_jobs_accepted=40,completed_job_slots_reused=True,cold1024_admission_tested=True,quota_refused_before_inference=True,models_loaded=False,actual_inference=False,synthetic_PNGs=0,PS5_connections=0)
        (a.out/'receipt.private.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    finally:
        try:call('POST','/api/admin/shutdown')
        except OSError:pass
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
        log.close()
if __name__=='__main__':main()
