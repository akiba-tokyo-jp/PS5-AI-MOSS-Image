"""Anima host contract checks. Loopback fixtures always fail; no inference/GPU.

Usage: python tests/anima_api_protocol.py --config TOOLCHAIN_JSON --out D:/...
The toolchain JSON contains build paths, never API credentials. Output is kept
outside the source tree. Every credential in these tests is freshly synthetic.
"""
import argparse
import concurrent.futures
import hashlib
import hmac
import http.client
import http.server
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time

R=Path(__file__).resolve().parents[1]
CHECKS=[]

def check(value,name):
    if not value:raise AssertionError(name)
    CHECKS.append(name)

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();out=a.out.resolve()
    if out==R or R in out.parents:raise RuntimeError('Host fixtures must be outside the source repository')
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(a.config.read_text());cc=Path(cfg['LLVM'])/'clang.exe'
    env=dict(os.environ);env['PATH']=str(cc.parent)+os.pathsep+env.get('PATH','')
    env['TEMP']=env['TMP']=str(out)
    token=secrets.token_hex(32);processes=[];logs=[];commands=[]
    g=R/'native/gateway';n=R/'native/app';sha=R/'native/sha256'
    def run(cmd):
        cmd=list(map(str,cmd));commands.append(cmd)
        r=subprocess.run(cmd,cwd=R,env=env,capture_output=True,text=True,errors='replace')
        if r.returncode:raise RuntimeError(r.stderr[-8000:])
        return r.stdout
    common=[cc,'--target=x86_64-w64-mingw32','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-unused-function']
    fixture=out/'anima-failure.c'
    fixture.write_text((R/'tests/reference_engine_failure_v1.c').read_text().replace('/app0/models/waiIllustriousSDXL_v170.safetensors','/app0/models/anima-turbo-v1.1-f16.safetensors'),encoding='utf-8',newline='\n')
    native_deps=[n/'reference_image.c',sha/'sha256.c',fixture]
    native_flags=['-I',n,'-I',sha,'-isystem',Path(cfg['SD'])/'thirdparty']
    run([*common,*native_flags,n/'server.c',*native_deps,'-lws2_32','-o',out/'native-failure.exe'])
    native_unit=out/'native-unit.c'
    native_unit.write_text('''#define main native_server_main
#include "server.c"
#undef main
int main(void){
 ps5_ai_request r;char e[768];
 if(listen_port!=8190 || strcmp(root_path,"/app0/ai-runtime"))return 1;
 if(!validate_request("{\\"prompt\\":\\"fixture\\"}",&r,e,sizeof(e)))return 2;
 if(r.steps!=8 || r.cfg_scale!=1 || r.manga_strength!=0 || r.manga_monochrome || strcmp(r.sampler,"euler") || r.width!=1024 || r.height!=1024)return 3;
 if(!validate_request("{\\"prompt\\":\\"fixture\\",\\"model\\":\\"anima-turbo-v1.1\\",\\"manga_strength\\":0,\\"manga_monochrome\\":false}",&r,e,sizeof(e)))return 4;
 puts("native defaults and compatibility no-op verified");return 0;}
''',encoding='utf-8',newline='\n')
    run([*common,*native_flags,native_unit,*native_deps,'-lws2_32','-o',out/'native-unit.exe'])
    check('native defaults' in run([out/'native-unit.exe']),'native parser defaults1024/8/CFG1/Euler/noSDXL-LoRA')
    class Upstream(http.server.BaseHTTPRequestHandler):
        model='/app0/models/anima-turbo-v1.1-f16.safetensors'
        posts=[];hosts=[]
        def log_message(self,*args):pass
        def reply(self,code,body):
            raw=json.dumps(body).encode();self.send_response(code)
            self.send_header('Content-Length',str(len(raw)));self.send_header('Connection','close');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            self.hosts.append(self.headers.get('Host'))
            if self.path=='/health':self.reply(200,{'engine_available':True,'gpu_available':True,'gpu_verified':False,'model_loaded':False,'model':self.model,'gpu_device':'HOST MOCK - NO GPU','features':{'text_to_image':True}})
            elif self.path=='/api/jobs/'+'b'*32:self.reply(200,{'job_id':'b'*32,'status':'failed','error':'Expected host-only failure; no inference'})
            else:self.reply(404,{'error':'fixture route absent'})
        def do_POST(self):
            self.hosts.append(self.headers.get('Host'))
            raw=self.rfile.read(int(self.headers.get('Content-Length','0')))
            self.posts.append({'path':self.path,'body':json.loads(raw),'auth_ok':self.headers.get('Authorization')=='Bearer '+token})
            self.reply(202,{'job_id':'b'*32,'status':'queued'})
    upstream=http.server.ThreadingHTTPServer(('127.0.0.1',0),Upstream)
    ut=threading.Thread(target=upstream.serve_forever,daemon=True);ut.start()
    up_port=upstream.server_address[1]
    for name,flag in [('gateway-protocol','-DPS5_AI_GATEWAY_PROTOCOL_TEST'),('gateway-forwarder',f'-DPS5_AI_UPSTREAM_PORT={up_port}')]:
        run([*common,'-I',g,'-I',sha,g/'server.c',g/'cpu_hooks.c',sha/'sha256.c',flag,'-lws2_32','-o',out/(name+'.exe')])
    gateway_unit=out/'gateway-unit.c'
    gateway_unit.write_text('''#define main gateway_main
#include "server.c"
#undef main
int main(void){ps5_ai_request r;int url;char e[768],p[64],sig[65];
 if(listen_port!=8191 || strcmp(root_path,"/data/ai/anima-openai-gateway") || strcmp(OPENAI_MODEL,"anima-turbo-v1.1"))return 1;
 if(!openai_validate("{\\"prompt\\":\\"fixture\\"}",&r,&url,e,sizeof(e),p))return 2;
 if(r.steps!=8 || r.cfg_scale!=1 || r.manga_strength || r.manga_monochrome || strcmp(r.sampler,"euler"))return 3;
 strcpy(api_token,"synthetic-unit-key");token_len=strlen(api_token);server_started_at=42;
 image_signature("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",123,sig);puts(sig);return 0;}
''',encoding='utf-8',newline='\n')
    run([*common,'-I',g,'-I',sha,'-DPS5_AI_GATEWAY_PROTOCOL_TEST',gateway_unit,g/'cpu_hooks.c',sha/'sha256.c','-lws2_32','-o',out/'gateway-unit.exe'])
    signature=run([out/'gateway-unit.exe']).strip()
    check(signature==hmac.new(b'synthetic-unit-key',b'ps5-image-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:123:42',hashlib.sha256).hexdigest(),'gateway isolated defaults and image-only HMAC match standard library')
    def request(port,method,path,body=None,auth=True,typ='application/json'):
        c=http.client.HTTPConnection('127.0.0.1',port,timeout=15)
        headers={'Content-Type':typ}
        if auth:headers['Authorization']='Bearer '+token
        raw=json.dumps(body).encode() if isinstance(body,dict) else body
        c.request(method,path,body=raw,headers=headers);r=c.getresponse();data=r.read();c.close()
        return r.status,json.loads(data)
    def launch(name):
        root=out/name;root.mkdir();(root/'config').mkdir();(root/'config/api-token').write_text(token,encoding='ascii')
        port=free_port();log=(out/(name+'.private.log')).open('wb');logs.append(log)
        p=subprocess.Popen([str(out/(name+'.exe')),'--root',str(root),'--bind','127.0.0.1','--port',str(port)],stdout=log,stderr=log,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        processes.append((p,port))
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if p.poll() is not None:raise RuntimeError(name+' exited before health')
            try:
                if request(port,'GET','/health')[0]==200:break
            except OSError:pass
            time.sleep(.05)
        else:raise TimeoutError(name+' startup')
        check(not (root/'config/api-token').exists(),name+' consumes one-use token')
        return root,port
    try:
        native,np=launch('native-failure');protocol,pp=launch('gateway-protocol');forward,fp=launch('gateway-forwarder')
        for port,name in [(np,'native'),(pp,'gateway')]:
            code,health=request(port,'GET','/health')
            check(code==200 and health['model_id']=='anima-turbo-v1.1',name+' explicit Anima health identity')
            check(health['features']=={'text_to_image':True,'img2img':False,'inpainting':False,'ip_adapter':False,'controlnet':False,'sdxl_manga_lora':False},name+' truthful text2img-only feature flags')
            check(request(port,'GET','/health',auth=False)[0]==401,name+' rejects missing Bearer')
        code,models=request(pp,'GET','/v1/models')
        check(code==200 and [m['id'] for m in models['data']]==['anima-turbo-v1.1'],'OpenAI discovery lists only Anima')
        check(request(pp,'GET','/v1/models/wai-illustrious-sdxl-v17.0')[0]==404,'WAI model discovery rejected')
        spec=json.loads((R/'docs/openapi.json').read_text())
        check(request(pp,'GET','/openapi.json')[1]==spec,'embedded OpenAPI equals docs JSON')
        check(spec['components']['schemas']['ImageRequest']['properties']['cfg_scale']['default']==1,'OpenAPI CFG1 default')
        invalid=[{'model':'wai-illustrious-sdxl-v17.0'},{'model':'gpt-image-1'},{'manga_strength':1},{'manga_strength':.001},{'manga_monochrome':True},
                 {'init_image_id':'a'*32},{'reference_image_id':'a'*32},{'mask_image_id':'a'*32},{'denoise_strength':.5},{'ip_adapter_strength':.7},
                 {'controlnet':'canny'},{'controlnet':'openpose'},{'control_image_id':'a'*32},{'control_strength':1},
                 {'init_image_id':None},{'reference_image_id':None},{'mask_image_id':None},{'controlnet':None}]
        for fields in invalid:
            key=next(iter(fields));native_body={'prompt':'contract fixture',**fields}
            check(request(np,'POST','/api/generate',native_body)[0]==400,'native rejects '+key+'='+str(fields[key]))
            check(request(pp,'POST','/api/generate',native_body)[0]==400,'gateway async rejects '+key+'='+str(fields[key]))
            code,error=request(pp,'POST','/v1/images/generations',native_body)
            check(code==400 and error['error']['param']==key,'OpenAI rejects incompatible '+key+'='+str(fields[key]))
        for body,key in [({'n':2},'n'),({'unknown':1},'unknown'),({'size':'065x64'},'size'),({'stream':True},'stream')]:
            code,error=request(pp,'POST','/v1/images/generations',{'prompt':'fixture',**body})
            check(code==400 and error['error']['param']==key,'OpenAI rejects '+key)
        check(request(pp,'POST','/v1/images/edits',b'',typ='multipart/form-data; boundary=fixture')[0]==400,'edits explicitly unsupported before native upload')
        for path in ['/api/inputs','/api/masks','/api/masks/grayscale']:
            check(request(np,'POST',path,b'not-an-image',typ='image/png')[0]==400,'native input store disabled '+path)
        check(not list((native/'jobs').glob('*.request.json')) and not list((protocol/'jobs').glob('*.request.json')),'incompatible requests enqueue zero jobs')
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future=executor.submit(request,pp,'POST','/v1/images/generations',{'prompt':'fixture','manga_strength':0,'manga_monochrome':False,'size':'64x64'})
            deadline=time.monotonic()+5
            while not list((protocol/'jobs').glob('*.request.json')):
                if time.monotonic()>deadline:raise TimeoutError('protocol admission')
                time.sleep(.02)
            check(request(pp,'GET','/health')[0]==200,'health responsive during synchronous generation wait')
            check(request(pp,'POST','/v1/images/generations',{'prompt':'second'})[0]==429,'single synchronous request admission')
            code,error=future.result();check(code==500 and error['error']['type']=='server_error','failed fixture never becomes an image')
        Upstream.model='/app0/models/waiIllustriousSDXL_v170.safetensors'
        code,h=request(fp,'GET','/health')
        check(code==200 and not h['engine_available'] and not h['gpu_available'] and not h['model_loaded'],'gateway denies WAI native model even when native says GPU available')
        check(request(fp,'POST','/v1/images/generations',{'prompt':'fixture'})[0]==503 and not Upstream.posts,'model mismatch sends zero native POSTs')
        Upstream.model='/app0/models/anima-turbo-v1.1-f16.safetensors'
        check(request(fp,'POST','/v1/images/generations',{'prompt':'fixture','model':'anima-turbo-v1.1'})[0]==500,'native failure forwarded truthfully')
        check(len(Upstream.posts)==1,'exactly one native generation POST and no retry')
        forwarded=Upstream.posts[0]
        check(forwarded['path']=='/api/generate' and forwarded['auth_ok'],'Bearer same-console forwarding')
        check(forwarded['body']['model']=='anima-turbo-v1.1' and forwarded['body']['steps']==8 and forwarded['body']['cfg_scale']==1 and forwarded['body']['sampler']=='euler' and forwarded['body']['manga_strength']==0,'forwarded explicit Anima identity and defaults')
        check(set(Upstream.hosts)=={f'127.0.0.1:{up_port}'},'HTTP Host uses actual configured upstream port')
        check(not list((forward/'output').glob('*.png')) and not list((protocol/'output').glob('*.png')),'zero manufactured PNGs')
        client_token=out/'synthetic-client-token';client_token.write_text(token)
        try:
            models=run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',R/'scripts/Invoke-AnimaOpenAI.ps1','-BaseUrl',f'http://127.0.0.1:{pp}/v1','-TokenFile',client_token,'-ListModels'])
            check('anima-turbo-v1.1' in models,'PowerShell Anima model-discovery client')
        finally:client_token.unlink(missing_ok=True)
    finally:
        for p,port in reversed(processes):
            if p.poll() is None:
                try:request(port,'POST','/api/admin/shutdown',b'')
                except OSError:pass
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=5)
        upstream.shutdown();upstream.server_close()
        for f in logs:f.close()
        for p in out.glob('*/config/api-token'):p.unlink(missing_ok=True)
    tested=['native/app/server.c','docs/openapi.json','scripts/Invoke-AnimaOpenAI.ps1','tests/anima_api_protocol.py']
    tested += [p.relative_to(R).as_posix() for p in sorted(g.iterdir()) if p.is_file()]
    result={'schema':'anima-host-api-protocol-v1','success':True,'checks':CHECKS,'count':len(CHECKS),'inference_performed':False,'GPU_used':False,'PS5_connections':0,'PNG_generated':False,'generation_POST_retries':0,'tested_source_sha256':{p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in tested}}
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
