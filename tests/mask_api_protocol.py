"""Host-only input leases + multipart forwarding. Never runs PS5/model inference."""
import concurrent.futures
import http.client
import io
import hashlib
import struct
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import time
import sys
from PIL import Image


OUT=Path(os.environ['PS5_WAI_MASK_TEST_OUT'])
checks=[]


def check(value,name):
    if not value:raise AssertionError(name)
    checks.append(name)


def wait_health(port,token,process):
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('Fixture exited: '+process.stderr.read().decode(errors='replace'))
        try:
            if request(port,token,'GET','/health')[0]==200:return
        except OSError:pass
        time.sleep(.05)
    raise TimeoutError('Host fixture startup')


def request(port,token,method,path,body=b'',kind='application/json',auth=True,length=None):
    headers={'Content-Type':kind}
    if auth:headers['Authorization']='Bearer '+token
    if length is not None:headers['Content-Length']=str(length)
    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=60)
    conn.request(method,path,body=body,headers=headers)
    response=conn.getresponse();raw=response.read();code=response.status;conn.close()
    return code,json.loads(raw)


def form(fields,files=(),boundary='reference-contract-0123456789'):
    pieces=[]
    for key,value in fields:
        pieces.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    for key,kind,data in files:
        pieces.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="ignored.png"\r\nContent-Type: {kind}\r\n\r\n'.encode()+data+b'\r\n')
    return b''.join(pieces)+f'--{boundary}--\r\n'.encode(),f'multipart/form-data; boundary={boundary}'


def png(mode='RGBA'):
    buf=io.BytesIO();Image.new(mode,(8,8),(20,100,220,127) if mode=='RGBA' else (20,100,220)).save(buf,format='PNG');return buf.getvalue()


def main():
    # Explicit host port; production defaults remain8188. No LAN/PS5 connections.
    with socket.socket() as s:s.bind(('127.0.0.1',18188))
    with socket.socket() as s:s.bind(('127.0.0.1',0));gateway_port=s.getsockname()[1]
    token=secrets.token_hex(32)
    with tempfile.TemporaryDirectory(prefix='reference-contract-',dir=OUT/'host') as folder:
        roots=[Path(folder)/name for name in ('native','gateway')]
        for root in roots:(root/'config').mkdir(parents=True);(root/'config/api-token').write_text(token)
        processes=[]
        try:
            for exe,root,port in [('native-failure.exe',roots[0],18188),('gateway-forwarder.exe',roots[1],gateway_port)]:
                p=subprocess.Popen([str(OUT/'host'/exe),'--root',str(root),'--bind','127.0.0.1','--port',str(port)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                processes.append(p);wait_health(port,token,p)
            expected_schema=json.loads((Path(__file__).resolve().parents[1]/'docs/openapi.json').read_text())
            check(request(gateway_port,token,'GET','/openapi.json')[1]==expected_schema,'served reference OpenAPI schema matches candidate')
            check(all(not (root/'config/api-token').exists() for root in roots),'both one-use tokens consumed')
            valid_png=png()
            call=lambda method,path,body=b'',kind='application/json',auth=True:request(18188,token,method,path,body,kind,auth)
            check(call('POST','/api/inputs',valid_png,'image/png',False)[0]==401,'upload authentication')
            check(call('POST','/api/inputs',valid_png,'image/jpeg')[0]==415,'signature/type mismatch')
            check(call('POST','/api/inputs',b'not png','image/png')[0]==415,'invalid image signature')
            check(call('POST','/api/inputs',valid_png[:32],'image/png')[0]==400,'truncated image rejected')
            check(request(18188,token,'POST','/api/inputs',kind='image/png',length=8*1024**2+1)[0]==413,'compressed body limit before upload')
            huge=io.BytesIO();Image.new('RGB',(2049,1)).save(huge,format='PNG')
            check(call('POST','/api/inputs',huge.getvalue(),'image/png')[0]==400,'dimension limit before decoding')
            code,meta=call('POST','/api/inputs',valid_png,'image/png');check(code==201 and meta['width']==8 and len(meta['sha256'])==64,'PNG canonical storage and digest')
            image_id=meta['image_id'];file=roots[0]/'inputs'/(image_id+'.rgb')
            check(file.stat().st_size==16+8*8*3,'normalized RGB storage size')
            raw=file.read_bytes();check(raw[16:19]==bytes(((20*127+255*128+127)//255,(100*127+255*128+127)//255,(220*127+255*128+127)//255)),'transparent pixels composited on white')
            jpg=io.BytesIO();Image.new('RGB',(8,5),(100,20,40)).save(jpg,format='JPEG')
            code,jmeta=call('POST','/api/inputs',jpg.getvalue(),'image/jpeg');check(code==201 and jmeta['height']==5,'JPEG full decoding')
            check(call('DELETE','/api/inputs/'+jmeta['image_id'])[0]==200,'unused input deletion')
            check(call('GET','/api/inputs/'+jmeta['image_id'])[0]==404,'deleted input unavailable')
            base={'prompt':'host protocol only','width':256,'height':256,'steps':20,'init_image_id':image_id,'denoise_strength':.6}
            invalid=[dict(base,init_image_id='../escape'),dict(base,init_image_id='0'*32),dict(base,denoise_strength=0),dict(base,denoise_strength=True),dict(base,ip_adapter_strength=.7),dict(base,init_image_path='/system/private')]
            for i,body in enumerate(invalid):check(call('POST','/api/generate',json.dumps(body).encode())[0] in (400,404),'invalid reference request '+str(i))
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future=executor.submit(call,'POST','/api/generate',json.dumps(base).encode())
                code,accepted=future.result();check(code==202,'native image request admitted once')
            check(call('DELETE','/api/inputs/'+image_id)[0]==409,'queued/running input lease prevents deletion')
            until=time.monotonic()+5
            while time.monotonic()<until:
                code,job=call('GET','/api/jobs/'+accepted['job_id'])
                if job['status']=='failed':break
                time.sleep(.03)
            check(job['status']=='failed' and job['mode']=='img2img' and job['init_image_sha256']==meta['sha256'],'honest failure with mode/input digest persisted')
            check(job['manga_monochrome'] is False and job['manga_strength']==1,'omitted monochrome defaults off while LoRA stays enabled')
            check(call('GET','/api/inputs/'+image_id)[1]['references']==0,'terminal job releases input lease')
            fields=[('model','wai-illustrious-sdxl-v17.0'),('prompt','host protocol fixture'),('size','256x256'),('steps','20'),('manga_strength','0.5'),('manga_monochrome','false')]
            for mode in ('img2img','ip_adapter','combined'):
                print('Host-only multipart forwarding '+mode,flush=True)
                body,kind=form(fields+[('reference_mode',mode)],[('image','image/png',valid_png)])
                code,result=request(gateway_port,token,'POST','/v1/images/edits',body,kind)
                check(code==500 and 'Native job failed' in result['error']['message'],'multipart '+mode+' forwards real host failure, no false PNG')
                native_jobs=[json.loads(p.read_text()) for p in (roots[0]/'jobs').glob('*.json') if not p.name.endswith('.request.json')]
                current=max(native_jobs,key=lambda j:(j['created_at'],j['job_id']))
                check(current['mode']==mode and current['manga_strength']==.5,'multipart mode/manga strength forwarded '+mode)
                check(current['manga_monochrome'] is False,'multipart monochrome off forwarded '+mode)
                check(len(list((roots[0]/'inputs').glob('*.rgb')))==1,'ephemeral uploaded input cleanup '+mode)
            body,kind=form(fields+[('reference_mode','combined')],[('image','image/png',valid_png),('reference_image','image/jpeg',jpg.getvalue())])
            code,_=request(gateway_port,token,'POST','/v1/images/edits',body,kind)
            check(code==500 and len(list((roots[0]/'inputs').glob('*.rgb')))==1,'two separate images forward and clean up')
            import openai
            before_sdk=len(list((roots[0]/'jobs').glob('*.request.json')))
            with openai.OpenAI(base_url=f'http://127.0.0.1:{gateway_port}/v1',api_key=token,max_retries=0,timeout=60,
                               http_client=openai.DefaultHttpxClient(trust_env=False)) as client:
                try:
                    client.images.edit(model='wai-illustrious-sdxl-v17.0',prompt='official SDK host fixture',
                        image=('reference.png',valid_png,'image/png'),size='256x256',response_format='b64_json',
                        extra_body={'reference_mode':'ip_adapter','ip_adapter_strength':.7,'steps':20,'manga_strength':.5})
                    raise AssertionError('Host fixture cannot return generated image')
                except openai.APIStatusError as e:
                    check(e.status_code==500 and 'Native job failed' in str(e),'official OpenAI SDK multipart reaches native host fixture')
            check(len(list((roots[0]/'jobs').glob('*.request.json')))==before_sdk+1,'official SDK submits generation exactly once')
            for enabled in (False,True):
                code,accepted=call('POST','/api/generate',json.dumps({'prompt':'host option fixture','manga_monochrome':enabled}).encode())
                check(code==202,'native boolean accepted '+str(enabled))
                check(call('GET','/api/jobs/'+accepted['job_id'])[1]['manga_monochrome'] is enabled,'native boolean persisted '+str(enabled))
                while call('GET','/api/jobs/'+accepted['job_id'])[1]['status'] not in ('failed','completed'):time.sleep(.05)
                code,_=request(gateway_port,token,'POST','/v1/images/generations',json.dumps({'prompt':'host option fixture','manga_monochrome':enabled}).encode())
                check(code==500,'gateway boolean accepted '+str(enabled))
                current=max([json.loads(p.read_text()) for p in (roots[0]/'jobs').glob('*.json') if not p.name.endswith('.request.json')],key=lambda j:(j['created_at'],j['job_id']))
                check(current['manga_monochrome'] is enabled,'gateway boolean forwarded '+str(enabled))
            for invalid_value in ('false',0,[]):
                check(call('POST','/api/generate',json.dumps({'prompt':'invalid flag','manga_monochrome':invalid_value}).encode())[0]==400,'native rejects nonboolean '+str(invalid_value))
                check(request(gateway_port,token,'POST','/v1/images/generations',json.dumps({'prompt':'invalid flag','manga_monochrome':invalid_value}).encode())[0]==400,'gateway rejects nonboolean '+str(invalid_value))
            bad=[form(fields+[('prompt','duplicate')],[('image','image/png',valid_png)]),form(fields+[('mask','unsupported')],[('image','image/png',valid_png)]),form(fields+[('steps','{"x":1}')],[('image','image/png',valid_png)]),form(fields,[('image','image/png',valid_png),('image[]','image/png',valid_png)]),form(fields,[('image','image/png',valid_png),('mask','image/png',valid_png)])]
            before=len(list((roots[0]/'jobs').glob('*.request.json')))
            for i,(body,kind) in enumerate(bad):check(request(gateway_port,token,'POST','/v1/images/edits',body,kind)[0]==400,'invalid multipart '+str(i))
            check(len(list((roots[0]/'jobs').glob('*.request.json')))==before,'invalid edits never submit generation')
            # Masked img2img transport, canonical values and atomic three-input leases.
            source_buf=io.BytesIO();Image.new('RGB',(256,256),(90,140,180)).save(source_buf,format='PNG');source_png=source_buf.getvalue()
            a=Image.new('RGBA',(256,256),(230,10,90,255));a.paste((10,200,30,0),(0,0,128,256))
            alpha_buf=io.BytesIO();a.save(alpha_buf,format='PNG');alpha_png=alpha_buf.getvalue()
            g=Image.new('L',(256,256),0);g.paste(255,(0,0,128,256))
            gray_buf=io.BytesIO();g.save(gray_buf,format='PNG');gray_png=gray_buf.getvalue()
            mask_bytes=g.tobytes();mask_sha=hashlib.sha256(b'PS5MSK1\0'+struct.pack('<II',256,256)+mask_bytes).hexdigest()
            check(call('POST','/api/masks',alpha_png,'image/png',False)[0]==401,'mask upload authentication')
            check(request(18188,token,'POST','/api/masks',kind='image/png',length=4*1024**2)[0]==413,'mask below4MiB compressed limit before upload')
            check(call('POST','/api/masks',source_png,'image/png')[0]==400,'opaque RGB cannot silently become alpha mask')
            check(call('POST','/api/masks',jpg.getvalue(),'image/jpeg')[0]==415,'mask requires PNG')
            opaque_buf=io.BytesIO();Image.new('RGBA',(256,256),(0,0,0,255)).save(opaque_buf,format='PNG')
            check(call('POST','/api/masks',opaque_buf.getvalue(),'image/png')[0]==400,'fully opaque alpha mask refuses empty edit')
            black_buf=io.BytesIO();Image.new('L',(256,256),0).save(black_buf,format='PNG')
            check(call('POST','/api/masks/grayscale',black_buf.getvalue(),'image/png')[0]==400,'all-black grayscale mask refuses empty edit')
            code,mask_alpha=call('POST','/api/masks',alpha_png,'image/png')
            check(code==201 and mask_alpha['kind']=='mask' and mask_alpha['sha256']==mask_sha,'alpha zero means edit; stored canonical digest')
            code,mask_gray=call('POST','/api/masks/grayscale',gray_png,'image/png')
            check(code==201 and mask_gray['sha256']==mask_sha and mask_gray['bytes']==16+256*256,'white means edit; equivalent one-byte mask storage')
            mask_path=roots[0]/'inputs'/(mask_gray['image_id']+'.mask')
            check(mask_path.read_bytes()==b'PS5MSK1\0'+struct.pack('<II',256,256)+mask_bytes,'stored mask values preserve polarity and exact pixels')
            code,source_meta=call('POST','/api/inputs',source_png,'image/png');check(code==201,'masked source upload')
            mask_request={'prompt':'host masked protocol only','width':256,'height':256,'init_image_id':source_meta['image_id'],'mask_image_id':mask_gray['image_id'],'denoise_strength':.8}
            bad_native=[dict(mask_request,width=128),dict(mask_request,mask_image_id=source_meta['image_id']),dict(mask_request,init_image_id=mask_alpha['image_id']),{'prompt':'host','mask_image_id':mask_gray['image_id']}]
            before_masks=len(list((roots[0]/'jobs').glob('*.request.json')))
            for i,value in enumerate(bad_native):check(call('POST','/api/generate',json.dumps(value).encode())[0]==400,'masked input kind/dimension validation '+str(i))
            check(len(list((roots[0]/'jobs').glob('*.request.json')))==before_masks,'invalid masks never admit native generation')
            code,admitted=call('POST','/api/generate',json.dumps(mask_request).encode());check(code==202,'masked native request admitted once')
            check(call('DELETE','/api/inputs/'+mask_gray['image_id'])[0]==409,'mask lease blocks queued/running deletion')
            check(call('DELETE','/api/inputs/'+source_meta['image_id'])[0]==409,'masked source lease blocks deletion')
            while True:
                _,masked_job=call('GET','/api/jobs/'+admitted['job_id'])
                if masked_job['status']=='failed':break
                time.sleep(.03)
            check(masked_job['inpainting'] and masked_job['mask_image_sha256']==mask_sha and not masked_job['mask_prepared'],'failed fixture persists mask provenance without claiming inference')
            check(call('GET','/api/inputs/'+mask_gray['image_id'])[1]['references']==0,'terminal mask lease released')
            for meta in (mask_alpha,mask_gray,source_meta):check(call('DELETE','/api/inputs/'+meta['image_id'])[0]==200,'delete unused mask/source')
            mask_fields=[('prompt','host inpaint fixture'),('size','256x256'),('reference_mode','inpaint')]
            for mode,data in [('alpha',alpha_png),('grayscale',gray_png)]:
                body,kind=form(mask_fields+[('mask_mode',mode)],[('image','image/png',source_png),('mask','image/png',data)])
                before_masks=len(list((roots[0]/'jobs').glob('*.request.json')))
                code,result=request(gateway_port,token,'POST','/v1/images/edits',body,kind)
                check(code==500 and 'Native job failed' in result['error']['message'],'real native host failure propagated for '+mode+' mask')
                jobs=[json.loads(p.read_text()) for p in (roots[0]/'jobs').glob('*.json') if not p.name.endswith('.request.json')]
                current=max(jobs,key=lambda j:(j['created_at'],j['job_id']))
                check(current['inpainting'] and current['mask_image_sha256']==mask_sha,'multipart mask mode/polarity forwarded '+mode)
                check(len(list((roots[0]/'jobs').glob('*.request.json')))==before_masks+1,'one native job per masked request '+mode)
                check(not list((roots[0]/'inputs').glob('*.mask')) and len(list((roots[0]/'inputs').glob('*.rgb')))==1,'ephemeral mask and source cleanup '+mode)
            before_masks=len(list((roots[0]/'jobs').glob('*.request.json')))
            with openai.OpenAI(base_url=f'http://127.0.0.1:{gateway_port}/v1',api_key=token,max_retries=0,timeout=60,
                               http_client=openai.DefaultHttpxClient(trust_env=False)) as client:
                try:
                    client.images.edit(model='wai-illustrious-sdxl-v17.0',prompt='SDK mask fixture',image=('input.png',source_png,'image/png'),
                        mask=('mask.png',alpha_png,'image/png'),size='256x256',extra_body={'denoise_strength':.8})
                    raise AssertionError('Host fixture cannot generate PNG')
                except openai.APIStatusError as e:check(e.status_code==500 and 'Native job failed' in str(e),'official SDK standard mask field reaches native host fixture')
            check(len(list((roots[0]/'jobs').glob('*.request.json')))==before_masks+1,'SDK masked edit submits one job only')
            bad_masks=[form([('prompt','bad'),('size','256x256'),('reference_mode','inpaint')],[('image','image/png',source_png)]),
                form(mask_fields+[('mask_mode','unknown')],[('image','image/png',source_png),('mask','image/png',alpha_png)]),
                form([('prompt','bad'),('size','256x256'),('reference_mode','ip_adapter')],[('image','image/png',source_png),('mask','image/png',alpha_png)]),
                form(mask_fields,[('image','image/png',source_png),('mask','image/png',alpha_png),('mask','image/png',alpha_png)]),
                form(mask_fields,[('image','image/png',source_png),('mask','image/jpeg',jpg.getvalue())]),
                form(mask_fields,[('image','image/png',valid_png),('mask','image/png',alpha_png)]),
                form([('prompt','bad'),('mask_mode','alpha')],[('image','image/png',source_png)])]
            before_masks=len(list((roots[0]/'jobs').glob('*.request.json')))
            for i,(body,kind) in enumerate(bad_masks):check(request(gateway_port,token,'POST','/v1/images/edits',body,kind)[0]==400,'invalid mask multipart '+str(i))
            check(len(list((roots[0]/'jobs').glob('*.request.json')))==before_masks,'invalid masked edits never submit generation')
            check(not list((roots[0]/'inputs').glob('*.mask')) and len(list((roots[0]/'inputs').glob('*.rgb')))==1,'rejected mask request cleans temporary uploads')
            # Fill the explicit32-image bound, then verify admission refuses growth.
            ids=[image_id]
            for i in range(31):
                code,value=call('POST','/api/inputs',valid_png,'image/png');check(code==201,'bounded input slot '+str(i+2));ids.append(value['image_id'])
            check(call('POST','/api/inputs',valid_png,'image/png')[0]==429,'32-input quota')
            for value in ids:check(call('DELETE','/api/inputs/'+value)[0]==200,'release input slot')
            check(not list((roots[0]/'inputs').glob('*.rgb')),'input storage returns empty')
            large=io.BytesIO();Image.new('RGB',(2048,2048),(35,80,120)).save(large,format='PNG');large_bytes=large.getvalue();large_ids=[]
            for i in range(10):
                code,value=call('POST','/api/inputs',large_bytes,'image/png');check(code==201,'large input quota admission '+str(i+1));large_ids.append(value['image_id'])
            check(call('POST','/api/inputs',large_bytes,'image/png')[0]==429,'128MiB byte quota, independent of slot count')
            for value in large_ids:check(call('DELETE','/api/inputs/'+value)[0]==200,'large input deletion')
            code,keep=call('POST','/api/inputs',valid_png,'image/png')
            code,expired=call('POST','/api/inputs',valid_png,'image/png')
            _,keep_mask=call('POST','/api/masks',alpha_png,'image/png')
            _,expired_mask=call('POST','/api/masks',alpha_png,'image/png')
            check(call('POST','/api/admin/shutdown')[0]==202,'native idle shutdown before recovery test')
            processes[0].communicate(timeout=5)
            expired_path=roots[0]/'inputs'/(expired['image_id']+'.rgb');old=time.time()-24*60*60-10;os.utime(expired_path,(old,old))
            partial=roots[0]/'inputs'/('f'*32+'.rgb.part');partial.write_bytes(b'interrupted upload')
            (roots[0]/'config/api-token').write_text(token)
            expired_mask_path=roots[0]/'inputs'/(expired_mask['image_id']+'.mask');os.utime(expired_mask_path,(old,old))
            mask_partial=roots[0]/'inputs'/('e'*32+'.mask.part');mask_partial.write_bytes(b'interrupted')
            processes[0]=subprocess.Popen([str(OUT/'host/native-failure.exe'),'--root',str(roots[0]),'--bind','127.0.0.1','--port','18188'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            wait_health(18188,token,processes[0])
            check(call('GET','/api/inputs/'+keep['image_id'])[1]['sha256']==keep['sha256'],'unexpired input digest/metadata recovered after restart')
            check(call('GET','/api/inputs/'+expired['image_id'])[0]==404 and not expired_path.exists(),'expired input removed after restart')
            check(not partial.exists(),'interrupted input temporary file removed after restart')
            check(call('GET','/api/inputs/'+keep_mask['image_id'])[1]['sha256']==mask_sha,'mask kind and digest recovered after restart')
            check(call('GET','/api/inputs/'+expired_mask['image_id'])[0]==404 and not expired_mask_path.exists(),'expired mask removed after restart')
            check(not mask_partial.exists(),'interrupted mask temporary file removed after restart')
            check(call('DELETE','/api/inputs/'+keep_mask['image_id'])[0]==200,'recovered mask can be deleted')
            check(call('DELETE','/api/inputs/'+keep['image_id'])[0]==200,'recovered unpinned input can be removed')
            check(not list((roots[0]/'output').glob('*.png')) and not list((roots[1]/'output').glob('*.png')),'zero inference/generated images on Windows')
            for port,p in [(gateway_port,processes[1]),(18188,processes[0])]:
                check(request(port,token,'POST','/api/admin/shutdown')[0]==202,'idle host fixture shutdown')
                p.communicate(timeout=5);check(p.returncode==0,'clean host process exit')
        finally:
            for p in processes:
                if p.poll() is None:p.terminate();p.communicate(timeout=5)
    receipt={'checks_passed':len(checks),'checks':checks,'Windows_inference':False,'PS5_connections':0,'test_generated_PNGs':0}
    (OUT/'host/protocol-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
