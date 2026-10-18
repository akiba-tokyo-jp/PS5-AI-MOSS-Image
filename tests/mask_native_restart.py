"""Native localhost restart/history protocol fixture; never PS5/model inference."""
import argparse, http.client, io, json, secrets, socket, subprocess, time
from pathlib import Path
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument('--native-exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert a.native_exe.is_file();a.out.mkdir(parents=True,exist_ok=False);root=a.out/'HOST_ONLY_SYNTHETIC';root.mkdir()
 for d in ['config','jobs','output','inputs','logs']:(root/d).mkdir()
 token=secrets.token_hex(32)
 with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
 checks=[];procs=[];logfiles=[]
 def check(ok,label):
  assert ok,label
  checks.append(label)
 def request(method,path,body=None,kind='application/json'):
  if isinstance(body,dict):body=json.dumps(body).encode()
  c=http.client.HTTPConnection('127.0.0.1',port,timeout=10);c.request(method,path,body=body,headers={'Authorization':'Bearer '+token,'Content-Type':kind})
  r=c.getresponse();raw=r.read();code=r.status;typ=r.getheader('Content-Type','');c.close()
  return code,(json.loads(raw) if 'application/json' in typ else raw)
 def png(mode,value):
  b=io.BytesIO();Image.new(mode,(256,256),value).save(b,format='PNG');return b.getvalue()
 def start():
  (root/'config/api-token').write_text(token)
  f=(a.out/f'run{len(procs)}.log').open('wb');logfiles.append(f)
  proc=subprocess.Popen([str(a.native_exe),'--root',str(root),'--bind','127.0.0.1','--port',str(port)],stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);procs.append(proc)
  until=time.monotonic()+10
  while time.monotonic()<until:
   assert proc.poll() is None,'Localhost fixture exited during startup'
   try:
    if request('GET','/health')[0]==200:break
   except OSError:pass
   time.sleep(.05)
  else:raise TimeoutError('Localhost fixture startup')
  check(not (root/'config/api-token').exists(),'one-use host token consumed '+str(len(procs)))
 def stop():
  check(request('POST','/api/admin/shutdown',b'')[0]==202,'normal host shutdown '+str(len(procs)));procs[-1].wait(timeout=8);check(procs[-1].returncode==0,'clean host exit '+str(len(procs)))
 def stored_job(i,status):
  return {'job_id':i,'status':status,'progress':0.5 if status=='running' else 1.0,'error':'','image_url':'/api/images/'+i if status=='completed' else None,
   'created_at':int(time.time())-10,'updated_at':int(time.time())-5,'gpu_used':status=='completed','generation_seconds':0,'peak_cpu_bytes':None,'peak_gpu_bytes':None,'mode':'img2img',
   'manga_strength':0,'manga_monochrome':False,'init_image_sha256':'','reference_image_sha256':'','denoise_strength':.6,'ip_adapter_strength':0,'actual_steps':1,'ip_adapter_tokens':0,'gpu_ip_attention_verified':False,'init_encode_seconds':0,'ip_adapter_seconds':0,
   'controlnet':'','control_image_sha256':'','control_strength':0,'gpu_controlnet_verified':False,'controlnet_evals':0,'controlnet_tensors':0,'controlnet_seconds':0,
   'inpainting':True,'mask_prepared':status=='completed','mask_image_sha256':'0'*64,'mask_denoise_evaluations':1,'mask_preserved_pixels':100,'mask_changed_pixels':10}
 # This is a handmade transport fixture, not a PS5-produced image or GPU proof.
 synthetic_png=png('RGB',(9,24,39));history='a'*32;pending='b'*32
 (root/'output'/f'{history}.png').write_bytes(synthetic_png)
 (root/'jobs'/f'{history}.json').write_text(json.dumps(stored_job(history,'completed')))
 (root/'jobs'/f'{pending}.json').write_text(json.dumps(stored_job(pending,'running')))
 try:
  start()
  code,j=request('GET','/api/jobs/'+pending);check(code==200 and j['status']=='failed' and 'Server restarted' in j['error'],'36-field pending job recovered as failed')
  code,j=request('GET','/api/jobs/'+history);check(code==200 and len(j)==36,'36-field historical job readable')
  code,data=request('GET','/api/images/'+history);check(code==200 and data==synthetic_png,'disk-only historical PNG eligibility uses64-field internal parser')
  code,rgb=request('POST','/api/inputs',png('RGB',(40,90,140)),'image/png');check(code==201 and rgb['kind']=='image','RGB upload admitted')
  code,mask=request('POST','/api/masks/grayscale',png('L',255),'image/png');check(code==201 and mask['kind']=='mask','grayscale mask admitted')
  body={'prompt':'HOST ONLY intentional failure, no inference','width':256,'height':256,'init_image_id':rgb['image_id'],'reference_image_id':rgb['image_id'],'controlnet':'openpose','control_image_id':rgb['image_id'],'mask_image_id':mask['image_id'],'manga_strength':0}
  for label,alter in [('RGB cannot be mask',{'mask_image_id':rgb['image_id']}),('mask cannot be control',{'control_image_id':mask['image_id']}),('dimensions must agree',{'width':128})]:
   code,_=request('POST','/api/generate',dict(body,**alter));check(code==400,label)
   check(request('GET','/api/inputs/'+rgb['image_id'])[1]['references']==0 and request('GET','/api/inputs/'+mask['image_id'])[1]['references']==0,'invalid request leaves zero leases '+label)
  code,j=request('POST','/api/generate',body);check(code==202 and len(j)==36 and j['inpainting'] and j['controlnet']=='openpose','combined request admitted once with36 job fields')
  check(request('GET','/api/inputs/'+rgb['image_id'])[1]['references']==3,'same RGB has init/reference/control three leases')
  check(request('GET','/api/inputs/'+mask['image_id'])[1]['references']==1,'mask has fourth independent lease')
  check(request('DELETE','/api/inputs/'+mask['image_id'])[0]==409,'live mask deletion refused')
  until=time.monotonic()+5
  while time.monotonic()<until:
   _,last=request('GET','/api/jobs/'+j['job_id'])
   if last['status']=='failed':break
   time.sleep(.03)
  else:raise TimeoutError('Intentional host failure result')
  check('no inference performed' in last['error'] and not last['gpu_used'],'host failure cannot claim GPU generation')
  check(last['mask_image_sha256']==mask['sha256'] and not last['mask_prepared'],'failed job preserves mask provenance without claiming mask computation')
  check(request('GET','/api/inputs/'+rgb['image_id'])[1]['references']==0 and request('GET','/api/inputs/'+mask['image_id'])[1]['references']==0,'all four role leases released after terminal failure')
  check(len(list((root/'jobs').glob('*.request.json')))==1,'exactly one accepted native generation POST')
  stop()
  (root/'jobs'/f'{pending}.json').write_text(json.dumps(stored_job(pending,'queued')))
  start()
  check(request('GET','/api/jobs/'+pending)[1]['status']=='failed','36-field queued job recovered after restart')
  code,got=request('GET','/api/inputs/'+mask['image_id']);check(code==200 and got['kind']=='mask' and got['sha256']==mask['sha256'],'mask canonical digest and kind recovered via raw FD')
  code,got=request('GET','/api/inputs/'+rgb['image_id']);check(code==200 and got['kind']=='image' and got['sha256']==rgb['sha256'],'RGB canonical digest and kind recovered via raw FD')
  code,data=request('GET','/api/images/'+history);check(code==200 and data==synthetic_png,'historical PNG retrieved after second restart')
  check(request('DELETE','/api/inputs/'+mask['image_id'])[0]==200,'recovered mask removable')
  check(request('DELETE','/api/inputs/'+rgb['image_id'])[0]==200,'recovered RGB removable')
  check(not list((root/'inputs').glob('*.mask')) and not list((root/'inputs').glob('*.rgb')),'input store empty after normal deletes')
  stop()
 finally:
  for proc in procs:
   if proc.poll() is None:
    try:request('POST','/api/admin/shutdown',b'');proc.wait(timeout=8)
    except Exception:proc.terminate();proc.wait(timeout=8)
  for f in logfiles:f.close()
 for f in a.out.glob('run*.log'):check(token.encode() not in f.read_bytes(),'host fixture log contains no credential '+f.name)
 result={'state':'passed','checks_passed':len(checks),'checks':checks,'accepted_generation_POSTs':1,'automatic_retries':0,'PS5_connections':0,'Windows_model_inference':False,'generated_PNGs':0,'handmade_history_PNG_is_transport_fixture_only':True,'native_exe':str(a.native_exe)}
 (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
