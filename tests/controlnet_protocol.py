"""Real host HTTP validation/lease/forwarding tests; engine intentionally fails."""
import io,json,os,secrets,subprocess,sys,time,urllib.error,urllib.request
from pathlib import Path
from PIL import Image
O=Path(os.environ['PS5_WAI_FIXTURE_DIR'])
T=Path(os.environ['PS5_WAI_TEST_OUT'])
if len(sys.argv)>1:T=Path(sys.argv[1])
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
TOKEN=secrets.token_hex(32);checks=[]
def call(port,path,body=None,method=None,typ='application/json',auth=True):
    raw=json.dumps(body).encode() if isinstance(body,dict) else body
    req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=raw,method=method,
        headers={'Content-Type':typ,**({'Authorization':'Bearer '+TOKEN} if auth else {})})
    try:
        with opener.open(req,timeout=8) as r:return r.status,json.loads(r.read())
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read())
def expect(name,actual,wanted):assert actual==wanted,(name,actual,wanted);checks.append(name)
def launch(name,exe,port):
    root=T/name;root.mkdir();(root/'config').mkdir();(root/'config/api-token').write_text(TOKEN+'\n')
    f=(T/(name+'.private.log')).open('wb')
    p=subprocess.Popen([str(O/exe),'--root',str(root),'--bind','127.0.0.1','--port',str(port)],stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(100):
        try:
            if call(port,'/health')[0]==200:return p,root
        except OSError:pass
        assert p.poll() is None,'Host fixture stopped before health';time.sleep(.05)
    raise RuntimeError('Host fixture health timeout')
def main():
    T.mkdir();processes=[]
    try:
        native,root=launch('native','native-failure.exe',18188);processes.append((native,18188))
        proto,_=launch('protocol','gateway-protocol.exe',18289);processes.append((proto,18289))
        forward,fr=launch('forwarder','gateway-forwarder.exe',18189);processes.append((forward,18189))
        expect('unauthenticated native denied',call(18188,'/health',auth=False)[0],401)
        b=io.BytesIO();Image.new('RGB',(256,256),'black').save(b,format='PNG')
        code,upload=call(18188,'/api/inputs',b.getvalue(),typ='image/png');expect('control map upload',code,201);iid=upload['image_id']
        valid={'prompt':'Host protocol test only','width':256,'height':256,'controlnet':'canny','control_image_id':iid,'control_strength':1,'manga_strength':0}
        invalid=[{'controlnet':'unknown'},{'controlnet':'../model'},{'controlnet':5},{'control_image_id':'x'*32},
                 {'control_image_id':'../escape'},{'control_strength':0},{'control_strength':2},{'control_strength':'1'},
                 {'controlnet':None},{'control_image_id':None},{'control_image_path':'/etc/passwd'}]
        for i,fields in enumerate(invalid):
            v=dict(valid);v.update(fields);expect(f'native invalid control{i}',call(18188,'/api/generate',v)[0],400)
        v=dict(valid);del v['control_image_id'];expect('native missing map rejected',call(18188,'/api/generate',v)[0],400)
        v=dict(valid,control_image_id='f'*32);expect('native unknown map rejected',call(18188,'/api/generate',v)[0],404)
        oa={'prompt':'Host protocol test only','model':'wai-illustrious-sdxl-v17.0','size':'256x256','n':1,'controlnet':'canny','control_image_id':iid,'control_strength':1}
        for i,fields in enumerate(invalid):
            v=dict(oa);v.update(fields)
            if fields=={'controlnet':None} or fields=={'control_image_id':None}:pass
            expect(f'OpenAI invalid control{i}',call(18289,'/v1/images/generations',v)[0],400)
        # All three roles may share one upload; each pin must be released once.
        same=dict(valid,init_image_id=iid,reference_image_id=iid)
        code,j=call(18188,'/api/generate',same);expect('native shared map accepted',code,202)
        expect('shared input three leases',call(18188,'/api/inputs/'+iid)[1]['references'],3)
        expect('leased input delete refused',call(18188,'/api/inputs/'+iid,method='DELETE')[0],409)
        for _ in range(60):
            job=call(18188,'/api/jobs/'+j['job_id'])[1]
            if job['status']=='failed':break
            time.sleep(.05)
        expect('intentional engine failure',job['status'],'failed');expect('controlnet persisted',job['controlnet'],'canny')
        expect('no GPU success manufactured',job['gpu_controlnet_verified'],False)
        expect('all leases released after failure',call(18188,'/api/inputs/'+iid)[1]['references'],0)
        # Actual gateway forwards exactly one POST to the real native fixture.
        expect('forwarded native failure visible',call(18189,'/v1/images/generations',oa)[0],500)
        files=list((root/'jobs').glob('*.request.json'));expect('exactly two native accepted POSTs',len(files),2)
        forwarded=[json.loads(p.read_text()) for p in files if 'init_image_id' not in json.loads(p.read_text())]
        expect('one forwarded native request',len(forwarded),1);f=forwarded[0]
        expect('forwarded control model',f['controlnet'],'canny');expect('forwarded map ID',f['control_image_id'],iid);expect('forwarded strength',f['control_strength'],1)
        expect('input delete after jobs',call(18188,'/api/inputs/'+iid,method='DELETE')[0],200)
        assert not list((root/'output').glob('*.png')) and not list((fr/'output').glob('*.png'))
        result={'state':'passed','checks':checks,'count':len(checks),'Windows_inference':False,'PS5_connections':0,'PNG_generated':False}
        (T/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:
        for p,port in reversed(processes):
            try:call(port,'/api/admin/shutdown',b'',method='POST')
            except OSError:pass
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=3)
if __name__=='__main__':main()
