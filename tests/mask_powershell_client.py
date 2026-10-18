"""Actual Windows PowerShell mask client; failing host fixture, no inference."""
import io
import json
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
from PIL import Image
import mask_api_protocol as t

R=Path(__file__).resolve().parents[1]
with socket.socket() as s:s.bind(('127.0.0.1',18188))
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
with tempfile.TemporaryDirectory(prefix='powershell-mask-',dir=t.OUT/'host') as folder:
    root=Path(folder);token=secrets.token_hex(32);key=root/'client-key';key.write_text(token)
    buf=io.BytesIO();Image.new('RGB',(256,256),(20,100,220)).save(buf,format='PNG')
    image=root/'input.png';image.write_bytes(buf.getvalue())
    processes=[]
    try:
        for name,p in [('native-failure.exe',18188),('gateway-forwarder.exe',port)]:
            target=root/name;(target/'config').mkdir(parents=True);(target/'config/api-token').write_text(token)
            proc=subprocess.Popen([str(t.OUT/'host'/name),'--root',str(target),'--bind','127.0.0.1','--port',str(p)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            processes.append(proc);t.wait_health(p,token,proc)
        for index,mask_mode in enumerate(('alpha','grayscale')):
            mask=root/(mask_mode+'.png')
            im=Image.new('RGBA' if mask_mode=='alpha' else 'L',(256,256),(255,255,255,0) if mask_mode=='alpha' else 255)
            im.save(mask,format='PNG')
            result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(R/'scripts/Invoke-WaiReference.ps1'),
                '-Image',str(image),'-Mask',str(mask),'-MaskMode',mask_mode,'-Prompt','PowerShell masked host fixture','-Mode','inpaint',
                '-BaseUrl',f'http://127.0.0.1:{port}/v1','-TokenFile',str(key),'-OutputDirectory',str(root/'output'),'-MangaStrength','0'],capture_output=True,timeout=60)
            raw=result.stdout+result.stderr
            assert result.returncode!=0 and b'Native job failed' in raw,'Actual PowerShell mask did not reach native fixture: '+raw.decode(errors='replace')
            assert token.encode() not in raw,'Client emitted credential'
            jobs=list((root/'native-failure.exe/jobs').glob('*.request.json'))
            assert len(jobs)==index+1,'Exactly one POST per PowerShell invocation'
            bodies=[json.loads(p.read_text()) for p in jobs]
            assert all(b['init_image_id'] and b['mask_image_id'] and not b.get('reference_image_id') and b['manga_strength']==0 for b in bodies)
            assert not list((root/'native-failure.exe/inputs').glob('*.rgb'))
            assert not list((root/'native-failure.exe/inputs').glob('*.mask'))
            assert not list(root.rglob('output/*.png'))
        for p,proc in [(port,processes[1]),(18188,processes[0])]:
            assert t.request(p,token,'POST','/api/admin/shutdown')[0]==202
            proc.communicate(timeout=5);assert proc.returncode==0
    finally:
        for proc in processes:
            if proc.poll() is None:proc.terminate();proc.communicate(timeout=5)
receipt={'actual_Windows_PowerShell_client_verified':True,'mask_modes':['alpha','grayscale'],
         'generation_POST_count':2,'automatic_retries':0,'credential_output':False,'Windows_inference':False,
         'PS5_connections':0,'generated_PNGs':0,'expected_host_engine_failure_propagated':True}
(t.OUT/'host/powershell-mask-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
