"""Run the real PowerShell client against an intentionally failing host engine."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import controlnet_protocol as h
R=Path(__file__).resolve().parents[1]
h.T=Path(os.environ['PS5_WAI_TEST_OUT'])
if len(sys.argv)>1:h.T=Path(sys.argv[1])
def main():
    h.T.mkdir();processes=[]
    try:
        native,root=h.launch('native','native-failure.exe',18188);processes.append((native,18188))
        gateway,_=h.launch('gateway','gateway-forwarder.exe',18189);processes.append((gateway,18189))
        shell=Path(os.environ.get('PS5_WAI_POWERSHELL') or shutil.which('pwsh') or shutil.which('powershell'))
        control_map=h.T/'control-map.png';h.Image.new('RGB',(256,256),'black').save(control_map)
        client=R/'scripts/Invoke-WaiControlNet.ps1'
        token=h.T/'client-token';token.write_text(h.TOKEN+'\n')
        args=[shell,'-NoProfile','-File',client,'-ControlNet','canny','-ControlImage',control_map,
              '-Prompt','Host client protocol test only','-MangaStrength','0','-BaseUrl','http://127.0.0.1:18189/v1','-NativeBaseUrl','http://127.0.0.1:18188','-TokenFile',token,'-OutputDirectory',h.T/'output']
        result=subprocess.run(list(map(str,args)),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
        raw=result.stdout+result.stderr;(h.T/'client.private.log').write_text(raw,encoding='utf-8')
        assert result.returncode!=0 and 'Expected host-only' in raw and 'failure; no inference performed' in raw,raw[-2000:]
        assert h.TOKEN not in raw
        requests=list((root/'jobs').glob('*.request.json'));assert len(requests)==1
        request=json.loads(requests[0].read_text());assert request['controlnet']=='canny' and request['width']==request['height']==256 and request['control_strength']==1
        record={'state':'passed','actual_PowerShell_client_executed':True,'generation_POSTs':1,'automatic_retries':0,
                'expected_engine_failure_observed':True,'credential_output':False,'Windows_inference':False,'PS5_connections':0,
                'client_sha256':hashlib.sha256(client.read_bytes()).hexdigest()}
        (h.T/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    finally:
        for p,port in reversed(processes):
            try:h.call(port,'/api/admin/shutdown',b'',method='POST')
            except OSError:pass
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=3)
if __name__=='__main__':main()
