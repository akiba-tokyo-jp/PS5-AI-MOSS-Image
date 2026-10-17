"""Host-only HTTP contract tests. Fixture NEVER generates images or uses GPU."""
import concurrent.futures,hashlib,hmac,http.client,json,os,secrets,socket,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('WAI_TEST_OUT',str(ROOT/'.local/tests/gateway')));OUT.mkdir(parents=True,exist_ok=True)
CHECKS=[]
def check(value,name):
    if not value:raise AssertionError(name)
    CHECKS.append(name)

with tempfile.TemporaryDirectory(prefix='gateway-test-',dir=OUT) as folder:
    root=Path(folder);(root/'config').mkdir();token=secrets.token_hex(32)
    (root/'config/api-token').write_text(token,encoding='ascii')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    p=subprocess.Popen([str(OUT/'protocol-test.exe'),'--root',str(root),'--bind','127.0.0.1','--port',str(port)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    def request(method,path,body=None,auth=True):
        c=http.client.HTTPConnection('127.0.0.1',port,timeout=10)
        headers={'Content-Type':'application/json'}
        if auth:headers['Authorization']='Bearer '+token
        raw=None if body is None else json.dumps(body).encode()
        c.request(method,path,body=raw,headers=headers);r=c.getresponse();data=r.read();c.close()
        return r.status,json.loads(data)
    try:
        deadline=time.monotonic()+10
        while True:
            if p.poll() is not None:raise RuntimeError(p.stderr.read().decode(errors='replace'))
            try:
                if request('GET','/health')[0]==200:break
            except OSError:pass
            if time.monotonic()>deadline:raise TimeoutError('host fixture startup')
            time.sleep(.05)
        check(not (root/'config/api-token').exists(),'one-use startup token consumed')
        code,body=request('GET','/v1/models',auth=False)
        check(code==401 and body['error']['type']=='authentication_error','OpenAI-shaped unauthorized error')
        code,body=request('GET','/v1/models')
        check(code==200 and body['object']=='list' and body['data'][0]['id']=='wai-illustrious-sdxl-v17.0','model discovery')
        check(request('GET','/v1/models/nonexistent')[0]==404,'unknown model404')
        check(request('GET','/openapi.json')[1]==json.loads((ROOT/'docs/openapi.json').read_text()),'served OpenAPI matches source')
        invalid=[({'prompt':'x','n':2},'n'),({'prompt':'x','size':'999999999999999x64'},'size'),({'prompt':'x','size':'065x64'},'size'),({'prompt':None},'prompt'),({'prompt':'x','unknown':None},'unknown'),({'prompt':'x','stream':True},'stream'),({'prompt':'x','background':'transparent'},'background'),({'prompt':'x','model':'gpt-image-1'},'model')]
        for body,param in invalid:
            code,result=request('POST','/v1/images/generations',body)
            check(code==400 and result['error']['param']==param,'rejected parameter '+param)
        check(not list((root/'jobs').glob('*.request.json')),'invalid requests never enqueue inference')
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future=executor.submit(request,'POST','/v1/images/generations',{'prompt':'protocol fixture','size':'64x64','response_format':'b64_json'})
            deadline=time.monotonic()+3
            while not list((root/'jobs').glob('*.request.json')):
                if time.monotonic()>deadline:raise TimeoutError('queue admission')
                time.sleep(.02)
            started=time.monotonic();check(request('GET','/health')[0]==200 and time.monotonic()-started<1,'health responsive while synchronous image request waits')
            check(request('POST','/v1/images/generations',{'prompt':'second'})[0]==429,'bounded synchronous concurrency429')
            code,body=future.result()
            check(code==500 and body['error']['type']=='server_error','backend failure propagates without fake image')
        check(len(list((root/'jobs').glob('*.request.json')))==1 and not list((root/'output').glob('*.png')),'one host fixture job; zero image files')
        code,_=request('POST','/api/admin/shutdown')
        check(code==202,'idle gateway shutdown')
        p.communicate(timeout=5);check(p.returncode==0,'cooperative CPU process exit')
    finally:
        if p.poll() is None:p.terminate();p.communicate(timeout=5)

# Compile helper unit checks from the same source; no network/inference calls.
unit=OUT/'protocol-unit.c'
unit.write_text('''#define main gateway_main
#include "server.c"
#undef main
int main(void){
 strcpy(api_token,"fixture key, never a real credential");token_len=strlen(api_token);server_started_at=42;
 char sig[65];image_signature("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",123,sig);puts(sig);
 unsigned char input[16384];char output[21848];for(size_t i=0;i<sizeof(input);i++)input[i]=(unsigned char)i;
 const size_t sizes[]={0,1,2,3,4,257,12288,12289,16384};
 for(size_t k=0;k<sizeof(sizes)/sizeof(sizes[0]);k++){size_t n=base64_block(input,sizes[k],output);struct ai_sha256 s;unsigned char d[32];ai_sha256_init(&s);ai_sha256_update(&s,output,n);ai_sha256_final(&s,d);for(size_t i=0;i<32;i++)printf("%02x",d[i]);puts("");}
 return 0;}
''',encoding='utf-8')
compiler=Path(os.environ['WAI_TEST_LLVM'])/'clang.exe'
cmd=[str(compiler),'--target=x86_64-w64-mingw32','-O2','-DPS5_AI_GATEWAY_PROTOCOL_TEST',str(unit),str(ROOT/'native/gateway/cpu_hooks.c'),str(ROOT/'native/sha256/sha256.c'),'-I',str(ROOT/'native/gateway'),'-I',str(ROOT/'native/sha256'),'-lws2_32','-o',str(OUT/'protocol-unit.exe')]
subprocess.run(cmd,check=True,capture_output=True)
lines=subprocess.check_output([str(OUT/'protocol-unit.exe')],text=True).splitlines()
check(lines[0]==hmac.new(b'fixture key, never a real credential',b'ps5-image-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:123:42',hashlib.sha256).hexdigest(),'HMAC-SHA256 matches Python standard library')
import base64
for n,line in zip([0,1,2,3,4,257,12288,12289,16384],lines[1:]):
    check(line==hashlib.sha256(base64.b64encode(bytes(i%256 for i in range(n)))).hexdigest(),'base64 matches Python at length '+str(n))
report={'schema':'openai-gateway-host-protocol-tests-v1','success':True,'checks':CHECKS,'inference_performed':False,'GPU_used':False}
(OUT/'host-protocol-tests.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
