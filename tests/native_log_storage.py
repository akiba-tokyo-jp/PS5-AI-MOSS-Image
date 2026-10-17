"""Meaningful host fault/rotation checks. No inference or real GPU calls."""
import os,hashlib,http.client,json,secrets,socket,subprocess,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'native/app'
OUT=Path(os.environ.get('WAI_TEST_FAULT_OUT',str(ROOT/'.local/tests/fault')));OUT.mkdir(parents=True,exist_ok=True)
CLANG=Path(os.environ['WAI_TEST_LLVM'])/'clang.exe'
checks=[]
def check(x,message):
    if not x:raise AssertionError(message)
    checks.append(message)
def compile(args):
    p=subprocess.run([str(CLANG),'--target=x86_64-w64-mingw32','-std=gnu11','-O2','-Wall','-Wextra','-Werror',*map(str,args)],capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stdout+p.stderr)
exe=OUT/'native-retained-fault-fixture.exe'
compile([SRC/'server.c',ROOT/'native/gateway/cpu_hooks.c','-I',SRC,'-DPS5_AI_RETAIN_FAULT_TEST','-lws2_32','-o',exe])
with tempfile.TemporaryDirectory(dir=OUT,prefix='http-') as folder:
    root=Path(folder);(root/'config').mkdir();token=secrets.token_hex(32);(root/'config/api-token').write_text(token)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    p=subprocess.Popen([str(exe),'--root',str(root),'--bind','127.0.0.1','--port',str(port)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    def request(method,path,body=None):
        c=http.client.HTTPConnection('127.0.0.1',port,timeout=5);headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}
        c.request(method,path,None if body is None else json.dumps(body).encode(),headers);r=c.getresponse();raw=r.read();c.close();return r.status,json.loads(raw)
    try:
        deadline=time.monotonic()+10
        while True:
            if p.poll() is not None:raise RuntimeError(p.stderr.read().decode(errors='replace'))
            try:
                if request('GET','/health')[0]==200:break
            except OSError:pass
            if time.monotonic()>deadline:raise TimeoutError('fixture startup')
            time.sleep(.05)
        code,j=request('POST','/api/generate',{'prompt':'host fault fixture','width':64,'height':64,'steps':1,'cfg_scale':5,'seed':1,'sampler':'euler_a'})
        check(code==202,'one fixture job accepted')
        deadline=time.monotonic()+5
        while True:
            code,last=request('GET','/api/jobs/'+j['job_id'])
            if last['status']=='failed':break
            if time.monotonic()>deadline:raise TimeoutError('retained worker still looks running')
            time.sleep(.05)
        check('injected_host_ENOSPC' in last['error'] and not last['gpu_used'],'parked worker publishes failed job without image success')
        code,h=request('GET','/health');check(h['status']=='degraded' and not h['engine_available'] and not h['gpu_available'],'health reflects retained engine failure')
        check(request('POST','/api/generate',{'prompt':'second'})[0]==503,'new generation rejected after retained failure')
        start=time.monotonic();code,_=request('POST','/api/admin/shutdown')
        check(code==409 and time.monotonic()-start<2,'shutdown refuses impossible worker join promptly')
        check(not list((root/'output').glob('*.png')),'host fixture produces zero PNGs')
    finally:
        p.terminate();p.communicate(timeout=5)

unit=OUT/'journal-test.c'
unit.write_text(r'''
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>
#define O_NOFOLLOW 0
static int injected_no_space;
static ssize_t fixture_write(int fd,const void *data,size_t n){if(injected_no_space){errno=ENOSPC;return -1;}return write(fd,data,n);}
static int fixture_rename(const char *a,const char *b){if(MoveFileExA(a,b,MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))return 0;errno=EIO;return -1;}
/* Model POSIX rename-while-open semantics; ordinary Windows CRT open lacks
 * FILE_SHARE_DELETE. This changes the host fixture only. */
static int fixture_open(const char *path,int flags,int mode){(void)mode;HANDLE h=CreateFileA(path,GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,NULL,(flags&O_EXCL)?CREATE_NEW:OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,NULL);if(h==INVALID_HANDLE_VALUE){errno=EIO;return -1;}int fd=_open_osfhandle((intptr_t)h,flags|O_BINARY);if(fd<0)CloseHandle(h);return fd;}
#define write fixture_write
#define rename fixture_rename
#define open fixture_open
#define fsync _commit
#include "bounded_journal.h"
#undef write
#undef rename
int main(int argc,char **argv){assert(argc==2);char path[1024],previous[1100];assert(snprintf(path,sizeof(path),"%s/journal.log",argv[1])>0);snprintf(previous,sizeof(previous),"%s.previous",path);
 int fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_BINARY,0600);assert(fd>=0);struct ps5_ai_bounded_journal cap={0};char line[512];memset(line,'A',sizeof(line));line[511]='\n';
 for(int i=0;i<100;i++){assert(ps5_ai_journal_room(&fd,path,&cap,sizeof(line),4096));assert(write(fd,line,sizeof(line))==(ssize_t)sizeof(line));cap.bytes+=sizeof(line);}
 struct stat a,b;assert(!stat(path,&a)&&!stat(previous,&b));assert(a.st_size<=4096&&b.st_size<=4096&&cap.rotations>10);
 printf("rotations=%llu current=%lld previous=%lld\n",(unsigned long long)cap.rotations,(long long)a.st_size,(long long)b.st_size);
 const off_t old=a.st_size;cap.bytes=4096;injected_no_space=1;errno=0;assert(!ps5_ai_journal_room(&fd,path,&cap,sizeof(line),4096)&&errno==ENOSPC);assert(!stat(path,&a)&&a.st_size==old);assert(close(fd)==0);puts("PASS: ENOSPC preserves original current file; no GPU call or cleanup");return 0;}
''',encoding='utf-8')
binary=OUT/'journal-test.exe';compile([unit,'-I',SRC,'-o',binary])
with tempfile.TemporaryDirectory(dir=OUT,prefix='rotation-') as folder:
    result=subprocess.check_output([str(binary),folder],text=True)
    check('PASS: ENOSPC preserves original' in result,'multiple journal rotations retain bounded current/previous and preserve current on injected ENOSPC')
report={'schema':'native-wai-log-storage-host-tests-v1','state':'PASS','checks':checks,'real_GPU_calls':0,'inference_performed':False,'rotation_result':result,'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [SRC/'server.c',SRC/'retained_fault.inc',SRC/'bounded_journal.h']]}
(OUT/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
