"""Actual shared API log rotation and rename-failure check; host only."""
import json,os,subprocess,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=Path(os.environ['WAI_TEST_STDIO_OUT']);O.mkdir(parents=True,exist_ok=True)
unit=O/'test.c';unit.write_text(r'''
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <assert.h>
#include <direct.h>
#include "bounded_stdio_log.h"
int main(int argc,char **argv){assert(argc==2);char path[1024],prev[1100];snprintf(path,sizeof(path),"%s/api.log",argv[1]);snprintf(prev,sizeof(prev),"%s.previous",path);FILE *f=NULL;char line[1024];memset(line,'A',sizeof(line));line[1023]='\n';
 for(unsigned i=0;i<100;i++){assert(ps5_ai_bounded_stdio_append(&f,path,line,sizeof(line),4096));struct stat a,b;assert(!stat(path,&a)&&a.st_size<=4096);if(i>=4)assert(!stat(prev,&b)&&b.st_size<=4096);}
 assert(!remove(prev)&&!mkdir(prev));assert(!ps5_ai_bounded_stdio_append(&f,path,line,sizeof(line),4096)&&f==NULL);struct stat st;assert(!stat(path,&st)&&st.st_size==4096);
 puts("PASS: repeated rotations keep both files <= limit; failed rename leaves full current unchanged and append fails; NULL-safe caller required; no GPU calls");}
''',encoding='utf8')
cc=Path(os.environ['WAI_TEST_LLVM'])/'clang.exe';exe=O/'test.exe'
p=subprocess.run([str(cc),'--target=x86_64-w64-mingw32','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-I',str(R/'native/app'),str(unit),'-o',str(exe)],capture_output=True,text=True);assert p.returncode==0,p.stderr
with tempfile.TemporaryDirectory(dir=O) as tmp:
 p=subprocess.run([str(exe),tmp],capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr
receipt={'state':'PASS','actual_source':'bounded_stdio_log.h','real_GPU_calls':0,'result':p.stdout.strip()};(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
