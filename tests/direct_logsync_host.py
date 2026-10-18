# SPDX-License-Identifier: MIT
"""Exercise actual pool journal functions with real host files, no GPU/inference."""
from pathlib import Path
import subprocess,os,json,hashlib,argparse
R=Path(__file__).resolve().parents[1];S=R/'native/app'
parser=argparse.ArgumentParser();parser.add_argument('--compiler',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);ARGS=parser.parse_args();O=ARGS.out.resolve()
def main():
 O.mkdir(parents=True,exist_ok=True);(O/'temp').mkdir(exist_ok=True)
 source=(S/'pool.c').read_text();actual=source[source.index('static void acquire('):source.index('static int open_journal(')]
 pre=r'''#include "phase_profile.h"
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <assert.h>
#include <windows.h>
static int active=1,debug_mode=0,syncs=0,fail_sync=0,retained=0;
static atomic_bool log_ok=ATOMIC_VAR_INIT(true);
static atomic_flag journal_lock=ATOMIC_FLAG_INIT;
static int journal=-1;
int ps5_ai_profile_enabled(void){return active;}
void ps5_ai_profile_add(unsigned i,uint64_t n,uint64_t b,int v){(void)i;(void)n;(void)b;assert(v);}
bool ps5_ai_native_journal_debug_enabled(void){return debug_mode;}
void ps5_ai_flex_retain_fault(const char *reason){assert(reason);retained++;}
static int checked_sync(int fd){syncs++;if(fail_sync){errno=EIO;return -1;}return _commit(fd);}
#define fsync checked_sync
#define PS5_AI_DIRECT_POOL_HOST_TEST 1
'''
 test=r'''
int main(int argc,char**argv){assert(argc==2);journal=_open(argv[1],_O_RDWR|_O_CREAT|_O_TRUNC|_O_BINARY,_S_IREAD|_S_IWRITE);assert(journal>=0);
assert(trace("BEGIN host_file_test GPU_calls=0"));assert(syncs==1);
for(int i=0;i<127;i++)assert(trace("alloc pointer=host bytes=%d",i));assert(syncs==1);
assert(trace("free pointer=host"));assert(syncs==2);
assert(ps5_ai_direct_pool_log_checkpoint());assert(syncs==3);
debug_mode=1;assert(trace("alloc pointer=debug"));assert(syncs==4);
debug_mode=0;active=0;assert(trace("free pointer=outside_job"));assert(syncs==5);
active=1;assert(trace("alloc pointer=new_job"));assert(syncs==6);
assert(trace("free pointer=pending"));assert(syncs==6);Sleep(1100);assert(trace("free pointer=time_boundary"));assert(syncs==7);
for(int i=0;i<66;i++)assert(trace("alloc pointer=%0990d",i));assert(syncs>=8);assert(ps5_ai_direct_pool_log_checkpoint());
char bytes[200000];assert(lseek(journal,0,SEEK_SET)==0);int n=read(journal,bytes,sizeof(bytes));assert(n>65536);int lines=0;for(int i=0;i<n;i++)if(bytes[i]=='\n')lines++;assert(lines==200);
assert(lseek(journal,0,SEEK_END)>=0);fail_sync=1;assert(!trace("fault_retained=1 host_injected_storage_error"));assert(!atomic_load(&log_ok));assert(!ps5_ai_direct_pool_log_checkpoint()&&retained==1);
close(journal);return 0;
}
'''
 (O/'test.c').write_text(pre+actual+test)
 env=dict(os.environ);env['PATH']=str(ARGS.compiler.resolve().parent)+os.pathsep+env['PATH'];env['TMP']=env['TEMP']=str(O/'temp')
 args=[ARGS.compiler.resolve(),'-std=c11','-O2','-Wall','-Wextra','-Werror','-UNDEBUG','-I',S,O/'test.c','-lwinpthread','-o',O/'test.exe']
 p=subprocess.run(list(map(str,args)),env=env,capture_output=True,text=True);assert p.returncode==0,p.stderr
 subprocess.run([str(O/'test.exe'),str(O/'actual-records.private.log')],env=env,check=True)
 r={'state':'passed','actual_source_fragment_sha256':hashlib.sha256(actual.encode()).hexdigest(),'checks':['128record flush bound','64KiB flush bound','1second boundary checked on next record','All200 checked records preserved','Debug and outside-job writes remain immediately synced','Required final checkpoint','Injected storage failure remains sticky and retains resources'],'Windows_inference':False,'PS5_connections':0}
 (O/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
