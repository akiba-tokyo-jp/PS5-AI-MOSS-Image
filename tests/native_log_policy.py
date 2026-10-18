"""Exercise the actual policy parser and driver logger with CPU-only fault mocks."""
import json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'native/app';O=Path(os.environ['WAI_TEST_LOG_OUT']);O.mkdir(parents=True,exist_ok=True)
B=Path(os.environ['WAI_TEST_LLVM'])
parser=O/'policy.cpp'
parser.write_text(r'''
#include "log_policy.h"
#include <cassert>
#include <cstdio>
int main(){unsigned checks=0;
 for(auto p:{"{}","{\"logging\":{}}","{\"logging\":{\"level\":\"normal\"}}"}){assert(!ps5_ai_log_policy_debug(nlohmann::json::parse(p)));checks++;}
 assert(ps5_ai_log_policy_debug(nlohmann::json::parse("{\"logging\":{\"level\":\"debug\"}}")));checks++;
 for(auto p:{"{\"logging\":null}","{\"logging\":[]}","{\"logging\":\"debug\"}","{\"logging\":{\"level\":1}}","{\"logging\":{\"level\":\"off\"}}"}){bool failed=false;try{(void)ps5_ai_log_policy_debug(nlohmann::json::parse(p));}catch(...){failed=true;}assert(failed);checks++;}
 printf("policy_checks=%u\n",checks);}
''',encoding='utf8')
driver=O/'driver.c'
driver.write_text(r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include "journal_metrics.h"
#define PS5_AI_BOUNDED_JOURNAL_V1_H
struct ps5_ai_bounded_journal{uint64_t bytes,rotations;int ready;};
static int arena_fd=3;
static atomic_bool arena_log_healthy=true;
static atomic_flag arena_log_mutex=ATOMIC_FLAG_INIT;
static bool debug_mode,fail_sync;
static unsigned writes,syncs;
static uint64_t now_ns=1000;
bool ps5_ai_native_journal_debug_enabled(void){return debug_mode;}
static int mock_clock(clockid_t id,struct timespec *t){(void)id;t->tv_sec=now_ns/1000000000;t->tv_nsec=now_ns%1000000000;return 0;}
static int mock_sync(int fd){assert(fd==3);syncs++;if(fail_sync){errno=EIO;return -1;}return 0;}
static ssize_t mock_write(int fd,const void *data,size_t n){assert(fd==2||fd==3);assert(data&&n);writes++;return (ssize_t)n;}
static int ps5_ai_journal_room(int *fd,const char *path,struct ps5_ai_bounded_journal *s,size_t n,uint64_t limit){(void)s;assert(*fd==3&&path&&n&&limit==4194304);return 1;}
#define clock_gettime mock_clock
#define fsync mock_sync
#define write mock_write
#include "driver_journal_impl.inc"
int main(){
 int touched=0;
 for(unsigned i=0;i<10000;i++)assert(ps5_ai_arena_trace_routine("ignored %n",&touched));
 assert(!touched&&!writes&&!syncs&&ps5_ai_native_arena_journal_suppressed()==10000);
 assert(ps5_ai_arena_trace("critical_normal boundary=1")&&writes==1&&syncs==1);
 debug_mode=true;unsigned prior=syncs;
 for(unsigned i=0;i<1023;i++)assert(ps5_ai_arena_trace_routine("success %u",i));
 assert(syncs==prior);assert(ps5_ai_arena_trace_routine("success last")&&syncs==prior+1);
 assert(ps5_ai_arena_trace("critical_debug boundary=1")&&syncs==prior+2);
 now_ns+=1000000000;prior=syncs;assert(ps5_ai_arena_trace_routine("elapsed")&&syncs==prior+1);
 debug_mode=false;prior=syncs;assert(ps5_ai_native_arena_journal_checkpoint()&&syncs==prior+1);
 fail_sync=true;assert(!ps5_ai_arena_trace("critical_failure injected_fsync=1")&&!atomic_load(&arena_log_healthy));
 assert(!ps5_ai_arena_trace_routine("sticky unhealthy must not become success"));
 puts("PASS: normal omits formatting/write/sync; debug batches; critical/checkpoint durable; fsync failure sticky; no GPU calls");
}
''',encoding='utf8')
env=dict(os.environ);env['PATH']=str(B)+os.pathsep+env.get('PATH','');outputs=[]
for source,cc,extra in [(parser,'clang++.exe',['-std=c++20','-isystem',str(Path(os.environ['WAI_TEST_SD'])/'thirdparty')]),(driver,'clang.exe',['-std=gnu11'])]:
    exe=source.with_suffix('.exe');command=[str(B/cc),'--target=x86_64-w64-mingw32','-O2','-Wall','-Wextra','-Werror','-I',str(S),*extra,str(source),'-o',str(exe)]
    p=subprocess.run(command,env=env,capture_output=True,text=True);assert p.returncode==0,p.stderr
    p=subprocess.run([str(exe)],env=env,capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr;outputs.append(p.stdout.strip())
receipt={'state':'PASS','real_GPU_calls':0,'actual_source_includes':['log_policy.h','driver_journal_impl.inc'],'results':outputs}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
