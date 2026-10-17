/* SPDX-License-Identifier: MIT
 * Caller already owns its existing journal lock. Keep current+previous files.
 * This manages CPU log descriptors only; no GPU/synchronization ownership.
 */
#ifndef PS5_AI_BOUNDED_JOURNAL_V1_H
#define PS5_AI_BOUNDED_JOURNAL_V1_H
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
struct ps5_ai_bounded_journal { uint64_t bytes, rotations; int ready; };
static int ps5_ai_journal_room(int *fd,const char *path,struct ps5_ai_bounded_journal *s,size_t needed,uint64_t limit) {
    if(!fd || *fd<0 || !path || !s || !needed || needed>limit || limit<4096){errno=EINVAL;return 0;}
    if(!s->ready){struct stat st;if(fstat(*fd,&st) || !S_ISREG(st.st_mode) || st.st_size<0)return 0;s->bytes=(uint64_t)st.st_size;s->ready=1;}
    if(s->bytes<=limit-needed)return 1;
    if(s->rotations==UINT64_MAX){errno=EOVERFLOW;return 0;}
    char previous[1024],part[1024],header[256];
    int a=snprintf(previous,sizeof(previous),"%s.previous",path);
    int b=snprintf(part,sizeof(part),"%s.next-%d-%llu",path,(int)getpid(),(unsigned long long)(s->rotations+1));
    int h=snprintf(header,sizeof(header),"BOUNDED_JOURNAL_CONTINUATION pid=%d rotation=%llu current_limit=%llu previous_segment_retained=1 older_history_discarded=1\n",(int)getpid(),(unsigned long long)(s->rotations+1),(unsigned long long)limit);
    if(a<0 || (size_t)a>=sizeof(previous) || b<0 || (size_t)b>=sizeof(part) || h<0 || (size_t)h>=sizeof(header) || (uint64_t)h>limit-needed){errno=EINVAL;return 0;}
    if(fsync(*fd))return 0;
    int next=open(part,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(next<0)return 0;
    size_t sent=0;int ok=1;
    while(sent<(size_t)h){ssize_t n=write(next,header+sent,(size_t)h-sent);if(n<0 && errno==EINTR)continue;if(n<=0){ok=0;break;}sent+=(size_t)n;}
    if(ok && fsync(next))ok=0;
    if(ok && rename(path,previous))ok=0;
    if(ok && rename(part,path))ok=0;
    if(!ok){int saved=errno;close(next);/* Keep only this known partial for diagnosis. */errno=saved;return 0;}
    int old=*fd;*fd=next;s->bytes=(uint64_t)h;s->rotations++;
    if(close(old))return 0;
    return 1;
}
#endif
