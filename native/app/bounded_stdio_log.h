/* SPDX-License-Identifier: MIT
 * Caller owns its existing log mutex. CPU log files only; no GPU operations.
 * Keep current and previous, each at most limit bytes for new bounded logs.
 */
#ifndef PS5_AI_BOUNDED_STDIO_LOG_H
#define PS5_AI_BOUNDED_STDIO_LOG_H
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
static int ps5_ai_bounded_stdio_append(FILE **file,const char *path,
                                     const char *line,size_t bytes,uint64_t limit) {
    if(!file||!path||!line||!bytes||bytes>limit)return 0;
    if(!*file)*file=fopen(path,"ab");
    if(!*file)return 0;
    struct stat st;
    if(fstat(fileno(*file),&st)||!S_ISREG(st.st_mode)||st.st_size<0)return 0;
    if((uint64_t)st.st_size>limit-bytes) {
        char previous[2048];int n=snprintf(previous,sizeof(previous),"%s.previous",path);
        if(n<0||(size_t)n>=sizeof(previous))return 0;
        int flushed=fflush(*file),closed=fclose(*file);*file=NULL;
        if(flushed||closed)return 0;
#ifdef _WIN32
        if(!MoveFileExA(path,previous,MOVEFILE_REPLACE_EXISTING))return 0;
#else
        if(rename(path,previous))return 0;
#endif
        *file=fopen(path,"ab");if(!*file)return 0;
        if(fstat(fileno(*file),&st)||!S_ISREG(st.st_mode)||st.st_size<0||
           (uint64_t)st.st_size>limit-bytes)return 0;
    }
    return fwrite(line,1,bytes,*file)==bytes && !fflush(*file);
}
#endif
