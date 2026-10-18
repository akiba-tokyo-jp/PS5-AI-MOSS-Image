/* SPDX-License-Identifier: MIT
 * Bounded, PNG/JPEG-only decode. Normalized RGB is the only stored input.
 */
#include "reference_image.h"
#include "sha256.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#define O_NOFOLLOW 0
#endif
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#define STBI_MAX_DIMENSIONS PS5_AI_INPUT_MAX_DIMENSION
/* Decoding is serialized by the HTTP accept loop. Bound malformed PNG zlib
 * expansion as well as dimensions; do not trust only the advertised IHDR. */
static size_t decoder_live_bytes;
static void *decoder_alloc(size_t size) {
    if(size>64U*1024U*1024U || decoder_live_bytes>64U*1024U*1024U-size)return NULL;
    size_t *p=malloc(sizeof(size_t)+size);if(!p)return NULL;
    *p=size;decoder_live_bytes+=size;return p+1;
}
static void decoder_free(void *data) {
    if(data){size_t *p=(size_t *)data-1;decoder_live_bytes-=*p;free(p);}
}
static void *decoder_realloc(void *data,size_t size) {
    if(!data)return decoder_alloc(size);
    size_t *old=(size_t *)data-1,old_size=*old;
    if(size>64U*1024U*1024U || decoder_live_bytes-old_size>64U*1024U*1024U-size)return NULL;
    size_t *p=realloc(old,sizeof(size_t)+size);if(!p)return NULL;
    *p=size;decoder_live_bytes=decoder_live_bytes-old_size+size;return p+1;
}
#define STBI_MALLOC(n) decoder_alloc(n)
#define STBI_REALLOC(p,n) decoder_realloc(p,n)
#define STBI_FREE(p) decoder_free(p)
#include "stb_image.h"

static int dimensions(uint32_t w, uint32_t h) {
    return w && h && w <= PS5_AI_INPUT_MAX_DIMENSION && h <= PS5_AI_INPUT_MAX_DIMENSION &&
           (uint64_t)w*h <= PS5_AI_INPUT_MAX_PIXELS;
}
static void encode32(unsigned char *p, uint32_t n) {
    for (unsigned i=0;i<4;i++) p[i]=(unsigned char)(n>>(8*i));
}
static uint32_t decode32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static FILE *open_read(const char *path) {
#ifdef _WIN32
    int fd=_open(path,_O_RDONLY|_O_BINARY);
#else
    int fd=open(path,O_RDONLY|O_NOFOLLOW);
#endif
    if(fd<0)return NULL;
    struct stat st;
    if(fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_size<16 ||
       (uint64_t)st.st_size>16+3*(uint64_t)PS5_AI_INPUT_MAX_PIXELS){
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return NULL;
    }
    FILE *f=fdopen(fd,"rb");
    if(!f){
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
    }
    return f;
}
static int header(FILE *f,uint32_t *w,uint32_t *h) {
    unsigned char buf[16];
    if(fread(buf,1,16,f)!=16 || memcmp(buf,"PS5RGB1\0",8))return 0;
    *w=decode32(buf+8);*h=decode32(buf+12);
    return dimensions(*w,*h);
}
int ps5_ai_input_decode(const unsigned char *data,size_t size,unsigned char **out,uint32_t *w,uint32_t *h) {
    *out=NULL;
    int png=size>=8 && !memcmp(data,"\211PNG\r\n\032\n",8);
    int jpeg=size>=3 && data[0]==255 && data[1]==216 && data[2]==255;
    if(!size || size>PS5_AI_INPUT_MAX_BYTES || (!png && !jpeg))return 0;
    int x,y,c;
    if(!stbi_info_from_memory(data,(int)size,&x,&y,&c) || !dimensions((uint32_t)x,(uint32_t)y))return 0;
    unsigned char *rgba=stbi_load_from_memory(data,(int)size,&x,&y,&c,4);
    if(!rgba || !dimensions((uint32_t)x,(uint32_t)y)){stbi_image_free(rgba);return 0;}
    size_t pixels=(size_t)x*y;
    unsigned char *rgb=malloc(pixels*3);
    if(!rgb){stbi_image_free(rgba);return 0;}
    for(size_t i=0;i<pixels;i++)for(unsigned k=0;k<3;k++)
        rgb[3*i+k]=(unsigned char)((rgba[4*i+k]*rgba[4*i+3]+255U*(255U-rgba[4*i+3])+127U)/255U);
    stbi_image_free(rgba);*out=rgb;*w=(uint32_t)x;*h=(uint32_t)y;return 1;
}
int ps5_ai_input_write(const char *path,const unsigned char *rgb,uint32_t w,uint32_t h) {
    if(!dimensions(w,h))return 0;
#ifdef _WIN32
    int fd=_open(path,_O_WRONLY|_O_CREAT|_O_EXCL|_O_BINARY,_S_IREAD|_S_IWRITE);
#else
    int fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
#endif
    if(fd<0)return 0;
    FILE *f=fdopen(fd,"wb");
    if(!f){
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        remove(path);return 0;
    }
    unsigned char buf[16];memcpy(buf,"PS5RGB1\0",8);encode32(buf+8,w);encode32(buf+12,h);
    size_t n=(size_t)w*h*3;
    int ok=fwrite(buf,1,16,f)==16 && fwrite(rgb,1,n,f)==n && !fflush(f);
#ifdef _WIN32
    if(ok && _commit(fd))ok=0;
#else
    if(ok && fsync(fd))ok=0;
#endif
    if(fclose(f))ok=0;
    if(!ok)remove(path);return ok;
}
int ps5_ai_input_read(const char *path,unsigned char **out,uint32_t *w,uint32_t *h) {
    *out=NULL;FILE *f=open_read(path);if(!f)return 0;
    int ok=header(f,w,h);size_t n=ok?(size_t)(*w)*(*h)*3:0;
    unsigned char *rgb=ok?malloc(n):NULL;
    ok=rgb && fread(rgb,1,n,f)==n && fgetc(f)==EOF && !ferror(f);
    if(fclose(f))ok=0;
    if(!ok){free(rgb);return 0;}*out=rgb;return 1;
}
int ps5_ai_input_info(const char *path,uint32_t *w,uint32_t *h,uint64_t *bytes,char hash[65]) {
    // A cached input is checked again during native startup. Use the same
    // raw-fd access that already verified the bootstrap/config on this PS5;
    // no fdopen/rewind or shared stdio positioning state is involved.
    *w=*h=0;*bytes=0;hash[0]=0;
#ifdef _WIN32
    int fd=_open(path,_O_RDONLY|_O_BINARY);
#else
    int fd=open(path,O_RDONLY|O_NOFOLLOW);
#endif
    if(fd<0)return 0;
    struct stat before,after;
    int ok=!fstat(fd,&before)&&S_ISREG(before.st_mode)&&before.st_size>=16&&
        (uint64_t)before.st_size<=16+3*(uint64_t)PS5_AI_INPUT_MAX_PIXELS;
    struct ai_sha256 ctx;ai_sha256_init(&ctx);
    unsigned char chunk[16384],digest[32];uint64_t total=0,expected=16;
    while(ok&&total<expected){
        size_t want=expected-total;if(want>sizeof(chunk))want=sizeof(chunk);
#ifdef _WIN32
        int n=_read(fd,chunk,(unsigned)want);
#else
        ssize_t n=read(fd,chunk,want);
#endif
        if(n<0&&errno==EINTR)continue;
        if(n<=0){ok=0;break;}
        // Read the header exactly before accepting the payload dimensions.
        if(total<16){
            // The kernel may return a short read; collect the small header.
            memcpy(digest+(size_t)total,chunk,(size_t)n);
        }
        ai_sha256_update(&ctx,chunk,(size_t)n);total+=(uint64_t)n;
        if(total==16&&expected==16){
            *w=decode32(digest+8);*h=decode32(digest+12);
            if(memcmp(digest,"PS5RGB1\0",8)||!dimensions(*w,*h)){ok=0;break;}
            expected=16+3*(uint64_t)(*w)*(*h);
            if((uint64_t)before.st_size!=expected){ok=0;break;}
        }
    }
    unsigned char extra;
#ifdef _WIN32
    int eof=ok?_read(fd,&extra,1):-1;
#else
    ssize_t eof=ok?read(fd,&extra,1):-1;
#endif
    if(total!=expected||eof!=0||fstat(fd,&after)||before.st_size!=after.st_size||before.st_dev!=after.st_dev||before.st_ino!=after.st_ino)ok=0;
#ifdef _WIN32
    if(_close(fd))ok=0;
#else
    if(close(fd))ok=0;
#endif
    *bytes=ok?expected:0;
    if(ok){ai_sha256_final(&ctx,digest);for(unsigned i=0;i<32;i++)sprintf(hash+2*i,"%02x",digest[i]);}
    return ok;
}
