/* SPDX-License-Identifier: GPL-3.0-or-later
 * Optional CPU VideoOut adapter. Tiling/glyph presentation follows the GPL3+
 * psllama status_screen and PS5_Vulkan demo renderer, with guarded ownership.
 * No constructor, AGC/Vulkan call, compute queue, engine call, free or teardown.
 */
#include "status.h"
#include "font.h"
#include "raster.h"
#include <ps5platform/videoout.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

size_t sceKernelGetDirectMemorySize(void);
int sceKernelAllocateDirectMemory(int64_t,int64_t,size_t,size_t,int,int64_t *);
int sceKernelMapDirectMemory(void **,size_t,int,int,int64_t,size_t);
/* Typed public API and order used by the pinned psllama/PS5_Vulkan renderers. */
int sceSystemServiceHideSplashScreen(void);
void ps5_ai_ggml_report(const char *, ...);

#define FRAME_BYTES ((size_t)PS5_AI_STATUS_FRAME_BYTES)
#define TOTAL_BYTES ((size_t)PS5_AI_STATUS_SCANOUT_BYTES)
#define ALIGNMENT ((size_t)2097152)
#define GUARD UINT64_C(0x81a29c7ed41f362b)
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static pthread_t owner_thread;
static atomic_int started,stop_requested;
static struct ps5_ai_status_snapshot_v1 latest;
static struct ps5_ai_status_display_state_v1 state={.abi_bytes=sizeof(state),.abi_version=1};
_Static_assert(PS5_VIDEO_OUT_ATTRIBUTE_BYTES==80,"VideoOut attribute ABI");
_Static_assert(sizeof(struct ps5_video_out_buffer)==32,"VideoOut buffer ABI");
_Static_assert(PS5_VIDEO_OUT_FLIP_STATUS_WORDS==16,"VideoOut flip status ABI");

static void commit(const struct ps5_ai_status_display_state_v1 *value) {
    if(!pthread_mutex_lock(&lock)){state=*value;(void)pthread_mutex_unlock(&lock);}
}
int ps5_ai_status_publish(const struct ps5_ai_status_snapshot_v1 *value) {
    if(!ps5_ai_status_snapshot_valid(value))return -1;
    /* Nonblocking to callers; they can publish the next measured snapshot. */
    if(pthread_mutex_trylock(&lock))return 1;
    latest=*value;int rc=pthread_mutex_unlock(&lock);return rc?-1:0;
}
int ps5_ai_status_display_state(struct ps5_ai_status_display_state_v1 *out) {
    if(!out)return -1;if(pthread_mutex_trylock(&lock))return 1;
    *out=state;return pthread_mutex_unlock(&lock)?-1:0;
}
void ps5_ai_status_request_stop(void) {atomic_store_explicit(&stop_requested,1,memory_order_release);}
static int monotonic(uint64_t *out) {
    struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t)||t.tv_sec<0||t.tv_nsec<0||t.tv_nsec>=1000000000)return 0;
    if((uint64_t)t.tv_sec>(UINT64_MAX-(uint64_t)t.tv_nsec)/UINT64_C(1000000000))return 0;
    *out=(uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;return 1;
}
static void put(unsigned char *frame,unsigned x,unsigned y,uint32_t color) {
    if(x>=PS5_AI_STATUS_WIDTH||y>=PS5_AI_STATUS_HEIGHT)return;size_t offset=ps5_ai_status_pixel_offset(x,y);
    if(offset<=FRAME_BYTES-sizeof(color))memcpy(frame+offset,&color,sizeof(color));
}
static void draw(unsigned char *frame,const struct ps5_ai_status_snapshot_v1 *snapshot) {
    char rows[24][128];size_t count=ps5_ai_status_format(snapshot,rows);
    memset(frame,0,FRAME_BYTES);
    for(size_t line=0;line<count;line++) {
        unsigned scale=line?3:4,x=48,y=40+(unsigned)line*38;
        for(const char *p=rows[line];*p&&x+6*scale<1872;p++,x+=6*scale) {
            unsigned char c=(unsigned char)*p;if(c>='a'&&c<='z')c-=32;
            const unsigned char *glyph=NULL;
            if(c>='0'&&c<='9')glyph=glyphs[c-'0'];
            else if(c>='A'&&c<='Z')glyph=glyphs[10+c-'A'];
            else {const char *at=strchr(punctuation,c);if(at)glyph=glyphs[36+(at-punctuation)];}
            if(!glyph)continue;
            for(unsigned col=0;col<5;col++)for(unsigned bit=0;bit<7;bit++)if(glyph[col]&(1u<<bit))
                for(unsigned dy=0;dy<scale;dy++)for(unsigned dx=0;dx<scale;dx++)put(frame,x+col*scale+dx,y+bit*scale+dy,UINT32_C(0xffe7ecf2));
        }
    }
    for(size_t offset=0;offset<FRAME_BYTES;offset+=64)__asm__ volatile("clflush (%0)"::"r"(frame+offset):"memory");
    __asm__ volatile("mfence":::"memory");
}
static int shown(int video,int64_t argument) {
    uint64_t begin,now;if(!monotonic(&begin))return 0;
    for(;;) {
        struct {uint64_t pre[2],words[16],post[2];} g;
        memset(&g,0,sizeof(g));g.pre[0]=g.pre[1]=g.post[0]=g.post[1]=GUARD;
        int rc=sceVideoOutGetFlipStatus(video,g.words);
        if(rc||g.pre[0]!=GUARD||g.pre[1]!=GUARD||g.post[0]!=GUARD||g.post[1]!=GUARD)return 0;
        int pending=sceVideoOutIsFlipPending(video);if(pending<0)return 0;
        if(!pending&&g.words[PS5_VIDEO_OUT_FLIP_STATUS_SHOWN_ARGUMENT]==(uint64_t)argument)return 1;
        if(!monotonic(&now)||now<begin||now-begin>=UINT64_C(2000000000))return 0;
        if(usleep(1000)&&errno!=EINTR)return 0;
    }
}
static void *display_owner(void *unused) {
    (void)unused;struct ps5_ai_status_display_state_v1 local={.abi_bytes=sizeof(local),.abi_version=1,.thread_started=1};
    struct ps5_ai_status_snapshot_v1 sample,view;
    if(pthread_mutex_lock(&lock))return NULL;sample=latest;(void)pthread_mutex_unlock(&lock);
    int stage=1,rc=0;uint64_t total=sceKernelGetDirectMemorySize();
    if(total<TOTAL_BYTES||total>INT64_MAX){rc=-1;goto unavailable;}
    struct {uint64_t pre[2];int64_t value;uint64_t post[2];} p={{GUARD,GUARD},-1,{GUARD,GUARD}};
    stage=2;rc=sceKernelAllocateDirectMemory(0,(int64_t)total,TOTAL_BYTES,ALIGNMENT,3,&p.value);
    if(p.pre[0]!=GUARD||p.pre[1]!=GUARD||p.post[0]!=GUARD||p.post[1]!=GUARD){local.ownership_uncertain=1;goto unavailable;}
    if(rc){if(p.value!=-1)local.ownership_uncertain=1;goto unavailable;}
    if(p.value<0||(uint64_t)p.value%ALIGNMENT||(uint64_t)p.value>total-TOTAL_BYTES){local.ownership_uncertain=1;goto unavailable;}
    local.physical_owned=1;commit(&local);
    struct {uint64_t pre[2];void *value;uint64_t post[2];} m={{GUARD,GUARD},NULL,{GUARD,GUARD}};
    stage=3;rc=sceKernelMapDirectMemory(&m.value,TOTAL_BYTES,0x33,0,p.value,ALIGNMENT);
    if(m.pre[0]!=GUARD||m.pre[1]!=GUARD||m.post[0]!=GUARD||m.post[1]!=GUARD){local.ownership_uncertain=1;goto unavailable;}
    if(rc){if(m.value)local.ownership_uncertain=1;goto unavailable;}
    uintptr_t address=(uintptr_t)m.value;
    if(!address||address%ALIGNMENT||address>UINTPTR_MAX-TOTAL_BYTES){local.ownership_uncertain=1;goto unavailable;}
    local.mapping_owned=1;commit(&local);
    unsigned char *pixels=m.value;memset(pixels,0,TOTAL_BYTES);
    stage=4;
    ps5_ai_ggml_report("BEFORE UI_HideSplashScreen public_provider=1 one_attempt=1 VideoOutOpen_not_yet_called=1");
    int splash_rc=sceSystemServiceHideSplashScreen(),splash_errno=errno;
    ps5_ai_ggml_report("RETURN UI_HideSplashScreen signed_rc=%d hex_rc=0x%08x errno_after_call=%d return_value_only=1 visible_effect_unverified=1 PS_button_response_unverified=1 no_retry=1",
                      splash_rc,(unsigned)splash_rc,splash_errno);
    errno=splash_errno;
    int video=sceVideoOutOpen(PS5_VIDEO_OUT_USER_SYSTEM,0,0,NULL);
    if(video<0){rc=video;goto unavailable;}
    local.video_opened=1;commit(&local);
    stage=5;rc=sceVideoOutSetFlipRate(video,0);if(rc)goto unavailable;
    struct {uint64_t pre[2];unsigned char bytes[80];uint64_t post[2];} a;
    memset(&a,0,sizeof(a));a.pre[0]=a.pre[1]=a.post[0]=a.post[1]=GUARD;
    sceVideoOutSetBufferAttribute2(a.bytes,UINT64_C(0x8000000022000000),0,1920,1080,0,0,0);
    if(a.pre[0]!=GUARD||a.pre[1]!=GUARD||a.post[0]!=GUARD||a.post[1]!=GUARD){local.ownership_uncertain=1;goto unavailable;}
    struct {uint64_t pre[2];struct ps5_video_out_buffer rows[2];uint64_t post[2];} b;
    memset(&b,0,sizeof(b));b.pre[0]=b.pre[1]=b.post[0]=b.post[1]=GUARD;b.rows[0].data=pixels;b.rows[1].data=pixels+FRAME_BYTES;
    stage=6;rc=sceVideoOutRegisterBuffers2(video,0,0,b.rows,2,a.bytes,0,NULL);
    if(rc||a.pre[0]!=GUARD||a.pre[1]!=GUARD||a.post[0]!=GUARD||a.post[1]!=GUARD||
       b.pre[0]!=GUARD||b.pre[1]!=GUARD||b.post[0]!=GUARD||b.post[1]!=GUARD||
       b.rows[0].data!=pixels||b.rows[1].data!=pixels+FRAME_BYTES||b.rows[0].metadata||b.rows[1].metadata||
       b.rows[0].reserved[0]||b.rows[0].reserved[1]||b.rows[1].reserved[0]||b.rows[1].reserved[1]){local.ownership_uncertain=1;goto unavailable;}
    local.buffers_registered=1;commit(&local);
    unsigned index=0;int64_t argument=1;
    while(!atomic_load_explicit(&stop_requested,memory_order_acquire)) {
        int refreshed=0;
        if(!pthread_mutex_trylock(&lock)){sample=latest;refreshed=!pthread_mutex_unlock(&lock);}
        uint64_t now=0;int time_valid=monotonic(&now);
        if(!ps5_ai_status_live_view(&view,&sample,now,refreshed&&time_valid)){rc=-1;goto unavailable;}
        struct timespec clock;
        if(!clock_gettime(CLOCK_REALTIME,&clock)&&clock.tv_sec>0&&clock.tv_nsec>=0&&clock.tv_nsec<1000000000){view.valid|=PS5_AI_STATUS_CLOCK;view.clock_unix_seconds=clock.tv_sec;}
        else view.valid&=~PS5_AI_STATUS_CLOCK;
        if(local.heartbeat==UINT64_MAX){rc=-1;goto unavailable;}
        view.sequence=++local.heartbeat;draw(pixels+(size_t)index*FRAME_BYTES,&view);
        stage=7;rc=sceVideoOutSubmitFlip(video,(int32_t)index,PS5_VIDEO_OUT_FLIP_VSYNC,argument);
        if(rc){local.ownership_uncertain=1;goto unavailable;}
        stage=8;if(!shown(video,argument)){rc=-1;local.ownership_uncertain=1;goto unavailable;}
        local.completed_flip_argument=(uint64_t)argument;local.ready=1;commit(&local);
        /* Only the other, inactive buffer is written after positive flip proof. */
        index^=1;if(argument==INT64_MAX){rc=-1;goto unavailable;}argument++;
        for(unsigned i=0;i<10&&!atomic_load_explicit(&stop_requested,memory_order_acquire);i++)
            if(usleep(100000)&&errno!=EINTR){rc=-1;goto unavailable;}
    }
    local.ready=0;local.stopped=1;commit(&local);return NULL;
unavailable:
    local.ready=0;local.display_fault=1;local.error_stage=stage;local.error_code=rc;local.stopped=1;commit(&local);
    /* Never free/unmap/unregister or close a potentially scanning buffer. The
     * API/engine continues; caller owns the later normal OS lifetime decision. */
    return NULL;
}
int ps5_ai_status_start_after_gate(const struct ps5_ai_status_gate_v1 *gate) {
    if(!gate||gate->abi_bytes!=sizeof(*gate)||gate->abi_version!=1||gate->reserved||
       gate->process_id!=(uint32_t)getpid()||gate->completed_gate_passed!=1||
       gate->completed_gate_bytes!=UINT64_C(67108864)||gate->explicit_AGC_attempts!=1)return -1;
    int expected=0;if(!atomic_compare_exchange_strong(&started,&expected,1))return -1;
    if(pthread_mutex_lock(&lock))return -1;
    memset(&latest,0,sizeof(latest));latest.abi_bytes=sizeof(latest);latest.abi_version=1;
    latest.process_id=gate->process_id;latest.valid=PS5_AI_STATUS_MODEL;
    memcpy(latest.model,"WAI-Illustrious-SDXL v17.0",sizeof("WAI-Illustrious-SDXL v17.0"));
    state.thread_started=1;(void)pthread_mutex_unlock(&lock);
    int rc=pthread_create(&owner_thread,NULL,display_owner,NULL);
    if(rc){if(!pthread_mutex_lock(&lock)){state.thread_started=0;state.display_fault=1;state.stopped=1;state.error_stage=0;state.error_code=rc;(void)pthread_mutex_unlock(&lock);}return -1;}
    return 0; /* Thread started only. Poll state.ready; never gate inference on it. */
}
