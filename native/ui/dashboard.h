/* SPDX-License-Identifier: GPL-3.0-or-later
 * CPU-only presentation. Existing snapshot validation, privacy filtering and
 * metric source rules remain in the unchanged status formatter.
 * Font masks have their own OFL-1.1 license; see generated font provenance.
 */
#ifndef WAI_STATUS_DASHBOARD_V1_H
#define WAI_STATUS_DASHBOARD_V1_H
#include "status.h"
#include "raster.h"
#include "dashboard_font.h"
#include "dashboard_brand.h"
#include <stdio.h>
#include <string.h>

struct wai_canvas { unsigned char *frame; int linear; };
enum { WAI_BG=0x0b1120, WAI_CARD=0x131e30, WAI_LINE=0x26344a,
       WAI_FG=0xebf1fa, WAI_MUTED=0x94a6bf, WAI_MINT=0x77e2c4,
       WAI_LILAC=0xb6a3fa, WAI_AMBER=0xf0c17c, WAI_RED=0xf89caa };

static inline void wai_pixel(struct wai_canvas *c,int x,int y,unsigned rgb) {
    if((unsigned)x>=1920u||(unsigned)y>=1080u)return;
    size_t offset=c->linear?((size_t)y*1920u+(unsigned)x)*4u:ps5_ai_status_pixel_offset((unsigned)x,(unsigned)y);
    if(offset<=PS5_AI_STATUS_FRAME_BYTES-4u){uint32_t color=UINT32_C(0xff000000)|rgb;memcpy(c->frame+offset,&color,4);}
}
static void wai_rect(struct wai_canvas *c,int x,int y,int w,int h,unsigned color) {
    if(w<=0||h<=0)return;
    for(int py=y;py<y+h;py++)for(int px=x;px<x+w;px++)wai_pixel(c,px,py,color);
}
static void wai_round(struct wai_canvas *c,int x,int y,int w,int h,int r,unsigned color) {
    if(w<=0||h<=0)return;if(r>w/2)r=w/2;if(r>h/2)r=h/2;
    for(int py=0;py<h;py++){
        int inset=0,dy=py<r?r-py-1:(py>=h-r?py-(h-r):0);
        if(py<r||py>=h-r){inset=r;while(inset>0&&(r-inset)*(r-inset)+dy*dy<r*r)inset--;inset++;}
        for(int px=inset;px<w-inset;px++)wai_pixel(c,x+px,y+py,color);
    }
}
static void wai_card(struct wai_canvas *c,int x,int y,int w,int h) {
    wai_round(c,x,y,w,h,22,WAI_LINE);wai_round(c,x+1,y+1,w-2,h-2,21,WAI_CARD);
}
static unsigned wai_mix(unsigned fg,unsigned bg,unsigned a) {
    unsigned r=(((fg>>16)&255)*a+((bg>>16)&255)*(255-a)+127)/255;
    unsigned g=(((fg>>8)&255)*a+((bg>>8)&255)*(255-a)+127)/255;
    unsigned b=((fg&255)*a+(bg&255)*(255-a)+127)/255;
    return (r<<16)|(g<<8)|b;
}
static void wai_text(struct wai_canvas *c,int x,int y,int width,unsigned face,
                     unsigned color,unsigned background,const char *text) {
    if(!text||face>=WAI_FONT_FACES)return;
    int end=x+width;
    for(const unsigned char *p=(const unsigned char *)text;*p;p++){
        unsigned ch=(*p>=32&&*p<=126)?*p:'?';
        const struct wai_glyph *g=&wai_font_glyphs[face][ch-32];
        if(x+(int)g->advance>end)break;
        for(unsigned gy=0;gy<g->height;gy++)for(unsigned gx=0;gx<g->width;gx++){
            unsigned a=wai_font_pixels[g->offset+(size_t)gy*g->width+gx];
            if(a)wai_pixel(c,x+g->x_offset+(int)gx,y+g->y_offset+(int)gy,wai_mix(color,background,a));
        }
        x+=g->advance;
    }
}
static const char *wai_find(char rows[][128],size_t n,const char *prefix) {
    size_t len=strlen(prefix);for(size_t i=0;i<n;i++)if(!strncmp(rows[i],prefix,len))return rows[i]+len;
    return NULL;
}
static int wai_technical_row(const char *line) {
    static const char *prefix[]={"PS5 AI MOSS Image","MODEL:","PS5 UTC ","PID:","STATE:","JOB:","PROGRESS:",
        "PROCESS CURRENT RSS:","PROCESS PEAK RSS:","PROCESS CPU:","GPU BUSY:","TEMPERATURE:",
        "DIRECT DATA USED:","DIRECT DATA PEAK:","DRIVER ARENA USED:","DISPATCH:"};
    for(unsigned i=0;i<sizeof(prefix)/sizeof(prefix[0]);i++)if(!strncmp(line,prefix[i],strlen(prefix[i])))return 1;
    return 0;
}

static int ps5_ai_dashboard_draw(unsigned char *frame,const struct ps5_ai_status_snapshot_v1 *s,
                                 int linear,int preview) {
    if(!frame||!ps5_ai_status_snapshot_valid(s))return 0;
    struct wai_canvas c={frame,linear};char rows[24][128],text[128];size_t n=ps5_ai_status_format(s,rows);
    /* Match the existing 16 MiB scanout allocation, then color only visible pixels. */
    memset(frame,0,(size_t)PS5_AI_STATUS_FRAME_BYTES);wai_rect(&c,0,0,1920,1080,WAI_BG);
    wai_round(&c,64,46,88,88,20,WAI_CARD);
    for(unsigned y=0;y<72;y++)for(unsigned x=0;x<72;x++)wai_pixel(&c,72+(int)x,54+(int)y,wai_brand_pixels[y*72+x]);
    wai_text(&c,176,40,920,3,WAI_FG,WAI_BG,"PS5 AI MOSS Image");
    const char *model=wai_find(rows,n,"MODEL: ");
    if(model)wai_text(&c,180,111,1080,0,WAI_MUTED,WAI_BG,model);
    /* Heartbeat describes the display loop only, never GPU utilization. */
    wai_round(&c,1468,51,12,12,6,s->sequence&1?WAI_MINT:0x3c8d7e);
    wai_text(&c,1498,41,350,0,WAI_MINT,WAI_BG,"DISPLAY LIVE");
    const char *clock=wai_find(rows,n,"PS5 UTC ");
    if(clock){snprintf(text,sizeof(text),"%s UTC",clock);wai_text(&c,1400,94,456,0,WAI_MUTED,WAI_BG,text);}
    wai_rect(&c,64,162,1792,1,WAI_LINE);

    int has_pool=wai_find(rows,n,"DIRECT DATA USED: ")!=NULL;
    int hero_w=has_pool?1056:1792;wai_card(&c,64,190,hero_w,300);
    const char *phase=wai_find(rows,n,"STATE: ");
    unsigned accent=s->phase==PS5_AI_STATUS_FAILED?WAI_RED:(s->phase==PS5_AI_STATUS_LOADING?WAI_AMBER:WAI_MINT);
    if(phase){
        wai_text(&c,100,220,hero_w-72,0,WAI_MUTED,WAI_CARD,"ENGINE STATUS");
        static const char *names[]={"Ready for a new image","Loading the model","Creating your image","Image completed","Generation failed","Stopping"};
        wai_text(&c,98,263,hero_w-68,3,accent,WAI_CARD,names[s->phase]);
    }else wai_text(&c,100,252,hero_w-72,2,WAI_FG,WAI_CARD,"Native image engine");
    const char *progress=wai_find(rows,n,"PROGRESS: ");
    if(progress){
        wai_text(&c,100,344,hero_w-72,1,WAI_FG,WAI_CARD,progress);
        wai_round(&c,100,399,hero_w-72,10,5,WAI_LINE);
        int fill=(int)((uint64_t)(hero_w-72)*s->progress_basis_points/10000u);
        if(fill>0)wai_round(&c,100,399,fill,10,5,accent);
    }
    const char *job=wai_find(rows,n,"JOB: ");
    if(job){snprintf(text,sizeof(text),"Job  %s",job);wai_text(&c,100,435,hero_w-72,0,WAI_MUTED,WAI_CARD,text);}

    if(has_pool){
        wai_card(&c,1144,190,712,300);
        wai_text(&c,1180,220,640,0,WAI_MUTED,WAI_CARD,"DIRECT DATA POOL");
        snprintf(text,sizeof(text),"%.2f / %.2f GiB",s->direct_used_bytes/1073741824.0,s->direct_reserved_bytes/1073741824.0);
        wai_text(&c,1178,265,640,3,WAI_FG,WAI_CARD,text);
        wai_text(&c,1180,344,640,0,WAI_MUTED,WAI_CARD,"This app's reserved pool");
        wai_round(&c,1180,397,640,12,6,WAI_LINE);
        int fill=(int)((double)s->direct_used_bytes/(double)s->direct_reserved_bytes*640.0);
        if(fill>0)wai_round(&c,1180,397,fill,12,6,WAI_LILAC);
        snprintf(text,sizeof(text),"%.1f%% used     Peak %.2f GiB",100.0*(double)s->direct_used_bytes/(double)s->direct_reserved_bytes,s->direct_peak_bytes/1073741824.0);
        wai_text(&c,1180,435,640,0,WAI_MUTED,WAI_CARD,text);
    }

    struct {const char *label;char value[96];unsigned face;} metrics[6];unsigned m=0;
    const char *rss=wai_find(rows,n,"PROCESS CURRENT RSS: "),*peak=wai_find(rows,n,"PROCESS PEAK RSS: ");
    if(rss||peak){metrics[m].label=rss?"PROCESS RSS":"PROCESS PEAK RSS";metrics[m].face=2;snprintf(metrics[m++].value,96,"%.2f GiB",s->cpu_rss_bytes/1073741824.0);}
    if(wai_find(rows,n,"DRIVER ARENA USED: ")){metrics[m].label="DRIVER ARENA";metrics[m].face=2;snprintf(metrics[m++].value,96,"%.1f / %.0f MiB",s->driver_used_bytes/1048576.0,s->driver_reserved_bytes/1048576.0);}
    if(wai_find(rows,n,"DISPATCH: ")){metrics[m].label="COMPLETED FENCES";metrics[m].face=2;snprintf(metrics[m++].value,96,"%llu",(unsigned long long)s->fences);}
    static const char *extra_prefix[]={"PROCESS CPU: ","GPU BUSY: ","TEMPERATURE: "};
    static const char *extra_label[]={"PROCESS CPU","GPU BUSY","TEMPERATURE"};
    for(unsigned i=0;i<3;i++){const char *v=wai_find(rows,n,extra_prefix[i]);if(v&&m<6){metrics[m].label=extra_label[i];metrics[m].face=2;snprintf(metrics[m++].value,96,"%s",v);}}
    if(m){int w=(1792-(int)(m-1)*24)/(int)m;for(unsigned i=0;i<m;i++){int x=64+(int)i*(w+24);wai_card(&c,x,514,w,154);wai_text(&c,x+30,538,w-60,0,WAI_MUTED,WAI_CARD,metrics[i].label);wai_text(&c,x+28,583,w-56,metrics[i].face,WAI_FG,WAI_CARD,metrics[i].value);}}

    unsigned log_count=0;for(size_t i=0;i<n;i++)if(!wai_technical_row(rows[i]))log_count++;
    if(log_count){wai_card(&c,64,692,1792,302);wai_text(&c,100,715,1720,1,WAI_FG,WAI_CARD,"Activity");unsigned line=0;
        for(size_t i=0;i<n&&line<4;i++)if(!wai_technical_row(rows[i])){int y=773+(int)line*44;wai_round(&c,100,y+10,6,6,3,WAI_LILAC);wai_text(&c,124,y,1686,0,WAI_MUTED,WAI_CARD,rows[i]);line++;}}
    const char *footer=preview?"DESIGN PREVIEW / SAMPLE TELEMETRY / NOT A PS5 SCREENSHOT":"PS5 native MOSS Image generation";
    wai_text(&c,64,1025,1320,0,preview?WAI_AMBER:WAI_MUTED,WAI_BG,footer);
    snprintf(text,sizeof(text),"Display frame  %llu",(unsigned long long)s->sequence);
    wai_text(&c,1490,1025,366,0,WAI_MUTED,WAI_BG,text);return 1;
}
#endif
