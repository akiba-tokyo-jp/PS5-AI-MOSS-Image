/* SPDX-License-Identifier: MIT
 * OpenAI Images protocol adapter. One bounded response worker, one image/job.
 * All inference is forwarded by engine_bridge.inc to the SAME PS5's API.
 */
#include "sha256.h"
#include "openapi_schema.inc"
#define OPENAI_MODEL "moss-image-sft"
#define IMAGE_URL_TTL 3600
#define OPENAI_WAIT_MS UINT64_C(7200000)

static void finish_client(socket_t fd, uint64_t deadline);
static void openai_error(socket_t fd, int code, const char *message, const char *param) {
    char q[4608], p[400], body[5300];
    if (!quote_json(q,sizeof(q),message)) strcpy(q,"\"Request failed\"");
    if (param) quote_json(p,sizeof(p),param); else strcpy(p,"null");
    const char *type=code==401?"authentication_error":code==429?"rate_limit_error":code>=500?"server_error":"invalid_request_error";
    snprintf(body,sizeof(body),"{\"error\":{\"message\":%s,\"type\":\"%s\",\"param\":%s,\"code\":null}}",q,type,p);
    send_json(fd,code,body);
}
static void image_signature(const char *id,int64_t expires,char hex[65]) {
    unsigned char key[64]={0},inner[32],digest[32],pad[64];struct ai_sha256 s;
    if(token_len>64){ai_sha256_init(&s);ai_sha256_update(&s,api_token,token_len);ai_sha256_final(&s,key);}
    else memcpy(key,api_token,token_len);
    char message[180];int n=snprintf(message,sizeof(message),"ps5-image-v1:%s:%" PRId64 ":%" PRId64,id,expires,server_started_at);
    for(size_t i=0;i<64;i++)pad[i]=key[i]^0x36;
    ai_sha256_init(&s);ai_sha256_update(&s,pad,64);ai_sha256_update(&s,message,(size_t)n);ai_sha256_final(&s,inner);
    for(size_t i=0;i<64;i++)pad[i]=key[i]^0x5c;
    ai_sha256_init(&s);ai_sha256_update(&s,pad,64);ai_sha256_update(&s,inner,32);ai_sha256_final(&s,digest);
    for(size_t i=0;i<32;i++)snprintf(hex+i*2,3,"%02x",digest[i]);
    wipe_secret(key,sizeof(key));wipe_secret(inner,sizeof(inner));wipe_secret(digest,sizeof(digest));wipe_secret(pad,sizeof(pad));
}
static int signed_image(const char *path,char id[33]) {
    char sig[65],extra;int64_t expires;
    if(sscanf(path,"/v1/images/%32[0-9a-f].png?expires=%10" SCNd64 "&signature=%64[0-9a-f]%c",id,&expires,sig,&extra)!=3 || !valid_id(id) || strlen(sig)!=64)return 0;
    int64_t now=(int64_t)time(NULL);if(expires<now || expires>now+IMAGE_URL_TTL)return 0;
    char expected[65];image_signature(id,expires,expected);unsigned diff=0;
    for(size_t i=0;i<64;i++)diff|=(unsigned char)sig[i]^(unsigned char)expected[i];
    return diff==0;
}
static void image_url(const char *id,char *url,size_t cap) {
    int64_t expires=(int64_t)time(NULL)+IMAGE_URL_TTL;char sig[65];image_signature(id,expires,sig);
    snprintf(url,cap,"http://%s:%d/v1/images/%s.png?expires=%" PRId64 "&signature=%s",bind_ip,listen_port,id,expires,sig);
}
static size_t base64_block(const unsigned char *in,size_t n,char *out) {
    static const char chars[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";size_t j=0;
    for(size_t i=0;i<n;i+=3){unsigned v=(unsigned)in[i]<<16;if(i+1<n)v|=(unsigned)in[i+1]<<8;if(i+2<n)v|=in[i+2];
        out[j++]=chars[(v>>18)&63];out[j++]=chars[(v>>12)&63];out[j++]=i+1<n?chars[(v>>6)&63]:'=';out[j++]=i+2<n?chars[v&63]:'=';}
    return j;
}
static int openai_validate(const char *body,ps5_ai_request *r,int *url_format,char *err,size_t cap,char param[64]) {
    object *o=malloc(sizeof(*o));if(!o){snprintf(err,cap,"Unable to parse request");return 0;}
    memset(r,0,sizeof(*r));r->width=r->height=1024;r->steps=PS5_AI_MOSS_DEFAULT_REQUEST_STEPS;r->cfg_scale=PS5_AI_MOSS_DEFAULT_REQUEST_CFG;r->flow_shift=5;r->seed=PS5_AI_MOSS_DEFAULT_REQUEST_SEED;strcpy(r->sampler,PS5_AI_MOSS_DEFAULT_REQUEST_SAMPLER);r->manga_strength=0;*url_format=0;param[0]=0;int ok=0;int64_t v;
    r->control_strength=1;r->denoise_strength=0.5;
    if(!parse_object(body,o)){snprintf(err,cap,"Malformed, duplicate, nested, or oversized JSON field");goto done;}
    for(size_t i=0;i<o->count;i++){
        field *f=&o->values[i];snprintf(param,64,"%s",f->key);
        const char *keys[]={"prompt","negative_prompt","model","n","size","response_format","quality","background","output_format","stream","user","safety_identifier","steps","cfg_scale","flow_shift","seed","sampler","manga_strength","manga_monochrome","init_image_id","reference_image_id","mask_image_id","denoise_strength","ip_adapter_strength","controlnet","control_image_id","control_strength"};
        int known=0;for(size_t k=0;k<sizeof(keys)/sizeof(keys[0]);k++)known|=!strcmp(f->key,keys[k]);
        if(!known){snprintf(err,cap,"Unsupported parameter: %.63s",f->key);goto done;}
        if((!PS5_AI_MOSS_IMG2IMG_TRIAL && (!strcmp(f->key,"init_image_id") || !strcmp(f->key,"denoise_strength"))) || !strcmp(f->key,"reference_image_id") || !strcmp(f->key,"mask_image_id") ||
           !strcmp(f->key,"ip_adapter_strength") || !strcmp(f->key,"controlnet") ||
           !strcmp(f->key,"control_image_id") || !strcmp(f->key,"control_strength")){
            snprintf(err,cap,"MOSS Image SFT currently supports text-to-image only; %.63s is unsupported",f->key);goto done;
        }
        if(f->type==5 && strcmp(f->key,"prompt")) continue;
        if(!strcmp(f->key,"prompt") || !strcmp(f->key,"negative_prompt")){
            if(f->type!=1 || strlen(f->value)>4096){snprintf(err,cap,"Prompt fields require at most4096 UTF-8 bytes");goto done;}
            strcpy(!strcmp(f->key,"prompt")?r->prompt:r->negative_prompt,f->value);
        }else if(!strcmp(f->key,"model")){
            if(f->type!=1 || strcmp(f->value,OPENAI_MODEL)){snprintf(err,cap,"Only model " OPENAI_MODEL " is available");goto done;}
        }else if(!strcmp(f->key,"n")){
            if(!get_int(f,1,1,&v)){snprintf(err,cap,"This PS5 gateway supports n=1 only");goto done;}
        }else if(!strcmp(f->key,"size")){
            if(f->type!=1){snprintf(err,cap,"size must be a string");goto done;}
            if(!strcmp(f->value,"auto"))continue;
            int w,h;char extra;char canonical[32];
            if(strlen(f->value)>9 || sscanf(f->value,"%dx%d%c",&w,&h,&extra)!=2 || w<64 || h<64 || w>1024 || h>1024 || w%16 || h%16){snprintf(err,cap,"size requires dimensions64..1024, multiples of16");goto done;}
            snprintf(canonical,sizeof(canonical),"%dx%d",w,h);if(strcmp(canonical,f->value)){snprintf(err,cap,"Invalid size syntax");goto done;}r->width=w;r->height=h;
        }else if(!strcmp(f->key,"response_format")){
            if(f->type!=1 || (strcmp(f->value,"b64_json") && strcmp(f->value,"url"))){snprintf(err,cap,"response_format must be b64_json or url");goto done;}*url_format=!strcmp(f->value,"url");
        }else if(!strcmp(f->key,"quality")){
            if(f->type!=1 || (strcmp(f->value,"auto") && strcmp(f->value,"standard"))){snprintf(err,cap,"Only quality auto/standard are supported; use steps extension for MOSS Image tuning");goto done;}
        }else if(!strcmp(f->key,"background")){
            if(f->type!=1 || (strcmp(f->value,"auto") && strcmp(f->value,"opaque"))){snprintf(err,cap,"Transparent output is not supported");goto done;}
        }else if(!strcmp(f->key,"output_format")){
            if(f->type!=1 || strcmp(f->value,"png")){snprintf(err,cap,"Only PNG output is supported");goto done;}
        }else if(!strcmp(f->key,"stream")){
            if(f->type!=4){snprintf(err,cap,"Streaming is not supported");goto done;}
        }else if(!strcmp(f->key,"user") || !strcmp(f->key,"safety_identifier")){
            if(f->type!=1 || strlen(f->value)>256){snprintf(err,cap,"User identifier must be a string up to256 bytes");goto done;}
            /* Accepted as metadata only, never logged; this is not moderation. */
        }else if(!strcmp(f->key,"steps")){
            if(!get_int(f,1,100,&v)){snprintf(err,cap,"steps must be1..100");goto done;}r->steps=(int)v;
        }else if(!strcmp(f->key,"cfg_scale")){
            if(!get_double(f,0,30,&r->cfg_scale)){snprintf(err,cap,"cfg_scale must be finite0..30");goto done;}
        }else if(!strcmp(f->key,"flow_shift")){
            if(!get_double(f,0,FLT_MAX,&r->flow_shift) || (float)r->flow_shift<=0){snprintf(err,cap,"flow_shift must remain a positive finite float");goto done;}
        }else if(!strcmp(f->key,"manga_strength")){
            if(!get_double(f,0,0,&r->manga_strength)){snprintf(err,cap,"The SDXL manga LoRA is incompatible with MOSS Image; manga_strength must be 0");goto done;}
        }else if(!strcmp(f->key,"manga_monochrome")){
            if(f->type!=4){snprintf(err,cap,"The SDXL manga LoRA is incompatible with MOSS Image; manga_monochrome must be false");goto done;}
        }else if(!strcmp(f->key,"init_image_id") || !strcmp(f->key,"reference_image_id") || !strcmp(f->key,"mask_image_id")){
            if(f->type!=1 || !valid_id(f->value)){snprintf(err,cap,"Invalid input image ID");goto done;}
            strcpy(!strcmp(f->key,"init_image_id")?r->init_image_id:!strcmp(f->key,"reference_image_id")?r->reference_image_id:r->mask_image_id,f->value);
        }else if(!strcmp(f->key,"denoise_strength")){
            if(!get_double(f,0.01,1,&r->denoise_strength)){snprintf(err,cap,"denoise_strength must be finite0.01..1");goto done;}
        }else if(!strcmp(f->key,"ip_adapter_strength")){
            if(!get_double(f,0.01,1.5,&r->ip_adapter_strength)){snprintf(err,cap,"ip_adapter_strength must be finite0.01..1.5");goto done;}
        }else if(!strcmp(f->key,"controlnet")){
            if(f->type!=1 || (strcmp(f->value,"canny") && strcmp(f->value,"openpose"))){snprintf(err,cap,"controlnet must be canny or openpose");goto done;}strcpy(r->controlnet,f->value);
        }else if(!strcmp(f->key,"control_image_id")){
            if(f->type!=1 || !valid_id(f->value)){snprintf(err,cap,"Invalid control image ID");goto done;}strcpy(r->control_image_id,f->value);
        }else if(!strcmp(f->key,"control_strength")){
            if(!get_double(f,0.01,1.5,&r->control_strength)){snprintf(err,cap,"control_strength must be finite0.01..1.5");goto done;}
        }else if(!strcmp(f->key,"seed")){
            if(!get_int(f,-1,INT64_MAX,&r->seed)){snprintf(err,cap,"Invalid seed");goto done;}
        }else if(!strcmp(f->key,"sampler")){
            if(f->type!=1 || !ps5_ai_moss_sampler_name_allowed(f->value)){snprintf(err,cap,"Unsupported sampler");goto done;}strcpy(r->sampler,f->value);
        }else{snprintf(err,cap,"Unsupported parameter: %.63s",f->key);goto done;}
    }
    if(!r->prompt[0]){strcpy(param,"prompt");snprintf(err,cap,"prompt is required and cannot be empty");goto done;}
    if(r->init_image_id[0] && (!PS5_AI_MOSS_IMG2IMG_TRIAL || !ps5_ai_moss_img2img_steps(r->denoise_strength))){snprintf(err,cap,PS5_AI_MOSS_SERVICE_MODE?"Img2img denoise_strength must remain finite0.01..1 as float":"MOSS img2img strength must retain the reference: finite0.01 <= float(strength) <0.75");goto done;}
    if((find_field(o,"denoise_strength") && find_field(o,"denoise_strength")->type!=5 && !r->init_image_id[0]) ||
       (find_field(o,"ip_adapter_strength") && find_field(o,"ip_adapter_strength")->type!=5 && !r->reference_image_id[0])){
        snprintf(err,cap,"Image strength requires its matching input image");goto done;
    }
    if(r->mask_image_id[0] && !r->init_image_id[0]){strcpy(param,"mask_image_id");snprintf(err,cap,"Mask requires an init image");goto done;}
    if((r->controlnet[0]!=0)!=(r->control_image_id[0]!=0) ||
       (find_field(o,"control_strength") && find_field(o,"control_strength")->type!=5 && !r->controlnet[0])){
        snprintf(err,cap,"ControlNet requires both its model name and prepared control image ID");goto done;
    }
    if(!ps5_ai_moss_sampler_request_allowed(r->width,r->height,r->steps,r->cfg_scale,r->flow_shift,r->sampler,r->seed)){
        snprintf(err,cap,PS5_AI_MOSS_SERVICE_MODE?"Turbo4 model profile requires64..1024 dimensions on16 grid,4steps,CFG1,shift5,Euler; arbitrary seed>=-1":"Sampler trial requires256/1024 square,30steps,CFG4,shift5,res_multistep,seed20261020");goto done;
    }
    ok=1;
done:free(o);return ok;
}
static void openai_models(socket_t fd,const char *path){
    char model[256];snprintf(model,sizeof(model),"{\"id\":\"" OPENAI_MODEL "\",\"object\":\"model\",\"created\":%" PRId64 ",\"owned_by\":\"local\"}",server_started_at);
    if(!strcmp(path,"/v1/models")){char out[320];snprintf(out,sizeof(out),"{\"object\":\"list\",\"data\":[%s]}",model);send_json(fd,200,out);}
    else if(!strcmp(path,"/v1/models/" OPENAI_MODEL))send_json(fd,200,model);
    else openai_error(fd,404,"Model not found","model");
}
static int openai_submit(socket_t fd,const char *body,int transient_inputs){
    ps5_ai_request r;char err[768],param[64];int url;
    if(!openai_validate(body,&r,&url,err,sizeof(err),param)){openai_error(fd,400,err,param[0]?param:NULL);return 0;}
    r.transient_inputs=transient_inputs;
    ps5_ai_engine_status status;memset(&status,0,sizeof(status));ps5_ai_engine_get_status(&status);
    if(!status.available || !status.gpu_available){openai_error(fd,503,"PS5 native GPU backend is unavailable",NULL);return 0;}
    lock_state();
    if(!openai_thread_ready){unlock_state();openai_error(fd,503,"Response worker is unavailable",NULL);return 0;}
    if(openai_busy || stopping){unlock_state();openai_error(fd,429,"One synchronous image request is already active; retry after completion",NULL);return 0;}
    if(!moss_storage_admitted()){unlock_state();openai_error(fd,507,"Output/job storage quota reached or metadata unavailable; archive output before a new job",NULL);return 0;}
    job *j=NULL;for(int i=0;i<MAX_JOBS;i++)if(!jobs[i].occupied){j=&jobs[i];break;}else if(!jobs[i].response_refs && strcmp(jobs[i].status,"queued") && strcmp(jobs[i].status,"running") && (!j || jobs[i].updated<j->updated))j=&jobs[i];
    if(!j){unlock_state();openai_error(fd,429,"Job queue is full",NULL);return 0;}
    uint64_t now;if(!monotonic_ms(&now)){unlock_state();openai_error(fd,500,"Monotonic clock unavailable",NULL);return 0;}
    memset(j,0,sizeof(*j));j->occupied=1;j->response_refs=1;j->request=r;j->created=j->updated=(int64_t)time(NULL);
    char existing[JOB_JSON_SIZE];do{++sequence;snprintf(j->id,sizeof(j->id),"%08" PRIx32 "%08" PRIx32 "%08" PRIx32 "%08" PRIx32,(uint32_t)j->created,(uint32_t)get_pid(),sequence,sequence^(uint32_t)(j->created>>32));}while(load_job_json(j->id,existing,sizeof(existing)));
    strcpy(j->status,"queued");char name[64];snprintf(name,sizeof(name),"%s.request.json",j->id);
    if(!atomic_write("jobs",name,body,strlen(body)) || !persist_job(j)){j->occupied=0;unlock_state();openai_error(fd,500,"Unable to persist job",NULL);return 0;}
    openai_pending_job=j;openai_pending_fd=fd;openai_url_format=url;openai_deadline=now+OPENAI_WAIT_MS;openai_busy=1;
    log_line("OpenAI job %s accepted; response_format=%s",j->id,url?"url":"b64_json");unlock_state();return 1;
}
static int openai_image_response(socket_t fd,const job *j,int url_format){
    char path[MAX_PATH_SIZE],name[64],url[512],prefix[320],headers[1600];snprintf(name,sizeof(name),"%s.png",j->id);
    if(!path_for(path,sizeof(path),"output",name))return 0;
    struct stat st;if(path_metadata(path,&st,0) || !S_ISREG(st.st_mode) || st.st_size<24 || st.st_size>32*1024*1024){openai_error(fd,500,"Invalid completed PNG",NULL);return 0;}
    image_url(j->id,url,sizeof(url));
    if(url_format){char response[800];snprintf(response,sizeof(response),"{\"created\":%" PRId64 ",\"data\":[{\"url\":\"%s\"}]}",j->created,url);
        int n=snprintf(headers,sizeof(headers),"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\nCache-Control: no-store\r\nx-request-id: %s\r\nX-PS5-Native-Job-ID: %s\r\nX-PS5-Seed: %" PRId64 "\r\n\r\n",strlen(response),j->id,j->native_job_id,j->request.seed);
        return n>0 && (size_t)n<sizeof(headers) && send_all(fd,headers,(size_t)n) && send_all(fd,response,strlen(response));}
    FILE *f=fopen(path,"rb");if(!f){openai_error(fd,500,"Completed PNG is unavailable",NULL);return 0;}
    int p=snprintf(prefix,sizeof(prefix),"{\"created\":%" PRId64 ",\"data\":[{\"b64_json\":\"",j->created);const char suffix[]="\"}]}";
    /* Parentheses are essential: base64 length is 4*ceil(bytes/3). */
    uint64_t total=(uint64_t)p+4*(((uint64_t)st.st_size+2)/3)+sizeof(suffix)-1;
    int h=snprintf(headers,sizeof(headers),"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %" PRIu64 "\r\nConnection: close\r\nCache-Control: no-store\r\nx-request-id: %s\r\nX-PS5-Native-Job-ID: %s\r\nX-PS5-Image-URL: %s\r\nX-PS5-Seed: %" PRId64 "\r\n\r\n",total,j->id,j->native_job_id,url,j->request.seed);
    unsigned char *in=malloc(12288);char *out=malloc(16384);int ok=in && out && h>0 && (size_t)h<sizeof(headers);
    if(ok)ok=send_all(fd,headers,(size_t)h) && send_all(fd,prefix,(size_t)p);
    uint64_t remaining=(uint64_t)st.st_size,until=0;if(!monotonic_ms(&until))ok=0;until+=60000;
    while(ok && remaining){uint64_t now;if(!monotonic_ms(&now) || now>=until){ok=0;break;}size_t want=remaining>12288?12288:(size_t)remaining;size_t n=fread(in,1,want,f);if(n!=want){ok=0;break;}ok=send_all(fd,out,base64_block(in,n,out));remaining-=n;}
    if(ok)ok=send_all(fd,suffix,sizeof(suffix)-1);free(in);free(out);fclose(f);return ok;
}
#ifdef _WIN32
static DWORD WINAPI openai_worker(void *unused)
#else
static void *openai_worker(void *unused)
#endif
{
    (void)unused;
    for(;;){
        job copy;job *owned=NULL;socket_t fd=BAD_SOCKET;int url=0,done=0,timedout=0;uint64_t now=0,deadline=0;
        lock_state();
        if(openai_busy){owned=openai_pending_job;copy=*owned;fd=openai_pending_fd;url=openai_url_format;deadline=openai_deadline;
            done=strcmp(copy.status,"queued") && strcmp(copy.status,"running");timedout=!monotonic_ms(&now) || now>=deadline;}
        int stop=stopping;unlock_state();
        if(owned && (done || timedout || stop)){
            if(done && !strcmp(copy.status,"completed") && copy.stats.gpu_used)openai_image_response(fd,&copy,url);
            else openai_error(fd,timedout?504:500,done && copy.error[0]?copy.error:"Image job did not complete; it may continue on the backend. Do not automatically retry.",NULL);
            uint64_t close_deadline=0;if(monotonic_ms(&close_deadline))finish_client(fd,close_deadline+1000);else close_socket(fd);
            lock_state();owned->response_refs=0;openai_busy=0;openai_pending_job=NULL;openai_pending_fd=BAD_SOCKET;unlock_state();
        }
        if(stop)break;pause_ms(50);
    }return 0;
}
