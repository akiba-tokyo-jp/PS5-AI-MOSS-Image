/* SPDX-License-Identifier: MIT
 * HTTP-only forwarding to the existing native engine on this SAME console.
 * No model loader, inference library, GPU API, external inference or fallback.
 */
static int backend_socket_error(void) {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
}
#ifndef PS5_AI_UPSTREAM_PORT
#define PS5_AI_UPSTREAM_PORT 8192
#endif
static socket_t backend_connect(void) {
    socket_t fd=socket(AF_INET,SOCK_STREAM,0);if(fd==BAD_SOCKET)return BAD_SOCKET;
    struct sockaddr_in a;memset(&a,0,sizeof(a));a.sin_family=AF_INET;a.sin_port=htons(PS5_AI_UPSTREAM_PORT);a.sin_addr=bind_address;
#ifdef _WIN32
    u_long mode=1;if(ioctlsocket(fd,FIONBIO,&mode)){close_socket(fd);return BAD_SOCKET;}
#else
    int flags=fcntl(fd,F_GETFL,0);if(flags<0 || fcntl(fd,F_SETFL,flags|O_NONBLOCK)){close_socket(fd);return BAD_SOCKET;}
#endif
    int rc=connect(fd,(struct sockaddr *)&a,sizeof(a));
    if(rc){int e=backend_socket_error();
#ifdef _WIN32
        if(e!=WSAEWOULDBLOCK && e!=WSAEINPROGRESS){close_socket(fd);return BAD_SOCKET;}
#else
        if(e!=EINPROGRESS && e!=EWOULDBLOCK){close_socket(fd);return BAD_SOCKET;}
#endif
        fd_set set;FD_ZERO(&set);FD_SET(fd,&set);struct timeval t={5,0};
        if(select((int)fd+1,NULL,&set,NULL,&t)<=0){close_socket(fd);return BAD_SOCKET;}
        int error=0;
#ifdef _WIN32
        int len=sizeof(error);
#else
        socklen_t len=sizeof(error);
#endif
        if(getsockopt(fd,SOL_SOCKET,SO_ERROR,(char *)&error,&len) || error){close_socket(fd);return BAD_SOCKET;}
    }
#ifdef _WIN32
    mode=0;if(ioctlsocket(fd,FIONBIO,&mode)){close_socket(fd);return BAD_SOCKET;}
#else
    if(fcntl(fd,F_SETFL,flags)){close_socket(fd);return BAD_SOCKET;}
#endif
    return fd;
}
/* Native API always supplies Content-Length and closes its connection. */
static int backend_http_bytes(const char *method,const char *path,const void *body,size_t body_n,const char *body_type,char *json,size_t cap,const char *png,int *http_status) {
    socket_t fd=backend_connect();if(fd==BAD_SOCKET)return 0;
    char header[1200],type_header[128]={0};
    if(body && snprintf(type_header,sizeof(type_header),"Content-Type: %s\r\n",body_type)>=(int)sizeof(type_header)){close_socket(fd);return 0;}
    int h=snprintf(header,sizeof(header),"%s %s HTTP/1.1\r\nHost: %s:%d\r\nAuthorization: Bearer %s\r\nConnection: close\r\n%sContent-Length: %zu\r\n\r\n",method,path,bind_ip,PS5_AI_UPSTREAM_PORT,api_token,type_header,body_n);
    int ok=h>0 && (size_t)h<sizeof(header) && send_all(fd,header,(size_t)h) && (!body_n || send_all(fd,body,body_n));wipe_secret(header,sizeof(header));
    uint64_t deadline=0;if(!monotonic_ms(&deadline))ok=0;deadline+=120000;
    char received[8193];size_t used=0;
    /* Read only through the header boundary, avoiding lost body-prefix bytes. */
    while(ok && used<8192){int n=recv_before_deadline(fd,received+used,1,deadline,5000);if(n!=1){ok=0;break;}used++;if(used>=4 && !memcmp(received+used-4,"\r\n\r\n",4))break;}
    received[used]=0;if(used<4 || memcmp(received+used-4,"\r\n\r\n",4))ok=0;
    int code=0;uint64_t length=0;int has_length=0;
    if(ok){char *p=strstr(received,"\r\n");if(sscanf(received,"HTTP/1.%*d %d",&code)!=1 || !p)ok=0;
        while(ok && p && p[2]){p+=2;char *end=strstr(p,"\r\n");if(!end){ok=0;break;}*end=0;char *colon=strchr(p,':');
            if(colon){*colon=0;char *v=colon+1;while(*v==' ')v++;
                if(equal_ci(p,"Content-Length")){char *last;errno=0;unsigned long long n=strtoull(v,&last,10);if(has_length++ || errno || *last || !*v)ok=0;length=(uint64_t)n;}
                if(equal_ci(p,"Transfer-Encoding"))ok=0;
            }p=end;}
        if(!has_length || length>(png?32*UINT64_C(1024)*1024:cap-1))ok=0;
    }
    int out=-1;char part[MAX_PATH_SIZE];part[0]=0;
    if(ok && png){if(code!=200 || snprintf(part,sizeof(part),"%s.backend.part",png)>=(int)sizeof(part))ok=0;
        else{
#ifdef _WIN32
            out=_open(part,_O_WRONLY|_O_CREAT|_O_EXCL|_O_BINARY,_S_IREAD|_S_IWRITE);
#else
            out=open(part,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
#endif
            if(out<0)ok=0;}}
    uint64_t total=0;char buf[16384];
    while(ok && total<length){int want=(int)(length-total>sizeof(buf)?sizeof(buf):length-total);int n=recv_before_deadline(fd,buf,want,deadline,5000);if(n<=0){ok=0;break;}
        if(png){size_t sent=0;while(sent<(size_t)n){
#ifdef _WIN32
            int w=_write(out,buf+sent,(unsigned)((size_t)n-sent));
#else
            ssize_t w=write(out,buf+sent,(size_t)n-sent);
#endif
            if(w<0 && errno==EINTR)continue;if(w<=0){ok=0;break;}sent+=(size_t)w;}}
        else memcpy(json+(size_t)total,buf,(size_t)n);total+=(uint64_t)n;
    }
    close_socket(fd);if(json && !png)json[(size_t)total]=0;
    if(out>=0){
#ifdef _WIN32
        if(_commit(out))ok=0;if(_close(out))ok=0;
#else
        if(fsync(out))ok=0;if(close(out))ok=0;
#endif
        if(ok && rename(part,png))ok=0;if(!ok)remove(part);
    }
    if(http_status)*http_status=code;return ok && total==length;
}

static int backend_http(const char *method,const char *path,const char *body,char *json,size_t cap,const char *png,int *status){
    return backend_http_bytes(method,path,body,body?strlen(body):0,"application/json",json,cap,png,status);
}
static int backend_upload_asset(const unsigned char *data,size_t size,const char *type,const char *path,char id[33],char *error,size_t cap){
    char raw[1024];int code=0;object *o=malloc(sizeof(*o));
    int ok=o && backend_http_bytes("POST",path,data,size,type,raw,sizeof(raw),NULL,&code) && code==201 && parse_object(raw,o);
    field *f=ok?find_field(o,"image_id"):NULL;ok=f && f->type==1 && valid_id(f->value);
    if(ok)strcpy(id,f->value);else snprintf(error,cap,"Native image upload unconfirmed or rejected (HTTP%d); generation was not submitted",code);
    free(o);return ok;
}
static int backend_upload_image(const unsigned char *data,size_t size,const char *type,char id[33],char *error,size_t cap){return backend_upload_asset(data,size,type,"/api/inputs",id,error,cap);}
static int backend_upload_mask(const unsigned char *data,size_t size,const char *type,int grayscale,char id[33],char *error,size_t cap){return backend_upload_asset(data,size,type,grayscale?"/api/masks/grayscale":"/api/masks",id,error,cap);}
static int backend_input_dimensions(const char *id,uint32_t *w,uint32_t *h){
    char path[64],raw[1024];int code=0;int64_t x,y;object *o=malloc(sizeof(*o));
    snprintf(path,sizeof(path),"/api/inputs/%s",id);
    int ok=o && valid_id(id) && backend_http("GET",path,NULL,raw,sizeof(raw),NULL,&code) && code==200 && parse_object(raw,o) &&
        get_int(find_field(o,"width"),1,2048,&x) && get_int(find_field(o,"height"),1,2048,&y);
    if(ok){*w=(uint32_t)x;*h=(uint32_t)y;}free(o);return ok;
}
static void backend_delete_input(const char *id){
    if(!valid_id(id))return;char path[64],raw[1024];int code=0;
    snprintf(path,sizeof(path),"/api/inputs/%s",id);
    /* HTTP409 while native inference still owns an input is retained for TTL cleanup. */
    (void)backend_http("DELETE",path,NULL,raw,sizeof(raw),NULL,&code);
}

/* Only for same-console /health responses. Unknown metadata is validated and
 * skipped with a depth bound. Public request/job parsers remain strict/flat.
 * Scratch storage belongs to the caller's heap object, not recursive stacks. */
static int skip_backend_metadata(const char **p,unsigned depth,char *scratch,size_t cap) {
    if(depth>8)return 0;
    skip_space(p);
    if(**p=='"')return parse_string(p,scratch,cap);
    if(**p=='-' || isdigit((unsigned char)**p))return parse_number(p,scratch,cap);
    if(!strncmp(*p,"true",4)){*p+=4;return 1;}
    if(!strncmp(*p,"false",5)){*p+=5;return 1;}
    if(!strncmp(*p,"null",4)){*p+=4;return 1;}
    char close=**p=='{'?'}':(**p=='['?']':0);int object_kind=**p=='{';
    if(!close)return 0;(*p)++;skip_space(p);if(**p==close){(*p)++;return 1;}
    for(;;){
        if(object_kind){if(!parse_string(p,scratch,cap))return 0;skip_space(p);if(*(*p)++!=':')return 0;skip_space(p);}
        if(!skip_backend_metadata(p,depth+1,scratch,cap))return 0;
        skip_space(p);if(**p==close){(*p)++;return 1;}
        if(*(*p)++!=',')return 0;skip_space(p);
    }
}

static int parse_backend_health_object(const char *p, object *obj) {
    memset(obj, 0, sizeof(*obj)); skip_space(&p);
    if (*p++ != '{') return 0;
    skip_space(&p);
    if (*p == '}') { ++p; skip_space(&p); return *p == 0; }
    for (;;) {
        if (obj->count == MAX_FIELDS || *p != '"') return 0;
        field *f = &obj->values[obj->count];
        if (!parse_string(&p, f->key, sizeof(f->key))) return 0;
        for (size_t i = 0; i < obj->count; ++i)
            if (!strcmp(f->key, obj->values[i].key)) return 0;
        skip_space(&p); if (*p++ != ':') return 0; skip_space(&p);
        if (*p == '"') {
            f->type = 1;
            if (!parse_string(&p, f->value, sizeof(f->value))) return 0;
        } else if (*p == '-' || isdigit((unsigned char)*p)) {
            f->type = 2;
            if (!parse_number(&p, f->value, sizeof(f->value))) return 0;
        } else if (!strncmp(p, "true", 4)) { f->type = 3; p += 4; }
        else if (!strncmp(p, "false", 5)) { f->type = 4; p += 5; }
        else if (!strncmp(p, "null", 4)) { f->type = 5; p += 4; }
        else if (*p=='{' || *p=='[') {
            f->type=6;
            if(!skip_backend_metadata(&p,0,f->value,sizeof(f->value)))return 0;
            f->value[0]=0;
        }
        else return 0;
        ++obj->count; skip_space(&p);
        if (*p == '}') { ++p; skip_space(&p); return *p == 0; }
        if (*p++ != ',') return 0;
        skip_space(&p);
    }
}

static int parse_backend_response_object(const char *path,const char *raw,object *o) {
    if(!strcmp(path,"/health"))return parse_backend_health_object(raw,o);
    /* Native202 admission and GET job snapshots now contain36 fields. This
     * private flat parser cap must not relax public request admission. */
    if(!strcmp(path,"/api/generate") || !strncmp(path,"/api/jobs/",10))
        return parse_object_limit(raw,o,MAX_BACKEND_JOB_FIELDS);
    return parse_object(raw,o);
}
static int backend_json(const char *method,const char *path,const char *body,object *o,int *code) {
    char *raw=malloc(16385);if(!raw)return 0;int ok=backend_http(method,path,body,raw,16385,NULL,code) && parse_backend_response_object(path,raw,o);free(raw);return ok;
}
static int backend_true(object *o,const char *key){field *f=find_field(o,key);return f && f->type==3;}
static const char *backend_text(object *o,const char *key){field *f=find_field(o,key);return f && f->type==1?f->value:"";}
static void backend_status(ps5_ai_engine_status *s) {
    memset(s,0,sizeof(*s));snprintf(s->backend,sizeof(s->backend),"HTTP gateway -> same-PS5 MOSS Image native API%d",PS5_AI_UPSTREAM_PORT);
    object *o=malloc(sizeof(*o));int code=0;
    if(!o || !backend_json("GET","/health",NULL,o,&code) || code!=200){snprintf(s->reason,sizeof(s->reason),"Same-PS5 MOSS Image native API%d is unavailable",PS5_AI_UPSTREAM_PORT);free(o);return;}
    int model_ok=!strcmp(backend_text(o,"model_id"),"moss-image-sft") && !strcmp(backend_text(o,"model"),"/app0/models/MOSS-Image-SFT-Q4_K.gguf");
    s->available=model_ok && backend_true(o,"engine_available");s->gpu_available=model_ok && backend_true(o,"gpu_available");s->gpu_verified=model_ok && backend_true(o,"gpu_verified");s->model_loaded=model_ok && backend_true(o,"model_loaded");
    snprintf(s->device,sizeof(s->device),"%s",backend_text(o,"gpu_device"));snprintf(s->driver,sizeof(s->driver),"%s",backend_text(o,"gpu_driver"));snprintf(s->model,sizeof(s->model),"%s",backend_text(o,"model"));
    if(!model_ok)snprintf(s->reason,sizeof(s->reason),"Native backend model is not configured MOSS Image SFT");else snprintf(s->reason,sizeof(s->reason),"%s",backend_text(o,"reason"));free(o);
}
int ps5_ai_engine_init(const char *config,char *error,size_t n){(void)config;ps5_ai_engine_status s;backend_status(&s);if(!s.available){snprintf(error,n,"%s",s.reason);return -1;}return 0;}
void ps5_ai_engine_get_status(ps5_ai_engine_status *s){backend_status(s);}
/* A full-edit latent mask legitimately skips the blend expression. Its zero
 * blend-evaluation count is not a missing GPU generation: require completed
 * native GPU proof, prepared mask and sane, nonempty final composition stats.
 * The same applies when NearestMax removes a tiny preserved pixel region. */
static int backend_mask_completion_verified(const ps5_ai_request *r,object *o,ps5_ai_stats *stats){
    if(!r->mask_image_id[0])return 1;
    int64_t evaluations,preserved,changed;
    if(strcmp(backend_text(o,"status"),"completed") || !backend_true(o,"gpu_used") ||
       !backend_true(o,"inpainting") || !backend_true(o,"mask_prepared") ||
       !get_int(find_field(o,"mask_denoise_evaluations"),0,INT_MAX,&evaluations) ||
       !get_int(find_field(o,"mask_preserved_pixels"),0,INT64_MAX,&preserved) ||
       !get_int(find_field(o,"mask_changed_pixels"),1,INT64_MAX,&changed) ||
       r->width<=0 || r->height<=0)return 0;
    const uint64_t pixels=(uint64_t)r->width*(uint64_t)r->height;
    if((uint64_t)preserved>pixels || (uint64_t)changed>pixels-(uint64_t)preserved)return 0;
    stats->mask_prepared=1;
    stats->mask_denoise_evaluations=(int)evaluations;
    stats->mask_preserved_pixels=(uint64_t)preserved;
    stats->mask_changed_pixels=(uint64_t)changed;
    return 1;
}
int ps5_ai_engine_generate(const ps5_ai_request *r,const char *output,ps5_ai_stats *stats,ps5_ai_progress_cb cb,void *user,char *error,size_t cap){
    memset(stats,0,sizeof(*stats));char *body=malloc(51000),*prompt=malloc(24585),*negative=malloc(24585);object *o=malloc(sizeof(*o));int ok=0,code=0;
    if(!body || !prompt || !negative || !o){snprintf(error,cap,"Insufficient memory forwarding request");goto done;}
    if(!quote_json(prompt,24585,r->prompt) || !quote_json(negative,24585,r->negative_prompt)){snprintf(error,cap,"Unable to encode prompt");goto done;}
    int n=snprintf(body,51000,"{\"model\":\"moss-image-sft\",\"prompt\":%s,\"negative_prompt\":%s,\"width\":%d,\"height\":%d,\"steps\":%d,\"cfg_scale\":%.17g,\"flow_shift\":%.17g,\"seed\":%" PRId64 ",\"sampler\":\"%s\",\"manga_strength\":%.17g,\"manga_monochrome\":%s",prompt,negative,r->width,r->height,r->steps,r->cfg_scale,r->flow_shift,r->seed,r->sampler,r->manga_strength,r->manga_monochrome?"true":"false");
    if(n<0 || n>MAX_BODY){snprintf(error,cap,"Encoded request too large");goto done;}
    if(r->init_image_id[0])n+=snprintf(body+n,51000-(size_t)n,",\"init_image_id\":\"%s\",\"denoise_strength\":%.17g",r->init_image_id,r->denoise_strength);
    if(r->reference_image_id[0])n+=snprintf(body+n,51000-(size_t)n,",\"reference_image_id\":\"%s\",\"ip_adapter_strength\":%.17g",r->reference_image_id,r->ip_adapter_strength);
    if(r->mask_image_id[0])n+=snprintf(body+n,51000-(size_t)n,",\"mask_image_id\":\"%s\"",r->mask_image_id);
    if(r->controlnet[0])n+=snprintf(body+n,51000-(size_t)n,",\"controlnet\":\"%s\",\"control_image_id\":\"%s\",\"control_strength\":%.17g",r->controlnet,r->control_image_id,r->control_strength);
    n+=snprintf(body+n,51000-(size_t)n,"}");
    if(n<0 || n>MAX_BODY){snprintf(error,cap,"Encoded native request exceeds24KiB");goto done;}
    /* Exactly one upstream POST. Never retry inference automatically. */
    if(!backend_json("POST","/api/generate",body,o,&code) || code!=202){snprintf(error,cap,"Native generation request unconfirmed or rejected (HTTP%d); no automatic retry",code);goto done;}
    char native_id[33];const char *id=backend_text(o,"job_id");if(!valid_id(id)){snprintf(error,cap,"Native job identity invalid; no retry");goto done;}strcpy(native_id,id);
    lock_state();snprintf(((job *)user)->native_job_id,33,"%s",native_id);unlock_state();
    log_line("Gateway job %s -> native job %s",((job *)user)->id,native_id);
    char path[100];snprintf(path,sizeof(path),"/api/jobs/%s",native_id);uint64_t deadline=0;if(!monotonic_ms(&deadline)){snprintf(error,cap,"Monotonic clock unavailable; native job may continue");goto done;}deadline+=OPENAI_WAIT_MS;
    for(;;){uint64_t now=0;if(!monotonic_ms(&now) || now>=deadline || stopping){snprintf(error,cap,"Native job wait stopped; native job may continue; no retry");goto done;}
        if(!backend_json("GET",path,NULL,o,&code) || code!=200){snprintf(error,cap,"Native job observation failed; no retry");goto done;}
        if(strcmp(backend_text(o,"job_id"),native_id)){snprintf(error,cap,"Native job identity changed");goto done;}
        const char *state=backend_text(o,"status");
        if(!strcmp(state,"failed")){snprintf(error,cap,"Native job failed: %.640s",backend_text(o,"error"));goto done;}
        double progress=0;get_double(find_field(o,"progress"),0,1,&progress);if(cb)cb(progress,user);
        if(!strcmp(state,"completed")){
            if(!backend_true(o,"gpu_used")){snprintf(error,cap,"Native job lacks completed GPU verification");goto done;}
            stats->gpu_used=1;get_double(find_field(o,"generation_seconds"),0,DBL_MAX,&stats->generation_seconds);int64_t v=0;
            if(get_int(find_field(o,"peak_cpu_bytes"),0,INT64_MAX,&v))stats->peak_cpu_bytes=(uint64_t)v;
            v=0;if(get_int(find_field(o,"peak_gpu_bytes"),0,INT64_MAX,&v))stats->peak_gpu_bytes=(uint64_t)v;
            stats->gpu_controlnet_verified=backend_true(o,"gpu_controlnet_verified");
            v=0;if(get_int(find_field(o,"controlnet_evals"),0,INT_MAX,&v))stats->controlnet_evals=(int)v;
            v=0;if(get_int(find_field(o,"controlnet_tensors"),0,INT_MAX,&v))stats->controlnet_tensors=(int)v;
            get_double(find_field(o,"controlnet_seconds"),0,DBL_MAX,&stats->controlnet_seconds);
            if(r->controlnet[0] && (!stats->gpu_controlnet_verified || stats->controlnet_evals<1 || stats->controlnet_tensors!=10)){
                snprintf(error,cap,"Native job lacks actual ControlNet GPU/residual proof");goto done;
            }
            if(!backend_mask_completion_verified(r,o,stats)){
                snprintf(error,cap,"Native job lacks actual masked diffusion/composition proof");goto done;
            }
            snprintf(path,sizeof(path),"/api/images/%s",native_id);
            if(!backend_http("GET",path,NULL,NULL,0,output,&code) || code!=200){snprintf(error,cap,"Native PNG retrieval failed");goto done;}
            ok=1;break;
        }
        if(strcmp(state,"running") && strcmp(state,"queued")){snprintf(error,cap,"Unknown native job state");goto done;}pause_ms(1000);
    }
done:
    if(r->transient_inputs){
        if(r->init_image_id[0])backend_delete_input(r->init_image_id);
        if(r->reference_image_id[0] && strcmp(r->init_image_id,r->reference_image_id))backend_delete_input(r->reference_image_id);
        if(r->mask_image_id[0] && strcmp(r->mask_image_id,r->init_image_id) && strcmp(r->mask_image_id,r->reference_image_id))backend_delete_input(r->mask_image_id);
        /* control_image_id is an existing caller-owned asset, never uploaded
         * by this multipart endpoint. Its native lease must be released there;
         * the gateway must not delete that persistent prepared control map. */
    }
    free(body);free(prompt);free(negative);free(o);return ok?0:-1;
}
/* Gateway shutdown must NEVER unload/stop/restart the native GPU application. */
void ps5_ai_engine_shutdown(void){}
int ps5_ai_engine_shutdown_checked(char *error,size_t n){(void)error;(void)n;return 0;}
