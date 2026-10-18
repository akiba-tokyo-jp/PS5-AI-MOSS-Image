/* SPDX-License-Identifier: MIT
 * Included by server.c. State mutex serializes leases with job admission.
 */
#define INPUT_SLOTS 32
#define INPUT_QUOTA (128U * UINT64_C(1024) * 1024)
#define INPUT_TTL (24*60*60)
typedef struct {char id[33],sha[65];uint32_t width,height;uint64_t bytes;int64_t expires;unsigned refs;int occupied;} input_record;
static input_record input_records[INPUT_SLOTS];
static int input_path(char *path,size_t cap,const char *id) {
    char name[40];if(!valid_id(id))return 0;
    snprintf(name,sizeof(name),"%s.rgb",id);return path_for(path,cap,"inputs",name);
}
static input_record *input_find(const char *id) {
    for(unsigned i=0;i<INPUT_SLOTS;i++)if(input_records[i].occupied && !strcmp(input_records[i].id,id))return &input_records[i];
    return NULL;
}
static void input_collect(void) {
    int64_t now=(int64_t)time(NULL);
    for(unsigned i=0;i<INPUT_SLOTS;i++){
        input_record *p=&input_records[i];char path[MAX_PATH_SIZE];
        if(p->occupied && !p->refs && p->expires<=now && input_path(path,sizeof(path),p->id) && !remove(path))p->occupied=0;
    }
}
static int input_pin_request(ps5_ai_request *r) {
    input_collect();const char *ids[]={r->init_image_id,r->reference_image_id,r->control_image_id};
    char *paths[]={r->init_image_path,r->reference_image_path,r->control_image_path};char *hashes[]={r->init_image_sha256,r->reference_image_sha256,r->control_image_sha256};
    input_record *found[3]={NULL,NULL,NULL};
    for(unsigned i=0;i<3;i++)if(ids[i][0]){
        found[i]=input_find(ids[i]);
        if(!found[i] || (!found[i]->refs && found[i]->expires<=(int64_t)time(NULL)) ||
           !input_path(paths[i],MAX_PATH_SIZE,ids[i]))return 0;
    }
    for(unsigned i=0;i<3;i++)if(found[i]){found[i]->refs++;strcpy(hashes[i],found[i]->sha);}
    return 1;
}
static void input_unpin_request(const ps5_ai_request *r) {
    const char *ids[]={r->init_image_id,r->reference_image_id,r->control_image_id};
    for(unsigned i=0;i<3;i++)if(ids[i][0]){input_record *p=input_find(ids[i]);if(p && p->refs)p->refs--;}
}
static void input_json(const input_record *p,char *out,size_t cap) {
    snprintf(out,cap,"{\"image_id\":\"%s\",\"width\":%u,\"height\":%u,\"bytes\":%" PRIu64 ",\"sha256\":\"%s\",\"expires_at\":%" PRId64 ",\"references\":%u}",
             p->id,p->width,p->height,p->bytes,p->sha,p->expires,p->refs);
}
static void upload_input(socket_t fd,const unsigned char *body,size_t size,const char *type) {
    int png=size>=8 && !memcmp(body,"\211PNG\r\n\032\n",8);
    int jpeg=size>=3 && body[0]==255 && body[1]==216 && body[2]==255;
    if((strcmp(type,"image/png") || !png) && (strcmp(type,"image/jpeg") || !jpeg)){
        send_error(fd,415,"Input requires matching PNG/JPEG Content-Type and signature");return;
    }
    uint32_t w,h;unsigned char *rgb=NULL;
    if(!ps5_ai_input_decode(body,size,&rgb,&w,&h)){send_error(fd,400,"Invalid image; maximum2048 pixels per dimension and4M pixels");return;}
    lock_state();input_collect();input_record *slot=NULL;uint64_t used=0,bytes=16+3*(uint64_t)w*h;
    for(unsigned i=0;i<INPUT_SLOTS;i++)if(input_records[i].occupied)used+=input_records[i].bytes;else if(!slot)slot=&input_records[i];
    if(!slot || used>INPUT_QUOTA || bytes>INPUT_QUOTA-used){unlock_state();free(rgb);send_error(fd,429,"Input image storage quota reached; delete unused inputs");return;}
    input_record value;memset(&value,0,sizeof(value));char path[MAX_PATH_SIZE],part[MAX_PATH_SIZE]={0};
    do{++sequence;snprintf(value.id,sizeof(value.id),"%08" PRIx32 "%08" PRIx32 "%08" PRIx32 "%08" PRIx32,
         (uint32_t)time(NULL),(uint32_t)get_pid(),sequence,sequence^UINT32_C(0x1a6e0000));
    }while(input_find(value.id));
    int ok=input_path(path,sizeof(path),value.id) && snprintf(part,sizeof(part),"%s.part",path)<(int)sizeof(part);
    struct stat st;
    if(ok && !path_metadata(path,&st,0))ok=0; /* Never overwrite another input. */
    if(ok)ok=ps5_ai_input_write(part,rgb,w,h);
    free(rgb);
    if(ok)ok=ps5_ai_input_info(part,&value.width,&value.height,&value.bytes,value.sha);
    if(ok)ok=!rename(part,path);
    if(!ok){if(part[0])remove(part);unlock_state();send_error(fd,500,"Unable to persist input image");return;}
    value.expires=(int64_t)time(NULL)+INPUT_TTL;value.occupied=1;*slot=value;
    char result[512];input_json(&value,result,sizeof(result));unlock_state();send_json(fd,201,result);
}
static void input_metadata(socket_t fd,const char *id,int deleting) {
    if(!valid_id(id)){send_error(fd,404,"Input image not found");return;}
    lock_state();input_collect();input_record *p=input_find(id);
    if(!p){unlock_state();send_error(fd,404,"Input image not found");return;}
    if(deleting){
        if(p->refs){unlock_state();send_error(fd,409,"Input image is referenced by a queued/running job");return;}
        char path[MAX_PATH_SIZE];if(!input_path(path,sizeof(path),id) || remove(path)){unlock_state();send_error(fd,500,"Unable to remove input image");return;}
        p->occupied=0;unlock_state();send_json(fd,200,"{\"deleted\":true}");return;
    }
    char result[512];input_json(p,result,sizeof(result));unlock_state();send_json(fd,200,result);
}
static int recover_inputs(void) {
    char directory[MAX_PATH_SIZE];if(!path_for(directory,sizeof(directory),"inputs",""))return 0;
#ifdef _WIN32
    char pattern[MAX_PATH_SIZE];snprintf(pattern,sizeof(pattern),"%s*.rgb*",directory);
    WIN32_FIND_DATAA entry;HANDLE find=FindFirstFileA(pattern,&entry);
    if(find==INVALID_HANDLE_VALUE)return GetLastError()==ERROR_FILE_NOT_FOUND;
    do{const char *name=entry.cFileName;
#else
    DIR *find=opendir(directory);if(!find)return 0;struct dirent *entry;
    while((entry=readdir(find))){const char *name=entry->d_name;
#endif
        size_t name_size=strlen(name);
        if((name_size!=36 || strcmp(name+32,".rgb")) && (name_size!=41 || strcmp(name+32,".rgb.part")))continue;
        input_record value;memset(&value,0,sizeof(value));memcpy(value.id,name,32);
        char path[MAX_PATH_SIZE];struct stat st;
        if(!input_path(path,sizeof(path),value.id))continue;
        if(name_size==41){
            /* No worker exists at startup. Only this app's exact temporary names. */
            if(snprintf(path,sizeof(path),"%s%s",directory,name)>=(int)sizeof(path) || remove(path))goto fail;
            continue;
        }
        if(path_metadata(path,&st,0))continue;
        if((int64_t)st.st_mtime+INPUT_TTL<=(int64_t)time(NULL)){if(remove(path))goto fail;continue;}
        STARTUP_STAGE("recovered_input_info_before",0);
        if(!ps5_ai_input_info(path,&value.width,&value.height,&value.bytes,value.sha)){STARTUP_STAGE("recovered_input_info_failed",0);goto fail;}
        STARTUP_STAGE("recovered_input_info_passed",1);
        value.expires=(int64_t)st.st_mtime+INPUT_TTL;value.occupied=1;uint64_t used=value.bytes;input_record *slot=NULL;
        for(unsigned i=0;i<INPUT_SLOTS;i++)if(input_records[i].occupied)used+=input_records[i].bytes;else if(!slot)slot=&input_records[i];
        if(!slot || used>INPUT_QUOTA)goto fail;
        *slot=value;
#ifdef _WIN32
    }while(FindNextFileA(find,&entry));FindClose(find);
#else
    }closedir(find);
#endif
    return 1;
fail:
#ifdef _WIN32
    FindClose(find);
#else
    closedir(find);
#endif
    return 0;
}
