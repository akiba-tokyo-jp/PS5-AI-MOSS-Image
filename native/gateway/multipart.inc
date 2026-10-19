/* SPDX-License-Identifier: MIT
 * Bounded multipart image edits, one source and optional separate IP reference.
 * Filenames are never interpreted. No remote URLs or automatic generation retry.
 */
#define EDIT_IMAGE_LIMIT (8U*1024U*1024U)
#define EDIT_MASK_LIMIT (4U*1024U*1024U)
#define MAX_EDIT_BODY (2U*EDIT_IMAGE_LIMIT+EDIT_MASK_LIMIT+MAX_BODY)
typedef struct {char name[64],type[64];const unsigned char *data;size_t size;int file;} edit_part;
static int backend_upload_image(const unsigned char *,size_t,const char *,char[33],char *,size_t);
static int backend_upload_mask(const unsigned char *,size_t,const char *,int,char[33],char *,size_t);
static int backend_input_dimensions(const char *,uint32_t *,uint32_t *);
static void backend_delete_input(const char *);
static const unsigned char *find_bytes(const unsigned char *p,size_t n,const char *needle,size_t size) {
    if(size>n)return NULL;
    for(size_t i=0;i<=n-size;i++)if(!memcmp(p+i,needle,size))return p+i;
    return NULL;
}
static int form_disposition(const char *value,edit_part *part) {
    const char *p=value;char kind[32];size_t used=0;
    while(*p && *p!=';'){if(used+1>=sizeof(kind))return 0;kind[used++]=*p++;}
    while(used && (kind[used-1]==' ' || kind[used-1]=='\t'))used--;kind[used]=0;
    if(!equal_ci(kind,"form-data"))return 0;
    int name_seen=0,filename_seen=0,encoded_filename_seen=0;
    while(*p){
        if(*p++!=';')return 0;while(*p==' ' || *p=='\t')p++;
        char key[32],text[2049];used=0;
        while(*p && *p!='=' && *p!=' ' && *p!='\t'){
            if(used+1>=sizeof(key))return 0;key[used++]=*p++;
        }
        key[used]=0;while(*p==' ' || *p=='\t')p++;if(*p++!='=')return 0;
        while(*p==' ' || *p=='\t')p++;used=0;
        if(*p=='\"'){
            p++;while(*p && *p!='\"'){
                if(*p=='\\'){p++;if(!*p)return 0;}
                if((unsigned char)*p<32 || used+1>=sizeof(text))return 0;text[used++]=*p++;
            }
            if(*p++!='\"')return 0;
        }else{
            while(*p && *p!=';'){
                if((unsigned char)*p<32 && *p!='\t')return 0;
                if(used+1>=sizeof(text))return 0;text[used++]=*p++;
            }
            while(used && (text[used-1]==' ' || text[used-1]=='\t'))used--;
        }
        text[used]=0;while(*p==' ' || *p=='\t')p++;if(*p && *p!=';')return 0;
        if(equal_ci(key,"name")){
            if(name_seen++ || !used || used>=sizeof(part->name))return 0;
            for(size_t i=0;i<used;i++)if(!isalnum((unsigned char)text[i]) && !strchr("_[]",text[i]))return 0;
            strcpy(part->name,text);
        }else if(equal_ci(key,"filename")){
            if(filename_seen++)return 0;part->file=1;
        }else if(equal_ci(key,"filename*")){
            if(encoded_filename_seen++)return 0;part->file=1;
        }else return 0;
    }
    return name_seen==1;
}
static int multipart_parts(const char *type,const unsigned char *body,size_t size,edit_part *parts,unsigned *count,char *error,size_t cap) {
    const char *prefix="multipart/form-data; boundary=";
    if(strncmp(type,prefix,strlen(prefix))){snprintf(error,cap,"Expected multipart/form-data with boundary");return 0;}
    const char *b=type+strlen(prefix);size_t n=strlen(b);
    if(n>=2 && b[0]=='\"' && b[n-1]=='\"'){b++;n-=2;}
    if(!n || n>70){snprintf(error,cap,"Invalid multipart boundary length");return 0;}
    for(size_t i=0;i<n;i++)if(!isalnum((unsigned char)b[i]) && !strchr("'()+_,-./:=?",b[i])){snprintf(error,cap,"Invalid multipart boundary");return 0;}
    char marker[80];memcpy(marker,"\r\n--",4);memcpy(marker+4,b,n);marker[n+4]=0;
    if(size<n+4 || memcmp(body,"--",2) || memcmp(body+2,b,n) || memcmp(body+2+n,"\r\n",2)){snprintf(error,cap,"Invalid initial multipart delimiter");return 0;}
    const unsigned char *p=body+n+4,*end=body+size;*count=0;
    while(p<end){
        if(*count>=24){snprintf(error,cap,"Too many multipart fields");return 0;}
        const unsigned char *h_end=find_bytes(p,(size_t)(end-p),"\r\n\r\n",4);
        if(!h_end || h_end-p>2048 || memchr(p,0,(size_t)(h_end-p))){snprintf(error,cap,"Invalid multipart part headers");return 0;}
        char headers[2050];memcpy(headers,p,(size_t)(h_end-p));headers[h_end-p]=0;
        edit_part *part=&parts[(*count)++];memset(part,0,sizeof(*part));
        int disposition=0,has_type=0;char *line=headers;
        for(;;){
            char *next=strstr(line,"\r\n");if(next)*next=0;char *colon=strchr(line,':');
            if(!colon){snprintf(error,cap,"Invalid part header");return 0;}*colon=0;char *value=colon+1;while(*value==' ' || *value=='\t')value++;
            if(equal_ci(line,"Content-Disposition")){
                if(disposition++ || !form_disposition(value,part)){snprintf(error,cap,"Invalid form-data disposition");return 0;}
            }else if(equal_ci(line,"Content-Type")){
                if(has_type++ || strlen(value)>=sizeof(part->type)){snprintf(error,cap,"Invalid image Content-Type");return 0;}strcpy(part->type,value);
            }else{snprintf(error,cap,"Unsupported multipart part header");return 0;}
            if(!next)break;line=next+2;
        }
        if(!disposition){snprintf(error,cap,"Missing form-data disposition");return 0;}
        for(unsigned i=0;i+1<*count;i++)if(!strcmp(parts[i].name,part->name) ||
            ((!strcmp(parts[i].name,"image") || !strcmp(parts[i].name,"image[]")) &&
             (!strcmp(part->name,"image") || !strcmp(part->name,"image[]")))){snprintf(error,cap,"Duplicate multipart field");return 0;}
        p=h_end+4;const unsigned char *next=find_bytes(p,(size_t)(end-p),marker,n+4);
        if(!next){snprintf(error,cap,"Missing closing multipart boundary");return 0;}
        part->data=p;part->size=(size_t)(next-p);p=next+n+4;
        if(part->file){
            int mask=!strcmp(part->name,"mask");
            if(!part->size || (mask?part->size>=EDIT_MASK_LIMIT:part->size>EDIT_IMAGE_LIMIT) ||
               (mask?strcmp(part->type,"image/png"):(strcmp(part->type,"image/png") && strcmp(part->type,"image/jpeg")))){snprintf(error,cap,"Image requires PNG/JPEG up to8MiB; mask requires PNG below4MiB");return 0;}
        }else if(part->size>4096 || memchr(part->data,0,part->size)){snprintf(error,cap,"Invalid/oversized text field");return 0;}
        if(end-p>=2 && !memcmp(p,"--",2)){
            p+=2;if(end-p==2 && !memcmp(p,"\r\n",2))p+=2;
            if(p!=end){snprintf(error,cap,"Unexpected multipart trailing bytes");return 0;}
            return 1;
        }
        if(end-p<2 || memcmp(p,"\r\n",2)){snprintf(error,cap,"Invalid part delimiter");return 0;}p+=2;
    }
    snprintf(error,cap,"Truncated multipart body");return 0;
}
static int multipart_json(edit_part *parts,unsigned count,const char *init,const char *reference,const char *mask,char *json,size_t cap,char *error,size_t error_cap) {
    size_t used=1;json[0]='{';int fields=0;
    for(unsigned i=0;i<count;i++){
        edit_part *p=&parts[i];if(p->file || !strcmp(p->name,"reference_mode") || !strcmp(p->name,"mask_mode"))continue;
        if(!strcmp(p->name,"init_image_id") || !strcmp(p->name,"reference_image_id") || !strcmp(p->name,"mask_image_id")){snprintf(error,error_cap,"Use uploaded image fields for edits");return 0;}
        char text[4097],encoded[24585];memcpy(text,p->data,p->size);text[p->size]=0;
        if(!valid_utf8((const unsigned char *)text)){snprintf(error,error_cap,"Invalid UTF-8 form field");return 0;}
        int numeric=!strcmp(p->name,"n") || !strcmp(p->name,"steps") || !strcmp(p->name,"cfg_scale") ||
            !strcmp(p->name,"seed") || !strcmp(p->name,"manga_strength") || !strcmp(p->name,"denoise_strength") || !strcmp(p->name,"ip_adapter_strength") || !strcmp(p->name,"control_strength");
        if(numeric){const char *q=text;char number[128];if(!parse_number(&q,number,sizeof(number)) || *q){snprintf(error,error_cap,"Invalid numeric form field");return 0;}strcpy(encoded,number);}
        else if(!strcmp(p->name,"stream")){
            if(strcmp(text,"false")){snprintf(error,error_cap,"Streaming is unsupported");return 0;}strcpy(encoded,"false");
        }else if(!strcmp(p->name,"manga_monochrome")){
            if(strcmp(text,"false") && strcmp(text,"true")){snprintf(error,error_cap,"manga_monochrome must be true or false");return 0;}strcpy(encoded,text);
        }else if(!quote_json(encoded,sizeof(encoded),text)){snprintf(error,error_cap,"Unable to encode form field");return 0;}
        int n=snprintf(json+used,cap-used,"%s\"%s\":%s",fields++?",":"",p->name,encoded);
        if(n<0 || (size_t)n>=cap-used){snprintf(error,error_cap,"Edited request exceeds JSON limit");return 0;}used+=(size_t)n;
    }
    const char *names[]={"init_image_id","reference_image_id","mask_image_id"},*ids[]={init,reference,mask};
    for(unsigned i=0;i<3;i++)if(ids[i][0]){
        int n=snprintf(json+used,cap-used,"%s\"%s\":\"%s\"",fields++?",":"",names[i],ids[i]);
        if(n<0 || (size_t)n>=cap-used){snprintf(error,error_cap,"Edited request exceeds JSON limit");return 0;}used+=(size_t)n;
    }
    if(used+2>cap)return 0;json[used++]='}';json[used]=0;return 1;
}
static int openai_edit_submit(socket_t fd,const unsigned char *body,size_t size,const char *type) {
    /* Keep the inherited parser source for a future MOSS Image-compatible backend,
     * but never upload or enqueue an unsupported SDXL image-conditioning job. */
    const int moss_image_edits_supported=0;
    if(!moss_image_edits_supported){openai_error(fd,400,"MOSS Image SFT currently supports text-to-image only; image edits, masks and IP-Adapter are unsupported","image");return 0;}
    edit_part parts[24];unsigned count;char error[768],param[64]={0},mode[32]="img2img";
    if(!multipart_parts(type,body,size,parts,&count,error,sizeof(error))){openai_error(fd,400,error,NULL);return 0;}
    edit_part *image=NULL,*reference_image=NULL,*mask=NULL;int grayscale=0,has_mask_mode=0;
    for(unsigned i=0;i<count;i++){
        edit_part *p=&parts[i];
        if(!strcmp(p->name,"image") || !strcmp(p->name,"image[]")){if(!p->file){openai_error(fd,400,"image must be a file","image");return 0;}image=p;}
        else if(!strcmp(p->name,"reference_image")){if(!p->file){openai_error(fd,400,"reference_image must be a file","reference_image");return 0;}reference_image=p;}
        else if(!strcmp(p->name,"mask")){if(!p->file){openai_error(fd,400,"mask must be a PNG file","mask");return 0;}mask=p;}
        else if(!strcmp(p->name,"mask_mode")){
            if(p->file || !((p->size==5&&!memcmp(p->data,"alpha",5)) || (p->size==9&&!memcmp(p->data,"grayscale",9)))){openai_error(fd,400,"mask_mode must be alpha or grayscale","mask_mode");return 0;}
            grayscale=p->size==9;has_mask_mode=1;
        }
        else if(!strcmp(p->name,"reference_mode")){
            if(p->file || !p->size || p->size>=sizeof(mode)){openai_error(fd,400,"Invalid reference_mode","reference_mode");return 0;}
            memcpy(mode,p->data,p->size);mode[p->size]=0;
        }else if(p->file){openai_error(fd,400,"Unsupported image field",p->name);return 0;}
    }
    if(!image){openai_error(fd,400,"One image file is required","image");return 0;}
    int init=!strcmp(mode,"img2img") || !strcmp(mode,"combined") || !strcmp(mode,"inpaint"),ip=!strcmp(mode,"ip_adapter") || !strcmp(mode,"combined");
    if((mask && !init) || (has_mask_mode && !mask) || (!strcmp(mode,"inpaint") && !mask)){openai_error(fd,400,"Mask requires img2img/combined; inpaint requires a mask","mask");return 0;}
    if((!init && !ip) || (reference_image && strcmp(mode,"combined"))){openai_error(fd,400,"reference_mode must be img2img, ip_adapter or combined; separate reference_image requires combined","reference_mode");return 0;}
    char *json=malloc(MAX_BODY+1);ps5_ai_request request;int url_format;
    const char placeholder[]="00000000000000000000000000000000";
    if(!json){openai_error(fd,500,"Out of memory",NULL);return 0;}
    if(!multipart_json(parts,count,init?placeholder:"",ip?placeholder:"",mask?placeholder:"",json,MAX_BODY+1,error,sizeof(error)) ||
       !openai_validate(json,&request,&url_format,error,sizeof(error),param)){
        openai_error(fd,400,error,param[0]?param:NULL);free(json);return 0;
    }
    lock_state();int busy=openai_busy || stopping || !openai_thread_ready;unlock_state();
    if(busy){openai_error(fd,429,"One image request is already active",NULL);free(json);return 0;}
    char first_id[33]={0},second_id[33]={0},mask_id[33]={0};
    if(!backend_upload_image(image->data,image->size,image->type,first_id,error,sizeof(error))){openai_error(fd,503,error,"image");free(json);return 0;}
    if(reference_image && !backend_upload_image(reference_image->data,reference_image->size,reference_image->type,second_id,error,sizeof(error))){
        backend_delete_input(first_id);openai_error(fd,503,error,"reference_image");free(json);return 0;
    }
    if(mask && !backend_upload_mask(mask->data,mask->size,mask->type,grayscale,mask_id,error,sizeof(error))){
        backend_delete_input(first_id);if(second_id[0])backend_delete_input(second_id);openai_error(fd,400,error,"mask");free(json);return 0;
    }
    if(mask){
        uint32_t iw,ih,mw,mh;
        if(!backend_input_dimensions(first_id,&iw,&ih) || !backend_input_dimensions(mask_id,&mw,&mh) ||
           iw!=mw || ih!=mh || iw!=(uint32_t)request.width || ih!=(uint32_t)request.height){
            backend_delete_input(first_id);if(second_id[0])backend_delete_input(second_id);backend_delete_input(mask_id);
            openai_error(fd,400,"Masked source, PNG mask and requested output must have identical dimensions","mask");free(json);return 0;
        }
    }
    int accepted=0;
    if(multipart_json(parts,count,init?first_id:"",ip?(reference_image?second_id:first_id):"",mask_id,json,MAX_BODY+1,error,sizeof(error)))accepted=openai_submit(fd,json,1);
    else openai_error(fd,400,"Unable to encode edited request",NULL);
    if(!accepted){backend_delete_input(first_id);if(second_id[0])backend_delete_input(second_id);if(mask_id[0])backend_delete_input(mask_id);}
    free(json);return accepted;
}
