#define main fixture_server_main
#include "server.c"
#undef main
static unsigned checks;
#define CHECK(v) do {if(!(v)){fprintf(stderr,"FAIL line %d\n",__LINE__);return 1;}checks++;}while(0)
static void fields(char *out,unsigned count){size_t used=0;out[used++]='{';for(unsigned i=0;i<count;i++)used+=(size_t)sprintf(out+used,"%s\"f%u\":%u",i?",":"",i,i);strcpy(out+used,"}");}
int main(int argc,char **argv){
 CHECK(argc==2);object *o=calloc(1,sizeof(*o));CHECK(o);char text[JOB_JSON_SIZE];
 fields(text,32);CHECK(parse_object(text,o)&&o->count==32);
 fields(text,33);CHECK(!parse_object(text,o));CHECK(parse_job_object(text,o)&&o->count==33);
 fields(text,36);CHECK(!parse_object(text,o));CHECK(parse_job_object(text,o)&&o->count==36);
 fields(text,64);CHECK(parse_job_object(text,o)&&o->count==64);
 fields(text,65);CHECK(!parse_job_object(text,o));
 CHECK(!parse_job_object("{\"a\":1,\"a\":2}",o));CHECK(!parse_job_object("{\"a\":{}}",o));
 job j={0};strcpy(j.id,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");strcpy(j.status,"running");
 strcpy(j.request.mask_image_id,"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");strcpy(j.request.mask_image_sha256,"cafeface");
 j.request.width=j.request.height=256;CHECK(serialize_job(&j,text,sizeof(text)));
 CHECK(parse_job_object(text,o)&&o->count==36);CHECK(!parse_object(text,o));CHECK(parse_job_object(text,o));
 CHECK(find_field(o,"inpainting")->type==3);CHECK(!strcmp(find_field(o,"mask_image_sha256")->value,"cafeface"));
 CHECK(!serialize_job(&j,text,16));
 memset(input_records,0,sizeof(input_records));
 input_record *rgb=&input_records[0],*mask=&input_records[1];
 memset(rgb->id,'1',32);memset(mask->id,'2',32);rgb->occupied=mask->occupied=1;mask->is_mask=1;
 rgb->expires=mask->expires=(int64_t)time(NULL)+1000;rgb->width=rgb->height=mask->width=mask->height=256;
 strcpy(rgb->sha,"rgb-sha");strcpy(mask->sha,"mask-sha");
 ps5_ai_request q={0};q.width=q.height=256;
 strcpy(q.init_image_id,rgb->id);strcpy(q.reference_image_id,rgb->id);strcpy(q.control_image_id,rgb->id);strcpy(q.mask_image_id,mask->id);
 CHECK(input_pin_request(&q)==1);CHECK(rgb->refs==3&&mask->refs==1);
 CHECK(strstr(q.mask_image_path,".mask")&&strstr(q.control_image_path,".rgb"));
 CHECK(!strcmp(q.mask_image_sha256,"mask-sha")&&!strcmp(q.control_image_sha256,"rgb-sha"));
 input_unpin_request(&q);CHECK(rgb->refs==0&&mask->refs==0);
 strcpy(q.mask_image_id,rgb->id);CHECK(input_pin_request(&q)==-1);CHECK(rgb->refs==0&&mask->refs==0);
 strcpy(q.mask_image_id,mask->id);strcpy(q.control_image_id,mask->id);CHECK(input_pin_request(&q)==-1);CHECK(rgb->refs==0&&mask->refs==0);
 strcpy(q.control_image_id,rgb->id);q.width=128;CHECK(input_pin_request(&q)==-1);CHECK(rgb->refs==0&&mask->refs==0);q.width=256;
 strcpy(q.mask_image_id,"ffffffffffffffffffffffffffffffff");CHECK(input_pin_request(&q)==0);CHECK(rgb->refs==0&&mask->refs==0);
 strcpy(q.mask_image_id,mask->id);q.init_image_id[0]=0;CHECK(input_pin_request(&q)==-1);CHECK(rgb->refs==0&&mask->refs==0);
 char path[1024],hash[65];uint32_t w,h;uint64_t bytes;unsigned char data[192];memset(data,19,sizeof(data));unsigned char *loaded=NULL;
 snprintf(path,sizeof(path),"%s/write.rgb",argv[1]);CHECK(ps5_ai_input_write(path,data,8,8));CHECK(!ps5_ai_input_write(path,data,8,8));
 CHECK(ps5_ai_input_info(path,&w,&h,&bytes,hash)&&w==8&&h==8&&bytes==208);CHECK(ps5_ai_input_read(path,&loaded,&w,&h)&&!memcmp(loaded,data,192));free(loaded);
 CHECK(!ps5_ai_mask_info(path,&w,&h,&bytes,hash));CHECK(!ps5_ai_mask_read(path,&loaded,&w,&h));
 snprintf(path,sizeof(path),"%s/write.mask",argv[1]);CHECK(ps5_ai_mask_write(path,data,8,8));CHECK(!ps5_ai_mask_write(path,data,8,8));
 CHECK(ps5_ai_mask_info(path,&w,&h,&bytes,hash)&&w==8&&h==8&&bytes==80);CHECK(ps5_ai_mask_read(path,&loaded,&w,&h)&&!memcmp(loaded,data,64));free(loaded);
 CHECK(!ps5_ai_input_info(path,&w,&h,&bytes,hash));CHECK(!ps5_ai_input_read(path,&loaded,&w,&h));
 for(unsigned i=0;i<4;i++){snprintf(path,sizeof(path),"%s/bad%u.mask",argv[1],i);CHECK(!ps5_ai_mask_info(path,&w,&h,&bytes,hash)&&!bytes&&!hash[0]);}
 free(o);printf("{\"checks_passed\":%u,\"PS5_connections\":0,\"model_inference\":false}\n",checks);return 0;
}
