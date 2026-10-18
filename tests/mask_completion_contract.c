#define main gateway_program_main
#include "server.c"
#undef main
#include <assert.h>
static unsigned checks;
static void protocol_case(object *o,const ps5_ai_request*r,const char*state,int gpu,int prepared,
                          int evaluations,long long preserved,long long changed,int expected){
 char text[2048];ps5_ai_stats stats={0};
 snprintf(text,sizeof(text),"{\"fixture\":\"SYNTHETIC_BACKEND_METADATA_NO_GPU\",\"status\":\"%s\",\"gpu_used\":%s,\"inpainting\":true,\"mask_prepared\":%s,\"mask_denoise_evaluations\":%d,\"mask_preserved_pixels\":%lld,\"mask_changed_pixels\":%lld}",state,gpu?"true":"false",prepared?"true":"false",evaluations,preserved,changed);
 assert(parse_backend_response_object("/api/jobs/11111111111111111111111111111111",text,o));
 assert(backend_mask_completion_verified(r,o,&stats)==expected);
 if(expected){assert(stats.mask_prepared);assert(stats.mask_denoise_evaluations==evaluations);
  assert(stats.mask_preserved_pixels==(uint64_t)preserved);assert(stats.mask_changed_pixels==(uint64_t)changed);}
 checks++;
}
int main(void){
 object *o=malloc(sizeof(*o));assert(o);ps5_ai_request request={0};
 request.width=request.height=256;memset(request.mask_image_id,'1',32);
 protocol_case(o,&request,"completed",1,1,17,32768,32000,1); /* Partial ROI, real blend evaluations. */
 protocol_case(o,&request,"completed",1,1,0,0,65536,1); /* Full editable mask. */
 protocol_case(o,&request,"completed",1,1,0,2,65000,1); /* Tiny preserved area lost in reduced mask. */
 protocol_case(o,&request,"completed",1,0,0,0,65536,0);
 protocol_case(o,&request,"completed",1,0,17,32768,32000,0);
 protocol_case(o,&request,"completed",1,1,0,0,0,0);
 protocol_case(o,&request,"completed",1,1,17,32768,0,0);
 protocol_case(o,&request,"completed",0,1,0,0,65536,0);
 protocol_case(o,&request,"running",1,1,0,0,65536,0);
 protocol_case(o,&request,"completed",1,1,-1,0,65536,0);
 protocol_case(o,&request,"completed",1,1,0,65537,1,0);
 protocol_case(o,&request,"completed",1,1,0,2,65535,0);
 protocol_case(o,&request,"completed",1,1,0,-1,65536,0);
 assert(parse_object("{\"status\":\"completed\",\"gpu_used\":true,\"inpainting\":true,\"mask_prepared\":true,\"mask_preserved_pixels\":0,\"mask_changed_pixels\":65536}",o));
 ps5_ai_stats stats={0};assert(!backend_mask_completion_verified(&request,o,&stats));checks++;
 assert(parse_object("{\"status\":\"completed\",\"gpu_used\":true,\"inpainting\":true,\"mask_prepared\":true,\"mask_denoise_evaluations\":0,\"mask_preserved_pixels\":0}",o));
 assert(!backend_mask_completion_verified(&request,o,&stats));checks++;
 assert(parse_object("{\"status\":\"completed\",\"gpu_used\":true,\"inpainting\":true,\"mask_prepared\":true,\"mask_denoise_evaluations\":0,\"mask_preserved_pixels\":0,\"mask_changed_pixels\":0.5}",o));
 assert(!backend_mask_completion_verified(&request,o,&stats));checks++;
 request.mask_image_id[0]=0;memset(&stats,0,sizeof(stats));
 assert(parse_object("{}",o));assert(backend_mask_completion_verified(&request,o,&stats));
 assert(!stats.mask_prepared&&!stats.mask_denoise_evaluations&&!stats.mask_changed_pixels&&!stats.mask_preserved_pixels);checks++;
 free(o);printf("{\"state\":\"passed\",\"backend_metadata_contract_checks\":%u,\"fixture_type\":\"SYNTHETIC_BACKEND_METADATA_ONLY\",\"GPU_inference_verified\":false,\"generated_PNGs\":0}\n",checks);return 0;
}
