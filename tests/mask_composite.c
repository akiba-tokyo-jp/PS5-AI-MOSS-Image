/* MIT. Numeric preservation/feathering checks; no inference or GPU. */
#include "reference_image.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
static int checks;
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL line%d\n",__LINE__);return 1;}checks++;}while(0)
int main(void){
    const unsigned char original[]={7,8,9,100,110,120,0,100,255};
    const unsigned char generated[]={250,240,230,3,5,7,255,200,0};
    const unsigned char mixed[]={0,255,128},expected[]={7,8,9,3,5,7,128,150,127};
    unsigned char out[9];uint64_t preserved,changed;memcpy(out,generated,9);
    CHECK(ps5_ai_mask_composite(out,original,mixed,3,1,&preserved,&changed));
    CHECK(!memcmp(out,expected,9));CHECK(preserved==1&&changed==2);
    const unsigned char black[]={0,0,0};memcpy(out,generated,9);
    CHECK(ps5_ai_mask_composite(out,original,black,3,1,&preserved,&changed));
    CHECK(!memcmp(out,original,9)&&preserved==3&&changed==0);
    const unsigned char white[]={255,255,255};memcpy(out,generated,9);
    CHECK(ps5_ai_mask_composite(out,original,white,3,1,&preserved,&changed));
    CHECK(!memcmp(out,generated,9)&&preserved==0&&changed==3);
    CHECK(!ps5_ai_mask_composite(out,original,white,0,1,&preserved,&changed));
    CHECK(!ps5_ai_mask_composite(NULL,original,white,3,1,&preserved,&changed));
    printf("{\"checks_passed\":%d,\"GPU_calls\":0,\"model_inference\":false}\n",checks);
    return 0;
}
