/* MIT. Host protocol tests ONLY: intentionally fail, never create an image. */
int ps5_ai_engine_init(const char *c,char *e,size_t n){(void)c;(void)e;(void)n;return 0;}
void ps5_ai_engine_get_status(ps5_ai_engine_status *s){memset(s,0,sizeof(*s));s->available=s->gpu_available=1;strcpy(s->backend,"PROTOCOL TEST FIXTURE - no inference or GPU");}
int ps5_ai_engine_generate(const ps5_ai_request *r,const char *o,ps5_ai_stats *s,ps5_ai_progress_cb cb,void *u,char *e,size_t n){(void)r;(void)o;(void)cb;(void)u;memset(s,0,sizeof(*s));pause_ms(1500);snprintf(e,n,"Intentional host protocol-test failure: no inference performed");return -1;}
void ps5_ai_engine_shutdown(void){}
int ps5_ai_engine_shutdown_checked(char *e,size_t n){(void)e;(void)n;return 0;}

static int backend_upload_image(const unsigned char *d,size_t n,const char *t,char id[33],char *e,size_t c){(void)d;(void)n;(void)t;(void)id;snprintf(e,c,"Host fixture does not upload or run inference");return 0;}
static void backend_delete_input(const char *id){(void)id;}
static int backend_upload_mask(const unsigned char *data,size_t size,const char *type,int grayscale,char id[33],char *error,size_t cap){
    (void)data;(void)size;(void)type;(void)grayscale;(void)id;snprintf(error,cap,"Host protocol fixture does not run masked inference");return 0;
}
static int backend_input_dimensions(const char *id,uint32_t *w,uint32_t *h){(void)id;(void)w;(void)h;return 0;}
