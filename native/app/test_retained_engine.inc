/* MIT. Windows HTTP contract fixture only; no GPU or image generation. */
int ps5_ai_engine_init(const char *c,char *e,size_t n){(void)c;(void)e;(void)n;return 0;}
void ps5_ai_engine_get_status(ps5_ai_engine_status *s){memset(s,0,sizeof(*s));s->available=s->gpu_available=s->gpu_verified=s->model_loaded=1;strcpy(s->backend,"HOST FAILURE FIXTURE - NO INFERENCE");}
int ps5_ai_engine_generate(const ps5_ai_request *r,const char *o,ps5_ai_stats *s,ps5_ai_progress_cb cb,void *u,char *e,size_t n){(void)r;(void)o;(void)s;(void)cb;(void)u;ps5_ai_native_api_retained_failure("injected_host_ENOSPC_before_GPU_submit");while(!stopping)pause_ms(50);snprintf(e,n,"Host fixture stopped; no inference performed");return -1;}
void ps5_ai_engine_shutdown(void){}
int ps5_ai_engine_shutdown_checked(char *e,size_t n){(void)e;(void)n;return 0;}
