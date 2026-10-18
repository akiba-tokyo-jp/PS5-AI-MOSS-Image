/* Exercise the actual native parser; no networking or model inference. */
#define main unused_server_entry
#include "../native/app/server.c"
#undef main
#include <assert.h>
/* Link-only sentinels. The parser test must never enter an engine/UI route. */
int ps5_ai_engine_init(const char *p, char *e, size_t n) {(void)p;(void)e;(void)n;abort();}
void ps5_ai_engine_get_status(ps5_ai_engine_status *s) {(void)s;abort();}
int ps5_ai_engine_generate(const ps5_ai_request *r,const char *p,ps5_ai_stats *s,ps5_ai_progress_cb cb,void *u,char *e,size_t n) {
    (void)r;(void)p;(void)s;(void)cb;(void)u;(void)e;(void)n;abort();
}
void ps5_ai_engine_shutdown(void) {abort();}
void ps5_ai_v3_shutdown_request_accepted(int r,int q) {(void)r;(void)q;abort();}
void ps5_ai_v3_worker_join_receipt(int j,int r,int q,int c) {(void)j;(void)r;(void)q;(void)c;abort();}
void ps5_ai_native_ui_log(const char *p) {(void)p;abort();}
void ps5_ai_native_ui_phase(unsigned p) {(void)p;abort();}
void ps5_ai_native_ui_job_begin(const char *p) {(void)p;abort();}
int main(void) {
    struct ps5_ai_request request;
    char error[512];
    assert(validate_request("{\"prompt\":\"a woman\"}", &request, error, sizeof(error)));
    assert(request.manga_strength == 1.0);
    assert(validate_request("{\"prompt\":\"a woman\",\"manga_strength\":0}", &request, error, sizeof(error)));
    assert(request.manga_strength == 0.0);
    assert(validate_request("{\"prompt\":\"a woman\",\"manga_strength\":1.5}", &request, error, sizeof(error)));
    assert(!validate_request("{\"prompt\":\"a woman\",\"manga_strength\":-1}", &request, error, sizeof(error)));
    assert(!validate_request("{\"prompt\":\"a woman\",\"manga_strength\":1.5001}", &request, error, sizeof(error)));
    assert(!validate_request("{\"prompt\":\"a woman\",\"manga_strength\":\"1\"}", &request, error, sizeof(error)));
    assert(!validate_request("{\"prompt\":\"a woman\",\"manga_strength\":true}", &request, error, sizeof(error)));
    assert(!validate_request("{\"prompt\":\"a woman\",\"manga_strength\":1e999}", &request, error, sizeof(error)));
    assert(!validate_request("{\"prompt\":\"a woman\",\"manga_strength\":0,\"manga_strength\":1}", &request, error, sizeof(error)));
    assert(!validate_request("{\"prompt\":\"a woman\",\"manga_strength\":{}}", &request, error, sizeof(error)));
    puts("PASS 10 actual manga request parser checks; no model inference");
    return 0;
}
