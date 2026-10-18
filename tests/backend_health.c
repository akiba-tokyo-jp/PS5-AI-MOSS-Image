/* SPDX-License-Identifier: GPL-3.0-or-later */

#define main gateway_unused_main
#include "../native/gateway/server.c"
#undef main
#include <assert.h>
int main(void){object *o=malloc(sizeof(*o));assert(o);unsigned n=0;
const char *good="{\"engine_available\":true,\"gpu_available\":true,\"tv_display\":{\"ready\":true,\"heartbeat\":86,\"last_presented_frame\":86,\"fault\":false},\"model_loaded\":false}";
assert(parse_backend_health_object(good,o)&&backend_true(o,"engine_available")&&backend_true(o,"gpu_available")&&!backend_true(o,"model_loaded"));n++;
assert(!parse_object(good,o));n++;
assert(parse_backend_health_object("{\"metadata\":[null,true,{\"x\":[1,\"escaped \\\" brace }\"]}],\"engine_available\":true}",o)&&backend_true(o,"engine_available"));n++;
assert(!parse_backend_health_object("{\"tv_display\":{\"x\":1,}}",o));n++;
assert(!parse_backend_health_object("{\"tv_display\":[1,]}",o));n++;
assert(!parse_backend_health_object("{\"engine_available\":true,\"engine_available\":false}",o));n++;
assert(!parse_backend_health_object("{\"tv_display\":[[[[[[[[[[0]]]]]]]]]]}",o));n++;
assert(parse_backend_health_object("{\"engine_available\":{\"ignored\":true}}",o)&&!backend_true(o,"engine_available"));n++;
assert(parse_backend_health_object("{\"tv_display\":null,\"engine_available\":true}",o)&&backend_true(o,"engine_available"));n++;
assert(!parse_backend_health_object("{\"tv_display\":{\"x\":true}} garbage",o));n++;
printf("{\"response_parser_checks\":%u,\"public_request_parser_still_rejects_nesting\":true,\"inference_performed\":false}\n",n);free(o);return 0;}
