/* SPDX-License-Identifier: MIT -- exact GGML arithmetic-mode log token. */
#pragma once
#include <cstring>
static inline int ps5_ai_backend_precision_token(const char* text) noexcept {
    if(!text)return -1;
    const char* field=std::strstr(text," | fp16: ");
    if(!field)return -1;
    field+=9;
    const char* end=field;
    while(*end && *end!=' ' && *end!='\t' && *end!='\r' && *end!='\n' && *end!='|')++end;
    const size_t n=static_cast<size_t>(end-field);
    if(n==1 && field[0]=='0')return 0;
    if((n==1 && field[0]=='1') || (n==4 && !std::memcmp(field,"dot2",4)))return 1;
    return -1;
}
