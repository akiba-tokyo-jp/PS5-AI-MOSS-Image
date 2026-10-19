/* SPDX-License-Identifier: MIT
 * Actual pinned upstream name converter plus tiny string/log host shims.
 * No model weight loading or inference.
 */
#include "moss_lora_contract.hpp"
#include "name_conversion.h"
#include "core/util.h"
#include <cstdarg>
#include <cstdio>
#include <sstream>
#include <iostream>
bool starts_with(const std::string& value,const std::string& prefix){return value.rfind(prefix,0)==0;}
bool ends_with(const std::string& value,const std::string& suffix){return value.size()>=suffix.size()&&value.compare(value.size()-suffix.size(),suffix.size(),suffix)==0;}
bool contains(const std::string& value,const std::string& needle){return value.find(needle)!=std::string::npos;}
void replace_all_chars(std::string& value,char from,char to){for(char& c:value)if(c==from)c=to;}
std::vector<std::string> split_string(const std::string& value,char delimiter){std::vector<std::string> out;std::stringstream input(value);std::string row;while(std::getline(input,row,delimiter))out.push_back(row);return out;}
std::string sd_format(const char* fmt,...){char text[4096];va_list args;va_start(args,fmt);int n=vsnprintf(text,sizeof(text),fmt,args);va_end(args);if(n<0||n>=(int)sizeof(text))throw std::runtime_error("format bounds");return text;}
void log_printf(sd_log_level_t,const char*,int,const char*,...){}
int main(){unsigned checked=0;for(const auto& [name,shape]:ps5_ai_moss_lora::expected()){
    if(convert_tensor_name(name,VERSION_MOSS_IMAGE)!=name||!is_diffusion_model_name(name))
        throw std::runtime_error("Canonical LoRA changed by actual upstream converter or module filter");
    ++checked;
}std::cout<<"Pinned upstream name conversion PASS "<<checked<<" canonical names; no model execution\n";}
