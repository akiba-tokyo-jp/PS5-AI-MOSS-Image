// SPDX-License-Identifier: GPL-3.0-or-later
// Host-only projection and selector guards. No models, network or inference.
#include "moss_dit_kernels.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>
int main(){
 unsigned checks=0;
 for(unsigned layer=0;layer<42;++layer)for(auto suffix:{"ffn_up_gate.weight","ffn_down.weight","qkv.weight","o.weight"}){
  char name[160];std::snprintf(name,sizeof(name),"model.diffusion_model.layers.%u.proj.%s",layer,suffix);
  const unsigned expected=suffix[0]=='f'?1:4;assert(moss_kernel::projection(name)==expected);++checks;
 }
 for(auto name:{"model.diffusion_model.layers.42.proj.qkv.weight","model.diffusion_model.layers.0.proj.qkv.lora_up.weight","model.diffusion_model.text_refiner.layers.0.proj.qkv.weight","model.diffusion_model.layers.0.norm1.weight","model.diffusion_model.layers.0.proj.ffn_up_gate.weight.extra","unrelated",""}){assert(moss_kernel::projection(name)==0);++checks;}
 for(unsigned mask=0;mask<16;++mask){moss_kernel::reset(mask);for(unsigned bit=1;bit<=8;bit<<=1){assert(moss_kernel::enabled(bit)==bool(mask&bit));++checks;}}
 std::printf("{\"selection_checks\":%u,\"success\":true,\"PS5_connections\":0}\n",checks);
}
