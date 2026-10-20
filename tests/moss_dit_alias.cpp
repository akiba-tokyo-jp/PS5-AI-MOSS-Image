#include "model/diffusion/moss_dit_ops.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
static unsigned checks;
static void check(bool v){++checks;if(!v)std::abort();}
int main(){
 using moss_dit_opt::storage_ranges_disjoint;
 // Same GGML sentinel address in separate VkBuffers is not an alias.
 check(storage_ranges_disjoint(false,0,4096,0,4096));
 check(!storage_ranges_disjoint(true,0,4096,0,4096));
 check(!storage_ranges_disjoint(true,0,4096,4095,4096));
 check(storage_ranges_disjoint(true,0,4096,4096,4096));
 check(storage_ranges_disjoint(true,4096,4096,0,4096));
 check(!storage_ranges_disjoint(true,UINT64_MAX-3,4,0,1));
 check(!storage_ranges_disjoint(true,0,1,UINT64_MAX-3,4));
 std::mt19937_64 rng(20261020);
 for(unsigned i=0;i<100000;++i){
  uint64_t a=rng()%1000000,b=rng()%1000000,n=1+rng()%100000,m=1+rng()%100000;
  bool overlap=a<b+m&&b<a+n;
  check(storage_ranges_disjoint(true,a,n,b,m)==!overlap);
  check(storage_ranges_disjoint(false,a,n,b,m));
 }
 std::printf("{\"checks\":%u,\"success\":true,\"graph_execution\":false,\"PS5_connections\":0}\n",checks);
}
