// SPDX-License-Identifier: MIT
#include "phase_profile.h"
#include <cassert>
#include <windows.h>
static int active=1;static unsigned calls[32];static uint64_t elapsed[32],sizes[32];
extern "C" int ps5_ai_profile_enabled(){return active;}
extern "C" void ps5_ai_profile_add(unsigned i,uint64_t ns,uint64_t bytes,int valid){assert(i<32&&valid);calls[i]++;elapsed[i]+=ns;sizes[i]+=bytes;}
int main(){assert(ps5_ai_profile_group("unet")==0);assert(ps5_ai_profile_group("control_net")==1);assert(ps5_ai_profile_group("clip")==2);
errno=EDOM;assert(ps5_ai_profile_clock()>0 && errno==EDOM);
{PS5AIProfileScope outer(0);{PS5AIProfileScope inner(1);Sleep(2);}}
assert(calls[0]==1&&calls[1]==1&&elapsed[0]>=elapsed[1]&&elapsed[1]>0);
int value=ps5_ai_profile_call(24,8192,[]{errno=ERANGE;return 17;});assert(value==17&&errno==ERANGE&&calls[24]==1&&sizes[24]==8192);
active=0;{PS5AIProfileScope disabled(2);Sleep(1);}assert(calls[2]==0);
}
