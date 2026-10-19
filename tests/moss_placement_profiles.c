/* SPDX-License-Identifier: MIT -- arithmetic only; no mapping/GPU/PS5 calls. */
#include "placement.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
    const uint64_t gib=UINT64_C(1)<<30;
    assert(PS5_AI_DATA_REGION_BYTES==gib);
    assert(PS5_AI_DATA_VA_BYTES==(PS5_AI_MOSS_TRIAL_10G?10:12)*gib);
    assert(PS5_AI_DRIVER_VA_BYTES==(PS5_AI_MOSS_TRIAL_10G?246:244)*gib);
    assert(PS5_AI_REFERENCE_VA_BASE+PS5_AI_DRIVER_VA_BYTES==PS5_AI_DATA_VA_BASE);
    assert(PS5_AI_DATA_VA_BASE+PS5_AI_DATA_VA_BYTES==PS5_AI_DATA_VA_END);
    assert(PS5_AI_DATA_VA_END==UINT64_C(0x8000000000));
    assert(PS5_AI_DATA_VA_BASE%UINT64_C(2097152)==0);
    assert(PS5_AI_DATA_VA_BYTES/PS5_AI_DATA_REGION_BYTES==(PS5_AI_MOSS_TRIAL_10G?10:12));
    printf("PASS8 placement arithmetic checks profile=%s data_GiB=%llu driver_GiB=%llu region_GiB=1; no physical allocation\n",
        PS5_AI_MOSS_TRIAL_10G?"explicit-trial10":"production12",
        (unsigned long long)(PS5_AI_DATA_VA_BYTES/gib),(unsigned long long)(PS5_AI_DRIVER_VA_BYTES/gib));
}
