/* SPDX-License-Identifier: GPL-3.0-or-later
 * Host-only check of each compiled copy of the native display metadata ABI.
 */
#include "status.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    if (strcmp(PS5_AI_STATUS_APP_NAME, "PS5 AI MOSS Image") ||
        strcmp(PS5_AI_STATUS_DEFAULT_MODEL, "MOSS-Image-SFT") ||
        PS5_AI_STATUS_ABI != 1u)
        return 1;
    printf("PASS MOSS display labels; status snapshot bytes=%zu; native GPU calls=0\n",
           sizeof(struct ps5_ai_status_snapshot_v1));
    return 0;
}
