# Pre-publication checks (2026-10-20)

2026-10-21 service update: [5 hardware requests](moss-service-20261021.md) verified first 1024, arbitrary text/seeds/rectangles, img2img, and generation 5. Fixed-request/max 4 limits below describe past diagnostics. Ordinary service uses [API](api.md)/[build](build.md). New publication checks passed 389 staged files, 388 source-inventory files, and 19 image records.

2026-10-21 update: implemented/verified [4 DiT kernel changes](moss-dit-kernels-20261021.md); README/verification JSON updated. Publication checker passed 368 staged files, 367 inventory files, 11 image records, file policy, Markdown links, and embedded OpenAPI equality. Tokens/their hashes, weights, raw logs, and SELF are excluded. New policy 94, public C++ selector 239, and SPIR-V validation passed. Hardware generation/numerics/memory/shutdown/adopted-idle scope is in the linked report. Below records earlier checks.

Public source/document consistency was checked on Windows, without inference or new PS5 generation. PS5 times/images/memory are in [comparisons](moss-turbo-memory-options-20261020.md) and [verification](verification.json).

| Check | Result |
|---|---|
| Native/OpenAI-compatible loopback | 134 passed: authentication, model mismatch rejection, 1-time forwarding, error propagation, embedded OpenAPI equality |
| Turbo4 + bounded-upload build contract | 5 tests passed |
| FFN-chunk profile/recipe | 90 passed |
| F16-storage policy/receipts | Policy 61; F32/F16 42 each passed; no GPU execution |
| Dashboard/backend health | Rendering-region guards and 10 parser checks passed |
| DiT policy/shapes/storage | Policy 39, actual-shape graph 6,913, overlap 200,007 passed; no host inference/GPU execution |
| Staged publication files | Checked secrets/personal paths/model/executable exclusion, source inventory, PNGs, Markdown links, embedded schema |

The [publication checker](../tools/check_publication.py) reads staged Git blobs, not just working files. [Source inventory](source-snapshot.json) lists public file sizes/SHAs, distinct from executable build proof or full consumed SDK/header freezes.

This historical release targets fixed-condition 10GiB Turbo4 + upload trials. An arbitrary-prompt prototype was excluded after its first request received HTTP507 before generation. Normal API shutdown and independent read-only observation confirmed no GPU app and preserved services/registration/model metadata. No firmware/system-area changes occurred.

Additional DiT hardware verification is in the [dedicated report](moss-dit-fusion-20261020.md): same-core ON/OFF each generated 1 image per 256→1024, passed existing gates/16 added cases, matched PNGs, and confirmed independent idle after normal shutdown.
