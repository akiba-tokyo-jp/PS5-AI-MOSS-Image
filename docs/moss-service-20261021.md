# Ordinary API PS5 hardware verification —2026-10-21

At user request, ordinary service removes the test-only first 256 request (1 preceding image), fixed prompt/negative/seed, and maximum 4 attempts per 1 native launch. It accepts arbitrary text/seeds, random seeds, rectangles, first 1024, and img2img. Each dimension accepts 64–1024 in 16-pixel increments; actual-capacity/numeric checks remain.

The model profile explicitly specifies Turbo4 Euler 4/CFG 1/flow 5/sigmas[1,.910,.717,.024,0], NSFW0.7 + Turbo1 rank 144 F32/runtime 1. It does not provide arbitrary sampling/LoRA. Service reports `service_mode:true`, `requires_256_first:false`, `max_generation_attempts:null`. Do not confuse inherited low-level production admission macro 0 with ordinary API admission.

## The 5 actual requests

FW 12.40, RADV NAVI21, Direct 10GiB/budget 9.5GiB. Public PowerShell client used the same native owner/gateway, sending 1 generation POST each. The first lazy-loaded the model. Values are PS5 `generation_seconds`.

| Order | Request | Seed input | Actual forwards | Seconds |
|---|---|---|---:|---:|
| 1 | text-to-image 1024×1024, no preceding 256, incl. loading | 20261020 | 4 | 110.459715 |
| 2 | Different text, empty negative, 512×768 | 17 | 4 | 42.180063 |
| 3 | img2img 512×512, strength0.5 | 12345 | 3 | 35.473797 |
| 4 | img2img 256×256, strength 1 | -1→new selection | 4 | 31.132021 |
| 5 | text-to-image 256×256, same garden text as 4 | -1→different seed | 4 | 3.513742 |

Request 5 succeeded without restart, verifying removal of the 4-attempt cap on hardware. First 1024 PNG matched the earlier adopted-kernel same-condition PNG. [JSON](moss-service-20261021.json) records arbitrary-seed examples, 2 resolved random seeds, actual prompts/negatives, PNG sizes/SHAs, and metrics. Img2img used an existing public benchmark PNG, not personal attachments.

![Arbitrary text, different seed, rectangular output](images/moss-service-aspect-512x768-20261021.png)

![512 img2img, strength 0.5](images/moss-service-edit-512-20261021.png)

![256 img2img, strength 1](images/moss-service-edit-full-256-20261021.png)

![Request 5 in the same launch, random seed](images/moss-service-fifth-256-20261021.png)

Img2img with `An ink drawing.` did not strongly convert to ink drawing; anime styling remained. Mechanical img2img operation and successful style conversion are separate. At strength 1, sigma 1 sets reference-latent contribution to 0, resampling the garden. Character locking or photorealism is not claimed.

## GPU, memory, and shutdown checks

All startup readbacks passed: original 14 + Turbo3 + bounded upload + fusion 16 + kernel 20. Across 5 jobs, verified GPU DiT/Attention/decoder completion, finite DiT outputs 4/4/3/4/4, Q4 F32 records 672/672/504/672/672, F16-accumulation records 0, nonfinite 0,504 applied adapter tensors, and graph-cache miss 1/hits actual-forwards−1. The 2 img2img jobs confirmed GPU VAE encode, input hashes, and sigma suffixes.

Native PNGs matched gateway/client bytes; all images passed decoding, dimension/pixel-variation checks, and visual inspection. Raw logs, tokens, and personal process identifiers are private.

Final cumulative owned Direct high-water was 9,629,646,848 bytes (about 8.97GiB). Final usage for jobs 4/5 matched at 4,897,161,216 bytes. This is process-wide, not per-job peak; do not add RSS/GPU-labeled values.5 jobs cannot establish long-term fragmentation/stability.

Normal shutdown plus independent absent-owner/same-boot/service observation preceded restart of the fully readback-verified same SELF. Initial GPU gates passed again; native/gateway returned ready, empty queues, unloaded model. System areas, firmware, GPU driver, and memory settings were unchanged.

## Public build and host checks

Verified fresh 75 core objects, 24 native constructors/6 runtime providers, and complete post-transfer SELF readback. Passed 60,015 host-contract checks, 55 service-HTTP checks (40 consecutive mocks, no inference), and 134 legacy-HTTP checks. Quota rejection was host-tested; hardware completed 5 in-quota jobs. No disk-fill test to 8GiB was performed.

Ordinary build entry:`tools/build_service.py`; config:`config/moss-service.json.example`. Shared-header changes after gateway build were comments only; reconstructed hashes and executable token sequences matched. The later automatic-freeze wrapper was verified through prepare-only (gateway build +75-object plan); measured native used the same lower-level commands. Cloning does not automatically install SDKs or register titles. See [build](build.md), [owner/token/startup](deployment.md), [API](api.md).

## Measurement scope

These 5 jobs differ in prompts, dimensions, encoding, and conditioning-cache state, so they are not mutual speed comparisons.3.51 seconds is neither 1024 timing nor typical new-prompt generation. Earlier identical-condition 1024 mean 45.127473 seconds is in the [speed comparison](moss-dit-kernels-20261021.md). All dimensions/prompts' memory fit, actual 2048 inputs/all PNG/JPEG formats, and long-term stability remain unverified.

Service uses private-LAN auto-bind (explicit override available), Bearer authentication, serialized GPU generation, and default 8GiB output/jobs allowance. This does not measure physical free space.32MiB/job accounting returns 507 before exceeding the allowance; outputs are not automatically deleted. These capacity/security/completion checks remain without first 256/attempt restrictions.

## Public verification images

The following are old/new images from [verification.json](verification.json). Historical trial results retain their original profiles.

- [moss-dit-kernels-1024-20261021.png](images/moss-dit-kernels-1024-20261021.png)
- [moss-dit-kernels-256-20261021.png](images/moss-dit-kernels-256-20261021.png)
- [moss-img2img-1024-3step-20261021.png](images/moss-img2img-1024-3step-20261021.png)
- [moss-img2img-256-1step-20261021.png](images/moss-img2img-256-1step-20261021.png)
- [moss-img2img-256-2step-20261021.png](images/moss-img2img-256-2step-20261021.png)
- [moss-service-aspect-512x768-20261021.png](images/moss-service-aspect-512x768-20261021.png)
- [moss-service-edit-512-20261021.png](images/moss-service-edit-512-20261021.png)
- [moss-service-edit-full-256-20261021.png](images/moss-service-edit-full-256-20261021.png)
- [moss-service-fifth-256-20261021.png](images/moss-service-fifth-256-20261021.png)
- [moss-turbo4-upload-1024-20261020.png](images/moss-turbo4-upload-1024-20261020.png)
- [moss-turbo4-upload-256-20261020.png](images/moss-turbo4-upload-256-20261020.png)
