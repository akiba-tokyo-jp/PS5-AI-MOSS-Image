# MOSS img2img hardware verification on PS5

2026-10-21. **Implemented img2img encoding references into initial VAE latents; verified PS5 generation and Windows PNG retrieval at 256 and 1024.** Native asynchronous and OpenAI Images-style multipart editing APIs are integrated. No extra learned model is required; the existing small VAE encoder is used.

![1024 output using a reference](images/moss-img2img-1024-3step-20261021.png)

[Original anime reference](images/moss-dit-kernels-1024-20261021.png). Silver hair, red jacket, starry sky, and composition carried over. Even with photorealistic requested, the 1024 output remained strongly anime-styled; successful photorealism or guaranteed identity is not claimed. Initial-image conditioning differs from IP-Adapter identity conditioning.

## Measurements

Source: 1024×1024 RGB PNG. A new process first generated ordinary 256, then ran the table top-to-bottom. PS5 generation_seconds, 1 observation each, model loaded, including input VAE encode/output decode. request.steps is 4, but img2img executes only the author's schedule suffix.

| Output | denoise_strength | Actual DiT forwards | Generation time | Input encode portion | Path |
|---|---:|---:|---:|---:|---|
| 256×256 | 0.25 | 2 | 3.454672 s | 0.862 s | Native asynchronous |
| 256×256 | 0.10 | 1 | 1.816666 s | 0.093 s | Native asynchronous |
| 1024×1024 | 0.50 | 3 | 40.833333 s | 4.425 s | OpenAI-style edits |

Total 1024 HTTP time was 43.875 seconds. Initial/reused input-encode states differ, and the 2 examples at 256 also use different step counts; these are not equal-work speedups against 4-forward text-to-image. The preceding ordinary 256 took 70.140223 seconds including loading and matched the previous same-condition PNG byte-for-byte.

![256,2 forwards](images/moss-img2img-256-2step-20261021.png)
![256,1 forward](images/moss-img2img-256-1step-20261021.png)

## Strength semantics and current boundaries

Both builds explicitly require `--moss-img2img-trial`. Without it, images are rejected and existing text-to-image retains 4 forwards. Current verification fixes prompt/seed 20261020, square 256/1024, maximum 4 attempts per 1 process including failures, and production admission 0.

`denoise_strength` selects steps using existing SD.cpp `floor(4 * float(strength)) + 1`, not continuous noise amplitude. Both double/float must be at least 0.01; float must be below 0.75. Near boundaries, float rounding applies.

| float(strength) | Actual forwards | Retained sigmas | Initial-latent coefficient |
|---|---:|---|---:|
| [0.01,0.25) | 1 | [0.024,0] | 0.976 |
| [0.25,0.50) | 2 | [0.717,0.024,0] | 0.283 |
| [0.50,0.75) | 3 | [0.910,0.717,0.024,0] | 0.090 |

The coefficient follows flow noise_scaling `(1-sigma)*latent + sigma*noise`. At 0.75 or above, sigma 1 removes the reference, so this trial rejects it. Strength 0.5 and 0.7 share a suffix. Reported actual steps, actual finite DiT outputs, and 168×actual-forwards Q4 F32 records were independently compared. Logged initial_sigma is predicted from the fixed schedule, not directly measured on GPU.

This verifies ordinary img2img only. Masks, inpainting, IP-Adapter, ControlNet, multiple references, and long-running arbitrary-prompt operation are unsupported.1–3 forwards and 256/1024 were tested in the listed combinations, not every strength×size combination.

## Processing, numeric, and capacity checks

- Uploads were normalized to RGB on PS5 and pinned by job-owned leases. Header/dimensions/RGB SHA 256 were recalculated and matched immediately before inference. Clients cannot supply arbitrary file paths.
- Confirmed completed encoder-weight witnesses on PS5 Vulkan, finite VAE encode/decode, actual DiT, DiT Attention, and decoder completion. Witnesses sample operations, not all GPU instructions.
- Confirmed finite DiT outputs ordinary 4/image 2/image 1/image 3; Q4 F32 records 672/336/168/504; F16 accumulation 0, invalid 0, NaN etc.0, LoRA applied 504/skipped 0. Original 14 + Turbo3 + upload + fusion 16 + kernel 20 numeric gates remained.
- Native uploads were deleted after job completion and reference count 0. Gateway uses its existing temporary-edits-input cleanup after completion. The 1024 base 64/native-GET PNGs were byte-identical, fully decoded, and visually checked.
- 1024 img2img owned Direct high-water: 6,277,365,760B (about 5.85GiB); final 4,920,754,176B. Physical reservation 10GiB, inference budget 9.5GiB, outside-pool reserve 1GiB unchanged. API GPU-labeled peak 6,308,823,040B and process RSS high-water 11,629,019,136B overlap and cannot be added.
- Host strength contract 7412; actual native/gateway parser flag OFF/ON 148; default API regression 134; multipart rejections 4. PS5 rejected 10 unsupported mask/IP, invalid image/strength, duplicate-input, and related cases before loading/generation. Host tests do not replace PS5 inference.

Raw logs, tokens, personal paths/PIDs, weights, and executable binaries are private. [Aggregate JSON](moss-img2img-20261021.json), [API examples](api.md), [build](build.md), [verification index](verification.json). After normal shutdown and independent process-absence checks, the same verified native/gateway restarted idle with all 4 attempts unused. The model is unloaded after restart, so the next request adds lazy loading. Firmware, system areas, GPU driver, and registration were unchanged.

## Earlier image and speed verification

Img2img changes input encoding and forward counts, so it does not replace the [kernel speed comparison](moss-dit-kernels-20261021.md). Earlier successful outputs remain: [moss-turbo 4-upload-256-20261020](images/moss-turbo4-upload-256-20261020.png) / [moss-turbo 4-upload-1024-20261020](images/moss-turbo4-upload-1024-20261020.png) / [moss-dit-kernels-256-20261021](images/moss-dit-kernels-256-20261021.png) / [moss-dit-kernels-1024-20261021](images/moss-dit-kernels-1024-20261021.png).

After measurement, gateway OpenAPI descriptions added negative_prompt/input dimensions. Inference/transfer source and compile flags matched the measured version; only schema JSON/embedded schema differed. Rebuilt/restarted gateway verified hardware-GET schema equality, 134 default-API regressions, and 10 hardware rejection cases. No additional generation measurement after this wording correction is claimed.
