# MOSS Image API

Build the ordinary service with `--service-mode`. It accepts arbitrary prompts/seeds, rectangular images, and img2img, without attempt limits or a first 256 prerequisite. [Hardware results](moss-service-20261021.md). Inference runs in the same PS5's native engine.

| Item | Current value |
|---|---|
| OpenAI Images compatibility | `http://<PS5_IP>:8193/v1` |
| Native asynchronous API | `http://<PS5_IP>:8192` |
| model | `moss-image-sft` |
| Output | PNG, 1 image per 1 request. Each dimension 64–1024, multiple of 16; rectangles supported |
| prompt / negative_prompt | At most 4096 UTF-8 bytes each. Prompt required; negative defaults to empty |
| seed | -1 or 0–9223372036854775807. Omitted/-1 selects a new native seed |
| Turbo4 profile | Euler, 4 steps, CFG1, flow_shift5 |
| Dedicated sigmas | `[1,0.910,0.717,0.024,0]`, without double flow shift |
| Adapter | Combined NSFW0.7 + Turbo1.0, rank 144 F32, runtime multiplier 1 |

No text, subject, or negative instructions are inserted automatically. CFG 1 skips the negative DiT forward. Sampling compatible with the Turbo4 adapter is fixed as an explicit model profile; other steps/CFG/samplers are not selected automatically. HTTP cannot replace model files or arbitrary LoRAs.

Ordinary service mode permits existing GPU fallbacks when text length/tensor shapes do not select some optimized kernels. Full kernel-path coverage is required only by diagnostic builds. Startup GPU readback, actual forward counts, Q4 F32 accumulation, finite values, GPU completion, and capacity checks remain mandatory; actual selections and the unused mask are logged.

## Authentication and status

All APIs require Bearer authentication and the same private IPv4 /24. Do not put tokens in URLs, logs, or source. A signed image URL itself grants image-only access. Follow the [startup procedure](deployment.md), passing the same token value to both processes through separate temporary files.

`GET /health` reports `service_mode: true`, `requires_256_first: false`, `max_generation_attempts: null`, and `model_profile: turbo4`. `status: ready` does not mean the model is loaded; check `model_loaded` separately. It also reports `features.img2img`, `running_jobs`, and `queued_jobs`.

## text-to-image

Send JSON to `POST /v1/images/generations`. The first request can specify 1024.

```json
{
  "model": "moss-image-sft",
  "prompt": "An anime illustration of a peaceful garden.",
  "size": "1024x1024",
  "n": 1,
  "response_format": "b64_json",
  "seed": -1,
  "steps": 4,
  "cfg_scale": 1,
  "flow_shift": 5,
  "sampler": "euler"
}
```

Success returns `created` and PNG data in `data[0].b64_json`. `response_format: url` returns an image-only HMAC-signed URL valid for at most 1 hour. Headers `X-PS5-Native-Job-ID` and `X-PS5-Seed` expose the native job and actual seed. Seeds are 64bit integers; preserve them as strings in JavaScript and similar environments to avoid precision loss.

Also available: `GET /v1/models`, `GET /v1/models/moss-image-sft`, and `GET /openapi.json`. OpenAI Images compatibility refers to this partial implementation. `seed`, `steps`, `cfg_scale`, `negative_prompt`, `flow_shift`, and `sampler` are extensions. It does not imply streaming or compatibility with every OpenAI model. `quality: auto/standard` does not switch models.

Example using the [Windows client](../scripts/Invoke-MossOpenAI.ps1). Actual PowerShell verification completed 5 generations and PNG downloads.

```powershell
./scripts/Invoke-MossOpenAI.ps1 -BaseUrl http://<PS5_IP>:8193/v1 -TokenFile D:/ps5-private/api-token -Prompt 'An anime illustration of an adult woman reading a book in a sunny library.' -Width 512 -Height 768 -Seed 17 -OutputDirectory D:/moss-output
```

## img2img

Send `POST /v1/images/edits` as `multipart/form-data`, with 1 PNG/JPEG `image`, `prompt`, the sampling fields above, `reference_mode: img2img`, and `denoise_strength`. Input is limited to 8MiB and 2048 per dimension. PS5 resizes to output dimensions and encodes with PS5 Vulkan VAE. External URLs, multiple images, masks, inpainting, IP-Adapter, and ControlNet are unsupported.

`denoise_strength` ranges from 0.01 to 1, default 0.5. It selects a suffix of the 4step schedule, not a continuous noise-strength adjustment. Floating-point intervals [0.01,0.25), [0.25,0.5), [0.5,0.75), and [0.75,1] perform 1, 2, 3, and 4 forwards respectively. **At 0.75 or above, initial sigma 1 gives the reference latent 0 contribution, effectively resampling.** Use values below 0.75 to retain source content. Each interval uses the same sigma suffix.

```powershell
./scripts/Invoke-MossOpenAI.ps1 -BaseUrl http://<PS5_IP>:8193/v1 -TokenFile D:/ps5-private/api-token -Prompt 'An ink drawing.' -ReferenceImage D:/images/reference.png -DenoiseStrength 0.5 -Width 512 -Height 512 -Seed 12345 -OutputDirectory D:/moss-output
```

This does not guarantee identity preservation or photorealistic conversion. See [current outputs](moss-service-20261021.md) and [earlier img2img results](moss-img2img-20261021.md).

## Native asynchronous API

`POST /api/generate` accepts `prompt`, `negative_prompt`, `width`, `height`, `steps`, `cfg_scale`, `seed`, `sampler`, and `flow_shift`. Save the `job_id` and resolved seed from the 202 response. `GET /api/jobs/{job_id}` returns queued/running/completed/failed, errors, `generation_seconds`, actual `actual_steps`, seed, dimensions, memory metrics, image URL, and other metadata. `GET /api/images/{job_id}` retrieves PNG data.

For img2img, send raw PNG/JPEG data to authenticated `POST /api/inputs`, then use the 201 `image_id` as `init_image_id` and supply `denoise_strength`. Jobs record `init_image_sha256` and `init_encode_seconds`. After job completion with 0 references, manual inputs can be removed with `DELETE /api/inputs/{image_id}`. The gateway cleans up inputs it creates after processing. HTTP does not accept local file paths.

## Serialization, storage, and errors

GPU generation is serialized. The job table has 32 slots and reuses completed slots; this is not a cumulative 32-job limit. Clients should verify an empty queue and POST only 1 job, checking the existing job and health after a timeout. Do not retry automatically. Unknown/unsupported parameters are rejected. See [OpenAPI](openapi.json).

Ordinary service mode allocates a default 8GiB allowance to each process's `output` + `jobs`. It accounts for 32MiB per new/unfinished job and returns 507 before generation if it would not fit. Configure with `--storage-limit-mib`. This does not measure physical free space or prevent consumption by existing files/other apps. Generated files are not deleted automatically; follow [deployment guidance](deployment.md) after archiving. Input storage has separate 128MiB /32-item /24-hour cleanup; [logs](logging.md) use bounded rotation.

Physical Direct, RSS, and GPU-labeled peaks overlap and cannot be summed. Generations that fail capacity, finite-value, or GPU-completion checks are reported as failures. Removing attempt limits does not guarantee arbitrary requests fit memory.
