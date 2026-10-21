# PS5 AI MOSS Image

**A native image-generation app running MOSS-Image-SFT on the PS5 GPU.** Windows can request image generation and img2img through an OpenAI Images-compatible API and retrieve PNGs. Windows handles builds and HTTP clients; all model inference runs on PS5.

The public version now supports ordinary API use. **No initial 256 prerequisite, fixed prompt/seed, or 4-attempt limit per 1 launch.** The first request can generate at 1024; rectangular images, arbitrary text/seeds, random seeds, and references are supported. [Hardware verification](docs/moss-service-20261021.md).

![512×768 output generated on PS5](docs/images/moss-service-aspect-512x768-20261021.png)

## Generate through the API

Prepare a registered/running PS5 app and gateway plus a private token file. See [deployment/startup](docs/deployment.md) and [building from source](docs/build.md). Substitute your own PS5 private IPv4 and token filename.

```powershell
./scripts/Invoke-MossOpenAI.ps1 -BaseUrl http://<PS5_IP>:8193/v1 -TokenFile D:/ps5-private/api-token -Prompt 'An anime illustration of a peaceful garden.' -Width 1024 -Height 1024 -OutputDirectory D:/moss-output
```

Omitting `-Seed` selects a random seed. `-NegativePrompt` defaults to empty; no subject/style instructions are inserted automatically. Add `-ReferenceImage D:/images/reference.png -DenoiseStrength 0.5` to the same client for a reference image. No preceding 256 request is required.

Each output dimension is 64–1024 and a multiple of 16. The current model profile is **Turbo4: Euler 4 steps / CFG 1 / a dedicated sigma sequence**. This is sampling for the supplied Turbo adapter, not a test-count restriction. Arbitrary steps and other samplers are unsupported. [API specification/examples](docs/api.md), [OpenAPI](docs/openapi.json).

## Hardware results for this update

FW 12.40 / RADV NAVI21. These 5 requests used the same native launch and public PowerShell client; PS5 generation times and downloaded PNGs were checked.

| Order | Request | PS5 generation time |
|---|---|---:|
| 1 | First 1024×1024 after startup, including model loading | 110.46 s |
| 2 | Different prompt / seed 17, 512×768 | 42.18 s |
| 3 | img2img 512×512, strength 0.5, actual 3 forwards | 35.47 s |
| 4 | img2img 256×256, strength 1, random seed | 31.13 s |
| 5 | text-to-image 256×256, random seed | 3.51 s |

Request 5 also succeeded without restart. Prompts, resolutions, encoding, and conditioning-cache states differ, so these are not mutual speed comparisons. Success for 5 requests does not guarantee long-term continuous operation or memory fit for all resolutions/prompts.

In an earlier identical-condition 1024 comparison, DiT kernel improvements reduced the mean from 47.481390 to 45.127473 seconds (4.96%). That measurement generated at 256 first and is distinct from the cold 1024 result above. [Speed comparison and image differences](docs/moss-dit-kernels-20261021.md).

## Architecture on PS5

![Processing architecture](docs/images/architecture.svg)

- OpenAI Images-compatible gateway: 8193, model `moss-image-sft`. It forwards 1 job to the same PS5's native asynchronous API 8192.
- Qwen3.5-9B Q4_K_M text encoding: PS5 CPU. MOSS Q4_K DiT and FLUX.2 small VAE: PS5 Vulkan/RADV. Img2img VAE encoding also runs on the PS5 GPU.
- Direct reservation 10GiB, inference budget 9.5GiB. Uses Q4 weights, FP16 data paths with explicit F32 accumulation, and a combined rank 144 F32 adapter.
- Retains Bearer authentication, same-/24 private LAN restriction, serialized GPU generation, initial numeric readback, finite-value, actual-capacity, ownership, and GPU-completion checks.
- Output + job metadata have a default 8GiB storage allowance, returning 507 before exceeding it. Generated files are not automatically deleted. Logs retain existing bounded rotation.

Masks, inpainting, IP-Adapter, ControlNet, and arbitrary LoRA uploads are unsupported. Img2img does not guarantee character identity. Strong photorealistic conversion remains unmet. With shared memory, RSS, Direct, and API GPU-labeled peaks overlap and cannot be summed.

## Setup and documentation

See [build](docs/build.md), [deployment/startup/shutdown](docs/deployment.md), [API](docs/api.md), [current status](docs/status.md), and [verification data](docs/verification.json), plus [models and terms](docs/moss-models.md), [Turbo adapter reproduction](docs/moss-turbo4-reproduction.md), and [third-party notices](THIRD_PARTY_NOTICES.md).

This source release requires a PS5 native Homebrew launch environment, registered title, SDKs, dependency libraries, and models obtained separately. It includes no general-purpose automatic jailbreak/title-registration installer or prebuilt binaries. Ordinary service entry points are `tools/build_service.py` and `config/moss-service.json.example`. Attempt limits in historical trial documents describe those diagnostic builds.

Integrates [OpenMOSS](https://github.com/OpenMOSS/MOSS-Image), [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), and PS5 Vulkan/Mesa/SDK components. Preserve per-file licenses/copyrights and model-specific terms.
