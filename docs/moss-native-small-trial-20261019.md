# MOSS small-image hardware test and quality-cause comparison (2026-10-19 JST)

> Historical test record. See [Turbo4/memory comparisons](moss-turbo-memory-options-20261020.md) and [verification status](status.md) for the currently adopted candidate and latest times.

**PS5 generated a normal-quality 256×256,50steps image using MOSS plus the specified LoRA; Windows retrieved it through the OpenAI Images-compatible API. PS5 generation took 148.133782 seconds including initial loading.**

Strong color stripes in the initial 4steps image disappeared when only steps increased to 50 using the same SELF/settings. Under these conditions, **insufficient 4steps sampling caused the issue**. This does not guarantee all-prompt/resolution quality or establish minimum steps. This explicit individual experiment retained production runtime admission 0.

## Comparison establishing the cause

Both tests fixed native SELF, model, LoRA, prompt, seed, sampler, CFG, flow shift, driver, pool, and reserve. Each used 1 POST in a new native process, 0 automatic retries. After 4steps, normal shutdown/owner absence preceded fresh launch of the same SELF for 50steps.

| Item | 4steps reproduction | 50steps comparison |
|---|---:|---:|
| PS5 generation_seconds | 68.525607 | **148.133782** |
| Image | Strong fine color stripes reproduced | **Stripes eliminated; normal quality verified for this case** |
| PNG bytes | 196,962 | 194,931 |
| 4px-period highpass phase mean span | 0.11117244 | 0.00619342 |
| LoRA loaded / matched / used / skipped | 504 / 504 / 504 / 0 | 504 / 504 / 504 / 0 |

The reproduced 4steps SHA 256 exactly matched the original corrected 4steps (4steps) image, confirming added GPU gates/test-contract changes did not alter it. The 4px metric fell 94.43%. This descriptive pixel metric supports visual comparison; it is not a general quality-acceptance standard.

Ordinary native/gateway/PowerShell defaults were **already 50 steps**.4 applied only to initial smoke contracts; ordinary defaults were not changed 4→50. The quality-comparison profile explicitly permits only 256×256 at 4 or 50steps, with 1 atomic claim per 1 process. Old 4steps-only profiles and production admission 0 remain unchanged.

## 50steps conditions and results

| Item | Measurement/fixed condition |
|---|---|
| Hardware | PS5 FW 12.40, Title ID `PPSA99190`, RADV NAVI21 |
| Engine | MOSS-Image-SFT Q4_K, Qwen3.5-9B Q4_K_M, FLUX.2 small VAE, external Qwen tokenizer |
| Selected adapter | `axiomofmind/Moss-image-nsfw-uncensored-lora`, rank 16 converted exactly to F32, runtime strength 0.7 |
| Inference placement | Qwen text encoding: PS5 CPU; DiT/VAE: PS5 Vulkan. Windows: API client |
| Request | 256×256,50steps, Euler, CFG 3, flow shift 5, seed 20261019, negative CFG skip layers `[22]` |
| Native job | `6ad6269b000001b00000000100000001`, `completed`, actual GPU completion observed |
| PS5 generation time | **148.133782 s**, first generation in this process including initial loading |
| OpenAI-compatible response | HTTP200,149.406 s; full Windows processing 150.859 s |
| PNG | RGB, 256×256,194,931 bytes, fully decoded; native-GET/OpenAI PNGs byte-identical |
| PNG SHA256 | `8e14a3df612932dcef0bbdef38eb3a7cbafd964b87329e8c5a6c7bf55bc07a65` |
| LoRA application | loaded/matched/used 504 each, skipped 0, `application_verified=true` |

![Normal PS5 MOSS256×256,50steps output](images/moss-256-50step-20261019.png)

Warm generation, 1024 images, continuous generation, and minimum steps were not measured.148.1 seconds is not a warm-model time.

## GPU and finite-value evidence

Observed completed DiT matrix operations, Flash Attention, and VAE decoder on hardware.50-step sequential CFG checked 100 DiT outputs, 1 VAE output, 3,854,336 values including text, with nonfinite 0. Limited witnesses/finite values do not prove numeric correctness of every GPU operation.

All 6 initialization GPU-readback gates passed: Q4_K vector, N9 matrix, K3072/M3072/N304 matrix, Flash Attention, BF16 N7 vector/N9 matrix. The large matrix compares 99 edge/interior values against an independent CPU-double oracle and checks all output finiteness. Each gate confirmed no CPU fallback, dispatch/submit/fence increments, mismatch 0/nonfinite 0. BF16 N7 covers 119 values; N9 covers 153.

Driver reports `fp16:0 / bf16:0`; FP16 arithmetic is disabled. This limited test handled BF16 storage through a software path, not proven hardware BF16 arithmetic support.

## Memory interpretation and 12GiB target

A separately authorized dedicated trial allocated Direct pool **10,737,418,240 bytes＝10GiB**. Outside-pool working reserve 1GiB, inference budget `Vulkan0=9.5`, 1-region/claim limit 1GiB. Native context reports **12,280,922,112 bytes＝11.4375GiB** total Direct, so production pool 12GiB + reserve 1GiB fails admission. **The 12GiB target remains unmet.**

| Recorded value | bytes | Interpretation |
|---|---:|---|
| API `peak_cpu_bytes` | 11,769,872,384 | Process peak RSS, not separate CPU-only memory |
| API `peak_gpu_bytes` | 4,867,227,648 | Sum upper bound of driver-arena/Direct-data owned-pool peaks; peak times may differ |
| Direct data peak | 4,835,262,464 | About 4.503GiB, shared-pool peak including CPU text weights |
| Post-generation Direct data used | 3,536,896,000 | About 3.294GiB |
| CPU text weights peak | 4,835,246,080 | Within the same Direct pool; live 0 after conditioning |

API GPU peak is not pure VRAM usage; CPU/GPU figures cannot be added. Independent global GPU memory measurement is `unavailable`. Small-image results do not establish maximum resolution or long-running memory headroom.

## Fix history and normal shutdown

The first test rejected 17 BF16-left matrix nodes as unclassified scratch admission, before model GPU dispatch. Type/shape/stride/32-bit-span checks were added only for BF16 matrix left operands, retaining the conservative 4B/element scratch bound. Admission was not disabled and the pool was not enlarged. Actual 42-layer host metadata tests and independent bounds verification preceded fresh compilation of 75 core sources, with 0 old objects reused.

The corrected first 4steps took 68.704665 seconds and technically produced a GPU PNG/API result, but stripes failed quality acceptance. The 4-versus 50 comparison above isolated this case's quality cause. Offline primary-source comparison found no clear mismatch in MOSS, FLOW/shift 5, FLUX.2 VAE BN rounding, 2×2 unpacking, or DiT axis order. This is not universal proof of defect-free behavior.

Quality-comparison SELF SHA 256: `cf61a7190e4e0f79b7aedb12038fdeb1d529dd90cf92a5a26185ce73cf3cabe5`. Core archive remained `c9809e6bbb04ef3a09044d7c5036e113004c37bc616541ca5d808d4c53acaab4`. New quality contracts/numeric gates changed native only; all 75 compiler-dependency records were audited and a separate receipt confirmed unchanged consumed core inputs. Old core-manifest hashes were not rewritten.

After 50steps, native/gateway closed through normal idle shutdown APIs. Beyond 202 acceptance, separate process reads confirmed both PIDs absent and an independent observer confirmed no GPU title. No additional generation or recording occurred.

## Record scope

Original logs, private diagnostic receipts, weights, secrets, and failed images are excluded from Git. Public material contains conditions, aggregates, and the successful 50-step PNG. Actual source/build/request/shutdown/independent-idle evidence remains in private development records.
