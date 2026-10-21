# Turbo4: upload, FFN chunking, and F16 LoRA measurements

The next result is [DiT invariant reuse/F32 fusion](moss-dit-fusion-20261020.md):1024 at 47.760891 seconds, 6.62% faster than the same-core control, identical PNG. Below preserves the preceding comparison.

2026-10-20 JST. F32 Turbo4 + bounded upload established the baseline; FFN chunking and F16 LoRA storage were tested separately. **F16 reduced owned Direct usage but did not verify faster warm 1024 generation.** This public version targets the verified limited 10GiB trial.

![Turbo4 + bounded upload, first 256](images/moss-turbo4-upload-256-20261020.png)

![1024 after 256 in the same process](images/moss-turbo4-upload-1024-20261020.png)

Both are actual F32-combination PNGs, byte-identical to earlier Turbo4-only outputs.

## Single-observation identical-request comparison

All inference ran on PS5:4-step Euler/CFG 1, seed 20261020, combined rank 144 NSFW0.7 + Turbo1.0, runtime multiplier 1.0. Author sigmas `[1.0, 0.910, 0.717, 0.024, 0.0]`, no double shift. Windows was API client only.

| Configuration | First 256 incl. lazy load | First 1024 after 256, same process | Difference from 1024 baseline |
|---|---:|---:|---:|
| F32 Turbo4 + bounded upload | 71.377746 s | **51.092988 s** | Baseline |
| Above + FFN token chunk 2048 | 71.294669 s | 55.195208 s | ＋4.102220 s,＋8.03% |
| F16 LoRA + dedicated FP32 shared arithmetic, FFN chunking OFF | 68.897951 s | 51.884730 s | ＋0.791742 s,＋1.55% |

1 observation per configuration/size; no OS-cache flush or temperature/clock control.1024 was neither standalone cold nor run 2 at 1024. F16 first 256 was 2.479795 seconds below baseline, not a proven repeatable cold-speed gain. Earlier [Turbo4-only results](moss-turbo4-results-20261020.md),71.564874/51.133430 seconds, remain separate history.

FFN chunking reduced 1024 workspace high-water but doubled Q4 F32 matrix records 672→1344 and increased time about 8%.256 PNG matched; 1024 differed, without major visible defects. Chunking stays OFF for speed baseline. The [experimental profile](../build-profiles/moss-performance-turbo-bounded-upload-ffn2048-fp16-f32accum.json) remains for comparisons.

## Read memory figures separately

These records describe first 1024 after 256 in the same process. All retained **physical Direct reservation 10,737,418,240B (10GiB)**.

| Record | F32 combined baseline | FFN chunk 2048 | F16 storage |
|---|---:|---:|---:|
| Owned Direct data high-water | 5,934,579,712B | 5,383,815,168B | 5,265,719,296B |
| Post-generation owned Direct data | 4,749,066,240B | 4,749,066,240B | 4,080,205,824B |
| API GPU-labeled high-water | 5,968,183,296B | 5,426,167,808B | 5,299,322,880B |
| API CPU/process `ru_maxrss × 1024` high-water | 11,595,284,480B | 11,788,664,832B | 11,578,638,336B |

F16 lowered owned Direct high-water/final usage/GPU-labeled figures by 668,860,416B (637.875MiB) each, matching halved coefficient payload. Process RSS high-water fell 15.875MiB. These distinct overlapping quantities cannot be summed as physical RAM/dedicated VRAM savings. No 637.875MiB reservation was returned to the OS.

The [bounded-upload](moss-bounded-weight-upload-20261020.md)8MiB per-worker cap remains. Host staging, GPU weight storage, workspace, RSS, and Direct reservation are separate quantities.

## F16 changes and verification scope

The [stream converter](../tools/models/convert_moss_lora_f16.py) preserves F32 input and creates a separate lossy F16 file: 61,089,117 values became F16 subnormal; 31,301 rounded to 0. This is not simply all-FP16 arithmetic. The [dedicated backend](../native/app/moss_lora_fp32_shared_backend.inc) reads F16 LoRA storage while retaining F32 RHS/shared arithmetic/accumulation. It does not replace Q4 base weights, VAE, or generic F16 pipelines.

Existing 14 numeric gates, 3 rank 144-composition gates, [bounded-upload gate](../native/app/moss_bounded_weight_upload_gate.inc), and [8 dedicated readbacks](../native/app/moss_lora_f16_storage_gate.inc) passed. Fixed non-dyadic inputs and independent subnormal tests compared F32/F16 storage at unchanged tolerances. Modules/specializations/handles for 4 compiled pipelines were recorded and matched both images. Verified 2016 private F16 LoRA matrix records/0 F32,4 finite DiT outputs, 672 Q4 F32 records/0 F16 accumulation/0 invalid, 504 LoRA used/0 skipped.

Both sizes matched native/OpenAI PNG bytes and showed no major visible defects. They differ from F32; equal quality/pixel equality is not claimed. The 1024 request had 1 positive-conditioning cache hit. Each request used 1 POST/0 retries; normal shutdown followed by independent native/gateway absence and preserved services/registration/model metadata.

F16 is a [default-OFF dedicated trial](../native/app/moss_lora_f16_storage_trial.h) limited to verified canonical rank 144 and N362/4202, not arbitrary prompts/shapes. Baseline uses ordinary F32 LoRA; F16 also changes the dedicated F32 shared path, so timing cannot isolate storage effects. [F16 profile](../build-profiles/moss-performance-combined-lora-f16-fp32shared.json) and [control F32 profile](../build-profiles/moss-performance-combined-lora-f32-fp32shared.json) are separate.

## Next optimizations and unverified scope

Of the warm F32 baseline 51.09 seconds, DiT graph consumed 44.50 seconds (87.1%); other runners 5.99 seconds. These nested CPU wall values do not make fence waits wholly removable synchronization. There is no evidence that shrinking reservation alone accelerates computation.

Candidates included omitting exact SCALE×1 for combined LoRA and fusing post-up ADD/q/k/v concatenation into limited epilogues. At 4steps these represented at most 1008 SCALE and 672 ADD +336 CONCAT graph nodes, but individual GPU times were unmeasured. The approximately 166.96GiB SCALE pass volume derived from N4202 is nominal reads/writes, not measured physical transfer. Omitting `a*1+0` requires signed-zero/FTZ checks and preservation of helper contiguity/alias dependencies. This section records preimplementation candidates. SCALE omission and ADD/packed-q/k/v fusion were implemented and verified in [subsequent tests](moss-dit-fusion-20261020.md); matrix-product epilogue integration remains.

Arbitrary-prompt 8/10GiB ordinary-operation profiles were excluded from this release.8GiB startup/reservation succeeded, but the first request received storage-check HTTP507 before job registration, so inference/continuous-operation success is not claimed. This historical release retained production admission 0 and explicit fixed-prompt/seed/max 4-attempt trials. Weights, raw logs, tokens, and prebuilt binaries are excluded from Git.
