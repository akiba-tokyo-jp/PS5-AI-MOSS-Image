# MOSS sampler, Spectrum and memory investigation (2026-10-20)

> 過去の試験記録です。現在の採用候補・最新の秒数は[Turbo4・メモリ比較](moss-turbo-memory-options-20261020.md)と[検証状態](status.md)を参照してください。

The guarded PS5 trials below reduced native generation time by executing fewer DiT forwards. These are single observations, not a quality-equivalent benchmark or a production default. The repository remains private and experimental, with production admission disabled. No new prompt, long-duration reliability or timing distribution is established.

## Actual PS5 results

The common base is MOSS-Image-SFT Q4_K, the selected rank-16 LoRA at 0.7, FP16-enabled operand/storage support with FP32 matrix accumulation, flow shift 5 and seed 20261020. Prompt, model files, LoRA, size and seed match across the corresponding requests. Text encoding runs on the PS5 CPU; DiT and VAE run through the PS5 Vulkan backend. Windows only requests and retrieves images.

| Guarded configuration | 256×256, first generation including lazy model loading | 1024×1024, first request after 256 in the same process | Actual DiT forwards per image |
|---|---:|---:|---:|
| Baseline text-cache version: 50 steps, Euler, CFG 3 | 146.016289 s | 1023.859406 s | 100 |
| 30 steps, `res_multistep`, `simple` schedule, CFG 4 | 113.804763 s | 617.365825 s | 60 |
| Spectrum: 50 Euler steps, CFG 3, 30 actual CFG steps plus 20 latent forecasts | 114.148733 s | 617.938351 s | 60 |

Each cell is **n=1**. Times are measured inside the PS5 engine, not HTTP round-trip times. The first 1024 request is not an application-cold-start measurement. The same-prompt positive and negative text-cache hits were confirmed in the complete current small-to-large interval for both trials. Disk cache, temperature and clock were not controlled, and there is no repeated timing distribution.

Relative to the cache baseline, the observed reductions were 22.06% / 39.70% for the 30-step sampler and 21.82% / 39.65% for Spectrum at 256 / 1024. The approximately 0.57 s difference between their 1024 observations is not evidence that one is faster in general.

The 30-step trial deliberately changes the sampler, schedule, CFG and step count. Spectrum retains the 50-step Euler/CFG 3 request but approximates 20 fully CFG-combined denoised latents on the PS5 CPU. Its bounded manual policy keeps eight warmup steps and two final steps, with alternating forecasts between them. It is the existing SD.cpp latent-cache boundary with new bounded/finite/order guards, not the ComfyUI extension's hidden-state/head implementation.

Both trials passed the 14 existing startup numerical gates and recorded 60 completed finite GPU DiT outputs, 9,960 Q4 FP32-accumulation selections, zero FP16-accumulation selections, zero invalid/nonfinite results, and all 504 selected LoRA tensors used with none skipped. The existing negative-branch layer-22 omission remains. No CPU replacement of the learned DiT is claimed. Successful execution and finite PNG output do not establish image equality or equal perceptual quality; these changed sampling/forecast paths require visual comparison and broader prompt testing. There is no baseline PNG-parity claim for either trial.

## Where the observed time went

For the 1024 baseline, DiT graph total was 1016.443356 s for 100 forwards. It was 610.256676 s / 60 forwards for the 30-step sampler and 610.401184 s / 60 forwards for Spectrum. Other-runner totals stayed near 6.0 s in all three runs after completed text conditions were reused. The 256 first-generation other-runner total remained near 40.9 s, so its fixed loading/text work limited the relative reduction.

These are nested CPU-wall phase metrics. Graph calls include internal synchronization; driver fence waits include outstanding GPU work and scheduling. They are not pure GPU timestamps or demonstrated removable overhead, and their durations must not be added to graph totals. The earlier bounded GPU timestamp profile identified FFN-up as a target, but does not turn these end-to-end measurements into a per-kernel GPU profile. See [the preceding optimization investigation](moss-additional-optimization-20261020.md).

## Memory scope and remaining candidates

Spectrum reduces forward count; it does not demonstrate lower inference memory. At 1024 its owned F32 history payload is 12 MiB and history plus a prediction payload peaks at 14 MiB. At 256 those values are 0.75 MiB and 0.875 MiB. These are bounded payloads, not whole-process heap/RSS measurements; allocator bookkeeping is separate.

The existing 10 GiB Direct reservation is physically allocated and mapped at startup. It remains reserved while the app is idle; freeing an internal workspace restores pool headroom without returning that entire physical reservation to the system. Physical backing is released by the checked pool/lifecycle release path. The API CPU/GPU-labelled cumulative peaks are not separate additive resident-RAM and VRAM totals. The baseline's roughly 4.503 GiB shared Direct high-water mark includes CPU text allocations; it is not a measurement of dedicated GPU VRAM.

The bounded-weight-upload proposal uses a grow-to-fit host Scratch allocation capped at 8 MiB **per worker** for eligible raw, same-format GPU weight uploads. It does not reduce model coefficient size, activations, the Direct reservation or CPU Qwen loading by that amount. A subsequent isolated 50-step first-256 trial confirmed bounded worker staging, byte-identical output and lower observed host heap-mapping/RSS highwaters, with practically unchanged single-run time. See [the measured bounded-upload scope](moss-bounded-weight-upload-20261020.md). This is not a Spectrum-plus-upload test. Heap mapping and process highwaters do not establish total physical RAM or VRAM savings.

A read-only full-coefficient audit found all 37,158,912 values in the existing rank-16 LoRA's 504 matrices exactly representable as BF16. Its F32 matrix payload is 148,635,648 bytes; BF16 storage could remove 74,317,824 bytes, or **70.875 MiB**, before headers, padding, routing and temporary buffers. This is a storage opportunity, not a measured pool or speed saving. Whole-file F16 is not lossless: 23,287 coefficients are not F16-exact.

Lossless coefficient storage does not by itself preserve computation. The current BF16-weight Vulkan matrix route converts a non-BF16 right-hand-side activation buffer to BF16. FP32 accumulation does not undo that rounding; the F16-enabled shared-memory path can also cast activation loads. Current LoRA contracts therefore remain F32. A distinct trial would need non-dyadic down/up GPU oracles, actual pipeline receipts, finite/LoRA counts and image comparisons. Decoding BF16 coefficients to F32 before multiplication could retain the current activation route, with additional transient workspace and decoding cost to measure. The separately audited Turbo/composed F32 matrices are not all promoted BF16 and cannot inherit the rank-16 lossless-halving conclusion. Turbo runtime results are reported separately in the [Turbo4 actual-results note](moss-turbo4-results-20261020.md). A backend `bf16: 0` capability line alone does not prove BF16 weights unusable: the actual BF16 GPU readback gates passed. The active F32 contract, right-hand-side rounding and non-BF16-exact composed coefficients remain distinct barriers.

An offline FFN token-chunk prototype reduced the synthetic whole-graph planner estimate by about 80–82 MiB at the two 1024 token shapes. It also multiplied Q4 matmul nodes by five and LoRA matmul nodes by about 3.7, plus concat work. That planner profile does not reproduce the actual backend's runtime workspace or physical peak. This is an unmeasured, potentially costly memory candidate, not an adopted speed optimization or a promised physical saving.

MOSS's dense DiT is not a mixture-of-experts model. Strata's CUDA/HIP and expert-cache mechanisms cannot be treated as a drop-in MOSS/PS5 optimization. The relevant borrowed design is bounded staging for same-format weight transfer; the PS5 adapter uses the existing GGML ownership/completion path rather than imported Strata CUDA/HIP code.

## Source and attribution

- Existing SD.cpp latent-cache/math reference: [leejet/stable-diffusion.cpp at `8cd731e7b4667eb8f40a153315b9a7125d09a58d`](https://github.com/leejet/stable-diffusion.cpp/tree/8cd731e7b4667eb8f40a153315b9a7125d09a58d/src/runtime), MIT. Reused files and notices remain credited to leejet and contributors; the MOSS hardening does not claim their algorithms as new.
- Spectrum method/reference: [Rkkss/ComfyUI-Spectrum-MOSS at `3fde0674b29342565e7f42a0674af48d65aabc4a`](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/tree/3fde0674b29342565e7f42a0674af48d65aabc4a), GPL-3.0-only. Its source also credits awdqwdasdg's Comfyui-Spectrum-Qwen2.1 and arXiv:2603.01623. Its benchmark numbers and hidden-state implementation are not PS5 results. The new MOSS latent-adapter header retains its own GPL-3.0-only statement.
- Bounded staging and promoted-data provenance reference: [Niko1221/Strata at `e88ee39d1d6be6e3364183af3b39b62a7ed6f211`](https://github.com/Niko1221/Strata/tree/e88ee39d1d6be6e3364183af3b39b62a7ed6f211), MIT, Copyright (c) 2026 Niko1221 and the Strata contributors; in particular `src/core/weights.cpp` and `include/strata/core/weights.hpp`.

Individual licenses remain unchanged. See [third-party notices](../THIRD_PARTY_NOTICES.md). The accompanying source-notice/provenance patch retains the upstream license texts and distinguishes reused code, new hardening and method/design references. Model weights, raw/private logs, credentials and personal file paths are excluded from this document. Public release remains a separate decision after the guarded trials are complete.
