# Additional MOSS optimization investigation (2026-10-20)

The reviewed cache version completed matched PS5 256×256 and 1024×1024 generations with baseline-identical PNG files and normal visual review. The same-prompt 1024 request after 256 was 40.389505 s faster in this single observation. The recovered uninterrupted current engine log confirms the expected cache hits, and normal exit plus an independent fresh idle check passed. The repository remains private and experimental.

## Preserved conditions

The comparisons use MOSS-Image Q4_K weights, the selected rank-16 LoRA at strength 0.7, 50 steps, CFG 3, Euler, flow shift 5 and seed 20261020. FP16 operand/storage support is enabled; the selected Q4 matrix pipelines retain FP32 accumulation. This is not a conversion of the model to FP32 weights or a change to FP16 accumulation.

The Direct pool remains 10 GiB with the existing 9.5 GiB inference budget. Existing allocation, ownership, completed-fence, readback and finite-output checks remain in force. No system or driver installation changes are part of this optimization.

## What the GPU profile established

The optional Vulkan timestamp implementation was validated on the PS5 with real query availability, 64-bit timestamps, a 10 ns reported timestamp period, completed GPU fences and explicit query-memory cache handling. The recorder adds no extra queue submissions or CPU waits.

Two complete DiT graphs were sampled, excluding the witness callback fragments. They covered the negative branch's 41 layers (layer 22 omitted) and the positive branch's 42 layers. Their actual token counts were 270 and 362 at 256×256.

FFN-up accounted for 31.32% of the sampled dispatch intervals, making it a useful optimization target. This is a bounded sample of parent-operation-attributed shader dispatches, including any conversion/reduction dispatches. It is not an exclusive decomposition of the whole generation, and it must not be extrapolated into a promised total speedup. Timestamp instrumentation can also perturb scheduling.

CPU phase measurements are nested. In particular, driver-fence waits include outstanding GPU execution and scheduling; their duration is not demonstrated avoidable synchronization overhead. Do not add fence-wait time to graph time.

## Kernel candidates that were not adopted

Candidate A reused the existing scalar Q4/F32-accumulation shader with a 256-thread FFN-up specialization. All eight isolated numerical/completion cases passed. The sum of the two warmed CPU-wall microtimes was 0.017096173 s for the baseline and 0.017100502 s for A: a difference of about 4.3 µs, not evidence of an improvement. A was disabled automatically, and no image-generation claim is made for it. Its performance at other shapes has not been established.

Candidate B reused the same existing shader with BK=64, a 128-thread workgroup and a verified physical subgroup size of 64. Its isolated numerical gates passed, but the nominal warmed microtime difference was only 81 ns across the two shapes. Actual image measurements therefore determined the adoption decision.

| Matched request and timing scope | Validated baseline | Candidate B | Observed difference |
|---|---:|---:|---:|
| 256×256, first generation including lazy model loading | 145.839851 s | 147.548952 s | +1.1719% |
| 1024×1024, first 1024 request after the 256 request | 1064.248911 s | 1083.242504 s | +1.7847% |

These are PS5-native engine generation times from one matched observation per configuration and size, using identical saved API requests. The 1024 comparison is not an application-cold-start measurement. The dimension-sensitive text cache missed on that resolution change.

Both B PNG files were byte-identical to their corresponding baseline files. The actual generation receipts confirmed 4,150 B FFN-up selections, the expected positive/negative layer and token coverage, 16,600 recorded Q4 FP32-accumulation selections, no FP16-accumulation selections, no nonfinite results, and all 504 selected LoRA tensors used.

B did not demonstrate an end-to-end improvement and was not adopted. This is a decision based on these measurements, not a claim that either specialization is intrinsically slower at every shape. There was no controlled thermal/clock measurement or repeated timing distribution.

## Resolution-independent MOSS text cache

The generic conditioning cache includes image width and height in its key. In the current pure-text MOSS path, Qwen3.5 returns text hidden states without using those dimensions. Image/timestep RoPE is built separately inside each DiT forward using the actual latent dimensions and text length.

The change reuses the existing completed-condition LRU cache. It ignores only width and height when all of the following hold:

- The caller is the same actual MOSS LLMEmbedder instance, with vision disabled.
- No reference-image or presentation-reference pointer is supplied, and pass-to-VLM is disabled.
- The cached output contains text cross-attention features only; all other 13 condition fields are empty.

All other cache keys, reference ownership/value comparisons, capacity and LRU behavior, and existing context/LoRA invalidation remain unchanged. The actual width and height are still passed unchanged to the encoder on a cache miss. The generic cache behavior for SD1, SD2, SDXL and other models remains dimension-sensitive.

The default pass-to-DiT flag is preserved as a key rather than rejected by itself: it is true for ordinary requests even when there are no reference tensors. Control/image-mode data attached after the conditioning-cache boundary is not cached by this change. This is not permission to reuse spatial conditions or reference features across resolutions.

This can avoid redundant text encoding when the same prompt is reused at another resolution. It does not accelerate DiT kernels, and it does not eliminate text encoding for a new prompt. Same-resolution repeated prompts already benefited from the existing cache. Earlier other-runner timings contain more than text encoding, so no exact time saving is promised from those aggregate numbers.

### Validation completed before PS5 execution

- 81 behavior checks executed the actual proposed cache header with source-extracted condition/parameter types and counting test encoders.
- 12 source-invariant checks verified preserved keys, reference comparisons, cache clearing/LRU and caller invalidation.
- Independent source review found no eligibility or invalidation blocker in the current path.
- An isolated, byte-identical copy of the actual diffusion-engine source passed PS5-target syntax checking with the actual FP32-accumulation/RTTI flags. Dependency output proved that the proposed cache header was selected. The original vendor inputs remained unchanged.

The selected cache version was subsequently built with all 75 core objects newly compiled, no older-core carryforward, and 15 native compiler commands. Actual artifact verification passed for the consumed cache header/overlay/source freeze, diffusion-engine dependency file and linked RTTI definition. These build checks do not establish text-embedding equality or cache-hit performance on the PS5.

### Actual cache-version observations

| Matched request and timing scope | Baseline | Cache version C | Observed difference |
|---|---:|---:|---:|
| 256×256 first generation, including lazy model loading | 145.839851 s | 146.016289 s | +0.176438 s (+0.1210%); practically unchanged in this observation |
| First 1024×1024 request after 256, same process and prompt | 1064.248911 s | 1023.859406 s | −40.389505 s (−3.7951%) |

The saved request bytes and corresponding output PNG bytes match the baseline at both sizes. Both images passed visual review. These are single matched observations without a controlled thermal/clock/cache state. The 1024 measurement is the first request at that resolution after 256, not an application-cold-start or repeated timing distribution.

The 14 admitted startup GPU gates passed; unused im2col diagnostics remain separate. Both generations recorded 16,600 Q4 FP32-accumulation selections, no FP16-accumulation selections, no nonfinite values, and all 504 LoRA tensors used with none skipped. B, GPU timestamps and limited fusion were disabled.

The matched 1024 CPU phase evidence shows where the observed reduction occurred:

| Nested CPU-wall metric | Baseline | Cache C |
|---|---:|---:|
| DiT graph total, 100 calls | 1016.603640 s | 1016.443356 s |
| Other-runner graph total | 46.183293 s / 51 calls | 5.997303 s / 49 calls |
| Other-runner weight preparation | 23.202890 s | 0.004230 s |
| Other-runner graph call including internal waits | 22.687026 s | 5.712260 s |

Rows are nested and must not be added. The two fewer other-runner calls and reduced preparation match the intended reuse of completed text conditions, while the DiT duration remained effectively unchanged. Other-runner metrics also include decoder work and are not pure text-encoder timings.

The full current engine log was recovered after the initially bounded capture left an incomplete interval. Offline review parsed all 623 nonempty records from the completed small generation through the completed first-large generation, with no gap. It identified the current configuration and exactly two current generations: zero request-scoped INFO cache hits for the initial 256 request, and exactly two for the following 1024 request. The two hits follow that large request and precede its completion, rather than belonging to an older file prefix. This confirms reuse of the completed positive and negative text conditions on the resolution change.

| Verification | Result |
|---|---|
| Fresh core/native artifact, dependencies and linked RTTI | Passed; 75 new core objects, 15 native commands, 215-file source freeze |
| Both actual generations and PNG/visual comparison | Passed |
| Observed same-prompt resolution-change total-time reduction | 40.389505 s (3.7951%) in one matched 1024 request |
| Exact request-scoped INFO cache hits | Passed; initial 256: 0, following 1024: exactly 2 in the uninterrupted current interval |
| Normal app/gateway exit and independent fresh idle confirmation | Passed; normal API exit, both PIDs absent, new observer023 reports no native GPU title |

After both requests, the native app and gateway exited through their normal APIs. A separate process-list check confirmed both owners absent. A newly built and executed read-only observer023 then confirmed the same boot and existing service/syscore/shellcore identities, no running GPU title (BigApp −1 and zero native eboot processes), and all three app registrations retained. This is an observed clean lifecycle for this run, not proof of long-duration stability.

This improvement applies to reuse of the same text condition across a resolution change. It is not a demonstrated faster DiT kernel or a benefit for an unseen prompt. API CPU/GPU-labelled peaks include shared allocations and must not be added as physical RAM or interpreted as pure VRAM. No memory quota or 12 GiB achievement is claimed.

Further candidates, including GPU-timestamp calibration of isolated kernels and temporary per-operation Q4→FP16 expansion, remain proposals. They are not included in the selected cache change.
