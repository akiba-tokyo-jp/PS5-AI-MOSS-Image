# MOSS performance implementation and comparison (2026-10-20)

> Historical test record. See [Turbo4/memory comparisons](moss-turbo-memory-options-20261020.md) and [verification status](status.md) for the currently adopted candidate and latest times.

> For subsequent implementation/hardware results, see [FP16 data + FP32 accumulation](moss-mixed-precision-20261020.md):1024 improved 1219.096190→1064.248911 seconds. The following preserves the initial 3-configuration record; references to ungenerated FP16 and retaining variant 2 describe that historical state.

**Implemented and compared 3 changes. FP32 baseline/submission-fusion configurations generated 256/1024 successfully with identical corresponding PNGs. The FP16 candidate failed startup numeric gates and generated no image.** The submission/fusion FP32 experimental profile is retained, but single-observation differences were only about 0.87% at 256 and 0.38% at 1024, not a major speedup or proof of repeatable gains/fastest configuration. At this stage the repository remained PRIVATE and production admission`0`.

In the earlier [1024 cold/warm comparison](moss-1024-cold-warm-20261020.md), DiT `graph_call_including_internal_wait` consumed 1167.878713 seconds of 1177.725365 warm generation, about 99.16%. This update implements internal observation, existing FP16-path comparison, and submission sizing/limited fusion: 3 areas. Historical times are not results for the new configuration.

## 1. Instrument graph_call internals

Shared `phase_profile.h` and engine aggregation expanded to schema 2 with 42 fields. In addition to graph/weight/workspace/upload/readback/cleanup, DiT and other runners separately record:

| Metric | Observed interval |
|---|---|
| `vulkan_submit_driver_call` | Around submission API calls |
| `vulkan_fence_wait_driver_call` | Around fence-wait API calls |
| `vulkan_dispatch_record_call` | Around dispatch-recording API calls |
| `vulkan_bind_pipeline_record_call` | Around pipeline-bind recording API calls |
| `vulkan_backend_graph_call` | Entire Vulkan backend graph call |

All use `CLOCK_MONOTONIC` **CPU wall time**, not GPU timestamps, kernel duration, or utilization. The 4 driver metrics include `vk_common_GetDeviceProcAddr` lookup and the call itself, excluding surrounding journal work. Fence waits may await unfinished GPU execution. Dispatch/bind metrics do not cover all command recording; short dispatch-recording time does not imply a fast kernel.

`graph_total`, `graph_call`, backend graph, submit, and fence waits are nested. **Do not sum them against generation time.** Runner membership is restored through thread-local scope; disabled/nested/exception behavior and invalid-clock sample counts are checked. Engine-generation scope excludes aggregation logs themselves. All 3 configurations share identical instrumentation.

Implementation: [phase_profile.h](../native/app/phase_profile.h), [engine](../native/app/engine_sdcpp.cpp), [GGML runner](../patches/stable-diffusion/overlay/src/core/ggml_runner.cpp), [Vulkan backend](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan.cpp).

## 2. Compare the existing FP16 path

No new weight format/kernel is introduced. Only PS5 FP16 startup removes `GGML_VK_DISABLE_F16` itself; it does not set `=0`. Baseline/submission-fusion retain FP32; builds with inconsistent configuration and `--precision` are rejected.

FP16 hardware checks require `shaderFloat16`/16bit storage, startup token `fp16: 1` or `fp16: dot2`, and no disabling variable. Existing MOSS readback tolerances are unchanged. NaN/Inf, mismatches, missing GPU-completion evidence, or failed memory admission are failures. FP16 can alter rounding/accumulation, so adoption requires same-condition real-PNG comparisons. Speedup, image equality, and all-FP16 arithmetic remain unconfirmed here.

## 3. Submission sizing and limited elementwise fusion

Only test configuration 2 enables the following.

```text
GGML_VK_PS5_SUBMIT_BATCH=2
GGML_VK_PS5_ELEMENTWISE_FUSION=1
```

Submission sizing applies a2× multiplier to existing adaptive FLOP and node-count thresholds. The node threshold changes 100→200; inclusive checks can actually allow 101→201 nodes per group. This is not a strict 200-dispatch cap; nodes and dispatches are not always 1-to-one. It does not guarantee halved computation, submissions, or time.

Limited fusion selects existing `unary_mul` only for adjacent `UNARY(SIGMOID/SILU) → MUL`. It requires F32, matching shapes, supported contiguity/strides/u32 spans, and existing consumer/dependency/alias checks. UNARY may be on either MUL operand. It does not enable GELU, GEMM, or general fusion.

Serialized submissions, disabled async/general fusion, scalar FP32, and disabled COOPMAT/COOPMAT2 remain mandatory. Invalid environment values/conflicts with `GGML_VK_MAX_NODES_PER_SUBMIT` are rejected. No extra scratch is needed; measured scratch admission, regions, completion, and ownership guards remain. Baseline/FP16 unset both experimental knobs.

New GPU-readback gates cover 4 SIGMOID/SILU + MUL cases using left/right views of shared backing. Each 8192×304-value case is checked against independent double-precision formulas with absolute/relative tolerance fixed at`0.00002`. Configuration 2 requires 1 limited-fusion selection; others require 0; CPU fallback is prohibited. Actual dispatch/submit/fence, finite values, mismatches, and durable receipts are checked before generation. **This describes implementation requirements, not hardware success for all configurations.**

Implementation: [option/metadata checks](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan-ps5-profile.h), Vulkan backend, runner admission, scratch bounds, and [dedicated gate](../native/app/moss_perf_gate.inc).

## Test contract and comparison conditions

| Variant | Profile | Precision | Added submission/fusion |
|---|---|---|---|
| 0 | `moss-performance-baseline-fp32.json` | FP32 | Disabled |
| 1 | `moss-performance-fp16.json` | FP16 | Disabled |
| 2 | `moss-performance-submit-fusion-fp32.json` | FP32 | Limited path above |

New `ps5-ai-moss-experimental-performance-v1` is explicit test opt-in. Existing 256/1-claim and 1024/2-claim contracts remain. The new contract permits at most 4 valid attempts per process. Invalid requests consume no claim; valid requests atomically claim before inference. Failures consume claims with no return/reset. A4-attempt cap does not instruct generating 4 images.

Fixed conditions:

- MOSS-Image-SFT Q4_K, Qwen3.5-9B Q4_K_M, FLUX.2 small VAE, selected rank 16 LoRA 0.7.
- Same prompt/negative as the previous 1024 baseline, seed 20261020,50steps, CFG 3, Euler, flow shift 5.
- 256×256 or 1024×1024, Direct pool 10GiB, outside reserve 1GiB, `Vulkan0=9.5` budget.
- Same driver/models/tokenizer/LoRA; this does not achieve the requested 12GiB target.

The client verifies 256 PNG/numerics/hardware completion/visual output and running/queued=0 before 1024 in the same process. Native counters do not judge visual quality. Manually request run 2 at 1024 only if needed. Each 1 POST is single-send, without automatic retry. On timeout, inspect job/health completion before deciding.

**1024 after the first 256 is the first new-shape request, not model-unloaded cold 1024.** Compare the same instrumented FP32 configuration and 256→1024 order, separately from previous fresh-process cold/warm pairs. Report `generation_seconds`, API roundtrip, and Windows wall separately. CPU/GPU-labeled peaks include cumulative shared-pool high-water marks; do not sum them or call them pure VRAM.

After each test, verify normal shutdown, absent owner, and independent idle before the next GPU title. No simultaneous GPU apps, system-area changes, or firmware changes.

## Verified host checks and cross-builds

- New contract 139, old 256 contract 72, old 1024 contract 42 checks passed. Actual atomic helpers verified invalid-request non-consumption, retained failed claims, concurrent cap 4, production 0, and legacy contracts. Host tests perform no inference/GPU computation.
- Freshly compiled all 75 core members with 0 old archive members reused; input hashes matched before/after.
- Verified 3 native SELF files: 24 initializers, 6 runtime providers, SELF integrity, requested precision/variant/flags, and unchanged inputs.
- Core, GGML Vulkan provider, RADV archive, and generated shader headers are byte-identical across 3 configurations. Native startup selects the variant. Equality itself does not prove hardware FP16/fusion use.
- Completed build: `optimization-v2`. Interrupted v1 is not successful and partial objects were not reused in v2. Original manifest/source-freeze records remain.

## Hardware comparison results

| Configuration | First 256 | First 1024 after 256, context loaded | Optional 1024 repeat | Numeric/quality/shutdown checks |
|---|---|---|---|---|
| Baseline FP32 | 153.972336 s | 1219.096190 s | Not measured | GPU generation/PNG, normal shutdown/idle verified |
| FP16 | No generation | No generation | No generation | Startup numeric gate rejected; normal OS shutdown/idle verified |
| Submission/fusion FP32 | 152.631656 s | 1214.444805 s | Not measured | Both sizes generated on GPU; PNG/shutdown/idle verified |

## # FP32 baseline observations

| Request | PS5 `generation_seconds` | API roundtrip | Windows wall |
|---|---:|---:|---:|
| First 256 including loading | 153.972336 s | 155.234 s | 155.781 s |
| First 1024, same process | 1219.096190 s | 1221.328 s | 1226.906 s |

At 1024, verified 50steps, 100 DiT forwards, Vulkan DiT/VAE completion, all 151 returned F32 outputs finite/0 nonfinite, and LoRA 504 loaded/matched/used/skip 0. Visual PNG inspection was normal; SHA 256 matched the prior 1024 output.

```text
484391514834f4788b5bceb8c9b5d0ba26ce023f0fa861a51b1a3d6bd2b28f27
```

Normal shutdown and independent idle followed 1024. 256→1024 changes conditioning-cache size conditions, unlike prior 1024→1024 warm runs. Differences from old warm times are not attributed directly to instrumentation slowdown or pure loading. Saved TAIL contained no explicit cache-hit line; unobserved hits are not invented.

Main 1024 CPU wall intervals follow. **Do not add parent/child metrics.**

| Metric | Count | Seconds |
|---|---:|---:|
| DiT graph calls including internal waits | 100 | 1169.098366 |
| DiT Vulkan backend graph call | 102 | 1168.964269 |
| DiT driver fence wait | 64691 | 1164.567069 |
| DiT driver submit | 64691 | 1.094833 |
| DiT dispatch record call | 310300 | 0.394776 |
| DiT pipeline bind record call | 310300 | 0.320132 |

Fence waits of 1164 seconds include GPU computation; they are not entirely removable synchronization overhead. Visible weight prepare 0.060834 seconds, activation upload 0.096877 seconds, and readback 0.056284 seconds cover measured calls only, not all hidden transfers.

API `peak_gpu_bytes`8,403,091,456B sums shared-Direct lifetime peak 8,369,520,640B and driver-arena lifetime peak 33,570,816B. CPU text weights also borrow Direct; this is not dedicated VRAM.19 temporary text claims totaling 4,835,246,080B overlapped remaining 256 Direct usage, exactly explaining the peak; all 19 returned. After 1024: Direct used 3,559,440,384B/free 7,177,977,856B, CPU text Direct live 0, no retained fault. This pair does not prove absence of all leaks/fragmentation.

## # FP16 candidate rejection and scope

Hardware FP16/16bit storage, absent disabling variable, and actual `fp16: 1` were confirmed. However, Q4_K `K3072/M17/N9` exceeded fixed `abs 0.001 + rel 0.01 × |expected|` for 41/153 values; maximum absolute error 0.0947755356, NaN/Inf 0. Startup admission rejected it, so generation POSTs 0 and image time/quality were unmeasured. Generic im 2col1/7 was not this direct rejection cause.

The Q4_K CPU oracle decodes stored quantized weights; it does not mix in differences from original unquantized weights. Source/linked shaders include half accumulation under FP16 quantized DEFAULT precision and extra half-staging rounding. These are candidate explanations, but individual contributions/runtime pipeline IDs were not separated experimentally. This does not establish impossible FP16 use or a confirmed driver defect.

FP16 arithmetic + F32 accumulation is source-feasible but unimplemented/unverified here. Tolerances are not relaxed; fixture-only changes do not prove model correctness. Normal OS shutdown was accepted, then absent PID/independent idle confirmed.

## # Submission/fusion 256 comparison

All 4 fixed cases checked 2,490,368 values each with mismatch 0/NaN/Inf 0, passing GPU readback/limited-fusion-selection gates.256 generation verified GPU completion, 50steps, all 103 outputs finite including DiT 100/VAE 1, NaN/Inf 0, LoRA 504 loaded/matched/used/skip 0.

| First 256 incl. loading | Baseline FP32 | Submission/fusion FP32 |
|---|---:|---:|
| PS5 generation | 153.972336 s | 152.631656 s |
| DiT graph calls incl. internal waits | 91.475941 s | 89.930709 s |
| DiT driver submit / fence count | 64258 | 62100 |
| DiT dispatch records | 335300 | 330950 |
| Limited-fusion selections | 0 | 4350 |

PNGs matched baseline bytes and decoded RGB/alpha exactly, with normal visual inspection. SHA 256:

```text
6b69b1915b9afa3375db3f66bb70d033014f88d6ac1892f0a3bc9004b577e52e
```

Total difference:−1.340680 seconds,−0.8707%. Each version was observed 1 time in a separate process including loading; temperature/clocks/OS cache were uncontrolled. This does not establish a large/repeatable speedup.

DiT dispatches decreased 4350 (1.2973%), matching fusion selections; submit/fence decreased 2158 (3.3583%). 2× thresholds did not halve submissions. FLOP/node/graph-end/almost-ready boundaries remain; individual trigger conditions were not measured. Combined submission/fusion differences cannot be independently assigned to each change.

DiT graph calls decreased 1.545232 seconds; submit-call CPU wall decreased about 0.044154 seconds. Fence-wait differences are not wholly wasted synchronization and parent/child scopes cannot be added.

## # Submission/fusion 1024 comparison

| First 1024 after 256, same process | Baseline FP32 | Submission/fusion FP32 |
|---|---:|---:|
| PS5 generation | 1219.096190 s | 1214.444805 s |
| API roundtrip | 1221.328 s | 1216.266 s |
| Windows wall | 1226.906 s | 1222.156 s |
| DiT graph calls incl. internal waits | 1169.098366 s | 1164.535786 s |
| DiT graph total | 1171.537857 s | 1166.958508 s |
| DiT driver submit / fence count | 64691 | 62471 |
| DiT dispatch records | 310300 | 305950 |
| Limited-fusion selections | 0 | 4350 |

Total generation decreased 4.651385 seconds (0.3815%); DiT graph calls decreased 4.562580 seconds. Dispatches decreased 4350 (1.4019%); submit/fence decreased 2220 (3.4317%). Each configuration was measured 1 time, without temperature/clock/OS-cache control; as with 256, single differences do not establish reliable gains.

Driver fence wait 1164.567069→1159.916232 seconds; submit CPU wall 1.094833→1.052372 seconds; other-runner graph total 46.107206→46.029770 seconds. These nested CPU wall metrics are not pure GPU duration or removable synchronization. Fixed 50steps/CFG 3 retains 100 DiT forwards and main matrix/Attention work. This limited path alone did not deliver major acceleration.

1024 PNG size 2,496,997B; full SHA 256 matched `484391…b28f27`. Baseline bytes/decoded RGB matched, with normal visual output. All 4 PNGs across 2 sizes matched their counterparts. No extra image copy was created; the existing successful 1024 asset is used.

Both configurations after 1024: Direct used 3,559,440,384B/free 7,177,977,856B, lifetime peak 8,369,520,640B, CPU text Direct live 0, no retained fault. Candidate API `peak_gpu_bytes`:8,402,403,328B. Its baseline difference reflects driver-arena lifetime peaks, not reduced dedicated VRAM.

## # Retained configuration and unverified scope

The installed experimental profile remains **variant 2: FP32 + submission sizing + limited fusion**. FP16 failed current tolerances and was not adopted; this does not rule out a future mixed-precision candidate.

Final native/gateway shutdown returned 202; absent PIDs/independent idle were confirmed. They were stopped. Authorized comparisons were complete; no extra 1024/warm repeats were run. max 4/production 0 remained, without returned failed claims, automatic retries, or additional driver/model/pool changes.

Long-running operation, means/variance, thermal/clock control, other prompts, FP16 image quality, minimum steps, and separate submission/fusion contributions remain unverified. Other models' speed/CFG-batching results are not MOSS measurements.

Client results, analyses, build inputs, normal shutdown, and independent-idle records remain private. Original logs, diagnostic JSON, credentials, and weights are excluded from Git.
