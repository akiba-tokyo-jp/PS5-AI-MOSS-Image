# Driver-aware MOSS optimization plan

2026-10-21. **Planning only: no implementation, build, deployment or new PS5
measurement is claimed here.** Execute hardware comparisons after the selected
driver revision, feature profile and qualification state are fixed. A source
merge alone does not enable an opt-in feature or update the statically linked
driver inside MOSS.

Follow [the driver source/integration policy](driver-integration.md). The
driver-only baseline is now recorded in [the 2026-10-22 sequence](moss-cumulative-driver-20261022.md);
the P1-P5 application candidates below remain plans. Preserving native
allocator members applies only within a compatible runtime ABI. A changed
driver ABI requires rebuilding and qualifying the matching native adapter.

The objective is lower actual MOSS generation time at 512 and 1024, preserving
the authored model recipe and numerical, image, memory and completion checks.
Include currently unused driver mechanisms when a justified application change
could exploit them. Do not treat a synthetic driver gain as a measured MOSS gain.

## Frozen control and measurements

Use the current ordinary service as the functional control: Q4_K base, F16
matrix scratch, F32 accumulation, four Euler steps, CFG 1, authored sigmas,
NSFW0.7 + Turbo1.0 rank144 F32 adapter, existing mask15 optimizations and kernel
mask15 with the adopted Attention occupancy selection. Keep physical Direct
10 GiB, inference budget 9.5 GiB, CPU thread setting, tokenizer, VAE, model
files and API behavior fixed while comparing arithmetic or driver factors.

Record a new matched control. Historical 1024 repeat 44.839066 s, DiT graph
wall about 38.56 s mean and VAE decode 6.11 s are context, not promised timings
for a new prompt or driver. The DiT graph wall includes pending GPU execution
and waits. Earlier operator percentages came from a different kernel baseline.

Every future result should identify source/driver/profile, linked archive and
application objects, shader variants, actual width/height/text-token count,
seed, steps, adapter identities/scales, cache state and memory scope. Report:

- Native generation time, including all candidate preparation or weight merges.
- Conditioning load/preparation, tokenization, CPU/GPU graph and output-copy
  time separately where the actual instrumentation can distinguish them.
- Sampling/DiT, VAE encode where applicable, and VAE decode wall time.
- Valid GPU dispatch intervals and their exact coverage. Existing two-complete-
  forward profiling is not a full four-forward GPU measurement.
- Thread CPU time if supported by the native runtime; unavailable counters are
  unknown. A 10 ns timestamp period is not a shader-clock measurement.
- Current and high-water owned Direct use with per-job or lifetime labels,
  cache/scratch residency, allocation failures and retained faults. Do not add
  overlapping CPU/GPU API peaks as physical memory.

Use identical instrumentation in each comparison. Separate diagnostic GPU/ISA
captures from uninstrumented end-to-end timing; do not silently pool them.
Aggregate timing/log summaries at job boundaries. Retain required fault and
ownership journals and their bounded-storage policy.

Warm pipeline timings require an earlier completed generation of the same
shape/recipe and prompt; a loaded model alone is insufficient. Reuse a needed
functional pilot as that warm-up where its conditions match. Label first-use
of a new resolution separately, match the prefix in both arms, and count all
warm-up requests when reporting how much hardware work was performed.

## Implementation candidates

All names below identify planned candidates, not existing command-line options.

| ID | MOSS-side change | First comparison and decision |
| --- | --- | --- |
| P1 | Observe actual DiT CPU work; if material, reduce optional CPU work and evaluate the qualified driver wait policy | Identical request under normal and bounded CPU-load conditions; then isolate one optional-work reduction. Distinguish power, bandwidth, cache and submission effects |
| P2 | Larger VAE decode tiles | Tile256/overlap0.5 versus tile512/overlap0.5 at 512 and 1024; untiled follows only after real workspace admission |
| P3 | Shape-aware weight-side LoRA merge with bounded reuse | Runtime adapter versus per-GEMM merge versus per-job retained subset; keep the Q4 file unchanged and include merge cost |
| P4 | Separate and reduce new-prompt conditioning costs | CPU load/graph breakdown first; GPU Qwen becomes an independent candidate after all actual operations and residency are qualified |
| P5 | Compiler-aware matrix/Attention kernels and real SiTU epilogue | Inspect actual hot ISA first, then isolate conversion/indexing, direct-Q4/predecode, tile/subgroup or epilogue changes |

### P1: keep progress and display liveness

The normal serialized backend already uses `waitForFences`. The inspected
driver spins initially and then sleeps while work is pending; changing that
policy belongs to a matched driver candidate, not an application-only API swap.
Do not stop the inference worker or the kick worker that makes GPU progress.
The gateway already sleeps between polls. Measure their actual CPU cost before
changing polling behavior or required logging.

A concrete optional application candidate is dirty-region dashboard drawing.
Current drawing clears and flushes the entire padded 16 MiB inactive frame
about once per second. Initialize both buffers, track each buffer's prior
content, update changed regions and flush the actual changed tiled cache lines
with the existing ordering and positive flip/ownership proof. Preserve the
clock, live heartbeat and update frequency. Merely slowing the publisher below
the current three-second freshness window would make the UI appear stale.

If a controlled load experiment is warranted, start with one bounded CPU
compute worker with a small working set during DiT; stop cooperatively at the
phase boundary. A memory-streaming arm is a separate diagnostic only when
needed. Compare the same prompt/seed/graph and completed GPU intervals. A
loaded-arm slowdown proves interference, not uniquely a power-budget cause.

Sources: [startup](../native/app/startup.c), [display](../native/ui/display.c),
[dashboard](../native/ui/dashboard.h), [status](../native/ui/status.h),
[kick](../native/app/kick_impl32.c), and [Vulkan backend](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan.cpp).

### P2: tile changes are not automatically quality-neutral

Use the existing tile selector and overlap. At 1024, 256 tiles give 49 tiles
and 512 gives 9; area work decreases from 3.0625x to 2.25x, not by 49/9.
The approximately 1.62 s saving calculated from 6.11 s is an area-only scenario.
At output512 the same choices also compare tiled and full-image decode, which
must be labeled explicitly. Capture tile dimensions/count, allocated workspace,
decode intervals, finite outputs, seams and fine detail. Test img2img encoding
separately only after the decoder candidate passes; never change its path
silently as part of a decode comparison.

Check actual allocation admission before GPU execution. Keep the existing tile
path available when a larger tile cannot be admitted safely. Uncertain GPU
completion is not a recoverable allocation refusal and must not trigger retries.

Source: [engine configuration](../native/app/engine_config.h); prepared upstream
`src/model/vae/vae.hpp` and `src/runtime/tiling.cpp` at the pinned dependency.

### P3: amortize merging, including at 512

For input K, output M, rank r and actual token count N, adapter runtime work is
`2*r*N*(K+M)` FLOPs; building `B*A` is `2*K*M*r` per materialization, before
additional memory work. Use real shapes: up/gate K3072/M16384, down
K8192/M3072 and separate q/k/v K3072/M3072. With 106 text tokens, 512 has
N1130; per-GEMM merging costs more arithmetic for these shapes. Do not select
merge simply because width/height are at least 512.

The first implementation should support three comparable execution modes:

1. Existing runtime adapter as the unchanged fallback/control.
2. Merge a selected projection into its F16 scratch on each use, computing
   the adapter contribution with F32 factors/accumulation before the F16 store.
3. Build and retain a fixed, admitted subset for one generation, reuse it for
   the remaining authored DiT forwards, and release after proven completion.

Mode 3 begins after conditioning and its temporary claims have completed and
been released; it ends before the next job's text encoding. Initially do not
retain merged weights across jobs or implement next-job preencoding. This
avoids relying on simultaneous text/DiT residency to obtain reuse.

Do not cache every dense weight: all main linears would require 8.859375 GiB
before other allocations. Examples of complete subsets are all FFN-down
weights at 1.96875 GiB and all q/k/v at 2.21484375 GiB; these are storage
calculations, not proof either subset fits the actual allocator. Choose a
fixed subset by measured saved time per byte under real admission. A simple
LRU smaller than the 42-layer streaming working set can thrash every forward.
Do not claim a smaller physical reservation when only scratch/cache live use
changes. Budget the build temporary and retained destination together.

The cache key includes model/base-weight version, projection/slice and shape,
adapter content versions and order, ranks, alpha/scale/strength, output dtype,
layout and kernel policy. Invalidate on any change. Preserve independent
q/k/v mapping. Apply each adapter exactly once; merged projections must not
also take the same runtime delta. Track effective adapter application and real
merge/runtime work separately rather than fabricating old runtime-GEMM counts.

Start real inference with the existing validated rank144 adapter. Add independent
small fixtures for zero strength, sign/strength changes, cancellation, rank16,
rank144 and an additional rank16 contribution. Later generic LoRA admission
must validate every supplied A/B pair and requested target; acceptance is an
explicit feature, not bypass of the current fixed-adapter identity policy.
Accepted-but-unmergeable projections use the runtime path; an unsupported
adapter is explicitly rejected, never silently ignored. Adapter changes cannot
reuse stale merged weights. No automatic public arbitrary-LoRA claim precedes
those checks and actual image verification.

Keep original startup gates and add combined-merge readbacks. The new F16
rounding/reassociation can change images: record comparison to F32 host
reference and runtime behavior, without increasing tolerances after seeing
failures. A gate failure rejects that merge shape/precision candidate.

Sources: [adapter](../patches/stable-diffusion/overlay/src/model/adapter/lora.hpp),
[current adapter contract](../native/app/moss_turbo_lora_contract.hpp),
[engine](../native/app/engine_sdcpp.cpp), and the Vulkan predecode path.

### P4: distinguish cache hits, loading and learned compute

Completed conditioning cache reuse already exists. Saved new-prompt conditions
took about 25-47 s; cached conditions were 0.00 s at the recorded precision.
Separate current Qwen load/preparation, tokenization, graph and copy costs
before choosing a CPU-loading or GPU-compute implementation. The roughly 23 s
other-runners preparation metric is not exclusive Qwen file I/O.

GPU Qwen requires actual graph coverage, especially gated-delta-net shapes,
and a new explicit backend/admission path. Preserve F32 numerical policy and
validate conditioning values before image generation. First execute text and
DiT sequentially with a measured residency plan, accounting for reload or
offload time. The conditional old CPU-text claims plus cached-text DiT peak
exceed 10 GiB; nominal free space does not prove overlap fits. No increase in
the physical reservation is part of this plan.

Preencoding the next job is deferred until CPU-interference and residency
results support it. Persistent pipeline caching is a separate process-cold
candidate requiring the finalized driver's verified cache identity/UUID and
bounded invalid-cache fallback. It cannot accelerate resident warm arithmetic.

Sources: [pipeline](../patches/stable-diffusion/overlay/src/pipeline/image.cpp),
[conditioner](../patches/stable-diffusion/overlay/src/conditioning/conditioner.hpp),
[Qwen implementation](../patches/stable-diffusion/overlay/src/model/te/llm.hpp),
and [engine configuration](../native/app/engine_config.h).

### P5: expose useful driver paths and actual shape coverage

Inspect final ISA and resources of current FFN, QKV/O, LoRA and Attention under
the chosen driver. If unused driver transforms can help, build an isolated
shader change that exposes legal F16-before-F32 selection or invariant
indexing. Preserve float controls, bounds, read-only semantics and F32
accumulation. Compare direct Q4_K with per-GEMM F16 predecode under the same
compiler; changing a broad mask that also changes LoRA/RMS is not an isolated
predecode comparison.

Re-evaluate a small set of legal tile/workgroup/subgroup variants for actual
shapes. Verify actual wave size and VGPR/LDS/scratch/spills; neither a NAVI21
label nor an API queue count proves a hardware instruction or parallel ring.
Do not enable unsupported dot2/cooperative matrix or impose a global wave size.

Main up/gate GEMM and standalone SiTU fusion already exist. A true SiTU
epilogue must have both outputs from the separate 8192-row halves and the full
adapter contribution available before the nonlinearity. Preserve SiTU; do not
substitute SwiGLU. Assess the added accumulator/register cost as well as saved
intermediate reads/writes.

There is also a concrete unexercised-shape issue: current optimized predecode
and shared-KV Attention selectors include a token ceiling of 4202. At 1024,
longer text can select the valid baseline GPU path. Record this selection.
After new shape/readback/capacity checks, evaluate a bounded extended optimized
domain around N4202/4203 and a representative longer sequence such as N4608.
Those are diagnostic shapes, not new public prompt or request limits. Keep
valid baseline fallback outside the optimized domain and retain ordinary
service admission. Do not merely remove a shape guard without validation.

Sources: [MOSS graph](../patches/stable-diffusion/overlay/src/model/diffusion/moss_image.hpp),
[Vulkan](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan.cpp),
[kernel checks](../native/app/moss_kernel_matrix_gate.inc), and
[kernel policy](../tools/moss_kernel_policy.py).

## Minimal comparison protocol

1. Freeze the qualified driver and active opt-in features, plus a known
   rollback MOSS artifact. Preserve its native platform/allocator/physical-
   device members. No console work overlaps the driver's own tests.
2. Establish the new driver with unchanged MOSS as a driver-only comparison.
   Keep the old exact application/core/shader objects for this relink.
3. Screen one application change at 512 then 1024 against a matched control.
   Reuse the same prompt and actual token count within each pair. Use seed
   20261020, empty negative prompt and the fixed four-step model recipe.
   There is no prerequisite 256 generation.
4. Initially use one short public, nonpersonal prompt, for example:
   `Anime illustration of an adult woman with silver hair and a red jacket,
   standing on a rooftop beneath a starry sky.` Record actual token counts;
   do not assume historical N4202 or historical generation time.
5. Only a promising, numerically valid candidate receives at least three
   independent paired repetitions with alternating order. Compare paired
   deltas, median and observed spread. Three repetitions are an initial
   decision screen, not a p95 or long-run stability claim. If the effect is
   within variation, add only a justified small confirmation or leave it
   unadopted; do not generate a large catalog to find a favorable result.
6. For driver-dependent application changes use old-driver/old-app,
   new-driver/old-app, old-driver/new-app and new-driver/new-app as needed.
   Do not add the separate percentage improvements to predict their combination.
7. Evaluate process-first 1024 in a fresh process with no prior image, plus
   repeated-prompt warm generation, only for finalists or cold/text/cache
   candidates. A 512 pilot in another process is not a 1024 cold request.
   New prompts and cached prompts form distinct cohorts.
8. After a winning individual change, confirm a second seed, a different text
   length and one rectangular/tail case. For math/tile/text changes also run
   one supported img2img case. Test invalidation/replacement for LoRA and
   changing prompt for text/cache separately. Combine winners incrementally.

## Acceptance and fallback

- Correctness first: existing startup/readback gates, finite learned outputs,
  authored forward counts, actual adapter use, positive completed fences,
  ownership and real capacity remain. Do not clamp NaNs or loosen tolerances.
- Driver-only or pure CPU/UI changes target exact generated pixels/PNG bytes
  where encoding is deterministic. Arithmetic/tile/encoder changes need
  numerical and visual comparison; do not promise byte equality or universal
  quality from one image. Keep raw fixtures and reproducible candidate outputs.
- Speed adoption requires a repeatable improvement in the intended end-to-end
  cohort beyond observed control variation, with no unexplained size/shape
  slowdown. A faster isolated kernel with a slower whole job is not a MOSS
  speed adoption. Small or uncertain effects remain optional/unadopted.
- Report memory benefit separately from speed. Keep the normal API and current
  reservation available; pre-execution admission can select the supported
  fallback. Uncertain GPU completion is not a condition for automatic fallback,
  buffer release or a repeated POST.
- Few paired samples do not qualify tail latency or lifetime operation. Any
  claim of routine use needs bounded mixed-request confirmation, known release
  and restart behavior and no unexplained current-use growth. Preserve the
  existing fail/retention path on uncertainty.
- The shared driver can independently qualify useful Compute/LLM work even
  when MOSS is neutral. This application's acceptance does not redefine the
  other repository's integration policy.

Initial order: release/profile identity and measurement coverage; P1 observation
and P2; P3 with within-generation reuse; P5 real-kernel/compiler/shape candidates.
For new-prompt-heavy use, P4 load/compute measurement starts equally early, with
GPU encoding implemented only after graph/residency qualification. Document
actual gains and non-gains; no proposed speedup is a measured result.
