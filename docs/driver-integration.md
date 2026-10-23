# Driver source and MOSS integration

Updated 2026-10-23 JST. Future MOSS driver integration uses
[PS5-Vulkan-Inference](https://github.com/akiba-tokyo-jp/PS5-Vulkan-Inference),
as selected by the owner. This is an unofficial downstream of the original
PS5 Vulkan/Mesa work. Keep upstream attribution and license notices.

The selected repository is currently private. Access is needed to reproduce
its profile; this public MOSS repository does not redistribute its source,
archives, native executables or proprietary SDK files. Repository selection
does not automatically activate a profile or change a running application.

## Measured baseline

| Item | Recorded selection |
|---|---|
| Driver revision | `872cc1264ad8ba5d6b888b092968fa5858a38c58` |
| Explicit profile | `cumulative-idle` |
| Profile builder at that revision | `driver/mesa/build_profile.py` |
| Mesa revision | `1ba9874f478657ccf6da72ce55f0260164cc0bd0` |
| Persistent shader/pipeline caches | Disabled before backend initialization |
| Integration | Static application relink; original application objects reused |
| Changed archive members | `radv_shader.c.o`, `winsys_ps5_radv_ps5_winsys.c.o` |
| Other archive members | All 702 preserved, including native allocator/platform specializations |
| Initializers/runtime providers | 24 / 6 |

The profile combines the completion-snapshot correctness prerequisite,
half selection before F32 widening, conservative late loop-invariant
addressing, and the cached idle-completion path. It does not enable the
[planned MOSS application optimizations](optimization-validation-plan.md).

The actual PS5 sequence was first 256 including lazy model load, followed by
two 1024 requests in the same process: **70.786748 / 44.835497 / 44.840077
seconds**. The 1024 mean is **44.837787 seconds**, compared with the historical
45.127473 seconds. All three PNGs match their previous reference bytes.
[Full conditions, phases, memory, image and lifecycle checks](moss-cumulative-driver-20261022.md),
[machine-readable record](moss-cumulative-driver-20261022.json).

This is one sequence with historical controls. It does not establish a causal
speedup, general numerical/quality equivalence, all-shape memory capacity or
long-running stability. The 256-first sequence is a comparison fixture;
ordinary API use does not require a preceding 256 image.

## Future integration procedure

1. Read the selected driver's applicable instructions and integration contract.
   Pin its exact revision and explicit profile. The revision above identifies
   this measurement; a newer commit or profile is not automatically covered.
2. Match Mesa, compile commands, archive provenance, headers and runtime ABI.
   For this compiler/completion-only profile, replace only verified matching
   members and retain MOSS native allocator/platform/physical-device behavior.
   Do not replace the complete archive with an arbitrary generic driver.
3. If the driver runtime or allocator ABI changes, rebuild the matching native
   adapter and qualify its capability, ownership, cleanup and fault-retention
   contract. Do not carry an old allocator object across an ABI change merely
   because the link succeeds. New offline candidates are separate from this
   hardware-qualified measurement scope.
4. Keep persistent caches disabled until a distinct cache identity and reuse
   policy have been independently qualified. Do not reuse caches across
   compiler/profile binaries or infer warm inference gains from cache plans.
5. Relink MOSS statically and record application/core/shader inputs, driver
   archive/member differences, SDK/toolchain, initializers, runtime imports
   and final executable identity. The [service build](build.md) consumes the
   configured libraries; it does not download or select driver profiles.
6. Preserve the known rollback artifact. For authorized native verification,
   use one identified GPU title, full transfer/readback, existing startup
   numerical checks, finite learned outputs, GPU completion and measured
   capacity. Send each image request once; inspect an ambiguous outcome before
   another request. Use ordinary shutdown and independent process confirmation.

Never weaken barriers or release uncertain GPU resources to improve timing.
Keep driver-only and application-only changes separate in comparisons. Retain
the model, four-step recipe, precision, memory placement and cache conditions
unless the next experiment explicitly isolates a change.

The measured candidate was restored to the original application after testing.
That restored native application and gateway were subsequently closed normally
at the owner's request; process absence was confirmed. This documentation
publication performs no PS5 launch, configuration change or image generation.
