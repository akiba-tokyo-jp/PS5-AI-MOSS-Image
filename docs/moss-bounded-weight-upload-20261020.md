# Host staging for GPU weight transfer: hardware memory observations

> Historical test record. See [Turbo4/memory comparisons](moss-turbo-memory-options-20261020.md) and [verification status](status.md) for the currently adopted candidate and latest times.

2026-10-20 JST. MOSS raw, same-format GPU weight transfer was chunked using host Scratch that grows as needed. **The 8MiB limit is per worker.** Existing Vulkan tensor-transfer/synchronization paths are retained; CPU code does not directly write GPU addresses. Model coefficients, inference precision, and sampling were unchanged.

## First 256 comparison with identical requests

| Condition/result | Baseline cache version | Bounded-transfer version |
|---|---:|---:|
| PS5 first generation, including lazy loading | 146.016289 s | 145.680067 s |
| Size / steps / sampler / CFG / flow shift | 256×256 /50 /Euler /3 /5 | Same |
| Seed / selected LoRA | 20261020 /rank 16 F32,0.7 | Same |
| Completed finite GPU DiT outputs | 100 | 100 |
| Q4 matrix F32 accumulation selections / F16 accumulation / invalid | 16600 /0 /0 | 16600 /0 /0 |
| Selected LoRA tensors used / skipped | 504 /0 | 504 /0 |
| API-returned/native-GET PNG | Same-request baseline | Byte-identical to baseline |

The difference was−0.336222 seconds (−0.2303%); these observations of 1 run each show essentially equivalent generation times, not a repeatable speedup. Existing 14 numeric gates and a 9MiB+68B transfer/GPU SCALE/CPU-readback gate crossing the 8MiB boundary also passed. Hardware retained PS5 CPU text encoding, Vulkan DiT/VAE, and PNG retrieval through the OpenAI-compatible API.

## Transfer paths actually exercised

| Completed load target | Tensor uploads / chunks | Largest original tensor | Maximum observed worker Scratch |
|---|---:|---:|---:|
| DiT base | 185 /462 | 113,246,208B (108MiB) | 8,388,608B (8MiB) |
| Selected rank 16 LoRA | 504 /504 | 1,048,576B | 1,048,576B |
| Relevant VAE portion | 104 /104 | 589,824B | 589,824B |

Across 3 completed loads, 793 tensor uploads /1070 chunks and 3,477,715,724B used the affected path. This sums load calls, not unique resident tensors. LoRA/VAE Scratch does not reserve 8MiB initially: each worker grows to its tensor's needs and reuses storage. Growth frees the old region first to avoid a temporary double-allocation peak.

Each load observed 3 workers; the maximum per file was also 3. Potential concurrent staging is 3×8MiB＝24MiB. The 9 worker instances summed across 3 loads are not simultaneous workers.24MiB is not measured simultaneous usage, whole-loader/process memory, or Direct reservation. This bound does not apply to unaffected CPU Qwen host weights, format conversion, mmap, zip, or other paths.

## Heap mapping and RSS observations

| Measurement point/accounting | Baseline cache version | Bounded-transfer version | Difference |
|---|---:|---:|---:|
| Pre-generation heap mapped / peak-mapped | 64 /64MiB | 64 /64MiB | 0 /0MiB |
| After model-context creation: heap mapped / peak-mapped | 240 /240MiB | 240 /240MiB | 0 /0MiB |
| Post-generation heap mapped capacity | 480MiB | 400MiB | −80MiB |
| process lifetime heap peak-mapped capacity | 777.3125MiB | 545.125MiB | −232.1875MiB |
| API `ru_maxrss × 1024` high-water mark | 11,787,255,808B | 11,560,042,496B | −227,213,312B (−216.6875MiB) |

Heap figures are allocator segment capacity, not live malloc payload or resident pages. The post-model-context point is not completion of all lazy large GPU weight transfers. Post-generation segments decreased 28→25; arenas stayed 8 and libc fallback 0.

Heap peaks and `ru_maxrss` span startup gates, CPU text encoding, and all generation stages. Do not attribute the observed decrease to individual tensors or loader-only physical savings. Heap-capacity and RSS differences cannot be added; later savings can also be hidden by earlier high-water marks. The baseline lacks equivalent worker-capacity telemetry, so exact per-tensor byte savings from its old vectors cannot be calculated.

## Direct and GPU-labeled values are unchanged

| Accounting | Value in both versions |
|---|---:|
| Physical Direct reservation | 10,737,418,240B (10GiB) |
| Owned Direct allocation high-water mark | 4,835,262,464B |
| Post-generation owned Direct live claims | 3,536,896,000B |
| CPU text shared Direct high-water mark | 4,835,246,080B; final live 0 |
| API GPU-labeled pool + driver high-water mark | 4,867,227,648B |

Reduced host staging did not return the startup-allocated/mapped 10GiB physical Direct reservation to the system. Shared Direct/driver accounting is not discrete-GPU dedicated VRAM residency and must not be added to CPU RSS. Whole-process physical RAM, resident VRAM, and total UMA savings remain unproven.

## Timing and verification scope

Loader total wall times at 0.01-second precision were CPU text 23.16→22.89 seconds, DiT 12.96→12.90 seconds, and VAE 0.38→0.38 seconds. CPU text is outside the GPU raw-transfer change and is not treated as a causal improvement. Read/memcpy/convert/backend-copy breakdowns divide worker totals by thread count; they are neither additive wall stages nor pure GPU time.

This experiment compared 1 first 256 generation per version, 50 steps with Spectrum disabled. It does not include speed distributions controlling temperature/clocks/OS disk cache, bounded-transfer 1024, Spectrum/Turbo combinations, or long-term operation. Smaller host staging and equal output are distinct from guaranteeing whole-model memory fit/quality under all conditions.

For bounded-staging design references and licenses, see [existing sources](moss-sampler-spectrum-memory-20261020.md#source-and-attribution) and [third-party notices](../THIRD_PARTY_NOTICES.md). This public document excludes weights, credentials, raw logs, and personal paths.
