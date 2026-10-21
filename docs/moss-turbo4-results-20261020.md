# MOSS Turbo4 hardware results (2026-10-20)

> Historical test record. See [Turbo4/memory comparisons](moss-turbo-memory-options-20261020.md) and [verification status](status.md) for the currently adopted candidate and latest times.

This page records the earlier Turbo4-only trial without bounded upload. Subsequent [combined/FFN-chunking/F16-storage comparisons](moss-turbo-memory-options-20261020.md) are separate measurements.

A dedicated PS5 trial generated 256/1024 with MOSS-Image-SFT Q4_K and combined rank 144 F32 NSFW0.7 + Turbo1.0 adapter, retrieving PNGs through the OpenAI Images-compatible API. Windows is the client; PS5 CPU performs text encoding and PS5 Vulkan performs DiT/Flash Attention/VAE.

| PS5 generation time | 50-step baseline cache version | Dedicated Turbo4 trial |
|---|---:|---:|
| First 256×256 incl. lazy load | 146.016289 s | **71.564874 s** |
| First 1024×1024 after 256, same process | 1023.859406 s | **51.133430 s** |

Single-observation reductions were 50.99% at 256 and 95.01% at 1024. Positive prompt/seed matched, but steps 50→4, CFG 3→1, adapter, and sigmas changed. This is neither equal-quality/identical-PNG comparison nor 20× acceleration of the same model's kernels alone. Each condition was observed 1 time with no OS-cache flush or temperature/clock control.1024 is first after 256, not standalone cold 1024 or run 2 at 1024.

## Fixed conditions and hardware checks

- 4-step Euler, CFG 1, seed 20261020. Already-shifted sigmas `[1.0, 0.910, 0.717, 0.024, 0.0]` receive no second flow shift 5. CFG 1 skips negative-conditioning forwards.
- Combined rank 16 NSFW coefficients 0.7 and rank 128 Turbo coefficients 1.0 into rank 144, preserving independent q/k/v; runtime multiplier 1.0. [Input revisions/full SHAs/converter/notices](moss-turbo4-reproduction.md) are fixed; original rank 16 is retained separately.
- FP16 data/FP32 accumulation retained. Existing 14 numeric gates +3 rank 144 GPU composition/readback gates passed. Each image verified 4 finite DiT outputs, 672 Q4 F32-accumulation records/0 F16/0 invalid/0 nonfinite, LoRA 504 loaded/matched/used/skip 0.
- Native/OpenAI PNG bytes matched for each image; 256/1024 visual checks found no major defects. Each used 1 POST/0 retries. This is not broad prompt-quality or long-running stability evidence.

1024 recorded 4 DiT graph calls/44.542630 seconds and 49 other-runner calls/5.987196 seconds. These are nested CPU wall times, not GPU timestamps or additive intervals. Complete current-log sections showed 0 cache hits at 256 and 1 positive-conditioning hit at 1024. CFG 1 has no negative forward.

![1024×1024,4steps generated in the dedicated PS5 Turbo4 trial](images/moss-turbo4-upload-1024-20261020.png)

Normal native/gateway API shutdown preceded separate absent-process observations. A fresh independent observer confirmed same boot, existing services/syscore/shellcore, no GPU title, 3 title registrations, and 5 selected models' metadata preserved. Forced signals/manual GPU releases: 0.

## Memory interpretation

Both baseline/Turbo4 physically reserved 10GiB Direct.1024 API GPU-labeled high-water: baseline 4,868,833,280B (about 4.53GiB), Turbo4 5,968,183,296B (about 5.56GiB). These are shared-Direct/driver upper bounds, not pure VRAM. API CPU values also include cumulative process peaks and cannot be summed. This 1024 owned Direct data peaked 5,934,579,712B and ended 4,749,066,240B, distinct from physical reservation 10,737,418,240B.

Combined rank 144 coefficient payload is 1,337,720,832B, 1,189,085,184B (about 1.11GiB) above rank 16. This storage difference is not measured physical RAM/VRAM growth. Transfers, temporary workspace, resolution, and high-water timing overlap; coefficient size alone cannot determine actual memory.

A separate [bounded-upload trial](moss-bounded-weight-upload-20261020.md) capped grow-to-fit staging at 8MiB/worker (potential 24MiB with 3 workers), lowering same 256 API `ru_maxrss × 1024` by 216.6875MiB. Direct reservation/peak/GPU-labeled peak stayed equal, not proof of physical RAM/VRAM reduction. **Bounded upload was disabled for Turbo4 here (Turbo4); do not add those savings to this result.**

BF16-weight readback gates passed, so backend `bf16: 0` alone does not imply BF16 is unusable. However, BF16 matrix RHS rounding and non-BF16-exact rank 144 coefficients remain concerns. Changing the F32 contract requires separate numeric/image comparisons.

Earlier [30-step sampler/Spectrum](moss-sampler-spectrum-memory-20261020.md) and [text-cache](moss-additional-optimization-20261020.md) results remain. This was also a guarded development trial, production admission 0; the 12GiB target remains unmet.
