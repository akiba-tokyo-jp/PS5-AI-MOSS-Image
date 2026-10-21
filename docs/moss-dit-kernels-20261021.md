# 4 DiT kernel changes and 1024 hardware comparison

2026-10-21 JST. Implemented 4 changes and generated 256→1024→1024 for each configuration on PS5. **Mean 1024 time decreased from 47.481390 to 45.127473 seconds, 4.96%.** The adopted configuration adds new kernel mask 15 and Attention occupancy adjustment to existing DiT optimization mask 15. After normal shutdown and independently verified process absence, the same verified SELF launched in a new process; all numeric checks and idle native/OpenAI-compatible gateway were confirmed. No generation POST followed restoration.

![Adopted PS5 1024 output](images/moss-dit-kernels-1024-20261021.png)

## Implementation and adoption decisions

| Priority | Change | Decision |
|---|---|---|
| 1 FFN | Decode Q4_K once per GEMM into existing F16 scratch; reuse with F32-accumulating matrix products. Preserve original Q4 weights | Adopted: control mean 47.481390→46.543395 seconds |
| 2 Attention | Select existing shared-KV shader for limited head 128/F16 K/V/F32-accumulation shapes, then remove unused AMD LDS occupancy padding only on this path | Original-limit candidate slowed to 51.578863 seconds mean and was rejected. Adjusted version improved RMS version 45.739973→45.127473 seconds and was adopted |
| 3 QKV/LoRA | Q4 predecode for QKV/O; retain independent q/k/v; process LoRA addition/packing with 4 elements/thread | Adopted: FFN version 46.543395→46.126669 seconds |
| 4 RMS/conversion | Fuse 512-lane F32 RMS reduction with weight/modulation or QK RoPE; retain epsilon 1e-6 and reduction topology | Adopted: QKV version 46.126669→45.739973 seconds |

These contributions compare cumulative configurations, not independent contributions across all combinations or statistical significance. Attention occupancy adjustment is limited to the target shader's workgroup resources; it does not change the GPU driver, OS memory allowance, or physical Direct reservation.

## Identical-condition timing

MOSS-Image-SFT Q4_K, PS5 CPU Qwen3.5-9B Q4_K_M, Vulkan DiT/small VAE, combined rank 144 **F32** adapter (NSFW0.7 + Turbo1.0, runtime 1.0). Fixed prompt/seed 20261020,4 Euler steps / CFG 1, author sigmas [1.0,0.910,0.717,0.024,0.0], no double shift. FFN chunks, F16 LoRA storage, Spectrum, approximate denoising cache, and general fusion are OFF.

| Configuration | First 256 incl. lazy load | First 1024 after 256 | Run 2 at 1024, same process | Mean of 2 runs at 1024 |
|---|---:|---:|---:|---:|
| Previous DiT fusion version (control) | 69.111579 | 47.800313 | 47.162467 | 47.481390 |
| FFN predecode | 69.561725 | 46.836615 | 46.250174 | 46.543395 |
| FFN + Attention shared KV, original occupancy limit | 69.471026 | 51.873448 | 51.284278 | 51.578863 |
| FFN＋QKV／LoRA | 69.778932 | 46.429972 | 45.823365 | 46.126669 |
| FFN＋QKV／LoRA＋RMS | 69.641481 | 46.068369 | 45.411577 | 45.739973 |
| All 4 changes + Attention occupancy adjustment | 69.449735 | 45.415880 | 44.839066 | 45.127473 |

All values are PS5 generation_seconds, not standalone cold 1024. Each configuration started a new native process at 256; each request used 1 POST and 0 automatic retries. The 2 PNGs at 1024 matched within each configuration. These are sequential means of 2 observations, not alternating A/B tests controlling temperature/clocks/OS disk cache. First 256 time increased slightly; no claim is made that every size or first load improved.

Mean 1024 DiT graph CPU wall decreased 40.906064→38.558765 seconds. This includes waits and is not a pure GPU timestamp. Other runners stayed near 6 seconds; most savings were in DiT. See [aggregate values](moss-dit-kernels-20261021.json) and [earlier GPU timestamps](moss-dit-gpu-profile-20261021.md).

## Numeric, image, and memory checks

- Retained original 14 numeric gates, Turbo3, upload, and 16 fusion cases. The adopted version also read back 20 added cases on PS5:0 mismatches, 0 nonfinite values, owned-buffer release and completed fences confirmed. Q4 products retain abs 0.001/rel 0.01; fusion/Attention retain abs 2e-5/rel 2e-5.
- Added cases: FFN 4, QKV 2, Attention 2, LoRA vec 4 contiguous/strided 6, RMS contiguous/strided 6. Matrix fixtures separately tested actual M/K and tokens 4,202; real generation executed all 42 layers. Fixture timing includes shader compilation and is not a generation-speed metric.
- Every generation confirmed 4 finite DiT outputs, 672 Q4 F32 records/0 F16 accumulation/0 invalid, 504 LoRA applications/0 skipped, and completed GPU DiT/Attention/VAE. Selected-path receipts were checked. Windows performed no learned inference.
- OpenAI-compatible/native-GET PNGs matched and passed full decoding/visual checks. The 2 adopted 1024 outputs were byte-identical.
- **Old and adopted PNGs are not identical.** Rounding changes from Q4→F16 temporary decoding/fusion propagate during denoising. At 1024, MAE was 3.5000/255 and PSNR 28.91dB. No obvious defects were seen, but these metrics and 1 prompt's visual inspection do not guarantee equal quality or all-prompt quality. NaN replacement/clamping, skipped steps, or low-precision accumulation were not used to hide results.

| Owned Direct data (1024) | Control | Adopted |
|---|---:|---:|
| Generation high-water mark | 6,107,971,584B | 6,208,634,880B |
| Post-generation | 4,749,066,240B | 4,849,729,536B |

Both high-water and final usage rose 96MiB. The adopted high-water mark was approximately 5.78GiB without accumulation on run 2. Physical Direct 10GiB, inference budget 9.5GiB, and outside-pool reserve 1GiB remained. Owned pool, API GPU-labeled values, and CPU/process RSS overlap and cannot be added. Long-term stability is unverified.

![Adopted 256 output](images/moss-dit-kernels-256-20261021.png)

Controls matched previous public PNGs: [control 256](images/moss-turbo4-upload-256-20261020.png) / [control 1024](images/moss-turbo4-upload-1024-20261020.png). Historical [verification](verification.json) is retained.

## Reproduction and change boundaries

Add core --moss-dit-optimizations --moss-dit-kernels and native --moss-dit-optimization-mask 15 --moss-dit-kernel-mask 15 --moss-fa-shmem-no-limit to the [Turbo4 + upload + F32 parent build](build.md). Old optimization mask and new kernel mask are distinct. New bits: FFN=1, Attention=2, QKV/LoRA=4, RMS=8. Occupancy adjustment requires Attention bit 2. Omitted options are OFF.

Verified fresh 75 core objects, 24 initializers/6 runtime providers, and source-freeze/dependency-closure hashes before/after. Measured source bytes/SELF remain private; the public source inventory separately describes the tree after documentation updates. Passed host policy 94, selector 239, parent 39, storage overlap 200,007, F32 recipe 93 checks, and SPIR-V validation. Host checks do not replace hardware inference.

Implementation: [Vulkan](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan.cpp), [RMS shader](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/vulkan-shaders/moss_dit_rms.comp), [LoRA vec 4 shader](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/vulkan-shaders/moss_dit_fused.comp), [GPU checks](../native/app/moss_kernel_gate.inc), [matrix/Attention checks](../native/app/moss_kernel_matrix_gate.inc), [policy](../tools/moss_kernel_policy.py), [host selector tests](../tests/moss_kernel_selection.cpp). Existing GGML Vulkan kernels are selected/integrated; RMS/LoRA use GLSL. This does not execute CUDA/Triton or the ComfyUI runtime. [Source/shader provenance](moss-optimization-provenance.json) is retained.

Retains production admission 0, fixed prompt/seed/256/1024, and maximum 4 attempts including failures. No firmware/system-area/registration/GPU-driver changes or manual release of uncertain GPU resources occurred. Further epilogue integration, arbitrary prompts, long-running operation, and temperature/clock-controlled comparisons remain.
