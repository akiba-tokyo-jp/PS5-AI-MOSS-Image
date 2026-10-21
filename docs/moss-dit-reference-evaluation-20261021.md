# Additional DiT references observed 2026-10-21

2026-10-21 update: using this evaluation, 4 changes for FFN, Attention, QKV/LoRA, and RMS were implemented and verified at 1024 on PS5. [Measurements and decisions](moss-dit-kernels-20261021.md). The following records the investigation-stage evaluation.

These are source-code findings, not PS5 benchmarks or adopted math changes.

| Reference | Confirmed mechanism | Current MOSS Turbo4 implications |
| --- | --- | --- |
| [Elastic-DiT](https://github.com/wangjiangshan0725/Elastic-DiT/tree/5be46a21b298e712bff7634bc90623e505060387) | Adaptive router skips blocks and selects reduced MLP widths; block feature caching. Released infer.py loads Qwen-Image plus a separate checkpoint, BF16 CUDA,30 steps. | Not a drop-in MOSS/GGML/Vulkan patch. Would require MOSS-specific compatible routing/weights or a separately validated adaptation. Changes computed function; no claimed2x on PS5. |
| [ComfyUI-CacheDiT](https://github.com/Jasonzzt/ComfyUI-CacheDiT/tree/1d92bbd86ec59aa6223fe2368849b7413a1acb93) | Approximate reuse of residuals across differing denoising steps. README verified example Z-Image-Turbo9 steps/warmup3; Qwen50; Klein20. | No published MOSS-specific preset found in inspected utils.py. Four steps leave little reuse after warmup. No inference precision/quality equivalence or1.4–1.6x PS5 claim. Distinct from our exact per-generation text/RoPE cache. |
| [ComfyUI MOSS source](https://github.com/Comfy-Org/ComfyUI/blob/1045d48eb4cc1afd27c419a4765555440f192eab/comfy/ldm/moss_image/model.py) | Fused rms_adaln; joint q/k RMS+RoPE; fused QKV base projection; SiTU can feed activation quantization directly; prefetch and memory compiler. | QKV base projection already packed here, independent q/k/v LoRA preserved. Our RMS reduction is still separate from weight/modulation and RoPE. Reduction+epilogue fusion is a candidate, but arithmetic grouping and aliasing need numerical tests. Prefetch benefit is conditional on actual transfer/idle gaps; resident weights do not imply new useful disk transfers. |
| [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen/tree/a3bac2cc284adfcae41a297932970ad65b878033) | Eager/CUDA/Triton/HIP/Ascend support matrix, including RMS and quantization kernels. | No Vulkan backend in published matrix. Port algorithms to existing PS5 backend where justified, not Python/Torch packages or CUDA binaries. INT8/FP8 changes require matching kernels, model/activation format and quality tests; PC hardware timings are not PS5 estimates. |

The existing4-step authored schedule, F32 accumulation, selectedrank144 adapter and numerical gates stay unchanged during timestamp identification. No external reference implementation is executed on the Windows client or imported into PS5 inference in this measurement pass.


## Spectrum-MOSS reassessment

Rechecked main of [Rkkss/ComfyUI-Spectrum-MOSS](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/tree/3fde0674b29342565e7f42a0674af48d65aabc4a) on 2026-10-21: the revision matched the earlier 50-step investigation. README testing uses 50 steps/Euler + sgm_uniform; approximately 2–2.5× is the author's reported result under their conditions, not a PS5/Turbo4 measurement.

[nodes.py](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/blob/3fde0674b29342565e7f42a0674af48d65aabc4a/nodes.py) stores the final Transformer block's hidden-state history and applies a time-dependent output head to predicted hidden states. This approximately replaces some Transformer calls; it does not directly accelerate the FFN/Attention kernels currently prioritized.

- Default warmup 8/tail 2: all 4 of 4 steps fall within warmup, with 0 forecasts.
- Default Chebyshev degree 4 requires 5 actual-history computations for readiness. Even shorter warmup gives at most 3 past computations at prediction time within a fresh 4-step generation, insufficient for readiness. Changing degree or related settings needs separate accuracy/quality comparison.
- Adaptive detail guards and local token-error p95/p99 horizon adjustment are useful references for future many-step approximate acceleration, not guarantees of prediction accuracy or image quality.
- Hidden-state history differs from the current PS5 latent predictor. Assuming B1×4096 image tokens×3072 and history 8, history payload alone is 384MiB F32 or 192MiB F16 per branch. This is shape-based estimation, not this ComfyUI version's measured PS5 peak.
- Some paths replace NaN/Inf predictions or clamp values, differing from PS5's nonfinite rejection. License: GPL-3.0-only. The full implementation was not imported and verification conditions were not weakened.

The earlier PS5 [50-step latent Spectrum](moss-sampler-spectrum-memory-20261020.md) reduced 1024 from 1023.859406 to 617.938351 seconds (39.65%). It is a distinct SD.cpp latent-boundary implementation with 30 actual CFG steps +20 predictions; it does not prove success/equal quality for ComfyUI's hidden-state version. It is disabled in current Turbo4.

This reassessment compared README/code/existing implementation and configuration requirements only. No PS5 connection, additional generation, or app/driver/memory-setting change occurred. Measured FFN GEMM and Flash Attention remain the priorities.
