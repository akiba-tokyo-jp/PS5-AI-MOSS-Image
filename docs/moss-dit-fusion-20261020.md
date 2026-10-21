# DiT invariant-computation reuse and F32 Vulkan fusion

2026-10-20 JST.4 changes were implemented and generated 256→1024 on PS5. A control disabling optimizations in the same core measured **51.146425→47.760891 seconds at 1024 (6.62% reduction)**. Both sizes were byte-identical to the control and earlier Turbo4 baseline. Each configuration/size was observed 1 time.

![Hardware 256 output; identical PNG for both configurations](images/moss-turbo4-upload-256-20261020.png)

![Hardware 1024 output; identical PNG for both configurations](images/moss-turbo4-upload-1024-20261020.png)

## Implemented changes

| Target | Change and verification scope |
|---|---|
| DiT text projection, 2 text-refine layers, RoPE | Compute at the start of 1 generation and reuse GPU results for the next 3 forwards. Separate from Qwen conditioning cache. Outputs of 42 layers depending on time/image are not reused |
| SiTU | Fuse the original 9 pointwise graph operations into an F32 Vulkan shader, preserving the activation formula and operation order |
| Norm, modulation, residual, Q/K RoPE | RMS reduction stays unchanged. Fuse weight multiplication + modulation, gated residual, and Q/K weight multiplication + RoPE reads/writes |
| LoRA | Omit SCALE at multiplier 1; fuse ADD into base output and packed output for independent q/k/v. Q4 matrix-product epilogue fusion remains unimplemented |

The cache is limited to 1 generation with the same runner/model/adapter/backend. Shapes and conditioning values must match exactly. It is cleared at new-generation start and after verified GPU completion. No cross-request reuse, approximate denoising cache, skipped steps, or CPU DiT offload is used.

~~~mermaid
flowchart LR
  Q[PS5 CPU Qwen] --> T[Text projection and 2 refine layers]
  T --> C[GPU cache within 1 generation]
  R[RoPE 1 time per 1 generation] --> D[4 DiT forwards]
  C --> D
  D --> F[F32 SiTU, norm, residual, LoRA fusion]
  F --> V[PS5 Vulkan VAE]
  V --> P[PNG and OpenAI-compatible API]
~~~

## Identical-condition measurements

MOSS-Image-SFT Q4_K, Qwen3.5-9B Q4_K_M, FLUX.2 small VAE, combined rank 144 **F32** adapter, NSFW0.7 + Turbo1.0, runtime multiplier 1. 4 Euler steps / CFG 1, seed 20261020, fixed prompt, author sigmas [1.0,0.910,0.717,0.024,0.0], no double shift. FFN chunking, F16 LoRA storage, Spectrum, and general fusion are OFF. All learned inference runs on PS5.

| Metric | Same core, optimizations disabled | All 4 enabled | Difference |
|---|---:|---:|---:|
| First 256, including lazy loading | 71.354991 s | 69.275742 s | −2.079249 s, −2.91% |
| First 1024 after 256 in the same process | 51.146425 s | 47.760891 s | −3.385534 s, −6.62% |
| 1024 DiT graph | 44.557450 s | 41.207966 s | −3.349484 s, −7.52% |
| 1024 other runners graph | 5.985294 s | 5.994667 s | ＋0.009373 s |
| 1024 DiT dispatch-record calls | 12,556 | 7,232 | −5,324 |
| 1024 DiT fence-wait calls | 2,916 | 1,867 | −1,049 |

The first 2 rows are native-job generation times. Graph/driver-call times are CPU wall measurements with nested relationships and cannot be added. Fence waits include GPU execution and are not entirely removable synchronization overhead.1024 was neither standalone cold nor a run 2 at 1024 run. Temperature/clocks/OS disk cache were uncontrolled; means, variance, and individual optimization contributions were not measured.

## Memory

Physical Direct reservation 10GiB, inference budget 9.5GiB, and outside-pool reserve 1GiB were preserved. This speedup did not reduce reservation size.

| 1024 record | Control | All 4 enabled |
|---|---:|---:|
| Owned Direct data high-water mark | 5,934,579,712B | 6,107,971,584B |
| Post-generation owned Direct data | 4,749,066,240B | 4,749,066,240B |
| API GPU-labeled high-water mark | 5,968,183,296B | 6,137,233,408B |
| API CPU/process RSS high-water mark | 11,578,556,416B | 11,540,660,224B |

Owned peak increased 173,391,872B (165.359375MiB). This is a whole-configuration difference including non-in-place fused-output allocation, not cache size alone. Cache was destroyed after GPU completion and final usage returned to the same value. CPU/GPU-labeled and owned Direct values overlap and cannot be summed as physical RAM/dedicated VRAM. These 2 generations do not establish long-term memory stability.

## Numeric and execution verification

- Existing 14 numeric gates, Turbo3, and bounded-upload gates remained.16 added fusion cases compared 125,436 values with original GPU graphs: 0 nonfinite values, 0 mismatches, maximum absolute error 0. Existing tolerances were unchanged.
- 1 Q/K RoPE case actually used different VkBuffers sharing the same GGML virtual address. Overlap is checked using actual VkBuffer identity and byte ranges, not sentinel-address equality alone.
- Each generation verified 4 finite DiT outputs, 672 Q4 F32 records /0 F16 accumulation /0 invalid, 504 LoRA tensors applied /0 skipped, and completed GPU DiT/Flash Attention/VAE.
- Each generation recorded cache miss 1/hit 3, RoPE construction 1, and 1,008 omitted SCALE operations. Actual GPU records include SiTU 170, each norm-modulation 168, residual 336, Q/K RoPE 340, LoRA ADD 504, and q/k/v pack 168. 2-branch packing passed synthetic fixtures but was not selected by this adapter's actual graph.
- Native-GET/OpenAI-compatible PNGs were compared, fully decoded, and visually checked.256 SHA: be9e6ebad5da8ee7a4e536a40e905307a2aa24bcced8e86b600af9fa0aaa2ad7;1024 SHA:9b9f359d8ab249a2b6ece206a6b4a0fe1bd8df0ce8cbe000d39fe8e1d515c31a.
- Host checks covered policy 39, model-shape graph 6,913, and storage-overlap 200,007 cases. They used no model weights, learned inference, or GPU execution.

Both configurations sent 1 POST per request with 0 automatic retries. After normal API shutdown, independent observation confirmed native/gateway absent, same boot, and preserved existing services, registration, and model metadata. No firmware update, system-area change, or manual release of ambiguous GPU resources occurred.

## Build and remaining scope

Add core --moss-dit-optimizations and native --moss-dit-optimization-mask 15 to the Turbo4 + upload + F32 parent in [build instructions](build.md). mask 0 is the same-core control; 1/3/7 are cumulative limited comparisons. Default is OFF. Production admission 0, fixed requests, maximum 4 attempts, actual capacity, finite values, GPU completion, and ownership conditions remain.

Implementation: [model/cache](../patches/stable-diffusion/overlay/src/model/diffusion/moss_image.hpp), [typed ops/storage checks](../patches/stable-diffusion/overlay/src/model/diffusion/moss_dit_ops.hpp), [Vulkan shader](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/vulkan-shaders/moss_dit_fused.comp), [GPU numeric checks](../native/app/moss_dit_gate.inc), and [LoRA](../patches/stable-diffusion/overlay/src/model/adapter/lora.hpp).

Design references are the Apache-2.0 [vLLM-Omni MOSS implementation](https://github.com/vllm-project/vllm-omni/blob/942a5daa6c094615812ed89749ef07da96b8edab/vllm_omni/diffusion/models/mossimage/ops.py) and same-revision transformer. This is GGML/GLSL matching current PS5 F32 operation order, not a direct CUDA/Triton/BF16 port. No KohakuMossTrainer or Nunchaku code was incorporated. See [provenance](moss-optimization-provenance.json).

Individual change timings, arbitrary prompts/shapes, long-running generation, and Q4 GEMM epilogue integration are unverified or unimplemented. PNG equality and 6.62% are not generalized to them.
