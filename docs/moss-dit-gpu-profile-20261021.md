# Profiling current Turbo4 DiT time on PS5

2026-10-21 JST. **At 1024, FFN Q4 matrix products accounted for about 40.15%, Flash Attention 25.41%, and QKV 15.01%.** Next priorities are FFN matrix products and Attention. ComfyUI RMS fusion is relevant, but RMS alone accounted for about 1.64%.

Percentages below are relative to **summed GPU dispatch intervals of 2 complete DiT forwards** sampled within the current 4-step generation, not an exclusive breakdown of the full 47-second generation. Each size used 1 generation. After excluding the first forward's witness-split 3-part view, forwards 2 and 3 were measured. All 4 forwards were fully computed.

## Hardware GPU breakdown

| Operation | 256:2 forwards total | Share of measured GPU intervals | 1024:2 forwards total | Share of measured GPU intervals |
|---|---:|---:|---:|---:|
| FFN up/gate Q4 matrix products | 0.476006 s | 32.83% | 5.236539 s | 26.41% |
| FFN down Q4 matrix products | 0.264833 s | 18.27% | 2.725899 s | 13.75% |
| Flash Attention | 0.076056 s | 5.25% | 5.039632 s | 25.41% |
| QKV Q4 matrix products | 0.273850 s | 18.89% | 2.975969 s | 15.01% |
| Attention output O Q4 matrix products | 0.097055 s | 6.69% | 0.995355 s | 5.02% |
| LoRA down/up matrix products and associated work | 0.132614 s | 9.15% | 1.263481 s | 6.37% |
| Contiguous conversion | 0.029168 s | 2.01% | 0.379149 s | 1.91% |
| RMS normalization | 0.027580 s | 1.90% | 0.324258 s | 1.64% |
| Other operations | 0.072630 s | 5.01% | 0.891265 s | 4.49% |
| **Measured GPU intervals total** | **1.449793 s** | **100%** | **19.831548 s** | **100%** |

Parent-graph classifications include conversion/reduction dispatches issued by that operation. Classifying the same timestamps by actual shader type gave 1024 matrix products 13.200663 seconds (66.56%), Attention 5.039632 seconds (25.41%), fused shaders 0.844777 seconds (4.26%), conversion/copy 0.415179 seconds (2.09%), and RMS 0.324258 seconds (1.64%). Parent-operation and shader classifications describe the same time and cannot be added.

Attention's share rose from about 5% at 256 to about 25% at 1024. Large-image priorities therefore target 1024 matrix products/Attention, not 256 rankings alone. Full values/counts are in the [aggregate JSON](moss-dit-gpu-profile-20261021.json).

## Full-generation, CPU wall, and numeric checks

| Record | 256 | First 1024 after 256 in the same process |
|---|---:|---:|
| Native generation_seconds | 69.118352 s, including lazy load | 47.777175 s |
| Previous uninstrumented version | 69.275742 s | 47.760891 s |
| CPU wall for 2 sampled forwards | 1.554019 s | 20.108488 s |
| Valid GPU timestamp intervals | 4,088 | 3,582 |
| Tokens per complete forward | 362 | 4,202 |
| Graph nodes per complete forward | 2,390 | 2,390 |
| owned Direct data peak | 4,858,265,600B | 6,107,971,584B |
| Post-generation owned Direct data | 4,725,997,568B | 4,749,066,240B |

This instrumentation build is not a new speed improvement. It is not a repeated comparison controlling temperature/clocks/cache, and timestamp insertion can affect scheduling. The speed baseline remains the [uninstrumented comparison](moss-dit-fusion-20261020.md).

Existing 1024 CPU wall timing recorded DiT graph_total 41.208481 seconds, other runners graph_total 6.006801 seconds, and DiT submit-driver calls 0.033984 seconds total. Fence wait 40.313510 seconds includes GPU computation and is not entirely removable CPU synchronization cost. The difference between sampled CPU wall and GPU interval sums is not pure synchronization time either.

- All learned inference remains on PS5: MOSS Q4_K, Qwen3.5-9B Q4_K_M, small VAE, combined rank 144 **F32** adapter, NSFW0.7 + Turbo1.0, runtime multiplier 1. 4 Euler steps / CFG 1 / seed 20261020, dedicated sigmas, no double shift.
- Complete forwards were selected after actual-graph verification of 42 layers, 42 of each main Q4 node type, and LoRA down 252/up 252. Each generation had 4 finite DiT outputs, 672 Q4 F32 records, 0 F16 accumulation, 504 LoRA applications /0 skipped.
- Existing 14 numeric gates, Turbo3, upload gate, and 16 fusion cases remained. The added timestamp startup gate also passed on actual PS5 GPU. Availability, ordering, range, overflow, and completed fences are checked; retrieval failure is not valid timing.
- Actual queue timestampValidBits 64 and period 10ns were queried. Added CPU waits 0/submissions 0. Query-result caches are processed/read only after existing GPU completion.
- Query pool 131,072B; fixed CPU profiling area 293,592B. Physical Direct 10GiB, inference budget 9.5GiB, and outside-pool reserve 1GiB are unchanged. API CPU/GPU-labeled values overlap and cannot be added.
- Both requests used 1 POST/0 automatic retries. Native/OpenAI-compatible PNGs matched, fully decoded, and were visually checked. **Both sizes were byte-identical to the uninstrumented version.**

![Hardware 256, byte-identical to the uninstrumented version](images/moss-turbo4-upload-256-20261020.png)

![Hardware 1024, byte-identical to the uninstrumented version](images/moss-turbo4-upload-1024-20261020.png)

After measurement, source classification added actual pipeline names split_k_reduce/fa_split_k_reduce to reduction labels. This label-only correction passed 1,638 host-metadata checks and 26 configuration-rejection cases. It was not remeasured on PS5; tables/JSON retain the original other classification. Other shaders accounted for 0.00425% at 1024 and 0.13661% at 256, including split-K reductions.

## Next optimization targets

1. **FFN matrix products**: up/gate + down account for about 40%. Examine current Q4_K tiles, workgroups, load reuse, and epilogues while preserving F32 accumulation and independent rank 144 adapter behavior.
2. **Flash Attention**: about 25% at 1024. Prioritize tiles, shared memory, and K/V layout for actual head 128/tokens 4,202 shapes. Lower-precision Attention requires separate numeric/quality testing.
3. **QKV/LoRA GEMM surroundings**: QKV about 15%, LoRA about 6%. Base QKV concatenation already exists. Examine redundant reads/writes and dispatches while preserving independent q/k/v LoRA.
4. **Further RMS/conversion fusion**: use ComfyUI as a reference, with lower priority given the small measured share. Changing RMS reduction order does not automatically preserve identical PNGs.

This investigation adopted none of these new arithmetic changes or approximate caches. Time savings are unmeasured. See [ComfyUI, E-DiT, and CacheDiT evaluation](moss-dit-reference-evaluation-20261021.md).

## Build for remeasurement

Add core --moss-dit-optimizations and native --moss-dit-optimization-mask 15 plus **--moss-dit-gpu-profile** to the [Turbo4 + upload + F32 build](build.md). Combining with the old 50-step/rank 16 --moss-gpu-profile is rejected. Retain default-OFF, maximum 4 attempts, production admission 0, and existing capacity/ownership/numeric conditions.

Implementation: [profiling header](../native/app/moss_gpu_profile.h), [dedicated policy](../native/app/moss_dit_gpu_profile_policy.h), [Vulkan recording](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan.cpp), [host classification tests](../tests/moss_dit_gpu_profile.py). Build-policy/actual-recipe AST 286 checks, prior-profile 59,801/59,796 checks, 15 rejection cases, and 17 source-ordering cases passed. Host tests do not replace GPU execution or learned inference.

## State after measurement

After both generations, normal API shutdown closed native/gateway. Independent observation confirmed absent processes, same boot, and preserved services/title registration/model metadata. The verified uninstrumented mask 15 app was restored with full-byte comparison and all numeric gates rechecked. Native 8192/OpenAI-compatible 8193 returned idle, with no jobs and the model unloaded. Total generation POSTs: 2; none after restoration. Firmware, system areas, and Direct reservation were unchanged.
