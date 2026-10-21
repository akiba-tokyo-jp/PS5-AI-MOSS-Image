# MOSS speedup using FP16 data paths and FP32 accumulation

> Historical test record. See [Turbo4/memory comparisons](moss-turbo-memory-options-20261020.md) and [verification status](status.md) for the currently adopted candidate and latest times.

2026-10-20 JST. PS5 generated same-condition 256×256/1024×1024 images; Windows retrieved PNGs through the OpenAI-compatible API. Combining FP16 data paths with explicit FP32 accumulation reduced 1024 PS5 generation from 1219.096190 to 1064.248911 seconds. Each configuration was measured 1 time, without repeated temperature/clock/OS-cache control.

## Comparison conditions and results

Preserved MOSS-Image-SFT Q4_K, Qwen3.5-9B Q4_K_M, FLUX.2 small VAE, selected rank 16 LoRA 0.7,50steps, CFG 3, Euler, flow shift 5, seed 20261020, and prompt. Direct pool 10GiB, budget 9.5GiB, outside reserve 1GiB, and RADV unchanged. PS5 CPU text encoding and Vulkan DiT/VAE perform inference; Windows is the client.

| PS5 `generation_seconds` | FP32 baseline | Submission tuning/limited fusion FP32 | FP16 data + FP32 accumulation |
|---|---:|---:|---:|
| First 256 incl. loading | 153.972336 s | 152.631656 s | **145.839851 s** |
| First 1024 after 256, same process | 1219.096190 s | 1214.444805 s | **1064.248911 s** |

Against FP32,256 saved 8.132485 seconds (5.28%);1024 saved 154.847279 seconds (12.70%), at 1024 about 20 min 19 s→17 min 44 s. Against submission tuning/limited fusion, 1024 saved 150.195894 seconds.

This 1024 was neither the first request in an unloaded process nor run 2 at 1024. It was the first 1024 after 256, including size-change CPU conditioning. Conditions differ from the earlier [1024 cold/warm](moss-1024-cold-warm-20261020.md) comparison.

## Bottleneck identified in logs

| 1024 interval | FP32 baseline | FP16 data + FP32 accumulation |
|---|---:|---:|
| DiT graph calls incl. internal waits | 1169.098366 s | 1014.144911 s |
| Other runners graph total | 46.107206 s | 46.183293 s |
| DiT dispatch records | 310300 | 310300 |
| DiT submit / fence calls | 64691 | 64691 |

DiT graph calls occupied about 95.9% of baseline generation; nearly all savings appeared there, not API communication or reduced recording. These CPU wall intervals are nested. Fence waits include GPU execution; they are not wholly removable synchronization overhead, and parent/child intervals must not be summed. Pure GPU timestamps and individual kernel durations were unmeasured.

## Implemented changes

- Explicit FP32 accumulation in MOSS Linear construction/runtime LoRA Linear; other models' default precision unchanged.
- FP16 startup removes `GGML_VK_DISABLE_F16` entirely; actual logs confirm `fp16: 1` and 16bit capabilities.
- Accumulation modes are counted for Q4 matrix pipelines actually recording commands. Every generation confirmed F32 accumulation 16600, F16 accumulation 0, invalid 0. Separate witnesses confirm GPU completion.
- Rebuilt 75 core objects and checked matching native compile definitions. Capacity, scratch, ownership, and numeric gates remain.

The first FP16 candidate exceeded tolerance for 41/153 Q4_K N9 values and was rejected before generation. The adopted version fixes accumulation, without relaxed tolerances. Q4 N9/N304 passed mismatch 0 at existing abs 0.001 + rel 0.01. VAE uses direct convolution; the unused im 2col diagnostic's mismatch 1/7 was not counted as passing. An FP16-enabled log does not mean every operation is FP16.

## Hardware images and memory

Checked 256:103 returned outputs/3,964,928 values; 1024:151 outputs/62,554,112 values; NaN/Inf 0. Each image had LoRA 504 loaded/matched/used, skipped 0, and completed GPU DiT/VAE. Both PNGs were opened and showed normal anime images; native/OpenAI-compatible PNGs were byte-identical. Pixel equality with FP32 is not claimed.

1024 PNG SHA 256: `97ff9462ebd81b772bb313a8d8bbc71212b83b0c4d34b655ba8393b81f7b9aab`,2,496,438 bytes. API CPU peak 11,787,223,040B; GPU-labeled peak 8,403,091,456B. The latter is an upper bound comprising lifetime shared-Direct peaks, including CPU text weights, and driver arena. Do not sum peaks as physical memory/dedicated VRAM. Direct peak 8,369,520,640B, final used 3,559,440,384B, CPU text Direct live 0, no retained faults.

Normal API shutdown plus absent PIDs and independent idle were verified. This experiment retained production admission 0 and maximum 4 attempts. It does not establish 12GiB allocation, long-term stability, other prompts, minimum steps, or repeatability.

![1024 generated on PS5 with mixed precision](images/moss-mixed-1024-20261020.png)

## Findings from another candidate

A candidate forcing FFN upgate to the existing medium 64 tile reached independent GPU gates. Numeric checks had mismatch 0/nonfinite 0, but explicit-selection verification rejected it; image POSTs 0.

Actual non-coopmat NAVI21 excludes large, leaving 2 small/medium candidates, while the added selector assumed 3. Existing selection already chooses medium 64 for this FFN shape; fixing candidate-count detection alone would not change tiles or improve speed. The candidate was rejected without invented timing. Native verified quiet GPU/empty pool, used normal OS shutdown, and independent observation confirmed idle.

Restored the successful mixed-precision SELF above with full readback equality. Restarted native/gateway became ready and rechecked 14 startup gates, 6 MOSS gates, and 4 performance gates at original tolerances. No extra image POST was sent. Normal API shutdown confirmed both PIDs absent and independent no-GPU-title/idle. The improved version was deployed but stopped. Firmware, system areas, and other titles were unchanged.

## Evidence location

Remaining candidates include F32 runtime-LoRA scale/add fusion, MOSS-specific CFG batching, and shape-limited Attention tuning. Logs at this stage do not separate Q4/Attention/LoRA GPU time, so contributions/savings are unknown. Submit API calls themselves take about 1.09 seconds, insufficient alone for a large improvement. Before major kernel work, use verified operation-level timing and small numeric/time comparisons.

Original measurements, source/artifact checks, numeric results, and shutdown evidence remain private. Raw logs, models, credentials, and private diagnostic JSON are excluded from Git.
