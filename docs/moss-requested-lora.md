# User-selected MOSS adapter — 2026-10-19

The requested configuration is **MOSS-Image-SFT plus
axiomofmind/Moss-image-nsfw-uncensored-lora**. The URL names an adapter, not a
replacement checkpoint. The author recommends SFT, reports compatibility with
Base, and records Base as its training parent.

Sources: [author model card](https://huggingface.co/axiomofmind/Moss-image-nsfw-uncensored-lora/blob/e0a4cd44a906a4296dca7fa200c76058d1938082/README.md),
[adapter config](https://huggingface.co/axiomofmind/Moss-image-nsfw-uncensored-lora/blob/e0a4cd44a906a4296dca7fa200c76058d1938082/adapter_config.json),
[license](https://huggingface.co/axiomofmind/Moss-image-nsfw-uncensored-lora/blob/e0a4cd44a906a4296dca7fa200c76058d1938082/LICENSE).

## Fixed identity and measured header metadata

| Item | Value |
|---|---|
| Revision | `e0a4cd44a906a4296dca7fa200c76058d1938082` |
| File | `pytorch_lora_weights.safetensors` |
| Complete file size | 74,377,208 bytes / 70.9326 MiB |
| Expected LFS SHA256 | `d62ce599744d3e3f07f55a1f539052a99acc38d34bf35c13518aa949fc856afa` |
| Tensor storage | 504 BF16 tensors, 74,317,824 bytes / 70.875 MiB |
| Exact F32 expansion of the same elements | 148,635,648 bytes / 141.75 MiB |
| Rank / alpha | 16 / 16 |
| Largest BF16 tensor | 524,288 bytes / 0.5 MiB |
| Targets | q, k, v, o, ffn_up_gate, ffn_down in 42 DiT layers |
| Author example | scale 0.7, 30 steps, CFG3, 1024 square |

Anonymous immutable HTTP Range inspection read only the 59,376-byte header
plus its 8-byte length. No full weights or sample images were downloaded. The
whole-file SHA is upstream LFS metadata, not a local full-file verification.

## Memory assessment

The adapter's persistent weight increment is relatively small. F32 weight
storage would be 141.75 MiB, while keeping BF16 would store 70.875 MiB; neither
number is an inference peak. Dtype conversion may briefly retain both, and
runtime LoRA adds intermediate/output tensors, graph scratch and allocator
alignment. A real runtime must measure these.

The selected four base component files plus the LoRA total 9,351,811,137 bytes,
about **8.7096 GiB** on disk. Against the requested12 GiB pool the arithmetic
remainder is about3.2904 GiB. This is not free VRAM or a fit guarantee. The large
Qwen encoder, its CPU memory and recurrent state, DiT/VAE workspace, driver and
native display still determine the real peak. CPU/GPU share console memory.

The existing Direct-capacity issue is unchanged: historical native reports of
12 GiB before AGC and11.4375 GiB during a later GPU context do not meet the
prepared12 GiB pool plus1 GiB outside-pool headroom check. The adapter does not
increase the reported physical quota. No new PS5 allocation was attempted.

## Concrete compatibility work before enabling

The prepared native engine does not yet connect the requested adapter. The
plan in [config/moss-lora.plan.json](../config/moss-lora.plan.json) is metadata,
not runtime configuration. The existing downloader still covers only the four
base components; it must not imply that this LoRA is downloaded or enabled.

1. MOSS has fused `proj.qkv` in
   `patches/stable-diffusion/overlay/src/model/diffusion/moss_image.hpp`, whereas
   this adapter has separate `proj.q`, `proj.k`, `proj.v`. The current name
   conversion and preprocessing lack that MOSS mapping. Simply loading the
   original file risks silently leaving the q/k/v tensors unused. All504 source
   tensors must be accounted for, with shape and alpha/rank checks, and failure
   must be explicit if any requested weight is unused.
2. The loader currently preserves the BF16 LoRA type. It does not automatically
   create an F32 copy. The PS5 Vulkan path needs a verified supported dtype or
   an explicit exact BF16-to-F32 conversion with its derived hash recorded.
3. Q4 base weights cause SD.cpp Auto to choose runtime LoRA: two low-rank matrix
   products and output addition. This avoids expanding the whole DiT to F32.
   Immediate merge instead creates dense F32 weight/delta buffers; the largest
   MOSS feed-forward matrix is192 MiB in F32, with a similar dense delta. The
   graph-wide peak cannot be inferred just from this one matrix.
4. Native configuration/API must explicitly pass the MOSS adapter. WAI's SDXL
   manga LoRA fields cannot be reused as if compatible. The current API remains
   base-only, with compile-time runtime admission0.
5. After console recovery, establish actual capacity, numerical adapter
   application, complete GPU compute and output PNG before declaring success.

No PS5 connection, setting/app/driver/memory change, inference or PNG generation
was performed for this assessment. The repository remains PRIVATE.
