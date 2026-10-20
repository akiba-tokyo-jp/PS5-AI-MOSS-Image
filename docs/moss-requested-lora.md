# User-selected MOSS adapter — 2026-10-19

The selected configuration is **MOSS-Image-SFT plus
axiomofmind/Moss-image-nsfw-uncensored-lora**. The URL identifies a LoRA, not a
replacement checkpoint. The author recommends SFT, reports Base compatibility,
and records Base as its training parent.

Sources: [model card](https://huggingface.co/axiomofmind/Moss-image-nsfw-uncensored-lora/blob/e0a4cd44a906a4296dca7fa200c76058d1938082/README.md),
[adapter config](https://huggingface.co/axiomofmind/Moss-image-nsfw-uncensored-lora/blob/e0a4cd44a906a4296dca7fa200c76058d1938082/adapter_config.json),
[license](https://huggingface.co/axiomofmind/Moss-image-nsfw-uncensored-lora/blob/e0a4cd44a906a4296dca7fa200c76058d1938082/LICENSE).

**The selected derived F32 rank16 adapter has now loaded and applied on PS5:
504 loaded/matched/used tensors, skipped0, finite GPU output and normal PNGs
were confirmed in the bounded50-step baseline trials.** This page retains the
2026-10-19 selection/conversion facts. The newer Turbo4 recipe composes this
adapter with a pinned rank128 Turbo LoRA as rank144 F32, and has separate
sampling, capacity and image evidence. See [current status](status.md),
[latest comparisons](moss-turbo-memory-options-20261020.md) and
[exact Turbo composition](moss-turbo4-reproduction.md).
Production admission remains OFF; these results do not establish every prompt,
ordinary continuous operation, equal quality or long-run stability.

## Fixed files and exact conversion

| Item | Verified local value |
|---|---|
| Source revision | `e0a4cd44a906a4296dca7fa200c76058d1938082` |
| Original file | `pytorch_lora_weights.safetensors`, 74,377,208 bytes |
| Full source SHA256 | `d62ce599744d3e3f07f55a1f539052a99acc38d34bf35c13518aa949fc856afa` |
| Source tensors | 504 BF16 tensors / 37,158,912 elements |
| Rank / alpha | 16 / 16; alpha/rank=1 |
| Targets | q, k, v, o, ffn_up_gate, ffn_down in 42 DiT layers |
| Canonical derivative | `moss-image-nsfw-uncensored-r16-f32.safetensors`, 148,704,512 bytes |
| Full derivative SHA256 | `907c1e0220c8aa301595b4a369f7b09447001306ffebb53da1fc58f24542cdf9` |
| F32 tensor payload | 148,635,648 bytes / 141.75 MiB |
| Configured strength | 0.7, runtime application |

The [downloader](../tools/download_moss.py) now covers the four locked base
components plus the raw selected adapter by default. `--base-only` explicitly
omits the adapter and does not prepare the complete selected configuration.
All five source files were downloaded to D and verified by complete SHA256
readback. The [converter](../tools/models/convert_moss_lora_bf16_f32.py) validates
the immutable source identity, all 504 names/shapes, rank/alpha and finite values.
It shifts each finite BF16 bit pattern into exact F32, then rereads **every
converted value** against its source and records the complete derivative hash.
It does not merge or dequantize the full base checkpoint. Raw BF16 is not staged
under the F32 filename.

## Independent q/k/v and native hookup

The selected DiT's actual local GGUF header matches all 504 derivative tensors
against 168 target weights. The base MOSS projection slices q/k/v in that order.
The converter preserves each independent A/B pair and maps it as follows:

| Original module | Canonical fused target |
|---|---|
| q | `proj.qkv.weight.lora_down/up` |
| k | `proj.qkv.weight.1.lora_down/up` |
| v | `proj.qkv.weight.2.lora_down/up` |

The existing SD.cpp runtime adapter computes each low-rank update separately and
concatenates its output slices. It does not reuse q's A matrix for k or v.
Other targets retain their corresponding o/ffn names. Native configuration passes
exactly this required adapter through `sd_lora_t`, with
`LORA_APPLY_AT_RUNTIME`; incompatible manga-extension fields are not reused. See
[configuration and staging](deployment.md).

The MOSS-only adapter contract rejects unexpected names, dtype, rank, shape or
missing model targets. Each generation clears its previous used set, and the
acceptance gate requires **loaded 504 / matched 504 / used 504 / skipped 0**.
Unused tensors fail before decoded output allocation. These are required runtime
values, not claimed PS5 observations. Native generation additionally requires
completed GPU math, finite/readback checks and successful PNG publication.
`/health` exposes `selected_moss_lora` receipt counters and
`application_verified`; configured status alone is not proof of application.

## Host evidence and remaining admission

`tests/moss_lora_checks.py` passes conversion/numeric oracles, all 504 contract
acceptances with 1,013 negative cases, actual pinned upstream name conversion,
strict-adapter syntax and 110 configuration checks with both examples. The
conversion oracles include all 65,280 finite BF16 encodings and independent q/k/v
numeric comparisons. The current PS5-target engine syntax check passes.
Full core 75/75 and the native SELF were rebuilt from the current LoRA/lifecycle
source; [status](status.md) records their identities. These are host checks,
not a learned-model inference test.

The four base runtime files plus the F32 derivative total 9,426,138,441 bytes
(about 8.779 GiB) on disk. The 141.75 MiB persistent F32 tensor payload does not
include LoRA intermediates, graph scratch, allocator alignment, Qwen CPU state,
DiT/VAE workspace, driver or display. Disk size is not a resident-memory peak.

The requested 12 GiB Direct pool plus 1 GiB outside-pool reserve is still a
blocker. Historical native reports of 12 GiB before AGC and 11.4375 GiB with a
later GPU context do not satisfy that check. They are not current free-memory
measurements or a permanent hardware limit. No new allocation or successful
MOSS readback has been established by these host receipts. Runtime guard 0 stays
in place until capacity, Qwen numerical output and actual GPU application/readback
are verified. Constructor side effects remain unknown; the guard concerns the
explicit app initialization path. See [lifecycle and recovery](deployment.md).

Apache-2.0 text is retained and included in the new native package. Model weights,
private evidence, credentials and build artifacts are not committed to Git.
