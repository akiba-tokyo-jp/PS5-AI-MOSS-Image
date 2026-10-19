# MOSS-Image models and memory plan

This project is prepared on Windows. PS5 model loading, generation, and the
proposed 12 GiB Direct reservation are **not yet verified**.

The requested configuration is **MOSS-Image-SFT plus the user-selected LoRA**.
See [adapter selection and memory assessment](moss-requested-lora.md). The LoRA
is an addition, not a replacement checkpoint, and is not yet connected to the
native runtime. The base candidate is **MOSS-Image-SFT**, the curated-data fine-tune of Base.
This is the project's default for ordinary generation; upstream does not state
that SFT is better for every task. Base remains an optional comparison/fine-tuning
starting point. Both use a 5B DiT, Qwen3.5-9B text encoder, and FLUX.2 VAE.
[Official models](https://github.com/OpenMOSS/MOSS-Image/tree/608158b568fa2e769d918ceccaaf2461f52b3e38)

## Locked files

[config/models.lock.json](../config/models.lock.json) fixes full repository
revisions, exact sizes, and expected SHA256 values. These expected checksums come
from upstream Git-LFS metadata; full model bytes have not been downloaded or
verified by this project yet.

| Component | Selected file | Bytes |
| --- | --- | ---: |
| DiT | `MOSS-Image-SFT-Q4_K.gguf` | 3,327,400,800 |
| Text encoder | `Qwen3.5-9B-Q4_K_M.gguf` | 5,680,522,464 |
| VAE | `flux2-vae_small.safetensors` | 249,521,340 |
| Tokenizer | official MOSS `tokenizer/tokenizer.json` | 19,989,325 |

The small FLUX.2 decoder matches the official MOSS model's decoder family.
Current SD.cpp detects the decoder width from its tensors. The full FLUX.2 VAE
is a recorded alternative, rather than a silent replacement. SDXL VAE/TAESD
cannot replace this VAE. Use the MOSS tokenizer: the standalone Qwen repository's
tokenizer has different bytes, and SD.cpp explicitly requires an external file.
[SD.cpp setup](https://github.com/leejet/stable-diffusion.cpp/blob/b13a48219d5cdf408edcb93da6cd69b612ed03d3/docs/moss_image.md)

## Explicit download or verification

The downloader uses Python 3.11 or later and only its standard library. Its
default location is `D:\PS5-AI\models\moss-image-sft-v1`; model weights and the
large tokenizer stay outside the repository. Neither command performs inference
or contacts the PS5.

```powershell
python tools/download_moss.py --verify
python tools/download_moss.py --download
```

`--verify` checks existing files locally and writes nothing. `--download` is an
explicit multi-GB operation. It verifies existing files, downloads missing files
once from immutable HTTPS URLs, streams SHA256, and publishes a final filename
only after the complete size and digest match. A verified same-volume hardlink
is used for atomic publication without replacing a racing writer; NTFS supports
this. Unsupported filesystems fail with the partial preserved.

A mismatched existing file, interrupted `.part`, or another tool's lock is
preserved. Inspect it before explicitly moving/removing it and running the tool
again. There is no automatic resume, network retry, model substitution, remote
code execution, API token use, or PS5 deployment. Never commit the models.

## 12 GiB is a target, not a measured fit

The four complete files total **8.640 GiB** on disk. Against a proposed 12 GiB
pool, the arithmetic difference is **3.360 GiB**. The former 9,422 MiB pool left
only **0.561 GiB** by the same arithmetic. File size is not runtime memory: the
GGUF includes an output head that feature extraction may not need, while tensor
conversion, graph scratch, activations, tokenizer objects, copies, alignment and
drivers add memory. CPU and GPU also share physical console memory with the OS.

For planning sensitivity only, adding 0.5–2 GiB of live scratch/copies to the file
total gives 9.14–10.64 GiB before other overhead. That range is not a prediction
or a PS5 measurement. Component lifetimes, bounded CPU encoding and actual peaks
must be measured; a larger pool alone does not establish stability.

### Confirmed single-allocation blocker

Bounded HTTP Range inspection of the selected GGUF's actual tensor metadata
found `token_embd.weight`: shape `[4096, 248320]`, **Q4_K**, **572,129,280 bytes
(545.625 MiB)**. This exceeds the old native allocator's 512 MiB single-allocation
limit even though it is small relative to the entire pool. Q4_0 has the same
545.625 MiB embedding, so changing between these two quantizations does not fix
that cap. Q3_K_S has a 416.797 MiB embedding, but is a lower-precision alternative
and is not selected automatically.

The file also contains a 795.703 MiB Q6_K output head; feature-only loading must
avoid unused tensors. Keeping Q4 quality requires a validated CPU embedding
gather/chunked-storage path or a separately reviewed allocation-policy change.
The 12 GiB pool does not remove this per-allocation problem.

The prepared source now addresses the **legacy** cap with 1 GiB Direct regions
and bounds CPU weight claims by the actual owned region size. A 545.625 MiB
embedding can therefore pass the new source-level size bound, subject to a
sufficient contiguous free span. This is not a successful native allocation:
the native cross-build has passed, while 1 GiB native mapping, embedding load
and numerical/readback checks remain unverified. `runtime_admitted=0` prevents explicit AGC/pool setup.

There is also a capacity admission issue separate from the embedding cap.
Earlier records from this console reported 12 GiB before AGC and 11.4375 GiB in
a later GPU context. The prepared allocator requires the 12 GiB pool **plus** a
1 GiB outside-pool reserve to fit the native-reported Direct size. If either
earlier value is reported again, it will reject the request before allocating
the pool. Those historical reports are not current free-memory measurements or
a permanent hardware limit, but 13 GiB of admission has not been established.
The outside-pool reserve is an availability check, not an allocated CPU arena.

## Engine compatibility and defaults

Upstream research revision `b13a48219d5cdf408edcb93da6cd69b612ed03d3` implements
MOSS, Qwen3.5 hybrid attention, GGUF layout handling, and the gated-delta-net
operator. The prepared native source instead retains base
`8cd731e7b4667eb8f40a153315b9a7125d09a58d` and applies a scoped backport from MOSS
support commit `68285df319457e2d79310b48513e4ba1bf8ebe9a`; it is not a wholesale
update to the research HEAD. Every reviewed native core member must be rebuilt
against the actual backported headers before linking. CPU
text-encoder output needs an independent numerical check; new Vulkan operators,
DiT output, and VAE decoding need PS5 validation.

The selected small VAE's actual safetensors header contains 250 F32 tensors and
one 8-byte I64 `bn.num_batches_tracked` training counter, with no BF16 tensors.
The largest decoder tensor is 5,308,416 bytes (`[384,384,3,3]`). The loader must
ignore the training counter rather than submit it as floating-point GPU math;
the header alone does not establish successful decoding.

The source baseline is flow-matching Euler, **50 steps, CFG 3, shift 5**, with
negative-branch DiT layer **22** skipped (zero based). Width and height must be
positive multiples of 16. Official examples use an empty negative prompt;
explicit caller prompts should be retained. A low-step smoke test establishes
integration only. It cannot establish normal image quality or generation speed.
[Official settings](https://github.com/OpenMOSS/MOSS-Image/blob/608158b568fa2e769d918ceccaaf2461f52b3e38/USERGUIDE.md)

Text-to-image does not require a Qwen vision `mmproj`, a chat server, or a 262K
chat context. Prompt enhancement is excluded from the SD.cpp path. Layout JSON
is text input and does not establish raster img2img, inpainting, ControlNet,
IP-Adapter, or compatibility with WAI's SDXL LoRAs.

## Licenses

MOSS weights/tokenizer and Qwen3.5 use Apache-2.0. The **FLUX.2 autoencoder** is
explicitly Apache-2.0; this does not apply the same license to the FLUX.2-dev DiT.
Keep all original license text, applicable notices, attribution and modification
records. See [model notices](../licenses/MOSS-models-NOTICE.txt) and the retained
Apache license files. Native SDK/driver notices inherited from Anima remain
separate obligations.
