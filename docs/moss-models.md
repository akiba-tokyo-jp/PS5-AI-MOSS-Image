# MOSS-Image models and memory plan

PS5 model loading, CPU text encoding, Vulkan DiT/VAE generation and the selected
derived F32 LoRA have now been confirmed in bounded256/1024 image trials.
The latest measured recipe uses10GiB Direct, Turbo4 plus bounded upload and a
composed rank144 F32 adapter. The requested12GiB reservation remains unmet;
ordinary service mode now removes fixed prompt/seed, prior256 and attempt caps.
Five same-process requests including cold1024 and img2img were verified;
long-run stability and all-prompt memory fit remain unproven.
See [current status](status.md) and [ordinary service results](moss-service-20261021.md).

The requested configuration is **MOSS-Image-SFT plus the user-selected LoRA**.
See [adapter selection and memory assessment](moss-requested-lora.md). The LoRA
is an addition, not a replacement checkpoint. Its exact derived F32 runtime adapter was applied on PS5 with504 loaded,
matched and used tensors, skipped0, in the measured rank16 baseline.
The later Turbo4 recipe composes it with the pinned Turbo adapter as rank144;
these are distinct configurations. The base candidate is **MOSS-Image-SFT**, the curated-data fine-tune of Base.
This is the project's default for ordinary generation; upstream does not state
that SFT is better for every task. Base remains an optional comparison/fine-tuning
starting point. Both use a 5B DiT, Qwen3.5-9B text encoder, and FLUX.2 VAE.
[Official models](https://github.com/OpenMOSS/MOSS-Image/tree/608158b568fa2e769d918ceccaaf2461f52b3e38)

## Locked files

[config/models.lock.json](../config/models.lock.json) fixes full repository
revisions, exact sizes, and SHA256 values. All four selected base files and the
raw selected LoRA have now been downloaded to D, with complete final-file SHA256
readback matching the immutable upstream identities. The exact F32 LoRA derivative
is also fully verified. None of these host file checks is a PS5 load or memory-fit
measurement.

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
explicit `--output` location is required outside the checkout; model weights and the
large tokenizer stay outside the repository. The default selection includes the
raw required LoRA; `--base-only` deliberately omits it and is incomplete for the
selected runtime. Neither command performs inference
or contacts the PS5.

```powershell
python tools/download_moss.py --verify --output D:/models/moss-image-sft
python tools/download_moss.py --download --output D:/models/moss-image-sft
```

`--verify` checks existing files locally and writes nothing. `--download` is an
explicit multi-GB operation. It verifies existing files, downloads missing files
once from immutable HTTPS URLs, streams SHA256, and publishes a final filename
only after the complete size and digest match. A verified same-volume hardlink
is used for atomic publication without replacing a racing writer; NTFS supports
this. Unsupported filesystems fail with the partial preserved.

The BF16 source is converted separately with the fixed
[exact converter](../tools/models/convert_moss_lora_bf16_f32.py); only its canonical
F32 derivative is staged. The runtime files (base4 plus derivative) total
9,426,138,441 bytes, about 8.779 GiB. The raw adapter remains a source artifact on D.

A mismatched existing file, interrupted `.part`, or another tool's lock is
preserved. Inspect it before explicitly moving/removing it and running the tool
again. There is no automatic resume, network retry, model substitution, remote
code execution, API token use, or PS5 deployment. Never commit the models.

## 12 GiB is a target, not a measured fit

The four base files total **8.640 GiB** on disk. Adding the canonical F32 LoRA
raises runtime file storage to about 8.779 GiB; this does not measure resident
memory or establish available VRAM. The
GGUF includes an output head that feature extraction may not need, while tensor
conversion, graph scratch, activations, tokenizer objects, copies, alignment and
drivers add memory. CPU and GPU also share physical console memory with the OS.

Component lifetimes, CPU encoding, LoRA intermediates and actual peaks have
been observed for the named bounded trials. They do not establish every prompt
or size, long-run stability or a physical-memory total. A larger pool or a
disk-size subtraction is not such evidence.

### Historical single-allocation blocker and tested correction

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
and bounds CPU weight claims by the actual owned region size. The545.625MiB embedding passed actual owned CPU parameter claims in the
bounded10GiB configuration, and the model produced the documented images.
Individual claims still require sufficient contiguous free span and checked
ownership. This is not evidence that12GiB fits, that a low-total-free allocator
always succeeds, or that every CPU/model operation is numerically proven.
Default production runtime_admitted=0 remains separate from explicit trial admission.

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
against the actual backported headers before linking. The tested builds passed their fixed GPU numerical/readback gates and
completed finite DiT outputs and VAE PNGs on PS5. This does not substitute for
independent full-model numerical equivalence or validation of other operators
and conditions.

The selected small VAE's actual safetensors header contains 250 F32 tensors and
one 8-byte I64 `bn.num_batches_tracked` training counter, with no BF16 tensors.
The largest decoder tensor is 5,308,416 bytes (`[384,384,3,3]`). The loader must
ignore the training counter rather than submit it as floating-point GPU math;
the header alone does not establish successful decoding.

The historical rank16 baseline is flow-matching Euler, **50 steps, CFG3, shift5**, with
negative-branch DiT layer **22** skipped (zero based). Width and height must be
positive multiples of 16. Official examples use an empty negative prompt;
explicit caller prompts should be retained. A low-step smoke test establishes
integration only. It cannot establish normal image quality or generation speed.
[Official settings](https://github.com/OpenMOSS/MOSS-Image/blob/608158b568fa2e769d918ceccaaf2461f52b3e38/USERGUIDE.md)

Text-to-image does not require a Qwen vision `mmproj`, a chat server, or a 262K
chat context. Prompt enhancement is excluded from the SD.cpp path. Layout JSON
is text input and does not establish raster img2img, inpainting, ControlNet,
IP-Adapter, or compatibility with unrelated SDXL LoRAs.

The explicit [img2img trial](moss-img2img-20261021.md) now uses the existing
small VAE encoder as well as its decoder. This does not add a vision-language
encoder, IP-Adapter or SDXL conditioning compatibility.

## Licenses

MOSS weights/tokenizer and Qwen3.5 use Apache-2.0. The **FLUX.2 autoencoder** is
explicitly Apache-2.0; this does not apply the same license to the FLUX.2-dev DiT.
Keep all original license text, applicable notices, attribution and modification
records. See [model notices](../licenses/MOSS-models-NOTICE.txt) and the retained
Apache license files. Native SDK/driver notices remain separate obligations.
