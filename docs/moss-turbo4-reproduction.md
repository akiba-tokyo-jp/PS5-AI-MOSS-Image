# Reproduce the Turbo4 coefficient adapter

This describes offline **weight-format preparation**, not model inference. The repository includes the two converters below; neither runs a diffusion model, opens the Q4 base checkpoint, trains weights, or needs CUDA/PyTorch. The BF16-to-F32 converter uses NumPy; the composition converter uses only Python's standard library. Use Python3.11 or later on an IEEE binary32 platform. Keep outputs outside the repository and use a filesystem that supports same-volume hard links: publication refuses to replace an existing file.

The resulting adapter is a project-specific derivative, **NSFW0.7 + Turbo1.0, rank144 F32**. It is not the upstream Turbo file or a merged base checkpoint. The original rank16 adapter remains separate. Successful conversion does not prove PS5 memory fit, GPU numerical correctness, image quality or speed.

The separate [PS5 actual-results note](moss-turbo4-results-20261020.md) records the tested256/1024 trial. It does not replace this pinned preparation contract or a full deployment procedure.

## Pinned inputs

Download the original files yourself from their authors. The links fix repository revisions; avoid substituting a later `main` or a similarly named merged checkpoint. Retain the corresponding upstream license/attribution material.

| Input | Revision / size | SHA256 |
|---|---|---|
| [axiomofmind original BF16 LoRA](https://huggingface.co/axiomofmind/Moss-image-nsfw-uncensored-lora/resolve/e0a4cd44a906a4296dca7fa200c76058d1938082/pytorch_lora_weights.safetensors) | `e0a4cd44a906a4296dca7fa200c76058d1938082`,74377208B | `d62ce599744d3e3f07f55a1f539052a99acc38d34bf35c13518aa949fc856afa` |
| [akhaliq frozen rank128 Turbo LoRA](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps/resolve/2ded95c37a029360245926fe72e475e30039bbfb/moss_image_sft_4step_v2.0.safetensors) | `2ded95c37a029360245926fe72e475e30039bbfb`,1189176752B | `154c7f1455c3bf9925945c2f1ea2e4f09d0d0c67256081384a094b2b585f913f` |

At that Turbo revision, `lora.safetensors` has the same selected size/SHA. The frozen `moss_image_sft_4step_v2.0.safetensors` name avoids confusing it with other later or merged files. The selected file has504 F32 matrices,252 scalar alpha values, rank128 and actual alpha128. The converter checks the exact full-file/header identities and every alpha payload.

The selected NSFW adapter's [license](https://huggingface.co/axiomofmind/Moss-image-nsfw-uncensored-lora/blob/e0a4cd44a906a4296dca7fa200c76058d1938082/LICENSE) and the pinned Turbo [model-card declaration](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps/blob/2ded95c37a029360245926fe72e475e30039bbfb/README.md) identify Apache-2.0. Keep applicable notices, identify this modification, and treat each base component's terms separately. The converter source is MIT; that does not relicense model weights. This repository does not redistribute either input or the composed weights. See the [component/modification notice](../licenses/MOSS-Turbo4-NOTICE.txt).

## Convert without a GPU

The following commands run from the repository root. Replace the example paths with your own locations; create a fresh output directory outside the checkout. Do not commit the receipt files: they deliberately record absolute input/output paths and local filesystem identities for a private stability audit.

```sh
python tools/models/convert_moss_lora_bf16_f32.py \
  --source /models/input/pytorch_lora_weights.safetensors \
  --output /models/derived/moss-image-nsfw-uncensored-r16-f32.safetensors \
  --receipt /models/derived/nsfw-conversion.private.json

python tools/models/compose_moss_turbo_lora.py \
  --nsfw /models/derived/moss-image-nsfw-uncensored-r16-f32.safetensors \
  --nsfw-sha256 907c1e0220c8aa301595b4a369f7b09447001306ffebb53da1fc58f24542cdf9 \
  --turbo /models/input/moss_image_sft_4step_v2.0.safetensors \
  --output /models/derived/moss-nsfw07-turbo4-r144-f32.safetensors \
  --receipt /models/derived/composition.private.json
```

Shell continuation syntax differs: use backticks in PowerShell, or put each command on one line. No inference server, PS5 connection, CUDA runtime or GPU is involved in these commands. The conversion reads and verifies the complete source and derived coefficients; reserve disk space for a temporary output as well as the final file. A failed `.partial` file is not a completed artifact and is never silently reused or promoted.

Expected first derivative:148704512B, SHA256`907c1e0220c8aa301595b4a369f7b09447001306ffebb53da1fc58f24542cdf9`.

Expected composed file: **1337791552B**, SHA256 **`126fed9dd5268fe20ab0085525a4b7b7f8ff767a28aea907df2089c33360d4d3`**. Verify the entire output against that value before staging it. A small header check alone is insufficient. The actual file generated during development matched this value; an independent full conversion on a second machine has not yet been performed.

## Fixed composition and runtime settings

For each independent q/k/v/o/FFN projection in42 layers, the converter concatenates the two down matrices by rows and the up matrices by columns, in this order:

```text
D = [D_NSFW_rank16 ; D_Turbo_rank128]
U = [F32(F32(0.7) * U_NSFW_rank16), U_Turbo_rank128]
runtime multiplier = 1.0
```

The final file has504 F32 matrices, rank/alpha144/144. NSFW's0.7 is already baked into its up coefficients; setting the composed runtime multiplier to0.7 would also weaken Turbo. q, k and v keep their own matrices. There is no base merge, Q4 dequantization/requantization or FP16 storage conversion. Baking0.7 changes floating-point rounding relative to scaling a completed matmul, so byte-identical images to the old adapter are not promised.

The dedicated PS5 trial uses **4 Euler steps, CFG1**, and the already shifted author-training-grid sigmas **`[1.0,0.910,0.717,0.024,0.0]`**. It keeps flow-shift5 as model configuration but does **not** apply that shift a second time to these explicit sigmas. CFG1 has no negative-conditioning forward; a negative-prompt field does not provide CFG guidance in this mode. The release model card contains several version labels/aliases, so this recipe is bound to the pinned file and explicit grid rather than the mutable model name. These are the project's exact trial settings, not a claim that every upstream runtime's `simple` scheduler produces the same grid.

The output safetensors header contains only tensor descriptors and fixed provenance/hashes/scales. Paths, timestamps, inode/device values and credentials are not embedded. Fixed tensor order, little-endian F32 encoding and deterministic header serialization make identical output bytes expected from the same verified inputs and source. The accompanying private JSON receipts intentionally differ between hosts and must not be published as portable artifacts.

The rank144 coefficient payload is1337720832B, **1189085184B larger** than the rank16 payload. This disk/storage difference is not measured physical RAM/VRAM demand. Native admission must still retain the original14 numerical gates, the3 rank144 synthetic GPU composition/readback gates, actual full model identity verification, and completed finite GPU witnesses. Actual successful generation requires4 DiT outputs/672 F32 Q4 recordings and loaded/matched/used504, skipped0; host conversion and synthetic gates do not substitute for those observations.
