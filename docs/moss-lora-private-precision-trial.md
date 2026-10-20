# LoRA private-pipeline precision experiment

This remains a default-off, bounded-shape experiment. The F16-storage build
has now passed the original14 + Turbo3 + upload gates and its additional eight
readbacks, including fixed subnormal inputs. Actual first256 including lazy load
took68.897951s; first1024 after256 in the same process took51.884730s. Four finite
DiT outputs,672 Q4 F32 recordings,2,016 selected private LoRA recordings and504
applied tensors were confirmed for each image. Compared with the ordinary F32
combined baseline, owned Direct usage fell637.875MiB, but warm1024 was1.55% slower
in this single comparison. This is not whole physical RAM/VRAM saving or equal
quality proof. The separate private-F32-storage learned-image control has not
been run. See [latest measured scope](moss-turbo-memory-options-20261020.md).

The existing combined Turbo4 and bounded-upload implementation remains the
parent. The optional FFN implementation is retained, but this experiment
requires FFN, Spectrum, operating-profile, sampler-trial, Q4-candidate and
operator-profile switches to be off. Direct reservation remains 10 GiB.

`--moss-lora-storage f32` selects the original composed F32 adapter with a
private F32-shared-math pipeline. `--moss-lora-storage f16` selects the separately
rounded F16 adapter with the same private F32 shared staging, arithmetic and
accumulation. Omitting the option leaves the existing pipeline selection and
adapter contract unchanged. The F16 conversion is deliberately lossy.

Both the fresh75 core builder and native builder take this option together
with their existing combined-parent switches. For example, add the following
to the existing FP16/F32-accumulation combined build command:

```text
--moss-turbo-trial --moss-bounded-weight-upload-trial --moss-f32-accumulation
--moss-lora-storage f32
```

The native command also retains `--precision fp16 --experimental-small-trial
--trial-10g --performance-trial-variant 1`. Repeat with `f16` for the candidate.
Each mode requires an independently fresh75 core. Old core carry-forward is
refused. Source-freeze metadata must include the exact fields returned by
`moss_lora_precision_policy.fields(mode, core=True)` in addition to the existing
feature fields, and must retain the fresh75/no-reuse declarations.

The CPU gateway takes `--moss-turbo-trial --moss-lora-storage f32` (or `f16`).
Its upload and Spectrum flags remain zero. The separate transport marker
identifies request/adapter metadata; the gateway performs no model loading or
GPU inference. The normal OpenAI-compatible request remains four Euler steps,
CFG1, the existing prompt and seed, and the explicit author sigma array
`[1.0, 0.910, 0.717, 0.024, 0.0]` without applying flow shift again.

## Converter and selected adapter

`tools/models/convert_moss_lora_f16.py` streams bounded chunks of the pinned
composed F32 file, uses IEEE binary16 round-to-nearest-even, refuses overflow
and nonfinite inputs, and creates a new fixed basename. It never overwrites
the F32 source. It rereads the output payload and full SHA256 and rechecks the
input. This is format conversion, not learned inference.

| Storage | File bytes | SHA256 |
|---|---:|---|
| F32 | 1,337,791,552 | `126fed9dd5268fe20ab0085525a4b7b7f8ff767a28aea907df2089c33360d4d3` |
| F16 | 668,930,824 | `7904da4b4437dd142b37d1d3b459f558eac1556b8f7e5b46941f6fd39fca735a` |

The F16 basename is `moss-nsfw07-turbo4-r144-f16.safetensors`. Its 504 tensors
retain rank144 with NSFW16 first and Turbo128 second. The baked component
strengths remain 0.7 and 1.0, and the global runtime multiplier remains 1.0.
The base Q4 model is not merged or requantized. Conversion reduces coefficient
payload by 668,860,416 bytes (637.875 MiB); this is not a measured process,
Direct-pool, physical-RAM or VRAM peak reduction.

The measured conversion contains 61,089,117 F16 subnormal coefficients and
31,301 coefficients rounded to zero. All outputs are finite and no overflow
occurred. Maximum coefficient absolute error is 7.62939453125e-6. These facts
do not prove acceptable generated images.

The native config parser requires the exact selected basename, whole-file SHA
and global multiplier. The configured SHA is an identity declaration; native
logging does not claim to rehash the file. Deployment must separately verify
the entire staged file and preserve the original F32 file and prior config.

## Admission and evidence

The original14 numerical gates, original3 rank144 gates and existing bounded
upload gate are preserved. Before allocating the new GPU fixture buffers, a
strict backend-admission API checks the physical device, serialized queue
policy, initialized private slots and completed ownership. Unsupported caps
produce a durable bounded refusal code and reject model loading.

The additional gate runs the original three fixed non-dyadic cases and one
separate fixed subnormal/control case in both storage formats: eight readback
events total. It compares all down/up outputs and Q/K/V concatenations against
independent FP64 oracles and the actual private-F32-storage GPU readbacks.
The tolerances stay `abs 0.001 + rel 0.01`. A failed baseline, candidate or
subnormal case is logged and rejects admission; the fixture is not tuned to
force a pass. Checked synchronization and ownership precede readback/release.

Four private pipelines reuse the already archived `matmul_f32_f32_fp32` and
`matmul_f16_f32_fp32` modules. No shader archive or device FP16 flag is changed.
Actual original/patched module hashes, physical required subgroup,
specialization and compiled handles are recorded. Canonical rank144 LoRA
nodes alone can select these pipelines; Q4, VAE and general F16 kernels retain
their existing selection. No extra GPU submission or wait is inserted.

Before image admission, require all eight successful readbacks and the final
`native_moss_lora_precision_gates_complete` receipt. Each image must retain
four completed DiT forwards, 672 Q4/F32-accumulation recordings, finite outputs,
all504 LoRA tensors applied, and exactly2,016 private LoRA matrix recordings
(four per canonical tensor). Pipeline identities must remain unchanged.

Run one256 image before the first1024 image for each selected storage mode,
using the existing one-POST/no-retry client and at most four attempts per
native process. Review the actual generated images; bit-identical PNGs and
quality parity are not required or promised. Compare F16 against the private
F32-storage baseline, because its shared arithmetic also differs from the
ordinary pre-experiment F32-weight route.

Host tests and target syntax checks cover contracts and code integration only.
They do not substitute for these PS5 admission and image checks.
