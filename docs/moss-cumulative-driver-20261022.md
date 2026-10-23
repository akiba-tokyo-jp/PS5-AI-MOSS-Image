# MOSS cumulative-idle driver: matching hardware measurement sequence

2026-10-22 JST. A fresh PS5 native launch completed the same sequence as the
[previous adopted kernel baseline](moss-dit-kernels-20261021.md): first 256,
then 1024 twice in the same process. All three requests succeeded. The 1024
mean was **44.837787 seconds**, compared with the historical
**45.127473 seconds**. This is one new sequence, not a controlled
alternating A/B; the difference does not establish a reproducible speedup.

![New-driver PS5 1024 output](images/moss-cumulative-driver-1024-20261022.png)

## Identical-condition timing

Values are PS5 `generation_seconds`, in seconds. The historical observations
are copied without changing their original record. The earlier 2026-10-22
two-request measurement is retained separately and is not pooled into these
three fresh observations.

| Condition | Previous recorded driver | New cumulative-idle driver | New minus previous |
|---|---:|---:|---:|
| First 256, including lazy model load | 69.449735 | 70.786748 | +1.337013 |
| First 1024 after 256 | 45.415880 | 44.835497 | -0.580383 |
| Second 1024, same process | 44.839066 | 44.840077 | +0.001011 |
| Mean of two 1024 requests | 45.127473 | 44.837787 | -0.289686 |

First 256 includes this process's initial lazy model loading, but excludes
application launch and startup numerical checks. The OS file cache was not
flushed; this is not a reboot/storage-cold measurement. Both 1024 requests
started with loaded models and used the same cached conditioning. A new
prompt can require fresh CPU text encoding and is outside this warm comparison.
The 256-first order is a comparison fixture, not an API prerequisite.

Each request used one OpenAI-compatible generation POST with zero automatic
retries. Windows downloaded and fully decoded the real PS5 output; no learned
inference ran on Windows. Temperature, clocks and power were not controlled.

## Driver integration and retained recipe

The driver source is [PS5-Vulkan-Inference](https://github.com/akiba-tokyo-jp/PS5-Vulkan-Inference)
at `872cc1264ad8ba5d6b888b092968fa5858a38c58`, explicitly selecting `cumulative-idle`. This
combines the completion-snapshot correctness prerequisite, H1 half selection
before F32 widening, H3 conservative late LICM, and cached idle completion.
Mesa remains `1ba9874f478657ccf6da72ce55f0260164cc0bd0`.

The previously verified relinked SELF was reused unchanged. Exact original
SELF reproduction had passed. Only `radv_shader.c.o` and
`winsys_ps5_radv_ps5_winsys.c.o` differ: all 702 other archive members retain
their identities, including application-specific memory/platform members.
Application objects, shaders, model, precision and settings were unchanged;
24 initializer slots and six runtime providers were preserved. Persistent
shader/pipeline caches remained **disabled**, before backend initialization.

Recipe: MOSS-Image-SFT Q4_K; CPU Qwen3.5-9B Q4_K_M; Vulkan DiT/small VAE;
F16 matrix scratch with F32 accumulation; NSFW0.7 + Turbo1.0 rank144 F32
adapter, runtime multiplier1; Euler4/CFG1/seed20261020; authored sigmas
[1,.910,.717,.024,0] without double shift. The same historical prompt was
used throughout. DiT optimization mask15, kernel mask15 and shared-KV
occupancy adjustment remain. Direct pool10GiB, inference budget9.5GiB,
VAE tile256/overlap0.5. No proposed P1-P5 application changes were included.

## Phase and memory observations

| Request | Windows API wall | Conditioning | Sampling | VAE decode | DiT graph CPU wall |
|---|---:|---:|---:|---:|---:|
| First 256 | 71.547000 | 38.09 | 26.16 | 0.78 | 26.064442 |
| First 1024 | 46.188000 | 0.00 | 38.54 | 5.91 | 38.458580 |
| Second 1024 | 46.062000 | 0.00 | 38.54 | 5.91 | 38.459726 |

These are existing wall timers, not exclusive GPU timestamps. Scopes are
nested and must not be added as an exclusive partition. Fence wait includes
the GPU work being awaited and is not entirely removable synchronization cost.

| Request | Owned Direct peak GiB | Owned Direct end GiB | API CPU-labelled peak bytes | API GPU-labelled peak bytes |
|---|---:|---:|---:|---:|
| First 256 | 4.618362 | 4.495178 | 11,593,252,864 | 4,990,287,872 |
| First 1024 | 5.782242 | 4.516663 | 11,626,807,296 | 6,240,026,624 |
| Second 1024 | 5.782242 | 4.516663 | 11,626,840,064 | 6,240,059,392 |

The physical pool is 10GiB. API labels overlap shared memory; adding them
does not measure physical consumption or discrete GPU VRAM. CPU text Direct
claims returned to zero after every request. All three requests used the
same identified native owner.

## GPU, image and lifecycle verification

- Startup retained the original14 gates, Turbo3/upload, 16 fusion and 20
  kernel GPU readbacks, with zero mismatches or nonfinite values.
- Every image had four finite DiT outputs, 672 Q4 F32 accumulation records,
  zero F16/invalid accumulation records, and 504 applied adapter tensors with
  zero skipped tensors. GPU completion passed and no fault retention occurred.
- Every PNG fully decoded at the requested dimensions. OpenAI gateway and
  native image bytes matched. All three matched their previous reference PNG;
  the two new 1024 PNGs also matched each other byte-for-byte. This applies
  to these fixed requests, not all prompts or model-quality equivalence.
- The candidate shut down normally. The saved original SELF was restored and
  fully read back, relaunched and passed its numerical admission. Native and
  gateway services were ready/idle with no queued/running jobs. Models were
  unloaded in the restored process. The tested driver was not left active
  as the default runtime; its repository is the selected source for future
  builds. Firmware, registrations, system regions and settings were unchanged.

After the measurement/restoration sequence, the restored native application
and gateway were closed normally at the owner's request. Independent process
inspection confirmed both owners and all native `eboot.bin` owners absent.
The earlier ready/idle restoration result above is a historical checkpoint,
not a claim that the application is currently running. No additional image
request was made for this shutdown or documentation publication.

![New-driver PS5 256 output](images/moss-cumulative-driver-256-20261022.png)

## Saved record and interpretation

[Machine-readable measurements](moss-cumulative-driver-20261022.json) preserve timings, stage scopes,
memory bytes, PNG hashes, driver/source pins and verification outcomes in a
public-safe form. Full private startup/generation, transfer/readback and
restoration receipts are retained on the development drive. No credentials,
raw prompts/logs or runtime binaries are included in this report.

The mean difference is -0.289686 seconds
(+0.642% time reduction by arithmetic).
The sequence has two warm observations per driver and historical controls;
neither a causal gain nor regression is established. Driver microbenchmark
percentages are not whole-image acceleration percentages. Further MOSS
application optimizations and long-running validation remain separate work.

See [driver source and integration policy](driver-integration.md) for future
builds. This measured revision is fixed; newer driver profiles or runtime ABI
changes are not implicitly qualified by these results.
