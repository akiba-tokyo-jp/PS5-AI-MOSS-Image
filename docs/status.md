# Current status

2026-10-22: [matching driver measurement sequence](moss-cumulative-driver-20261022.md)
and [JSON](moss-cumulative-driver-20261022.json) preserve a fresh PS5 launch
with explicit `cumulative-idle` from PS5-Vulkan-Inference at
`872cc1264ad8ba5d6b888b092968fa5858a38c58`. First256 including lazy load:
70.786748s; same-process1024: 44.835497s and 44.840077s, mean44.837787s.
Historical corresponding mean:45.127473s. All three PNGs match their old
references. Persistent caches remain disabled; application/model/precision
settings are unchanged. The sequence is not an alternating A/B or a long-run
qualification. [PS5-Vulkan-Inference is the selected source](driver-integration.md)
for future driver builds. The restored native application and gateway were
subsequently closed normally at the owner's request; their process absence
was independently confirmed. Publishing this documentation does not relaunch
the PS5 application.

Planned next work: [driver-aware optimization and validation](optimization-validation-plan.md)
covers CPU interference, VAE tiles, bounded weight-side LoRA reuse, text
conditioning and real-kernel/compiler tuning at 512 and 1024. It is a plan,
not an implemented or hardware-verified improvement. Application comparisons
must use a frozen driver/profile. The completed driver-only sequence above
does not implement these application candidates.

2026-10-21: verified ordinary MOSS API on PS5 FW 12.40/RADV NAVI21. **Actual arbitrary prompts/seeds, rectangles, first 1024, and generation 5 succeeded.** Removed initial 256/fixed-prompt-seed/4-attempt restrictions. [Details/PNGs](moss-service-20261021.md), [machine-readable data](moss-service-20261021.json).

| Feature | Status |
|---|---|
| OpenAI-style generation/base 64 PNG | Verified on PS5 with public PowerShell client |
| Native asynchronous jobs/image equality | Verified for all 5 jobs |
| First 1024 after startup | 110.459715 s incl. lazy load, no preceding 256 |
| Arbitrary text/omitted negative/different seed/512×768 | Verified actual output |
| Random seed | Different seeds for 2 requests; headers/native jobs matched |
| Img2img | Verified 512 strength 0.5/3 forwards and 256 strength 1/4; earlier 256/1024 also tested |
| Generation count | No software attempt cap; 5 jobs in one launch succeeded; long-running operation unverified |
| PNG/JPEG input up to 2048/dimension; output 64–1024 in 16 increments | Admission policy; not every dimension/input format hardware-tested |
| Default 8GiB storage/507 admission |5 in-quota hardware jobs; host quota rejection verified. No hardware 8GiB fill test |
| Normal shutdown/restart | Independent owner absence, same boot/services, all restart readbacks verified |
| Masks/inpainting/IP-Adapter/ControlNet/arbitrary LoRA | Unsupported |
| Identity locking/strong photorealistic conversion | No guarantee; photorealism unmet |

An additional arbitrary-text 1024 test exposed diagnostic rejection requiring every optimized kernel despite completed GPU computation. Ordinary service now permits shape-dependent existing GPU fallbacks while retaining numeric/capacity/completion checks. Rebuilt with fresh 75 core objects through the service wrapper; verified a new character's1024 PNG on PS5 (124.476684 seconds incl. loading). Different prompts prevent a speed comparison.

Turbo4 keeps Euler 4/CFG 1, author sigmas, NSFW0.7 + Turbo1 F32 rank 144/runtime 1 as the model profile, separate from sample text/attempt caps. Direct 10GiB/budget 9.5GiB and original 14 + Turbo3 + upload + fusion 16 + kernel 20 readback/numeric/capacity/ownership checks remain. All 5 jobs matched finite DiT/GPU completion, 168×actual-forwards Q4 F32 records, and 504 applied LoRA tensors.

Legacy `production_runtime_admitted:false` describes the inherited low-level ABI; ordinary admission uses independent `service_mode:true`. Diagnostic limits/historical comparisons remain unchanged. Use [API](api.md)/[service build](build.md).

DiT kernel baseline: [identical-condition 1024 mean 45.127473 seconds](moss-dit-kernels-20261021.md). This update's cold 1024 took 110.459715 seconds. The 5 differing-prompt/cache jobs are not speedup comparisons or evidence of long-term stability/all-request memory fit/new acceleration. Further optimization remains possible.

Public Git excludes weights, tokens, private controllers, raw logs, and prebuilt binaries. This source release requires registered native Homebrew plus SDK/dependencies. [Deployment/owner/token](deployment.md), [third-party notices](../THIRD_PARTY_NOTICES.md).
