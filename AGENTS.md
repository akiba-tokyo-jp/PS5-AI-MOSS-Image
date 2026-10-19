# PS5 AI MOSS Image development

Keep this repository PRIVATE until an explicit user instruction authorizes public release. The controlled 256x256/50-step trial produced a visually normal GPU PNG; this single condition does not establish production readiness or broad quality coverage.

Work on Windows/D: while the console is unavailable. Do not access or alter the PS5 merely to prepare this project. WAI, Anima and MOSS must never run concurrently on the console. Preserve their separate title IDs, ports and files.

The user requests a MOSS Direct Memory target of 12GiB. This is a requested target, not a measured PS5 allocation. CPU and GPU share the console's physical memory; account for CPU text encoding, GPU scratch, VAE, display, and OS memory. Do not silently reduce the requested pool or claim a successful allocation. Keep model-admission checks until real capacity/readback and per-tensor bounds are verified.

Use D: for new builds, dependencies, downloaded models and diagnostics. Never commit credentials, local logs or model weights. Use existing WAI/Anima success only as provenance, never as evidence that MOSS works. Do not update firmware or write system areas as preparation. No recording is required.

# Current experimental result and cause investigation (2026-10-19)

The user reported a reboot, explicitly approved a separate10GiB small trial and MOSS registration, then requested artifact-cause identification. The completed controlled pair used the same native SELF and identical model/LoRA/prompt/seed/CFG/flow/sampler/driver/pool/reserve. Each fresh process accepted one request only.4steps reproduced the original artifact PNG byte-for-byte in68.525607s;50steps removed the dense colored artifacts in148.133782s including initial model loading. Within this tested condition, the cause was insufficient sampling at4steps. Minimum acceptable steps, other prompts/resolutions, warm timings and continuous operation remain unmeasured. Ordinary native/gateway/client defaults were already50;4was the initial smoke-only contract. See docs/moss-native-small-trial-20261019.md.

Both requests verified PS5 CPU text encoding, Vulkan DiT/VAE completion, selected LoRA504 loaded/matched/used with0skipped, finite witnesses and byte-identical native/OpenAI PNG retrieval. The50-step request checked100DiT outputs,1VAE output and3,854,336values including text with0nonfinite. Six targeted GPU readback gates passed; do not generalize this to all operators or arbitrary models.

The native context reported11.4375GiB total. The requested production12GiB pool plus1GiB outside reserve remains unmet. The approved10GiB trial does not achieve that target. Production admission remains0. The original small-trial profile still allows256/4only; the explicit quality-comparison profile allows256/4or50only, CFG3, Euler, shift5, seed20261019 and configured LoRA0.7, with a single atomic generation claim per process. Failure never resets that claim. No additional generation or new build is implied by documentation updates.

Native432/gateway435 were normally closed after50steps, both PIDs were independently absent, and observer005 verified no GPU title. Root owns any later live-console action. Do not retry POST automatically, change model/driver/pool, generate larger images or publish the repository without the relevant authorization. No recording or raw-log upload is required. Only the authorized normal50-step PNG is included as a small README asset; failed images, credentials, weights and raw diagnostic logs remain outside Git.

# Historical pre-reboot instruction (superseded facts preserved)

The following records the earlier waiting/preparation scope. Its statements about no reboot report, unstaged models and unverified LoRA predate the verified experimental result above; they are not the current deployment status.

The user is returning home and will report after rebooting the PS5. Until that explicit human report, do not connect to, poll, launch, deploy, restart or generate on the PS5. A successful ping or port response is not the required human confirmation.

After the report, first establish a fresh boot, recovered services and no prior GPU-title owner. The authorized scope is at most one small-image request per model, one GPU title at a time: Anima512x512, then MOSS256x256 only if all prerequisites pass. No large-image tests, benchmark sweeps, automatic retries or recording. Ambiguous completion/ownership stops the sequence. Read docs/reboot-smoke-scope-20261019.md and the local authoritative D:/PS5-AI/reboot-smoke-20261019/plan.private.json before live work.

Anima's staged mmap=false candidate changes only that field and is not proof of the previous cause. MOSS is currently not ready: model/adapter staging, LoRA connection, runtime admission and the12GiB capacity method are unverified. Do not bypass the guard, silently lower the requested target, or launch into known unmet capacity conditions merely because the requested output is small. Preserve WAI and the other title, system regions and firmware. Keep repositories PRIVATE until real normal operation is verified. Never expose credentials or perform Windows/cloud model inference.
