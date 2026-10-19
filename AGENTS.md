# PS5 AI MOSS Image development

Keep this repository PRIVATE until an explicit user instruction authorizes public release. The256/50-step quality trial and1024/50-step cold/warm pair produced visually normal GPU PNGs. These measured conditions do not establish production readiness, long-running stability or broad quality coverage.

Work on Windows/D: while the console is unavailable. Do not access or alter the PS5 merely to prepare this project. WAI, Anima and MOSS must never run concurrently on the console. Preserve their separate title IDs, ports and files.

The user requests a MOSS Direct Memory target of 12GiB. This is a requested target, not a measured PS5 allocation. CPU and GPU share the console's physical memory; account for CPU text encoding, GPU scratch, VAE, display, and OS memory. Do not silently reduce the requested pool or claim a successful allocation. Keep model-admission checks until real capacity/readback and per-tensor bounds are verified.

Use D: for new builds, dependencies, downloaded models and diagnostics. Never commit credentials, local logs or model weights. Use existing WAI/Anima success only as provenance, never as evidence that MOSS works. Do not update firmware or write system areas as preparation. No recording is required.

# Historical256 result and cause investigation (2026-10-19)

The user reported a reboot, explicitly approved a separate10GiB small trial and MOSS registration, then requested artifact-cause identification. The completed controlled pair used the same native SELF and identical model/LoRA/prompt/seed/CFG/flow/sampler/driver/pool/reserve. Each fresh process accepted one request only.4steps reproduced the original artifact PNG byte-for-byte in68.525607s;50steps removed the dense colored artifacts in148.133782s including initial model loading. Within this tested condition, the cause was insufficient sampling at4steps. Minimum acceptable steps, other prompts/resolutions, warm timings and continuous operation remain unmeasured. Ordinary native/gateway/client defaults were already50;4was the initial smoke-only contract. See docs/moss-native-small-trial-20261019.md.

Both requests verified PS5 CPU text encoding, Vulkan DiT/VAE completion, selected LoRA504 loaded/matched/used with0skipped, finite witnesses and byte-identical native/OpenAI PNG retrieval. The50-step request checked100DiT outputs,1VAE output and3,854,336values including text with0nonfinite. Six targeted GPU readback gates passed; do not generalize this to all operators or arbitrary models.

The native context reported11.4375GiB total. The requested production12GiB pool plus1GiB outside reserve remains unmet. The approved10GiB trial does not achieve that target. Production admission remains0. The original small-trial profile still allows256/4only; the explicit quality-comparison profile allows256/4or50only, CFG3, Euler, shift5, seed20261019 and configured LoRA0.7, with a single atomic generation claim per process. Failure never resets that claim. No additional generation or new build is implied by documentation updates.

Native432/gateway435 were normally closed after50steps, both PIDs were independently absent, and observer005 verified no GPU title. Root owns any later live-console action. Do not retry POST automatically, change model/driver/pool, generate larger images or publish the repository without the relevant authorization. No recording or raw-log upload is required. Only the authorized normal50-step PNG is included as a small README asset; failed images, credentials, weights and raw diagnostic logs remain outside Git.

# Completed1024 cold/warm benchmark (2026-10-20)

The latest user instruction authorized1024x1024 first generation including loading and one second generation in the same process, with a different anime image from Anima. That exact two-attempt benchmark is now COMPLETE. Do not start additional POSTs from this authorization. Cold generation1242.706350s and same-process warm1177.725365s both completed on PS5; API/native PNGs are byte-identical, and both phase images have the sameSHA256484391514834f4788b5bceb8c9b5d0ba26ce023f0fa861a51b1a3d6bd2b28f27. Root visually accepted both1024 images. See docs/moss-1024-cold-warm-20261020.md.

Native466/gateway469 and the same SELF/request were maintained for both phases:1024x1024,50steps,CFG3,flow shift5,Euler,seed20261020,LoRA0.7,pool10GiB,reserve1GiB. Each phase had onePOST/no retry. The dedicated native counter permits at most two attempts, failures consume claims, and a third is refused. Old256 profiles and production admission0 remain unchanged. The repository stays PRIVATE.

Cold includes first loading in this process, but OS disk cache was not flushed. Warm means the model context was already loaded; it does not imply zero CPU conditioning, weight reread/transfer, workspace preparation or synchronization. Cold-minus-warm is not a pure loading-time measurement. API memory peaks include cumulative process/pool high-water marks, and the GPU-labelled pool also contains CPU text weights; do not add CPU/GPU peaks or label them pure VRAM.

The real DiT plan used up to1,185,459,208B in9chunks, max chunk275,382,272B, all below the1GiB per-claim limit. The earlier fake buft used1GiB blocks and reported754MB/1chunk; this was not a strict real-workspace bound. Native startup uses64MiB suballocation blocks, and all actual runtime region/scratch checks remained active. Real tiled VAE workspace132,251,656B/3chunks and scratchx/y1,572,864B also differ from the synthetic F32 estimate. Do not turn these metadata results into aggregate physical/fragmentation/driver-headroom guarantees.

Both APIs accepted normal shutdown202; both PIDs were independently absent and observer007 verified idle/no GPU title. No further generation, source change, rebuild, recording or public release is implied. Only one approved normal1024 PNG was added; model weights, credentials and raw diagnostic logs remain outside Git.

# Historical pre-reboot instruction (superseded facts preserved)

The following records the earlier waiting/preparation scope. Its statements about no reboot report, unstaged models and unverified LoRA predate the verified experimental result above; they are not the current deployment status.

The user is returning home and will report after rebooting the PS5. Until that explicit human report, do not connect to, poll, launch, deploy, restart or generate on the PS5. A successful ping or port response is not the required human confirmation.

After the report, first establish a fresh boot, recovered services and no prior GPU-title owner. The authorized scope is at most one small-image request per model, one GPU title at a time: Anima512x512, then MOSS256x256 only if all prerequisites pass. No large-image tests, benchmark sweeps, automatic retries or recording. Ambiguous completion/ownership stops the sequence. Read docs/reboot-smoke-scope-20261019.md and the local authoritative D:/PS5-AI/reboot-smoke-20261019/plan.private.json before live work.

Anima's staged mmap=false candidate changes only that field and is not proof of the previous cause. MOSS is currently not ready: model/adapter staging, LoRA connection, runtime admission and the12GiB capacity method are unverified. Do not bypass the guard, silently lower the requested target, or launch into known unmet capacity conditions merely because the requested output is small. Preserve WAI and the other title, system regions and firmware. Keep repositories PRIVATE until real normal operation is verified. Never expose credentials or perform Windows/cloud model inference.
