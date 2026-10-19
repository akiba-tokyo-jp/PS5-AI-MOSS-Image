# PS5 AI MOSS Image development

Keep this repository PRIVATE while PS5 native MOSS inference is failing or unverified. The user wants public release only after normal real-console operation is confirmed. A successful host build is not PS5 generation success.

Work on Windows/D: while the console is unavailable. Do not access or alter the PS5 merely to prepare this project. WAI, Anima and MOSS must never run concurrently on the console. Preserve their separate title IDs, ports and files.

The user requests a MOSS Direct Memory target of 12GiB. This is a requested target, not a measured PS5 allocation. CPU and GPU share the console's physical memory; account for CPU text encoding, GPU scratch, VAE, display, and OS memory. Do not silently reduce the requested pool or claim a successful allocation. Keep model-admission checks until real capacity/readback and per-tensor bounds are verified.

Use D: for new builds, dependencies, downloaded models and diagnostics. Never commit credentials, local logs or model weights. Use existing WAI/Anima success only as provenance, never as evidence that MOSS works. Do not update firmware or write system areas as preparation. No recording is required.

# Current live-test instruction (2026-10-19)

The user is returning home and will report after rebooting the PS5. Until that explicit human report, do not connect to, poll, launch, deploy, restart or generate on the PS5. A successful ping or port response is not the required human confirmation.

After the report, first establish a fresh boot, recovered services and no prior GPU-title owner. The authorized scope is at most one small-image request per model, one GPU title at a time: Anima512x512, then MOSS256x256 only if all prerequisites pass. No large-image tests, benchmark sweeps, automatic retries or recording. Ambiguous completion/ownership stops the sequence. Read docs/reboot-smoke-scope-20261019.md and the local authoritative D:/PS5-AI/reboot-smoke-20261019/plan.private.json before live work.

Anima's staged mmap=false candidate changes only that field and is not proof of the previous cause. MOSS is currently not ready: model/adapter staging, LoRA connection, runtime admission and the12GiB capacity method are unverified. Do not bypass the guard, silently lower the requested target, or launch into known unmet capacity conditions merely because the requested output is small. Preserve WAI and the other title, system regions and firmware. Keep repositories PRIVATE until real normal operation is verified. Never expose credentials or perform Windows/cloud model inference.
