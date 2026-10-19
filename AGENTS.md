# PS5 AI MOSS Image development

Keep this repository PRIVATE while PS5 native MOSS inference is failing or unverified. The user wants public release only after normal real-console operation is confirmed. A successful host build is not PS5 generation success.

Work on Windows/D: while the console is unavailable. Do not access or alter the PS5 merely to prepare this project. WAI, Anima and MOSS must never run concurrently on the console. Preserve their separate title IDs, ports and files.

The user requests a MOSS Direct Memory target of 12GiB. This is a requested target, not a measured PS5 allocation. CPU and GPU share the console's physical memory; account for CPU text encoding, GPU scratch, VAE, display, and OS memory. Do not silently reduce the requested pool or claim a successful allocation. Keep model-admission checks until real capacity/readback and per-tensor bounds are verified.

Use D: for new builds, dependencies, downloaded models and diagnostics. Never commit credentials, local logs or model weights. Use existing WAI/Anima success only as provenance, never as evidence that MOSS works. Do not update firmware or write system areas as preparation. No recording is required.
