# MOSS Image deployment, startup, and shutdown

Ordinary service uses native title `PPSA99190`, native API 8192, and CPU gateway 8193. A native Homebrew launch environment and title registration are required. Verified on FW 12.40. No general-purpose title-registration/jailbreak installer is included. Sending a SELF to WebSrv's ELF loader does not launch a native title.

## File placement

Place title files at `/data/ai/native-moss-image-sft/PPSA99190`, using existing native-launcher mounts/registration. Do not overwrite system areas or other titles.

```text
PPSA99190/
  eboot.bin
  sce_sys/param.json
  sce_sys/icon0.png
  sce_module/libc.prx
  models/
    MOSS-Image-SFT-Q4_K.gguf
    Qwen3.5-9B-Q4_K_M.gguf
    flux2-vae_small.safetensors
    tokenizer_qwen3.5.json
    moss-nsfw07-turbo4-r144-f32.safetensors
  licenses/
  ai-runtime/
    config.json
    config/
    logs/
    jobs/
    output/
```

Deploy the [service configuration](../config/moss-service.json.example) as `ai-runtime/config.json`. Follow the [model lock](../config/models.lock.json) and [adapter composition](moss-turbo4-reproduction.md); verify full sizes/SHAs before and after transfer. Check runtime-library redistribution terms separately.

Match the compiled layout: physical Direct 10GiB, inner budget 9.5GiB, outside-pool reserve 1GiB, F32 rank 144 adapter/runtime 1. Do not reuse old 12GiB/rank 16 settings. Keep executables/configuration in a private backup and verify complete transferred files before switching.

## Owner verification and token handoff

Native starts as a UID 1 title. Set `ai-runtime` and its config/logs/jobs/output directories to actual UID 1 ownership and 0700; config.json to 0600. Avoid symlinks and verify ownership/real paths using the existing launcher/management shell. If native compatibility stat cannot report ownership, do not interpret a reported UID 0 as the actual owner. Retain bootstrap that verifies UID 1 through trusted external observation.

1. Generate a sufficiently random token in a private Windows file, without exposing it on screen, in logs, URLs, or arguments. The client reads this file too. Native and gateway use the same value.
2. Confirm other GPU titles have exited and the current-boot owner/existing services are healthy. Launch the title 1 time through the registered native launcher **with no native `ai-runtime/config/api-token` file present**.
3. Externally verify that the newly created PID/timestamped bootstrap journal and owner witness belong to this UID 1/PPSA99190 process, along with runtime ownership/mode and the witness's full byte identity. Token wait lasts at most 5 minutes; GPU/API work has not started. Do not skip checks based on an old journal or PID.
4. For that verified process, hand off a temporary file containing the same token at `ai-runtime/config/api-token` with UID 1/0600. Native consumes/deletes it and retains it in memory. Placing the token before launch violates bootstrap conditions. Do not repeatedly supply it to a timed-out process.
5. Verify startup GPU readback, actual pool allocation, and engine readiness. Initial model lazy-loading occurs on the image request.

This requires process/UID/startup/mount management in the existing Homebrew environment. Do not reuse personal PIDs/nonces or a root controller as a public installer. If the environment cannot verify ownership, provide a suitable provisioner rather than removing the owner guard.

## CPU gateway

Prepare owner-only config/logs/jobs/output directories at `/data/ai/moss-openai-gateway`; place the same token in `config/api-token` with gateway-owner/0600 permissions. Gateway reads/deletes it at startup. It is a CPU ELF, not a GPU title, so existing WebSrv `/elfldr` can launch this ELF 1 time.

```text
<gateway.elf> --root /data/ai/moss-openai-gateway --bind auto --port 8193
```

Bind defaults to auto. If private IPv4 detection fails, startup exits; specify `--bind <PS5_PRIVATE_IP>`. Native and gateway must use the same LAN address. Gateway forwards only to that address's same-PS5 native 8192 and never retries generation POSTs. Existing WebSrv 8080 is preserved.

Check both authenticated `GET /health` endpoints for `service_mode:true`, `requires_256_first:false`, `max_generation_attempts:null`, ready state, and empty queues. The [client](../scripts/Invoke-MossOpenAI.ps1) can request 1024 directly on the first run. No fixed prompt or preceding 256 operation is required.

## Storage and operations

Each process has a default 8GiB allowance for output + jobs, configurable at startup with `--storage-limit-mib`. Accounting reserves 32MiB per unfinished/next job and returns 507 before generation if insufficient. This neither measures nor guarantees physical free space. Gateway also stores PNG copies/metadata; manage both roots. Input storage and [bounded logs](logging.md) are separate.

PNGs are not deleted automatically. Near the allowance, stop new client requests, verify completion/empty queues, download required PNGs/metadata completely, then manage only old output/job pairs in the relevant root while stopped. Do not recursively delete the root itself, active token/configuration, or other titles' data. Check free space before raising limits.

## Shutdown and recovery

Verify processing finished, queues empty, and GPU completion; send authenticated `POST /api/admin/shutdown` 1 time each to gateway and native.202 acknowledges a shutdown request, not completed termination. Independently verify both owners absent, same-boot idle, and existing services. Next launch requires new owner verification and temporary-token handoff.

On timeout, check health/existing jobs without automatic retry. If GPU state is unknown, faults retained, or shutdown rejected, do not substitute force-free or kill for normal shutdown. Use trusted existing OS close/recovery and do not overlap another GPU title. After these 5 requests, normal shutdown, independent absence of both owners, same-artifact restart, and all GPU gates were verified.
