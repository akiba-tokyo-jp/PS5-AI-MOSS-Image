# Build MOSS Image from source

The ordinary API entry point is `tools/build_service.py`. It applies a service profile to native and gateway builds, removing fixed prompts/seeds, the first 256 prerequisite, and the maximum 4 attempts. [Hardware verification](moss-service-20261021.md). Historical low-level trial commands describe diagnostic builds.

## Dependencies and configuration

Cross-compile on Windows x64. Match pinned revisions in [dependencies.lock.json](../dependencies.lock.json), [native profile](../build-profiles/native.json), and [core profile](../build-profiles/moss-core.json), then copy the [build configuration example](../config/build.json.example) to a private workspace. Specify actual paths for SDK, LLVM, PS5 Mesa/RADV, Vulkan-Headers, SD.cpp, shader toolchain, dependency archives, and native packager.

SDKs, libraries, weights, and generated shaders are excluded from Git. This is not an installer that automatically downloads them after cloning. Do not mix native and CPU-gateway SDK ABIs; match 16KiB pages, emulated TLS, `GGML_MAX_NAME=160`, constructors, and imports. Use the public overlay and pinned base, not merely an existing core archive's filename. See [model preparation](moss-models.md), [adapter reproduction](moss-turbo4-reproduction.md), and [third-party notices](../THIRD_PARTY_NOTICES.md).

## Ordinary service build

Choose a new output directory. Store dependency configuration and build artifacts outside the source checkout on a drive with sufficient capacity.

```powershell
python -B tools/build_service.py --config D:/ps5-dev/build.json --out D:/ps5-dev/build/moss-service --jobs 2
```

The wrapper builds the gateway and generates its schema, records tracked/non-ignored checkout inputs in a private freeze, freshly compiles 75 core objects, and builds native. Even where low-level profile names still contain trial, `--service-mode` disables attempt limits and the 256 prerequisite.

Selected configuration: F32 accumulation, Turbo4, bounded weight uploads, DiT optimization mask 15, kernel mask 15, FA shared-KV occupancy adjustment, img2img, Direct pool 10GiB / inference budget 9.5GiB. Do not mix in FFN chunk 2048, F16 LoRA storage, GPU profiling, or alternate sampling candidates. The model uses a rank 144 F32 adapter with runtime multiplier 1. Numeric, capacity, GPU-completion, and owner guards remain.

Outputs include `native/title/PPSA99190/eboot.bin`, the CPU ELF in the gateway build directory, manifests, and `moss-service.json.example`. See gateway manifest `artifact.path` for the exact ELF path. Building does not connect, deploy, or launch on PS5.

If managing an existing complete [freeze format](../config/moss-source-freeze.json.example), specify `--source-freeze PRIVATE_JSON`. Do not modify source during building. The public source snapshot and records of actually consumed dependency closure are separate.

`--prepare-only` builds the gateway, checks the freeze and 75-object core plan, then exits. It does not establish native/core compilation or hardware success. The hardware artifact was built with the wrapper's lower-level commands, verifying 75 compiles, native 24 constructors /6 runtime providers, and complete SELF readback. Subsequently, the wrapper's automatic freeze also completed all 75 core/native/gateway builds, PS5 startup, and 1024 generation.

## Deployment and verification

Use [config/moss-service.json.example](../config/moss-service.json.example). The old `moss-native.json.example` describes a historical 12GiB/rank 16 configuration and must not be reused unchanged. Configuration JSON alone cannot change compiled pool/VA layout.

Follow [deployment/startup](deployment.md) and launch as a native Homebrew title. API readiness follows numeric readback; the model lazy-loads on the first image request. The first request can specify 1024 directly. Host policy/HTTP tests and compilation do not replace PS5 GPU execution. See [PNGs and measurements](moss-service-20261021.md).

Historical default-OFF diagnostics remain independent. Do not import fixed requests/attempt caps from [DiT kernel comparisons](moss-dit-kernels-20261021.md), [GPU timing](moss-dit-gpu-profile-20261021.md), or [memory-option comparisons](moss-turbo-memory-options-20261020.md) into ordinary service mode.
