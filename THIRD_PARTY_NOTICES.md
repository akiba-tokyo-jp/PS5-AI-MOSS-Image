# Third-party notices

This integration contains GPL-3.0-only files alongside GPL-3.0-or-later, MIT, BSD and other components. File-specific notices remain in force; a later-version option does not extend to GPL-3.0-only files.
See [LICENSE](LICENSE), [component provenance](docs/moss-optimization-provenance.json) and the copyright/SPDX notices in each source.
This is a source publication, not a distribution of Sony SDKs, firmware, model weights or runtime binaries.

| Component | Origin / role | License |
|---|---|---|
| stable-diffusion.cpp | leejet; inference and model management | MIT |
| MOSS latent Spectrum adapter | Existing SD.cpp `runtime/spectrum.hpp` route, plus bounded state, finite checks, strict scheduling and PS5 completion receipts added 2026-10-20 | Original MIT notices retained; new `moss_spectrum.hpp` is GPL-3.0-only |
| ComfyUI-Spectrum-MOSS | Rkkss, commit `3fde0674b29342565e7f42a0674af48d65aabc4a`; pinned method/reference, not the compiled ComfyUI hidden-state implementation | GPL-3.0-only; see `licenses/ComfyUI-Spectrum-MOSS-GPL-3.0-only.txt` |
| Strata | Niko1221 and contributors, commit `e88ee39d1d6be6e3364183af3b39b62a7ed6f211`; bounded staging design reference for a new same-format GGML upload adapter, without CUDA/HIP code | MIT; see `licenses/Strata-MIT.txt` |
| vLLM-Omni MOSS | Revision 942a5daa6c094615812ed89749ef07da96b8edab; per-request text/RoPE cache and pointwise fusion design/test reference. New F32 GGML/GLSL implementation; upstream CUDA/Triton code not imported | Apache-2.0 reference; new files retain their own notices |
| GGML | ggml contributors; tensor CPU/Vulkan execution | MIT; modifications forming part of the integrated GPL application retain original notices |
| PS5_Vulkan | Mihawk-99, BlackBearReloaded and contributors; native startup, tools, PS5 GPU reference | GPL-3.0-or-later, file-specific notices |
| PS5_Mesa / Mesa | RADV, NIR, ACO, Vulkan runtime | Predominantly MIT; individual files / bundled licenses govern |
| PS5_PayloadSDK | Mihawk-99, public payload SDK; platform/thread/TLS/placement code | GPL-3.0-or-later, BSD and other file-specific notices |
| PS5 payload SDK | John Törnblom and contributors; payload CRT / stubs | GPL-3.0-or-later and file-specific notices |
| cobanov/PS5LM | Native compatibility and lifecycle reference; five-function libc compatibility adaptation | GPL-3.0 and source-specific notices |
| mansanitizer/psllama | CPU VideoOut tiling/status reference | GPL-3.0; see upstream and source comments |
| WAI Dashboard Glyphs | Embedded bitmap derivative of Liberation Sans; original Google / Red Hat notices retained | SIL OFL-1.1; see licenses/WAI-Dashboard-Glyphs-OFL.txt |
| libc++ / libc++abi / libunwind / LLVM builtins | Native C++ runtime and compiler | Apache-2.0 WITH LLVM-exception |
| Vulkan-Headers | Khronos headers | Apache-2.0 / MIT, file-specific |
| stb_image_write / nlohmann json / oniguruma | Pulled through pinned stable-diffusion.cpp, not separate vendored copies here | MIT / public-domain option, MIT, BSD respectively; retain upstream files/notices when building |
| NumPy | Optional Windows host weight-format conversion dependency, not vendored | Follow the license notices in the installed NumPy distribution |

Pinned revisions are recorded in [dependencies.lock.json](dependencies.lock.json).
The bridge, startup guards, ownership tracking, HTTP integration, bounded logging and UI adapt and connect these projects.
Claims of authorship do not extend to their inference algorithms, model weights, driver or platform runtime.

The current icon is the project owner's supplied character image, resized for the home screen and dashboard. See assets/sce_sys/icon-provenance.json. It is an owner-supplied artwork asset, not a MOSS generation benchmark.
Keep applicable artwork and model/output terms when reusing images. Private recording URLs and authentication credentials are excluded.

Dashboard glyph masks retain OFL-1.1 separately from application code. Their derivative name is WAI Dashboard Glyphs.
The static masks and existing app-icon thumbnail need no runtime font library or additional GPU rendering context.

The experimental MOSS Spectrum path forecasts fully CFG-combined denoised latents through the existing SD.cpp cache boundary. It is not the ComfyUI extension's hidden-state/head port, and its performance or quality claims do not transfer to PS5. Reference revisions and modified paths are recorded in [MOSS optimization provenance](docs/moss-optimization-provenance.json).
The bounded-loader implementation preserves raw same-format bytes through the existing GGML backend. Strata supplied the bounded-staging design reference; its CUDA/HIP loader and conversion kernels are not included here.
