# Third-party notices

The integrated application is GPL-3.0-or-later. Individual MIT, BSD and other notices remain in force.
See [LICENSE](LICENSE), [licenses](licenses) and the copyright/SPDX notices in each source.
This is a source publication, not a distribution of Sony SDKs, firmware, model weights or runtime binaries.

| Component | Origin / role | License |
|---|---|---|
| stable-diffusion.cpp | leejet; inference and model management | MIT |
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
| WAI / Illustrious / TAESDXL | Separately obtained models | Separate model terms; see [models](docs/models.md) |

Pinned revisions are recorded in [dependencies.lock.json](dependencies.lock.json).
The bridge, startup guards, ownership tracking, HTTP integration, bounded logging and UI adapt and connect these projects.
Claims of authorship do not extend to their inference algorithms, model weights, driver or platform runtime.

The icon and sample images were generated on the PS5 with WAI v17 and TAESDXL. The 512-pixel icon is a Lanczos resize of the verified 1024 PNG; no Windows inference was used.
Keep model attribution and applicable model/output terms when reusing artwork.
No third-party application screenshot, user photograph, private recording URL or authentication credential is included.

Dashboard glyph masks retain OFL-1.1 separately from application code. Their derivative name is WAI Dashboard Glyphs.
The static masks and existing app-icon thumbnail need no runtime font library or additional GPU rendering context.
