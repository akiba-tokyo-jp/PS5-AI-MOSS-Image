# MOSS status display

The PS5 CPU draws a dark navy background and cards for status, memory, and recent logs. No display-only Vulkan context or external GUI is added. The display title is PS5 AI MOSS Image. The icon uses a character image supplied by the project owner; [asset provenance](../assets/sce_sys/icon-provenance.json) is retained.

Available values include app name, model, PS5 UTC clock, display heartbeat, engine state, job progress, the app's own Direct pool usage/reservation/peak, completed fences, and filtered recent logs. Unavailable fields are omitted. CPU/GPU utilization and temperature are not estimated when unmeasured. Direct values do not represent PS5-wide or total dedicated VRAM usage.

Display heartbeat and actual GPU-generation completion are separate. `GET /health` `tv_display` is reported when VideoOut completion is verified and is null during temporary mutex contention. `gpu_verified` depends on actual image-generation GPU completion; an updating screen alone does not prove generation success.

The font bitmap is OFL-1.1. Original copyright and the derivative font's distinct name are retained in [font provenance](../native/ui/font-provenance.json) and [license](../licenses/WAI-Dashboard-Glyphs-OFL.txt). The derivative name in the license does not identify the app's supported model. This MOSS documentation excludes photos and old dashboard benchmarks for other models.

`tests/dashboard.py` verifies host rendering and metadata, not PS5 learned-model inference. Hardware startup, display, generation, and normal shutdown are checked independently.
