# Masked image editing

`POST /v1/images/edits` accepts the OpenAI-compatible multipart `image` and
`mask` files. Use the same dimensions for the source, mask and requested output.
The existing WAI checkpoint performs img2img with a latent denoise mask;
no separate inpainting model is required.

- `mask_mode=alpha` (default): transparent PNG pixels are edited, opaque pixels
  are preserved, following the OpenAI image mask convention.
- `mask_mode=grayscale`: white pixels are edited and black pixels are preserved.
- Intermediate values feather the final RGB composition. The latent mask is
  rounded at 0.5. A mask containing no editable pixels is rejected.
- `reference_mode=inpaint` is an img2img alias requiring a mask. Masks are also
  accepted with `img2img` or `combined`, and require an initial image.
- `denoise_strength` controls how strongly the source changes; it is separate
  from the requested step count. The result reports `actual_steps`.

The PS5 restores zero-mask RGB pixels exactly after genuine masked diffusion.
This preservation operation runs on the PS5 CPU. UNet and enabled ControlNet
inference retain their Vulkan backend. The native job includes `inpainting`,
`mask_prepared`, `mask_image_sha256`, `mask_denoise_evaluations`,
`mask_preserved_pixels` and `mask_changed_pixels`.

`mask_denoise_evaluations` counts actual latent preservation blend executions,
not all denoiser evaluations. A fully editable latent mask legitimately has
zero preservation blends; it must still have prepared its mask, completed real
GPU inference and passed finite-output checks. `mask_changed_pixels` counts
actual changed RGB pixels rather than the size of the editable region.

For ControlNet, first upload a prepared PNG/JPEG control map to the authenticated
native `POST /api/inputs` endpoint, then pass its returned `control_image_id`,
`controlnet=openpose` (or `canny`) and numeric `control_strength` alongside the
multipart mask request. `control_image` file uploads and automatic pose/edge
extraction are not implemented. The gateway deletes its temporary source,
reference and mask uploads after the terminal job, while the caller retains
ownership of an existing control ID. Delete that ID only after its leases return
to zero. `combined` can express IP-Adapter plus masks and ControlNet, but protocol
support alone does not establish adequate runtime memory for every combination.

Native clients may upload masks with `POST /api/masks` (alpha) or
`POST /api/masks/grayscale`, then use `mask_image_id` in `/api/generate`.
Mask uploads require PNG below 4 MiB. Authenticated input leases prevent deleting
an image while a queued or running job uses it. Invalid roles or dimensions are
rejected before admitting a generation job. Existing jobs have 36 response
fields; public requests remain limited to 32 fields and the internal job parser
accepts up to 64 fields.

`scripts/Invoke-WaiReference.ps1` supports `-Mask`, `-MaskMode` and `-Mode inpaint`.
The client sends one generation POST and performs no automatic retry. After a
timeout, inspect the native queue and job before sending another request.

Host fixtures in `tools/build_mask_controlnet_tests.py` and `tests/mask_*.py`
compile the actual native and gateway sources with an intentional failure
engine. They check input ownership, exact mask polarity/composition, restart
recovery, the OpenAI SDK/PowerShell transport and combined mask/ControlNet
forwarding. These localhost fixtures never perform model inference and cannot
serve as evidence of PS5 GPU image generation.
