# MOSS logging and storage

`logging.level: "normal"` reduces routine records; explicitly select `"debug"` only for detailed investigation. Configuration is read once before AGC/worker initialization; unknown values/types are rejected. This does not imply detailed logging of every dispatch during normal operation.

Startup, engine, and arena journals retain startup, required capacity/ownership/GPU-completion checks, numeric gates, job results, and faults. Routine suppression and bounded rotation are separate mechanisms. Journal-write timing is nested CPU wall time, not pure GPU time or entirely removable wait time.

Bounded journals/stdio retain current and previous segments, rotating before exceeding limits. Limits are defined per caller; partial records are not reported as successful. Existing large files are not appended indefinitely. Write, synchronization, or metadata-check failure is not converted into generation success.

Ordinary service removes software generation-count caps and limits each process's output + job metadata to a default 8GiB. It accounts for 32MiB per new/unfinished job and rejects with 507 before generation. This is not filesystem-free-space measurement and cannot prevent other consumption. Successful PNGs are not automatically deleted; archive and manage while stopped. [Deployment/storage allowance](deployment.md). Historical maximum 4-attempt records concern diagnostic builds.

When collecting full logs, verify current owner, boot, inode, device, unchanged size/mtime before/after, no gaps, and restored permissions. Do not describe a rotated fragment missing its beginning as complete evidence. Keep raw logs private; publish only summarized conditions, times, and verification scope.

Do not publish authentication tokens, secret hashes, personal paths, or raw process records. Direct reservation, API GPU-labeled peak, process RSS, and heap mapped capacity are different quantities; do not add them as total physical RAM/dedicated VRAM. See [service verification](moss-service-20261021.md) and [status](status.md).
