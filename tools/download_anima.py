# SPDX-License-Identifier: MIT
"""Download exact official Anima v1.1 split weights; no inference or PS5 access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

REPO = "circlestone-labs/Anima"
REVISION = "f973fc41ec7545364ac9776c2440285f43ff2a30"
FILES = [
    ("split_files/diffusion_models/anima-turbo-v1.1.safetensors", 4182230656,
     "fba11953276b57edf59d1dc4f1857ac05aa079c56f982b4d7c20298d57d3f7eb"),
    ("split_files/text_encoders/qwen_3_06b_base.safetensors", 1192135096,
     "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"),
    ("split_files/vae/qwen_image_vae.safetensors", 253806246,
     "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"),
]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, size, expected in FILES:
        target = output / Path(source).name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".part")
            request = urllib.request.Request(
                f"https://huggingface.co/{REPO}/resolve/{REVISION}/{source}",
                headers={"User-Agent": "PS5-AI-Anima-model-download/1.0"})
            digest = hashlib.sha256()
            total = 0
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as stream:
                while chunk := response.read(8 << 20):
                    stream.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if total != size or digest.hexdigest() != expected:
                raise RuntimeError("Official model hash/size mismatch: " + target.name)
            temporary.rename(target)
        elif target.stat().st_size != size or sha256(target) != expected:
            raise RuntimeError("Existing model differs: " + target.name)
        rows.append({"source": source, "file": target.name, "bytes": size, "sha256": expected})
        print(json.dumps({"verified": target.name, "bytes": size}), flush=True)
    license_file = output / "LICENSE.md"
    if not license_file.exists():
        with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{REVISION}/LICENSE.md", timeout=60) as response:
            data = response.read(1 << 20)
        if len(data) != 18259:
            raise RuntimeError("Unexpected model license length")
        with license_file.open("xb") as stream:
            stream.write(data)
    manifest = output / "official-models.json"
    if manifest.exists():
        raise RuntimeError("Manifest exists; preserve it and choose a new output directory")
    manifest.write_text(json.dumps({"repo": REPO, "revision": REVISION, "files": rows,
                                    "inference": False}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
