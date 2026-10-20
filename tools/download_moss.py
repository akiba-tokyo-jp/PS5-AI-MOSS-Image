# SPDX-License-Identifier: MIT
"""Verify or download locked MOSS files. No inference, PS5 access or credentials."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import urllib.parse
import urllib.request

LOCK = Path(__file__).resolve().parents[1] / "config" / "models.lock.json"
REPOSITORY = LOCK.parents[1]
CHUNK = 8 << 20


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_lock(path=LOCK):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "ps5-ai-moss-image-model-lock-v1":
        raise ValueError("Unrecognized model lock schema")
    files = data.get("files", [])
    if len(files) != 4 or {f.get("role") for f in files} != {"diffusion", "text_encoder", "vae", "tokenizer"}:
        raise ValueError("Expected exactly the four selected model components")
    adapters = data.get("requested_adapters", [])
    if (len(adapters) != 1 or adapters[0].get("role") != "lora" or
            adapters[0].get("repository") != "axiomofmind/Moss-image-nsfw-uncensored-lora" or
            adapters[0].get("target_tensor_count") != 504):
        raise ValueError("Expected the user-selected complete MOSS rank16 adapter")
    seen = set()
    for entry in [*files, *adapters]:
        repo = entry["repository"]
        revision = entry["revision"]
        remote = entry["remote_path"]
        relative = entry["relative_install_path"]
        for value in (remote, relative):
            parts = PurePosixPath(value)
            if (not value or parts.is_absolute() or "\\" in value or ":" in value
                    or any(p in {"", ".", ".."} for p in value.split("/"))):
                raise ValueError("Model paths must be safe relative POSIX paths")
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
                or not re.fullmatch(r"[0-9a-f]{40}", revision)
                or not re.fullmatch(r"[0-9a-f]{64}", entry["expected_sha256"])
                or type(entry["expected_bytes"]) is not int
                or not 0 < entry["expected_bytes"] <= 12 * 1024**3):
            raise ValueError("Invalid immutable model identity")
        expected_url = (f"https://huggingface.co/{repo}/resolve/{revision}/"
                        + urllib.parse.quote(remote, safe="/"))
        if entry["url"] != expected_url or relative in seen:
            raise ValueError("Unpinned URL or duplicate destination")
        seen.add(relative)
    return data


def reject_links(path):
    """Reject symlinks/junctions at existing target or ancestor components."""
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError("Refusing a symlink or Windows reparse point")


def checked_target(output, entry):
    candidate = output.joinpath(*PurePosixPath(entry["relative_install_path"]).parts)
    reject_links(candidate)
    if not candidate.resolve().is_relative_to(output):
        raise ValueError("Model destination escapes output directory")
    return candidate


def verify_file(path, entry):
    reject_links(path)
    if not path.is_file():
        raise ValueError("Model file is missing or not regular")
    if path.stat().st_size != entry["expected_bytes"]:
        raise ValueError("Model size differs from lock; refusing replacement")
    digest = sha256(path)
    if digest != entry["expected_sha256"]:
        raise ValueError("Model SHA256 differs from lock; refusing replacement")
    return {"role": entry["role"], "file": entry["relative_install_path"],
            "bytes": entry["expected_bytes"], "sha256": digest, "verified": True}


def download_file(target, entry):
    """One HTTP request, same-volume partial, full hash before atomic publish."""
    partial = target.with_name(target.name + ".part")
    reject_links(partial)
    if partial.exists():
        raise ValueError("Partial already exists; preserve/inspect it before retrying")
    target.parent.mkdir(parents=True, exist_ok=True)
    reject_links(target.parent)
    request = urllib.request.Request(entry["url"], headers={
        "User-Agent": "PS5-AI-MOSS-Image-model-download/1.0",
        "Accept-Encoding": "identity",
    })
    digest = hashlib.sha256()
    total = 0
    # Open exclusively before networking so a conflicting partial is never touched.
    with partial.open("xb") as stream:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200 or urllib.parse.urlsplit(response.url).scheme != "https":
                raise ValueError("Expected a complete HTTPS model response")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != entry["expected_bytes"]:
                raise ValueError("HTTP size differs from lock")
            while chunk := response.read(CHUNK):
                total += len(chunk)
                if total > entry["expected_bytes"]:
                    raise ValueError("Response exceeds locked size")
                stream.write(chunk)
                digest.update(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    if total != entry["expected_bytes"] or digest.hexdigest() != entry["expected_sha256"]:
        raise ValueError("Downloaded bytes differ from lock; partial preserved")
    reject_links(target)
    if target.exists():
        raise ValueError("Final model appeared concurrently; partial preserved")
    # Atomic no-clobber publication, including if another writer races the check.
    # Same-volume hardlink support (NTFS by default) is required. On unsupported
    # filesystems this fails closed and preserves the verified partial.
    os.link(partial, target)
    partial.unlink()
    return verify_file(target, entry)


def operate(output, data, download=False):
    reject_links(output)
    output = output.resolve()
    if output.is_relative_to(REPOSITORY):
        raise ValueError("Keep model weights and the large tokenizer outside the repository")
    lock_fd = None
    lock_path = output / ".download-moss.lock"
    if download:
        output.mkdir(parents=True, exist_ok=True)
        reject_links(output)
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        rows = []
        for entry in data["files"]:
            target = checked_target(output, entry)
            if target.exists():
                row = verify_file(target, entry)
            elif download:
                row = download_file(target, entry)
            else:
                raise ValueError("A locked model component is missing")
            rows.append(row)
            print(json.dumps(row), flush=True)
        return rows
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify", action="store_true", help="Local hash/size checks only; no network or writes")
    action.add_argument("--download", action="store_true", help="Explicitly download missing locked files; no inference")
    parser.add_argument("--output", type=Path, required=True, help="Explicit model directory outside the repository")
    parser.add_argument("--base-only", action="store_true", help="Explicitly omit the required requested adapter; not a complete runnable selected configuration")
    args = parser.parse_args()
    try:
        data = load_lock()
        if not args.base_only:
            data = dict(data, files=[*data["files"], *data["requested_adapters"]])
        rows = operate(args.output, data, download=args.download)
    except Exception as error:
        # Transport exceptions can contain temporary signed redirect URLs. Avoid
        # printing those query strings or any environment/credential contents.
        message = str(error) if isinstance(error, (ValueError, FileExistsError)) else type(error).__name__
        print(json.dumps({"ok": False, "error": message, "automatic_retry": False}), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "components_verified": len(rows), "inference": False,
                      "PS5_access": False, "automatic_retry": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
