# SPDX-License-Identifier: MIT
"""Write a local exact model-transfer plan; never connects to PS5 or copies GB files.

This plan does not claim remote deployment, current full hash verification, memory
admission or inference. The transport must hash actual bytes while transferring
and verify complete remote size/SHA before publishing any model/config candidate.
"""
import argparse
import json
from pathlib import Path


def prepare(models_root, output):
    repository = Path(__file__).resolve().parents[2]
    root, output = Path(models_root).resolve(), Path(output).resolve()
    if root.is_relative_to(repository) or output.is_relative_to(repository) or output.exists():
        raise ValueError("Keep models and a fresh stage plan outside the repository")
    lock = json.loads((repository / "config/models.lock.json").read_text(encoding="utf-8"))
    runtime = [*lock["files"], *lock["derived_adapters"]]
    if len(runtime) != 5:
        raise ValueError("Four base components plus exactly one derived selected LoRA required")
    rows, names = [], set()
    remote = "/data/ai/native-moss-image-sft/PPSA99190/models/"
    for entry in runtime:
        relative = entry["relative_install_path"]
        source = root.joinpath(*relative.split("/"))
        if not source.resolve().is_relative_to(root) or not source.is_file():
            raise ValueError("Selected model/derived adapter is absent or escapes its root")
        if source.stat().st_size != entry["expected_bytes"]:
            raise ValueError("Local selected model size mismatch")
        name = entry.get("stage_basename", Path(relative).name)
        if name != Path(name).name or name in names:
            raise ValueError("Ambiguous stage basename")
        names.add(name)
        stat = source.stat()
        rows.append({"role": entry["role"], "source": str(source), "bytes": stat.st_size,
                     "source_mtime_ns": stat.st_mtime_ns, "source_ctime_ns": stat.st_ctime_ns,
                     "expected_sha256": entry["expected_sha256"], "remote": remote + name,
                     "app0_path": "/app0/models/" + name, "current_full_hash_rechecked_by_this_tool": False,
                     "transfer_and_full_remote_readback_verification_required": True})
    record = {"schema": "moss-selected-model-stage-plan-v1", "state": "HOST_PLAN_ONLY_NO_TRANSFER",
              "title_id": "PPSA99190", "native_port": 8192, "gateway_port": 8193,
              "runtime_admitted": False, "required_runtime_files": rows,
              "raw_BF16_adapter_is_not_staged": True, "copies_or_hardlinks_created": 0,
              "PS5_access": False, "inference": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True, help="Explicit verified model root outside the repository")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.models_root, args.output), indent=2))
