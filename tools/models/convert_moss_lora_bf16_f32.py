# SPDX-License-Identifier: MIT
"""Exact selected MOSS LoRA conversion; no model inference or PS5 access.

The selected PEFT adapter has independent q/k/v A matrices. Keep all three
low-rank pairs as the existing SD.cpp fused-output slices; never share A or
expand/merge the full quantized checkpoint. BF16 -> F32 is a bit-exact shift.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct

import numpy as np

SOURCE_BYTES = 74377208
SOURCE_SHA = "d62ce599744d3e3f07f55a1f539052a99acc38d34bf35c13518aa949fc856afa"
REVISION = "e0a4cd44a906a4296dca7fa200c76058d1938082"
TARGET_NAME = "moss-image-nsfw-uncensored-r16-f32.safetensors"
PREFIX = "lora.model.diffusion_model."
SCHEMA = "ps5-ai-moss-lora-f32-slices-v1"
RANK = 16
DEPTH = 42


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate safetensors key")
        result[key] = value
    return result


def read_header(stream):
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError("Missing safetensors prefix")
    length = struct.unpack("<Q", raw)[0]
    if not 2 <= length <= 4 * 1024 * 1024:
        raise ValueError("Invalid bounded safetensors header length")
    header_bytes = stream.read(length)
    if len(header_bytes) != length:
        raise ValueError("Truncated safetensors header")
    header = json.loads(header_bytes, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("Object header required")
    return header, 8 + length


def mapped_name_shape(name, row):
    match = re.fullmatch(r"transformer\.layers\.(\d+)\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)\.lora_([AB])\.weight", name)
    if not match:
        raise ValueError("Unrecognized selected-adapter tensor name")
    layer, module, side = match.groups()
    if str(int(layer)) != layer or not 0 <= int(layer) < DEPTH:
        raise ValueError("Invalid MOSS layer index")
    input_dim = 8192 if module == "ffn_down" else 3072
    output_dim = 16384 if module == "ffn_up_gate" else 3072
    expected_shape = [RANK, input_dim] if side == "A" else [output_dim, RANK]
    if row.get("dtype") != "BF16" or row.get("shape") != expected_shape:
        raise ValueError("Selected-adapter dtype/shape mismatch")
    suffix = "lora_down" if side == "A" else "lora_up"
    if module in ("q", "k", "v"):
        target = "qkv.weight" + ({"q": "", "k": ".1", "v": ".2"}[module])
    else:
        target = module + ".weight"
    return PREFIX + "layers." + layer + ".proj." + target + "." + suffix


def selected_rows(header, source_data_bytes):
    metadata = header.get("__metadata__", {})
    adapter = json.loads(metadata.get("lora_adapter_metadata", "{}"))
    if adapter.get("r") != 16 or adapter.get("lora_alpha") != 16 or adapter.get("peft_type") != "LORA":
        raise ValueError("Selected rank16/alpha16 LORA metadata required")
    result = []
    seen = set()
    for name, row in header.items():
        if name == "__metadata__":
            continue
        target = mapped_name_shape(name, row)
        if target in seen:
            raise ValueError("Duplicate converted tensor")
        seen.add(target)
        offsets = row.get("data_offsets")
        count = int(np.prod(row["shape"], dtype=np.int64))
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(x) is not int for x in offsets) or
                offsets[0] < 0 or offsets[1] - offsets[0] != count * 2):
            raise ValueError("Invalid exact tensor byte range")
        result.append((name, target, row, count))
    if len(result) != DEPTH * 6 * 2:
        raise ValueError("All504 selected tensors are required; no partial adapter")
    expected = {f"transformer.layers.{layer}.proj.{module}.lora_{side}.weight"
                for layer in range(DEPTH)
                for module in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
                for side in ("A", "B")}
    if {r[0] for r in result} != expected:
        raise ValueError("Selected-adapter target set mismatch")
    result.sort(key=lambda item: item[2]["data_offsets"][0])
    position = 0
    for _, _, row, _ in result:
        if row["data_offsets"][0] != position:
            raise ValueError("Overlapping or sparse tensor payload")
        position = row["data_offsets"][1]
    if position != source_data_bytes:
        raise ValueError("Unexpected adapter payload/trailing bytes")
    return result


def expand_bits(bits):
    values = np.asarray(bits, dtype="<u2")
    if np.any((values & np.uint16(0x7f80)) == np.uint16(0x7f80)):
        raise ValueError("Nonfinite adapter weights are forbidden")
    return (values.astype("<u4") << np.uint32(16)).tobytes()


def convert(source, output, receipt):
    source, output, receipt = map(lambda p: Path(p).resolve(), (source, output, receipt))
    repository = Path(__file__).resolve().parents[2]
    if output.name != TARGET_NAME or output.is_relative_to(repository) or receipt.is_relative_to(repository):
        raise ValueError("Use fixed derivative filename and output/receipt outside source repository")
    if output == source or output.exists() or receipt.exists():
        raise FileExistsError("Preserve existing input/output/receipt; choose a new reviewed run")
    source_stat = source.stat()
    if source_stat.st_size != SOURCE_BYTES:
        raise ValueError("Locked source size differs")
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SOURCE_SHA:
            raise ValueError("Locked source SHA256 differs")
    with source.open("rb") as stream:
        header, source_data_start = read_header(stream)
    rows = selected_rows(header, SOURCE_BYTES - source_data_start)
    converted = {"__metadata__": {
        "format": "pt", "ps5_ai_moss_lora_schema": SCHEMA,
        "source_sha256": SOURCE_SHA, "source_revision": REVISION,
        "rank": "16", "alpha": "16", "target_tensor_count": "504",
        "conversion": "finite BF16 payload shifted16bits to exact F32; independent q/k/v output slices",
    }}
    position = 0
    for _, target, row, count in rows:
        converted[target] = {"dtype": "F32", "shape": row["shape"], "data_offsets": [position, position + count * 4]}
        position += count * 4
    encoded = json.dumps(converted, ensure_ascii=True, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    prefix = struct.pack("<Q", len(encoded)) + encoded
    partial = output.with_name(output.name + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(prefix)
    with source.open("rb") as src, partial.open("xb") as dst:
        dst.write(prefix)
        for _, _, row, count in rows:
            src.seek(source_data_start + row["data_offsets"][0])
            remaining = count * 2
            while remaining:
                raw = src.read(min(remaining, 2 * 1024 * 1024))
                if not raw or len(raw) % 2:
                    raise ValueError("Source changed/truncated during conversion")
                data = expand_bits(np.frombuffer(raw, dtype="<u2"))
                dst.write(data)
                digest.update(data)
                remaining -= len(raw)
        dst.flush()
        os.fsync(dst.fileno())
    # Re-read every output value against the original source; no approximate cast.
    with source.open("rb") as src, partial.open("rb") as dst:
        reread, start = read_header(dst)
        if reread != converted or start != len(prefix):
            raise ValueError("Converted header readback differs")
        for _, target, row, count in rows:
            src.seek(source_data_start + row["data_offsets"][0])
            dst.seek(start + converted[target]["data_offsets"][0])
            remaining = count * 2
            while remaining:
                raw = src.read(min(remaining, 2 * 1024 * 1024))
                if not raw or len(raw) % 2 or dst.read(len(raw) * 2) != expand_bits(np.frombuffer(raw, dtype="<u2")):
                    raise ValueError("Full-value readback does not match exact source expansion")
                remaining -= len(raw)
    if source.stat().st_size != source_stat.st_size or source.stat().st_mtime_ns != source_stat.st_mtime_ns:
        raise ValueError("Source identity changed during conversion")
    with partial.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest.hexdigest():
            raise ValueError("Derivative full SHA readback differs")
    os.link(partial, output)  # Atomic no-overwrite publication on the same NTFS volume.
    partial.unlink()
    record = {"schema": SCHEMA, "source_revision": REVISION, "source_sha256": SOURCE_SHA,
              "source_path": str(source), "output_path": str(output), "output_bytes": output.stat().st_size,
              "output_sha256": digest.hexdigest(), "tensor_count": len(rows), "rank": 16, "alpha": 16,
              "source_dtype": "BF16", "target_dtype": "F32", "tensor_payload_bytes": position,
              "all_tensor_values_exact_reread": True, "independent_qkv_A_B_preserved": True,
              "base_checkpoint_merged": False, "PS5_access": False, "inference": False}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output, args.receipt), indent=2))


if __name__ == "__main__":
    main()
