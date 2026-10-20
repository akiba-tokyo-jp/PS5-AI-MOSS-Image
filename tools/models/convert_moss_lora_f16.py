# SPDX-License-Identifier: MIT
"""Bounded, pinned coefficient-format preparation. This never runs a model.

This is an explicitly LOSSY experiment. It does not alter the Q4 base, adapter
strengths, inference profile, installed files, or the original F32 adapter.
NumPy is required solely for IEEE binary16 conversion and error statistics.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

import numpy as np

SOURCE_BYTES = 1337791552
SOURCE_SHA256 = "126fed9dd5268fe20ab0085525a4b7b7f8ff767a28aea907df2089c33360d4d3"
SOURCE_HEADER_BYTES = 70712
SOURCE_HEADER_SHA256 = "7ed8089c08aef1a42196fea9f9bdfe30990c3ed280cbd9a97a55de8e4faa06bd"
SOURCE_SCHEMA = "ps5-ai-moss-nsfw07-turbo1-r144-f32-v1"
SOURCE_NAME = "moss-nsfw07-turbo4-r144-f32.safetensors"
OUTPUT_NAME = "moss-nsfw07-turbo4-r144-f16.safetensors"
SCHEMA = "ps5-ai-moss-nsfw07-turbo1-r144-f16-storage-v1"
POLICY = "composed_r144_f16_storage_f32acc_non_dyadic_gate_v1"
PAYLOAD_F32_BYTES = 1337720832
MAX_SOURCE_CHUNK_BYTES = 2 * 1024 * 1024
NSFW_SHA256 = "907c1e0220c8aa301595b4a369f7b09447001306ffebb53da1fc58f24542cdf9"
TURBO_SHA256 = "154c7f1455c3bf9925945c2f1ea2e4f09d0d0c67256081384a094b2b585f913f"


def unique_object(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("Duplicate header key")
        result[k] = v
    return result


def canonical_catalog():
    result = {}
    for layer in range(42):
        for module in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
            qkv = module in ("q", "k", "v")
            suffix = {"q": "", "k": ".1", "v": ".2"}.get(module, "")
            base = f"lora.model.diffusion_model.layers.{layer}.proj.{('qkv' if qkv else module)}.weight{suffix}"
            result[base + ".lora_down"] = [144, 8192 if module == "ffn_down" else 3072]
            result[base + ".lora_up"] = [16384 if module == "ffn_up_gate" else 3072, 144]
    return result


def read_header(stream):
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("Truncated header prefix")
    n = struct.unpack("<Q", prefix)[0]
    if not 2 <= n <= 4 * 1024 * 1024:
        raise ValueError("Header must be bounded")
    raw = stream.read(n)
    if len(raw) != n:
        raise ValueError("Truncated header")
    h = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(h, dict):
        raise ValueError("Object header required")
    return h, 8 + n, raw


def validate_source_header(header, raw, file_bytes):
    if file_bytes != SOURCE_BYTES or len(raw) != SOURCE_HEADER_BYTES or hashlib.sha256(raw).hexdigest() != SOURCE_HEADER_SHA256:
        raise ValueError("Pinned composed source/header identity differs")
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or metadata.get("ps5_ai_moss_lora_schema") != SOURCE_SCHEMA:
        raise ValueError("Pinned composed F32 schema required")
    rows = {k: v for k, v in header.items() if k != "__metadata__"}
    catalog = canonical_catalog()
    if set(rows) != set(catalog):
        raise ValueError("Exactly 504 canonical rank144 tensors required")
    spans = []
    for name, shape in catalog.items():
        row = rows[name]
        if not isinstance(row, dict) or set(row) != {"dtype", "shape", "data_offsets"} or row["dtype"] != "F32" or row["shape"] != shape:
            raise ValueError("Exact F32 rank144 source shapes required")
        offsets = row["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets) or offsets[0] < 0 or offsets[1] - offsets[0] != math.prod(shape) * 4:
            raise ValueError("Invalid source tensor span")
        spans.append(tuple(offsets))
    position = 0
    for begin, end in sorted(spans):
        if begin != position:
            raise ValueError("Sparse or overlapping source payload forbidden")
        position = end
    if position != PAYLOAD_F32_BYTES or 8 + len(raw) + position != file_bytes:
        raise ValueError("Missing/trailing payload forbidden")
    return rows


def output_header(rows):
    # Fixed public metadata only. No timestamps, host paths, UID or receipt paths.
    out = {"__metadata__": {
        "ps5_ai_moss_lora_schema": SCHEMA,
        "storage_policy": POLICY,
        "source_sha256": SOURCE_SHA256,
        "source_header_sha256": SOURCE_HEADER_SHA256,
        "source_nsfw_sha256": NSFW_SHA256,
        "source_turbo_sha256": TURBO_SHA256,
        "conversion": "IEEE754_binary32_to_binary16_RNE_lossy",
        "component_order": "NSFW16_then_Turbo128",
        "baked_up_scales": "0.7,1.0",
        "runtime_global_scale": "1.0",
        "rank": "144", "target_tensor_count": "504",
        "base_model_merged": "false",
        "GPU_numerical_admission": "required_not_asserted_by_converter",
        "image_quality_parity": "not_asserted",
    }}
    position = 0
    for name, row in sorted(rows.items()):
        size = math.prod(row["shape"]) * 2
        out[name] = {"dtype": "F16", "shape": row["shape"], "data_offsets": [position, position + size]}
        position += size
    raw = json.dumps(out, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    raw += b" " * (-len(raw) % 8)
    return out, raw, position


def empty_statistics():
    return {"values": 0, "exact_values": 0, "changed_values": 0, "nonfinite_source": 0,
            "overflowed_values": 0, "rounded_to_zero": 0, "F16_subnormal_values": 0,
            "max_absolute_error": 0.0, "max_relative_error_nonzero_source": 0.0,
            "sum_squared_error": 0.0}


def round_block(raw):
    if not raw or len(raw) % 4 or len(raw) > MAX_SOURCE_CHUNK_BYTES:
        raise ValueError("Aligned nonempty bounded F32 chunk required")
    source = np.frombuffer(raw, dtype="<f4")
    if not bool(np.isfinite(source).all()):
        raise ValueError("Nonfinite source coefficient forbidden")
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        half = source.astype("<f2")
    if not bool(np.isfinite(half).all()):
        raise ValueError("F16 coefficient overflow forbidden")
    restored = half.astype("<f4")
    source64 = source.astype(np.float64)
    error = restored.astype(np.float64)
    error -= source64
    np.abs(error, out=error)
    nonzero = source != 0
    stats = empty_statistics()
    stats.update(values=int(source.size), exact_values=int(np.count_nonzero(source == restored)),
                 changed_values=int(np.count_nonzero(source != restored)),
                 rounded_to_zero=int(np.count_nonzero(nonzero & (restored == 0))),
                 F16_subnormal_values=int(np.count_nonzero((np.abs(restored) < 2 ** -14) & (restored != 0))),
                 max_absolute_error=float(error.max(initial=0)),
                 sum_squared_error=float(np.dot(error, error)))
    if bool(nonzero.any()):
        stats["max_relative_error_nonzero_source"] = float(np.max(error[nonzero] / np.abs(source64[nonzero]), initial=0))
    return half.tobytes(), stats


def merge_statistics(total, part):
    for key in total:
        total[key] = max(total[key], part[key]) if key.startswith("max_") else total[key] + part[key]


def final_statistics(stats):
    return {**stats, "root_mean_squared_error": math.sqrt(stats["sum_squared_error"] / stats["values"]) if stats["values"] else 0.0}


def convert_payload(source, destination, source_start, rows, chunk_bytes=MAX_SOURCE_CHUNK_BYTES):
    # Internal generic streaming primitive also used by tiny host fixtures.
    if type(chunk_bytes) is not int or chunk_bytes < 4 or chunk_bytes > MAX_SOURCE_CHUNK_BYTES or chunk_bytes % 4:
        raise ValueError("Invalid chunk size")
    total, per_tensor, digest, written = empty_statistics(), {}, hashlib.sha256(), 0
    for name, row in sorted(rows.items()):
        begin, end = row["data_offsets"]
        source.seek(source_start + begin)
        left, tensor_stats = end - begin, empty_statistics()
        while left:
            n = min(chunk_bytes, left)
            raw = source.read(n)
            if len(raw) != n:
                raise ValueError("Source changed or truncated")
            rounded, stats = round_block(raw)
            if destination is not None:
                destination.write(rounded)
            digest.update(rounded)
            written += len(rounded)
            merge_statistics(tensor_stats, stats)
            left -= n
        merge_statistics(total, tensor_stats)
        per_tensor[name] = {"source_dtype": "F32", "output_dtype": "F16", "shape": row["shape"], **final_statistics(tensor_stats)}
    return written, digest.hexdigest(), final_statistics(total), per_tensor


def file_record(path):
    p = Path(path).resolve(strict=True)
    with p.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": digest}


def stat_identity(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def publish_no_replace(partial, output):
    os.link(partial, output)
    partial.unlink()


def prepare(source, receipt, output=None, analyze_only=False):
    source = Path(source).resolve(strict=True)
    receipt = Path(receipt).resolve()
    output = Path(output).resolve() if output is not None else None
    if type(analyze_only) is not bool or analyze_only != (output is None):
        raise ValueError("Choose explicit analyze-only OR new output")
    if source.name != SOURCE_NAME or source == receipt or receipt.exists() or receipt.suffix != ".json":
        raise ValueError("Pinned source and fresh JSON receipt required")
    if output is not None and (output.name != OUTPUT_NAME or output.exists() or len({source, receipt, output}) != 3):
        raise ValueError("A fresh distinct fixed derivative basename is required")
    if not receipt.parent.is_dir() or (output is not None and not output.parent.is_dir()):
        raise ValueError("Create explicit destination directories first")
    source_before = stat_identity(source)
    pin = file_record(source)
    if pin["bytes"] != SOURCE_BYTES or pin["sha256"] != SOURCE_SHA256:
        raise ValueError("Pinned source whole-file identity differs")
    with source.open("rb") as f:
        header, start, raw = read_header(f)
    rows = validate_source_header(header, raw, pin["bytes"])
    _, out_raw, expected_payload = output_header(rows)
    partial = output.with_name(output.name + ".partial") if output is not None else None
    if partial is not None and partial.exists():
        raise ValueError("Partial exists; no automatic retry/overwrite")
    destination = partial.open("xb") if partial is not None else None
    try:
        if destination is not None:
            destination.write(struct.pack("<Q", len(out_raw)))
            destination.write(out_raw)
        with source.open("rb") as f:
            written, payload_sha, statistics, per_tensor = convert_payload(f, destination, start, rows)
        if written != expected_payload or statistics["values"] * 2 != expected_payload:
            raise ValueError("Converted payload count differs")
        if destination is not None:
            destination.flush()
            os.fsync(destination.fileno())
            destination.close()
            destination = None
        # Full SHA and file stability must hold after conversion, including analysis.
        if stat_identity(source) != source_before or file_record(source) != pin:
            raise ValueError("Source changed during coefficient preparation")
        out_pin = None
        if partial is not None:
            with partial.open("rb") as f:
                parsed, out_start, reraw = read_header(f)
                if reraw != out_raw or {v["dtype"] for k, v in parsed.items() if k != "__metadata__"} != {"F16"}:
                    raise ValueError("Output header reread mismatch")
                reread_sha = hashlib.file_digest(f, "sha256").hexdigest()
            if reread_sha != payload_sha or partial.stat().st_size != out_start + expected_payload:
                raise ValueError("Whole output payload reread mismatch")
            publish_no_replace(partial, output)
            out_pin = file_record(output)
        result = {"schema": "moss-lora-f16-storage-coefficient-preparation-v1", "analyze_only": analyze_only,
                  "learned_inference_executed": False, "GPU_admission_asserted": False,
                  "lossless": False, "image_parity_asserted": False, "policy": POLICY,
                  "converter": file_record(__file__), "numpy_version": np.__version__, "source": pin,
                  "source_header_sha256": hashlib.sha256(raw).hexdigest(),
                  "output": out_pin, "expected_output_header_bytes": len(out_raw),
                  "expected_output_header_sha256": hashlib.sha256(out_raw).hexdigest(),
                  "output_payload_sha256": payload_sha, "output_payload_bytes": expected_payload,
                  "coefficient_payload_bytes_saved": PAYLOAD_F32_BYTES - expected_payload,
                  "component_order": ["NSFW", "Turbo"], "component_ranks": [16, 128],
                  "baked_up_scales": [0.7, 1.0], "runtime_global_scale": 1.0,
                  "max_source_chunk_bytes": MAX_SOURCE_CHUNK_BYTES,
                  "scratch_scope": "bounded per chunk arrays; not measured process RSS or PS5 memory",
                  "statistics": statistics, "tensors": per_tensor}
        with receipt.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
        return result
    finally:
        if destination is not None:
            destination.close()
        # Failed partials remain evidence; never publish, remove, or retry them here.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--receipt", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output")
    mode.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    result = prepare(args.source, args.receipt, args.output, args.analyze_only)
    print(json.dumps({k: result[k] for k in ("analyze_only", "output", "output_payload_bytes", "coefficient_payload_bytes_saved", "statistics")}, sort_keys=True))


if __name__ == "__main__":
    main()
