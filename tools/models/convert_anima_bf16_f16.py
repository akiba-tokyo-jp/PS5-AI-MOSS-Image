# SPDX-License-Identifier: MIT
"""Pinned, streaming BF16 -> F16 storage conversion. No model inference.

Only two officially hashed GPU weight files are accepted. Source files stay
unchanged; complete staged outputs are reread and independently checked before
atomic publication. The CPU Qwen3 text encoder is deliberately not converted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np

# The CLI sets all three paths explicitly before any filesystem operation.
OFFICIAL: Path
DERIVED: Path
DIAG: Path
REPO = "circlestone-labs/Anima"
REVISION = "f973fc41ec7545364ac9776c2440285f43ff2a30"
CHUNK_BYTES = 8 << 20
FILES = (
    ("split_files/diffusion_models/anima-turbo-v1.1.safetensors",
     4182230656, "fba11953276b57edf59d1dc4f1857ac05aa079c56f982b4d7c20298d57d3f7eb",
     "anima-turbo-v1.1-f16.safetensors"),
    ("split_files/vae/qwen_image_vae.safetensors",
     253806246, "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
     "qwen_image_vae-f16.safetensors"),
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(name: str, data: dict) -> None:
    target = DIAG / name
    part = target.with_suffix(target.suffix + ".tmp")
    part.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(part, target)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate safetensors JSON key")
        result[key] = value
    return result


def read_header(stream, expected_dtype: str):
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("Short safetensors prefix")
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= 32 << 20:
        raise ValueError("Unsafe header length")
    raw = stream.read(length)
    if len(raw) != length or not raw.startswith(b"{"):
        raise ValueError("Short or invalid header")
    header = json.loads(raw, object_pairs_hook=unique_pairs)
    if not isinstance(header, dict):
        raise ValueError("Header must be an object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in metadata.items()):
        raise ValueError("Invalid metadata")
    rows = []
    for name, row in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(row, dict) or row.get("dtype") != expected_dtype:
            raise ValueError(f"Unexpected dtype for {name}")
        shape, offsets = row.get("shape"), row.get("data_offsets")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid shape for {name}")
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(n) is not int or n < 0 for n in offsets)):
            raise ValueError(f"Invalid offsets for {name}")
        start, end = offsets
        if end < start or start % 2 or end - start != math.prod(shape) * 2:
            raise ValueError(f"Shape/offset mismatch for {name}")
        rows.append((start, end, name, math.prod(shape)))
    rows.sort()
    next_offset = 0
    for start, end, name, _ in rows:
        if start != next_offset:
            raise ValueError(f"Hole or overlap before {name}")
        next_offset = end
    if not rows or os.fstat(stream.fileno()).st_size != 8 + length + next_offset:
        raise ValueError("Data extent differs from file size")
    return header, rows, prefix + raw, 8 + length


def expected_half_bits(bfloat_bits: np.ndarray) -> np.ndarray:
    """Independent integer IEEE RNE reference, including subnormals/sign.

    BF16 has only seven fraction bits. Normal binary16 values are therefore
    exact; only values below binary16's normal range require rounding.
    NaN payloads are not normalized here because source NaNs are rejected.
    """
    raw = bfloat_bits.astype(np.uint32)
    sign = ((raw & 0x8000) >> 0).astype(np.uint16)
    exponent = ((raw >> 7) & 255).astype(np.int32) - 127
    fraction = raw & 127
    result = np.zeros(raw.shape, dtype=np.uint16)
    normal = (exponent >= -14) & (exponent <= 15)
    result[normal] = (((exponent[normal] + 15).astype(np.uint32) << 10) |
                      (fraction[normal] << 3)).astype(np.uint16)
    for e in range(-25, -14):
        selected = exponent == e
        if not np.any(selected):
            continue
        significand = fraction[selected] + 128
        shift = e + 17
        if shift >= 0:
            rounded = significand << shift
        else:
            right = -shift
            quotient = significand >> right
            remainder = significand & ((1 << right) - 1)
            halfway = 1 << (right - 1)
            rounded = quotient + ((remainder > halfway) |
                                   ((remainder == halfway) & ((quotient & 1) != 0)))
        result[selected] = rounded.astype(np.uint16)
    result[exponent > 15] = 0x7c00
    return result | sign


def validate_rounding_reference() -> dict:
    raw = np.arange(65536, dtype=np.uint16)
    source = (raw.astype(np.uint32) << 16).view(np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        actual = source.astype("<f2").view("<u2")
    expected = expected_half_bits(raw)
    selected = np.isfinite(source)
    mismatches = int(np.count_nonzero(actual[selected] != expected[selected]))
    if mismatches:
        raise ValueError("NumPy conversion differs from independent RNE reference")
    return {"all_bf16_encodings": 65536, "finite_encodings_checked": int(selected.sum()),
            "bit_mismatches": mismatches, "reference": "integer IEEE round-to-nearest ties-to-even"}


def empty_stats() -> dict:
    return {"elements": 0, "source_nonfinite": 0, "source_subnormal": 0,
            "f16_overflow": 0, "f16_nonfinite": 0, "underflow_to_zero": 0,
            "f16_subnormal": 0, "changed_values": 0, "rounding_bit_mismatches": 0,
            "zero_sign_mismatches": 0, "min": None, "max": None,
            "max_abs_source": 0.0, "max_abs_rounding_error": 0.0,
            "max_relative_rounding_error_nonzero_source": 0.0,
            "sum_abs_rounding_error": 0.0, "sum_squared_rounding_error": 0.0}


COUNT_FIELDS = ("elements", "source_nonfinite", "source_subnormal", "f16_overflow",
                "f16_nonfinite", "underflow_to_zero", "f16_subnormal", "changed_values",
                "rounding_bit_mismatches", "zero_sign_mismatches")
MAX_FIELDS = ("max_abs_source", "max_abs_rounding_error",
              "max_relative_rounding_error_nonzero_source")


def accumulate(total: dict, row: dict) -> None:
    for key in COUNT_FIELDS:
        total[key] += row[key]
    for key in MAX_FIELDS:
        total[key] = max(total[key], row[key])
    for key in ("sum_abs_rounding_error", "sum_squared_rounding_error"):
        total[key] += row[key]
    if row["min"] is not None:
        total["min"] = row["min"] if total["min"] is None else min(total["min"], row["min"])
        total["max"] = row["max"] if total["max"] is None else max(total["max"], row["max"])


_LOOKUP = None


def lookup_tables():
    global _LOOKUP
    if _LOOKUP is not None:
        return _LOOKUP
    raw = np.arange(65536, dtype=np.uint16)
    values = (raw.astype(np.uint32) << 16).view(np.float32)
    expected = expected_half_bits(raw)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        numpy_half = values.astype("<f2")
    finite = np.isfinite(values)
    if np.any(expected[finite] != numpy_half.view("<u2")[finite]):
        raise ValueError("LUT differs from exhaustive independent IEEE RNE reference")
    restored = expected.view("<f2").astype(np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        difference = np.abs(restored - values.astype(np.float64))
        relative = np.where(values != 0, difference / np.abs(values.astype(np.float64)), 0.0)
    _LOOKUP = {
        "half_bits": expected, "values": values, "absolute_error": difference,
        "squared_error": difference * difference, "relative_error": relative,
        "source_nonfinite": ~finite,
        "source_subnormal": ((raw & 0x7f80) == 0) & ((raw & 0x7f) != 0),
        "f16_overflow": finite & (np.abs(values) > 65504.0),
        "f16_nonfinite": ~np.isfinite(restored),
        "underflow_to_zero": (values != 0) & (restored == 0),
        "f16_subnormal": ((expected & 0x7c00) == 0) & ((expected & 0x3ff) != 0),
        "changed_values": values != restored,
        "rounding_bit_mismatches": (expected != numpy_half.view("<u2")) & finite,
        "zero_sign_mismatches": (values == 0) & (np.signbit(values) != np.signbit(restored)),
    }
    return _LOOKUP


def inspect_chunk(raw: bytes, converted: bytes | None = None):
    table = lookup_tables()
    bfloat = np.frombuffer(raw, dtype="<u2")
    counts = np.bincount(bfloat, minlength=65536)
    used = np.flatnonzero(counts)
    row = empty_stats()
    row["elements"] = int(bfloat.size)
    for name in COUNT_FIELDS:
        if name != "elements":
            row[name] = int(counts[table[name]].sum())
    if row["source_nonfinite"] or row["f16_overflow"] or row["f16_nonfinite"]:
        raise ValueError("Source nonfinite or F16 range failure")
    if row["rounding_bit_mismatches"] or row["zero_sign_mismatches"]:
        raise ValueError("Rounding/sign validation failed")
    half = table["half_bits"][bfloat].astype("<u2", copy=False)
    output = half.tobytes()
    if converted is not None and output != converted:
        raise ValueError("Persisted output differs from intended conversion")
    if used.size:
        values = table["values"][used]
        row["min"], row["max"] = float(values.min()), float(values.max())
        row["max_abs_source"] = float(np.abs(values).max())
        row["max_abs_rounding_error"] = float(table["absolute_error"][used].max())
        row["max_relative_rounding_error_nonzero_source"] = float(table["relative_error"][used].max())
        weights = counts[used].astype(np.float64)
        row["sum_abs_rounding_error"] = float(np.dot(weights, table["absolute_error"][used]))
        row["sum_squared_rounding_error"] = float(np.dot(weights, table["squared_error"][used]))
    return output, row



def finish_stats(stats: dict) -> dict:
    if stats["elements"]:
        stats["mean_abs_rounding_error"] = stats["sum_abs_rounding_error"] / stats["elements"]
        stats["rms_rounding_error"] = math.sqrt(stats["sum_squared_rounding_error"] / stats["elements"])
    return stats


def convert_one(source_path: str, expected_size: int, expected_sha: str, target_name: str) -> dict:
    source = OFFICIAL / Path(source_path).name
    target = DERIVED / target_name
    part = target.with_suffix(target.suffix + ".part")
    if target.exists():
        raise FileExistsError("Refusing to overwrite a completed derived model")
    if source.stat().st_size != expected_size:
        raise ValueError("Official source size mismatch")
    source_stat = source.stat()
    total, tensors = empty_stats(), []
    started, last_progress = time.monotonic(), 0.0
    with source.open("rb") as src:
        header, rows, original_prefix, source_data_start = read_header(src, "BF16")
        result_header = {}
        next_offset = 0
        new_offsets = {}
        for start, end, name, _ in rows:
            new_offsets[name] = [next_offset, next_offset + end - start]
            next_offset += end - start
        for name, descriptor in header.items():
            result_header[name] = descriptor if name == "__metadata__" else dict(
                descriptor, dtype="F16", data_offsets=new_offsets[name])
        new_json = json.dumps(result_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        new_json += b" " * ((-len(new_json)) % 8)
        new_prefix = struct.pack("<Q", len(new_json)) + new_json
        resumed_bytes = part.stat().st_size if part.exists() else 0
        expected_target_size = len(new_prefix) + next_offset
        if resumed_bytes and not len(new_prefix) <= resumed_bytes <= expected_target_size:
            raise ValueError("Staged file has an invalid length")
        source_stream_sha = hashlib.sha256(original_prefix)
        derived_stream_sha = hashlib.sha256(new_prefix)
        with part.open("r+b" if resumed_bytes else "xb") as dst:
            if resumed_bytes:
                if dst.read(len(new_prefix)) != new_prefix:
                    raise ValueError("Staged metadata differs from intended conversion")
            else:
                dst.write(new_prefix)
            for start, end, name, elements in rows:
                if src.tell() != source_data_start + start:
                    raise ValueError("Source stream position mismatch")
                tensor_stats = empty_stats()
                remaining = end - start
                while remaining:
                    raw = src.read(min(CHUNK_BYTES, remaining))
                    if not raw or len(raw) % 2:
                        raise ValueError("Short/unaligned source tensor")
                    converted, stats = inspect_chunk(raw)
                    preserved = min(len(converted), max(0, resumed_bytes - dst.tell()))
                    if preserved and dst.read(preserved) != converted[:preserved]:
                        raise ValueError("Preserved staged bytes differ from intended conversion")
                    if preserved < len(converted):
                        dst.write(converted[preserved:])
                    source_stream_sha.update(raw)
                    derived_stream_sha.update(converted)
                    accumulate(tensor_stats, stats)
                    remaining -= len(raw)
                    if time.monotonic() - last_progress >= 15:
                        save("progress.json", {"state": "converting_or_reusing_verified_bytes",
                             "file": target_name, "data_bytes_done": src.tell() - source_data_start,
                             "data_bytes_total": next_offset, "reused_staged_bytes": resumed_bytes,
                             "elapsed_seconds": time.monotonic() - started})
                        last_progress = time.monotonic()
                if tensor_stats["elements"] != elements:
                    raise ValueError("Tensor element count mismatch")
                accumulate(total, tensor_stats)
                tensors.append({"name": name, "shape": header[name]["shape"], **finish_stats(tensor_stats)})
            if src.read(1):
                raise ValueError("Unexpected trailing source bytes")
            dst.flush()
            os.fsync(dst.fileno())
        if source_stream_sha.hexdigest() != expected_sha:
            raise ValueError("Source hash differs from pinned official source")
    derived_sha = derived_stream_sha.hexdigest()
    save("progress.json", {"state": "reread_verifying_derived_sha", "file": target_name,
                                    "expected_bytes": expected_target_size,
                                    "elapsed_seconds": time.monotonic() - started})
    if sha(part) != derived_sha:
        raise ValueError("Persisted derived SHA differs from intended complete converted bytes")
    with part.open("rb") as verified:
        observed_header, observed_rows, _, _ = read_header(verified, "F16")
    if observed_header != result_header or observed_rows != rows:
        raise ValueError("Persisted metadata/shape/offset mismatch")
    if (part.stat().st_size != expected_target_size or source.stat().st_size != source_stat.st_size or
            source.stat().st_mtime_ns != source_stat.st_mtime_ns):
        raise ValueError("Final source/output stability check failed")
    os.replace(part, target)
    result = {"source_repo": REPO, "source_revision": REVISION, "source_path": source_path,
              "source_filename": source.name, "source_bytes": expected_size, "source_sha256": expected_sha,
              "derived_filename": target.name, "derived_bytes": target.stat().st_size,
              "derived_sha256": derived_sha, "source_dtype": "BF16", "derived_dtype": "F16",
              "is_official_weight_file": False, "is_derived_storage_conversion": True,
              "metadata_preserved": True, "tensor_names_shapes_order_preserved": True,
              "tensor_count": len(rows), "stats": finish_stats(total),
              "reused_staged_bytes": resumed_bytes,
              "reread_validation": {"all_source_values_checked": True,
                  "all_derived_bytes_reread_and_sha_matched_to_intended_value_bytes": True,
                  "independent_integer_rounding_reference_exhaustive_all_bf16_encodings": True,
                  "preserved_partial_bytes_compared_exactly_before_reuse": True,
                  "source_sha_verified_during_complete_conversion": True},
              "elapsed_seconds": time.monotonic() - started}
    save(target.stem + "-tensor-stats.json", {"file": target.name, "tensors": tensors})
    save(target.stem + "-verified.json", result)
    print(json.dumps({"state": "verified", "file": target.name, "bytes": result["derived_bytes"],
                      "sha256": derived_sha, "tensor_count": len(rows), "stats": result["stats"]}), flush=True)
    return result



def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    if DIAG.joinpath("complete.json").exists():
        raise FileExistsError("Completed conversion already exists; inspect its manifest instead of rerunning")
    started = time.monotonic()
    reference = validate_rounding_reference()
    save("rounding-reference.json", reference)
    files = [convert_one(*entry) for entry in FILES]
    manifest = {"state": "verified", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "converter_filename": Path(__file__).name, "converter_sha256": sha(Path(__file__)),
                "python_version": sys.version, "numpy_version": np.__version__,
                "chunk_bytes": CHUNK_BYTES, "rounding_reference": reference,
                "conversion_method": "exhaustively verified BF16-to-F16 LUT; per-chunk code histogram",
                "validated_original_converter_sha256": "ac719409fbdbf4a361680de18f25c9b769e89d61b70883740f807033d23f93fc",
                "files": files, "text_encoder_converted": False,
                "precision_scope": "F16 weight storage; does not assert FP16 arithmetic or inference quality",
                "license": "CircleStone Labs Non-Commercial License v1.2 plus applicable NVIDIA model terms",
                "license_source": f"https://huggingface.co/{REPO}/blob/{REVISION}/LICENSE.md",
                "derived_hashes_are_not_official_hashes": True,
                "model_inference": False, "PS5_calls": 0, "network_calls": 0,
                "elapsed_seconds": time.monotonic() - started}
    save("complete.json", manifest)
    save("progress.json", {"state": "completed", "files": len(files),
                                   "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True,
                        help="Directory containing the two original official SHA-verified BF16 files")
    parser.add_argument("--derived-dir", type=Path, required=True,
                        help="New converted weight directory outside this source repository")
    parser.add_argument("--diagnostics-dir", type=Path, required=True,
                        help="New conversion-manifest directory outside this source repository")
    args = parser.parse_args()
    OFFICIAL = args.official_dir.resolve()
    DERIVED = args.derived_dir.resolve()
    DIAG = args.diagnostics_dir.resolve()
    repository = Path(__file__).resolve().parents[2]
    if not OFFICIAL.is_dir():
        parser.error("--official-dir must already exist")
    for output in (DERIVED, DIAG):
        if output == repository or repository in output.parents:
            parser.error("Converted weights and diagnostics must be outside the source repository")
        if output == OFFICIAL or OFFICIAL in output.parents:
            parser.error("Output directories must not be inside the official source directory")
    if DERIVED == DIAG or DERIVED in DIAG.parents or DIAG in DERIVED.parents:
        parser.error("Derived weights and diagnostics need separate directories")
    try:
        main()
    except Exception as exc:
        DIAG.mkdir(parents=True, exist_ok=True)
        save("error.json", {"state": "failed", "error_type": type(exc).__name__,
                                     "error": str(exc), "PS5_calls": 0, "model_inference": False})
        raise
