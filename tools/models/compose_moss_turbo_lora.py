# SPDX-License-Identifier: MIT
"""Prepare one F32 rank144 coefficient adapter; never execute a model.

This is a proposed offline converter, not admission of these weights on PS5.
It combines the selected F32 rank16 adapter at 0.7 and the pinned rank128
Turbo adapter at 1.0 by independent per-projection low-rank concatenation.
The Q4 base is never opened or merged. A new GPU rank144 numerical gate and
physical memory admission remain mandatory before use.
"""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

TURBO_REVISION = "2ded95c37a029360245926fe72e475e30039bbfb"
TURBO_NAME = "moss_image_sft_4step_v2.0.safetensors"
TURBO_SHA256 = "154c7f1455c3bf9925945c2f1ea2e4f09d0d0c67256081384a094b2b585f913f"
TURBO_BYTES = 1189176752
TURBO_HEADER_BYTES = 90552
TURBO_HEADER_SHA256 = "8a89c6346252ef1aa6b4779b03f2d525ea30cc63a7d7eb55503fc5fc88392ce5"
NSFW_ORIGIN_SHA256 = "d62ce599744d3e3f07f55a1f539052a99acc38d34bf35c13518aa949fc856afa"
NSFW_REVISION = "e0a4cd44a906a4296dca7fa200c76058d1938082"
NSFW_SCHEMA = "ps5-ai-moss-lora-f32-slices-v1"
NSFW_F32_SHA256 = "907c1e0220c8aa301595b4a369f7b09447001306ffebb53da1fc58f24542cdf9"
NSFW_F32_BYTES = 148704512
SCHEMA = "ps5-ai-moss-nsfw07-turbo1-r144-f32-v1"
OUTPUT_NAME = "moss-nsfw07-turbo4-r144-f32.safetensors"
NSFW_RANK, TURBO_RANK, RANK, DEPTH = 16, 128, 144, 42
MATRIX_COUNT = 504
MATRIX_PAYLOAD_BYTES = 1337720832
EXTRA_MATRIX_BYTES_VS_RANK16 = 1189085184
MODULES = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
MAX_BLOCK_BYTES = 2 * 1024 * 1024


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate safetensors key")
        result[key] = value
    return result


def read_header(stream):
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("Missing safetensors length")
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= 4 * 1024 * 1024:
        raise ValueError("Invalid bounded header length")
    raw = stream.read(length)
    if len(raw) != length:
        raise ValueError("Truncated safetensors header")
    header = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("Object header required")
    return header, 8 + length, raw


def names(layer, module, side):
    weight = "qkv.weight" + {"q": "", "k": ".1", "v": ".2"}[module] if module in ("q", "k", "v") else module + ".weight"
    normalized = f"lora.model.diffusion_model.layers.{layer}.proj.{weight}.lora_{side}"
    turbo = f"diffusion_model.layers.{layer}.proj.{module}.lora_{side}.weight"
    return normalized, turbo


def catalog():
    for layer in range(DEPTH):
        for module in MODULES:
            inputs = 8192 if module == "ffn_down" else 3072
            outputs = 16384 if module == "ffn_up_gate" else 3072
            for side in ("down", "up"):
                normalized, turbo = names(layer, module, side)
                yield layer, module, side, inputs, outputs, normalized, turbo


def validate_rows(header, expected, payload_bytes):
    rows = {k: v for k, v in header.items() if k != "__metadata__"}
    if set(rows) != set(expected):
        raise ValueError("Exact complete adapter tensor set required")
    ranges = []
    for name, shape in expected.items():
        row = rows[name]
        if not isinstance(row, dict) or row.get("dtype") != "F32" or row.get("shape") != shape:
            raise ValueError("Adapter tensor must match exact F32 shape")
        offsets = row.get("data_offsets")
        count = math.prod(shape)
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(x) is not int for x in offsets) or offsets[0] < 0 or
                offsets[1] - offsets[0] != count * 4):
            raise ValueError("Invalid exact adapter tensor byte range")
        ranges.append((offsets[0], offsets[1]))
    position = 0
    for begin, end in sorted(ranges):
        if begin != position:
            raise ValueError("Sparse/overlapping tensor payload forbidden")
        position = end
    if position != payload_bytes:
        raise ValueError("Trailing/missing tensor payload forbidden")
    return rows


def validate_nsfw_header(header, payload_bytes):
    metadata = header.get("__metadata__", {})
    required = {"ps5_ai_moss_lora_schema": NSFW_SCHEMA,
                "source_sha256": NSFW_ORIGIN_SHA256, "source_revision": NSFW_REVISION,
                "rank": "16", "alpha": "16", "target_tensor_count": "504"}
    if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in required.items()):
        raise ValueError("Selected exact-expanded NSFW rank16 provenance required")
    expected = {}
    for _, _, side, inputs, outputs, normal, _ in catalog():
        expected[normal] = [NSFW_RANK, inputs] if side == "down" else [outputs, NSFW_RANK]
    return validate_rows(header, expected, payload_bytes)


def validate_turbo_header(header, raw_header, file_bytes, data_start):
    if (file_bytes != TURBO_BYTES or len(raw_header) != TURBO_HEADER_BYTES or
            hashlib.sha256(raw_header).hexdigest() != TURBO_HEADER_SHA256):
        raise ValueError("Pinned Turbo header identity differs")
    expected = {}
    for layer, module, side, inputs, outputs, _, turbo in catalog():
        expected[turbo] = [TURBO_RANK, inputs] if side == "down" else [outputs, TURBO_RANK]
        expected[f"diffusion_model.layers.{layer}.proj.{module}.alpha"] = []
    return validate_rows(header, expected, file_bytes - data_start)


def f32(value):
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("Nonfinite/overflowed coefficient forbidden") from exc
    if not math.isfinite(result):
        raise ValueError("Nonfinite/overflowed coefficient forbidden")
    return result


def floats(raw):
    if len(raw) % 4:
        raise ValueError("Misaligned F32 payload")
    values = array("f")
    if values.itemsize != 4:
        raise ValueError("Host array F32 width differs")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite adapter coefficient forbidden")
    return values


def raw_floats(values):
    if sys.byteorder != "little":
        values = array("f", values)
        values.byteswap()
    return values.tobytes()


def read_exact(stream, count):
    raw = stream.read(count)
    if len(raw) != count:
        raise ValueError("Source changed/truncated during coefficient preparation")
    return raw


def down_chunks(left, right, columns, left_rank, right_rank, max_bytes=MAX_BLOCK_BYTES):
    """Row concatenation. Finite values retain their exact input F32 bits."""
    if columns <= 0 or left_rank <= 0 or right_rank <= 0 or max_bytes < 4:
        raise ValueError("Invalid bounded down-matrix dimensions")
    block = max_bytes // 4 * 4
    for stream, count in ((left, left_rank * columns * 4), (right, right_rank * columns * 4)):
        while count:
            raw = read_exact(stream, min(block, count))
            floats(raw)
            yield raw
            count -= len(raw)


def up_chunks(left, right, rows, left_rank, right_rank, left_scale, right_scale,
              max_bytes=MAX_BLOCK_BYTES):
    """Column concatenation with F32 coefficient multiplication, no inference.

    Baking 0.7 into coefficients changes IEEE rounding relative to multiplying
    a completed LoRA matmul. Algebra is preserved; bitwise image parity is not
    asserted. A future actual Vulkan composed-linear numerical gate is needed.
    """
    left_scale, right_scale = f32(left_scale), f32(right_scale)
    width = left_rank + right_rank
    if min(rows, left_rank, right_rank) <= 0 or max_bytes < width * 4:
        raise ValueError("Invalid bounded up-matrix dimensions")
    rows_per_block = max_bytes // (width * 4)
    for begin in range(0, rows, rows_per_block):
        count = min(rows_per_block, rows - begin)
        l = floats(read_exact(left, count * left_rank * 4))
        r = floats(read_exact(right, count * right_rank * 4))
        out = array("f")
        for row in range(count):
            ls = l[row * left_rank:(row + 1) * left_rank]
            rs = r[row * right_rank:(row + 1) * right_rank]
            out.extend(ls if left_scale == 1.0 else (f32(float(x) * left_scale) for x in ls))
            out.extend(rs if right_scale == 1.0 else (f32(float(x) * right_scale) for x in rs))
        yield raw_floats(out)


def source_identity(path, expected_sha, expected_bytes=None):
    path = Path(path).resolve(strict=True)
    st = path.stat()
    if not path.is_file() or (expected_bytes is not None and st.st_size != expected_bytes):
        raise ValueError("Pinned source is missing or has wrong byte count")
    with path.open("rb") as stream:
        actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("Pinned source SHA256 differs")
    return {"path": str(path), "bytes": st.st_size, "sha256": actual_sha,
            "device": st.st_dev, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns}


def verify_alpha(stream, data_start, row):
    stream.seek(data_start + row["data_offsets"][0])
    alpha = struct.unpack("<f", read_exact(stream, 4))[0]
    if not math.isfinite(alpha) or alpha != 128.0:
        raise ValueError("Every actual pinned Turbo alpha must equal128; card claims alone are insufficient")
    return alpha


def coefficient_chunks(nsfw, turbo, ns_start, t_start, ns_row, t_row, side, inputs, outputs):
    nsfw.seek(ns_start + ns_row["data_offsets"][0])
    turbo.seek(t_start + t_row["data_offsets"][0])
    if side == "down":
        return down_chunks(nsfw, turbo, inputs, NSFW_RANK, TURBO_RANK)
    return up_chunks(nsfw, turbo, outputs, NSFW_RANK, TURBO_RANK, 0.7, 1.0)


def publish_no_replace(partial, output):
    """Both paths are on the output volume; os.link refuses an existing name."""
    os.link(partial, output)
    Path(partial).unlink()


def prepare(nsfw_path, nsfw_sha256, turbo_path, output, receipt):
    if nsfw_sha256 != NSFW_F32_SHA256:
        raise ValueError("The selected actual NSFW F32 derivative SHA256 is locked")
    output, receipt = Path(output).resolve(), Path(receipt).resolve()
    nsfw_path, turbo_path = Path(nsfw_path).resolve(strict=True), Path(turbo_path).resolve(strict=True)
    repository = Path(__file__).resolve().parents[2]
    if (output.name != OUTPUT_NAME or len({nsfw_path, turbo_path, output, receipt}) != 4 or
            output.exists() or receipt.exists() or
            output.is_relative_to(repository) or receipt.is_relative_to(repository)):
        raise ValueError("Preserve inputs and use a fresh fixed derivative basename/receipt")
    if output.suffix != ".safetensors" or receipt.suffix != ".json":
        raise ValueError("Explicit safetensors output and JSON receipt required")
    ns_pin = source_identity(nsfw_path, nsfw_sha256, NSFW_F32_BYTES)
    t_pin = source_identity(turbo_path, TURBO_SHA256, TURBO_BYTES)
    with nsfw_path.open("rb") as stream:
        ns_header, ns_start, _ = read_header(stream)
    with turbo_path.open("rb") as stream:
        t_header, t_start, t_raw = read_header(stream)
    ns_rows = validate_nsfw_header(ns_header, ns_pin["bytes"] - ns_start)
    t_rows = validate_turbo_header(t_header, t_raw, t_pin["bytes"], t_start)
    # Read all252 actual alpha payloads before creating any derivative.
    with turbo_path.open("rb") as stream:
        for layer in range(DEPTH):
            for module in MODULES:
                verify_alpha(stream, t_start, t_rows[f"diffusion_model.layers.{layer}.proj.{module}.alpha"])
    composed = {"__metadata__": {
        "format": "pt", "ps5_ai_moss_lora_schema": SCHEMA, "rank": "144", "alpha": "144",
        "target_tensor_count": "504", "runtime_multiplier": "1.0",
        "nsfw_derivative_sha256": nsfw_sha256, "nsfw_origin_sha256": NSFW_ORIGIN_SHA256,
        "nsfw_revision": NSFW_REVISION, "nsfw_component_multiplier": "0.7",
        "turbo_sha256": TURBO_SHA256, "turbo_revision": TURBO_REVISION,
        "turbo_component_multiplier": "1.0", "turbo_alpha_verified": "128",
        "composition": "D=[D16;D128]; U=[float32(0.7*U16),U128]; separate q/k/v; no base merge",
        "rounding": "FP32 coefficient scaling; not bitwise equivalent to post-matmul0.7",
    }}
    position = 0
    for _, _, side, inputs, outputs, normal, _ in catalog():
        shape = [RANK, inputs] if side == "down" else [outputs, RANK]
        size = math.prod(shape) * 4
        composed[normal] = {"dtype": "F32", "shape": shape, "data_offsets": [position, position + size]}
        position += size
    if position != MATRIX_PAYLOAD_BYTES:
        raise ValueError("Internal exact rank144 payload budget differs")
    encoded = json.dumps(composed, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    encoded += b" " * (-len(encoded) % 8)
    prefix = struct.pack("<Q", len(encoded)) + encoded
    partial = output.with_name(output.name + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(prefix)
    with nsfw_path.open("rb") as ns, turbo_path.open("rb") as ts, partial.open("xb") as dst:
        dst.write(prefix)
        for _, _, side, inputs, outputs, normal, tname in catalog():
            for chunk in coefficient_chunks(ns, ts, ns_start, t_start, ns_rows[normal], t_rows[tname], side, inputs, outputs):
                dst.write(chunk)
                digest.update(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    # Re-read the complete output and regenerate expected coefficient chunks.
    with nsfw_path.open("rb") as ns, turbo_path.open("rb") as ts, partial.open("rb") as dst:
        reread, output_start, _ = read_header(dst)
        if reread != composed or output_start != len(prefix):
            raise ValueError("Derivative header readback differs")
        for _, _, side, inputs, outputs, normal, tname in catalog():
            for chunk in coefficient_chunks(ns, ts, ns_start, t_start, ns_rows[normal], t_rows[tname], side, inputs, outputs):
                if dst.read(len(chunk)) != chunk:
                    raise ValueError("Full coefficient readback differs")
        if dst.read(1):
            raise ValueError("Unexpected derivative trailing bytes")
    if source_identity(nsfw_path, nsfw_sha256) != ns_pin or source_identity(turbo_path, TURBO_SHA256, TURBO_BYTES) != t_pin:
        raise ValueError("Source identity changed during preparation")
    with partial.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest.hexdigest():
            raise ValueError("Derivative full SHA256 readback differs")
    # Same-volume hard link publishes atomically without replacing a destination.
    publish_no_replace(partial, output)
    record = {"schema": SCHEMA, "inputs": [ns_pin, t_pin], "output": {
        "path": str(output), "bytes": output.stat().st_size, "sha256": digest.hexdigest()},
        "rank": RANK, "matrix_count": MATRIX_COUNT, "matrix_payload_bytes": position,
        "extra_matrix_bytes_vs_rank16": EXTRA_MATRIX_BYTES_VS_RANK16,
        "full_coefficient_reread": True, "actual_turbo_alphas_verified": 252,
        "components": [{"identity": "selected NSFW F32 rank16", "strength": 0.7},
                       {"identity": "pinned Turbo4 F32 rank128", "strength": 1.0}],
        "runtime_multiplier": 1.0, "base_checkpoint_opened": False,
        "base_checkpoint_merged_or_requantized": False, "tensor_precision": "F32",
        "bitwise_original_post_matmul_scaling_equivalent": False,
        "physical_memory_budget_measured": False, "PS5_access": False, "model_inference": False}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsfw", type=Path, required=True)
    parser.add_argument("--nsfw-sha256", required=True)
    parser.add_argument("--turbo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.nsfw, args.nsfw_sha256, args.turbo, args.output, args.receipt), indent=2))


if __name__ == "__main__":
    main()
