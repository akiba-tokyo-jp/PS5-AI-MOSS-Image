# SPDX-License-Identifier: MIT
"""Tiny numeric/metadata host oracles only; no learned-model inference."""
import copy
import importlib.util
from pathlib import Path
import struct
import unittest

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "tools/models/convert_moss_lora_bf16_f32.py"
SPEC = importlib.util.spec_from_file_location("moss_lora_conversion", TOOL)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def fixture_header():
    header = {"__metadata__": {"lora_adapter_metadata": '{"r":16,"lora_alpha":16,"peft_type":"LORA"}'}}
    offset = 0
    for layer in range(42):
        for module in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
            for side in ("A", "B"):
                shape = [16, 8192 if module == "ffn_down" else 3072] if side == "A" else [16384 if module == "ffn_up_gate" else 3072, 16]
                count = shape[0] * shape[1] * 2
                header[f"transformer.layers.{layer}.proj.{module}.lora_{side}.weight"] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + count]}
                offset += count
    return header, offset


class ConversionTests(unittest.TestCase):
    def test_all504_names_shapes_and_bytes(self):
        header, size = fixture_header()
        rows = MOD.selected_rows(header, size)
        self.assertEqual(len(rows), 504)
        self.assertEqual(len({r[1] for r in rows}), 504)
        self.assertEqual(size, 74317824)

    def test_ordered_qkv_slices_preserve_separate_pairs(self):
        header, _ = fixture_header()
        base = "lora.model.diffusion_model.layers.0.proj.qkv.weight"
        for module, suffix in (("q", ""), ("k", ".1"), ("v", ".2")):
            for side, target in (("A", "lora_down"), ("B", "lora_up")):
                key = f"transformer.layers.0.proj.{module}.lora_{side}.weight"
                self.assertEqual(MOD.mapped_name_shape(key, header[key]), base + suffix + "." + target)

    def test_all_finite_bf16_against_independent_arithmetic_oracle(self):
        bits = np.arange(65536, dtype="<u2")
        bits = bits[(bits & 0x7f80) != 0x7f80]
        decoded = np.frombuffer(MOD.expand_bits(bits), dtype="<u4")
        exp = ((bits.astype(np.uint32) >> 7) & 255).astype(np.int32)
        frac = (bits & 127).astype(np.float64)
        magnitude = np.ldexp(np.where(exp == 0, frac / 128, 1 + frac / 128), np.where(exp == 0, -126, exp - 127))
        reference = np.copysign(magnitude, np.where(bits & 0x8000, -1.0, 1.0)).astype("<f4").view("<u4")
        self.assertEqual(len(bits), 65280)
        self.assertTrue(np.array_equal(decoded, reference))

    def test_infinity_nan_rejected(self):
        for bits in (0x7f80, 0xff80, 0x7fc1, 0xffff):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                MOD.expand_bits(np.array([bits], dtype="<u2"))

    def test_qkv_concat_matches_three_independent_updates(self):
        # Different A matrices are essential. Sharing q's A for k/v is wrong.
        rng = np.random.default_rng(20261019)
        x = rng.integers(-3, 4, (8, 5)).astype(np.float32)
        aa = [rng.integers(-3, 4, (16, 8)).astype(np.float32) for _ in range(3)]
        bb = [rng.integers(-3, 4, (7, 16)).astype(np.float32) for _ in range(3)]
        alpha, rank, strength = 16, 16, 0.7
        unfused = [strength * alpha / rank * (b @ (a @ x)) for a, b in zip(aa, bb)]
        fused = np.concatenate(unfused, axis=0)
        independent = strength * np.concatenate([b @ a for a, b in zip(aa, bb)], axis=0) @ x
        self.assertTrue(np.allclose(fused, independent, atol=0.0002, rtol=0.00001))
        wrong = np.concatenate([bb[i] @ (aa[0] @ x) for i in range(3)], axis=0)
        self.assertFalse(np.allclose(fused / strength, wrong))

    def test_missing_extra_bad_dtype_shape_rank_and_offsets_rejected(self):
        base, size = fixture_header()
        key = "transformer.layers.0.proj.q.lora_A.weight"
        mutations = []
        value = copy.deepcopy(base); del value[key]; mutations.append(value)
        value = copy.deepcopy(base); value["unrecognized"] = value[key]; mutations.append(value)
        value = copy.deepcopy(base); value[key]["dtype"] = "F16"; mutations.append(value)
        value = copy.deepcopy(base); value[key]["shape"] = [15, 3072]; mutations.append(value)
        value = copy.deepcopy(base); value[key]["data_offsets"] = [0, 2]; mutations.append(value)
        value = copy.deepcopy(base); value["__metadata__"]["lora_adapter_metadata"] = '{"r":16,"lora_alpha":8,"peft_type":"LORA"}'; mutations.append(value)
        for index, value in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                MOD.selected_rows(value, size)
        with self.assertRaises(ValueError):
            MOD.selected_rows(base, size + 2)


if __name__ == "__main__":
    unittest.main()
