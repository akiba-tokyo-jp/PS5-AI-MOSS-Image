# SPDX-License-Identifier: GPL-3.0-or-later
"""Execute the production conditioning-cache header with a counting encoder.

The fixture retains actual upstream parameter/result declarations, checked below
against --vendor. Tensor/model execution is a host mock; this does not measure
model embeddings, GPU correctness, or image quality. All outputs stay on D:.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def record(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def declaration(source, marker):
    """Extract a named simple type including nested methods/braces."""
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        depth += (source[index] == "{") - (source[index] == "}")
        if depth == 0:
            require(source[index:index + 2] == "};", "Type terminator changed")
            return source[start:index + 2]
    raise RuntimeError("Unclosed declaration: " + marker)


def compare_declarations(vendor, fixture):
    fixture_text = fixture.read_text(encoding="utf-8-sig")
    locations = {
        "src/model.h": ["enum SDVersion"],
        "src/model/diffusion/model.hpp": ["struct RefImageParams"],
        "src/conditioning/conditioner.hpp": [
            "struct SDCondition", "enum class MiniMaxH3PresentationKind",
            "struct MiniMaxH3PresentationItem", "struct ConditionerParams"],
    }
    for relative, markers in locations.items():
        actual = (vendor / relative).read_text(encoding="utf-8-sig")
        for marker in markers:
            # Whitespace-only normalization accommodates upstream indentation.
            normalize = lambda value: re.sub(r"\s+", "", value)
            require(normalize(declaration(actual, marker)) ==
                    normalize(declaration(fixture_text, marker)),
                    "Fixture no longer matches actual upstream " + marker)
    return [vendor / relative for relative in locations]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--llvm", type=Path, required=True,
                        help="LLVM bin directory or clang++ executable")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.drive.upper() == "D:" and not out.exists() and
            not out.is_relative_to(ROOT), "Fresh D output outside repo required")
    fixture_root = ROOT / "tests/fixtures/moss_conditioning_cache"
    fixture = fixture_root / "conditioning/conditioner.hpp"
    vendor = args.vendor.resolve()
    declarations = compare_declarations(vendor, fixture)
    header = ROOT / "patches/stable-diffusion/overlay/src/conditioning/conditioning_cache.h"
    source = fixture_root / "cache_behavior.cpp"
    compiler = args.llvm.resolve()
    if compiler.is_dir():
        compiler = compiler / "clang++.exe"
    require(compiler.is_file() and header.is_file(), "Missing compiler/header")
    inputs = [Path(__file__).resolve(), source, fixture, header, compiler,
              *declarations]
    before = [record(path) for path in inputs]
    out.mkdir(parents=True)
    command = [str(compiler), "-std=c++17", "-O0", "-Wall", "-Wextra",
               "-Werror", "-frtti", "-static", "-I" + str(fixture_root),
               "-I" + str(header.parent), str(source), "-o", str(out / "cache_behavior.exe")]
    compiled = subprocess.run(command, capture_output=True, text=True)
    (out / "compile.log").write_text(compiled.stdout + compiled.stderr, encoding="utf-8")
    require(compiled.returncode == 0, "Host compilation failed; see compile.log")
    ran = subprocess.run([str(out / "cache_behavior.exe")], capture_output=True, text=True)
    (out / "run.log").write_text(ran.stdout + ran.stderr, encoding="utf-8")
    require(ran.returncode == 0 and
            ran.stdout.strip() == "PASS 81 actual-cache behavior checks; no model/GPU/PS5",
            "Cache regression failed; see run.log")
    require(before == [record(path) for path in inputs], "Inputs changed during test")
    (out / "receipt.private.json").write_text(json.dumps({
        "schema": "moss-actual-header-cache-host-regression-v1",
        "checks": 81, "declaration_matches": 6,
        "state": "host_actual_header_mock_encoder_PASS",
        "compile_command": command, "inputs": before,
        "model_embedding_equality_measured": False, "PS5_calls": 0,
        "source_stable": True,
    }, indent=2) + "\n", encoding="utf-8")
    print("PASS 81 actual-header cache checks; mock encoder, no model/GPU/PS5")


if __name__ == "__main__":
    main()
