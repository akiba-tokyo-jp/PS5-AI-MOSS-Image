#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Host shape, conversion and actual pinned name-converter checks; no inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm", type=Path, required=True)
    parser.add_argument("--sd", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out, llvm, sd = args.out.resolve(), args.llvm.resolve(), args.sd.resolve()
    if out.is_relative_to(root) or (out / "receipt.json").exists():
        raise ValueError("Use a fresh output directory outside the repository")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEMP=str(out), TMP=str(out))
    env["PATH"] = str(llvm) + os.pathsep + env.get("PATH", "")
    commands, results = [], []

    def run(label, command):
        command = list(map(str, command))
        commands.append(command)
        result = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                text=True, timeout=120)
        (out / (label + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(label + " failed: " + (result.stdout + result.stderr)[-4000:])
        results.append({"check": label, "exit_code": 0, "stdout": result.stdout.strip()})

    includes = ["-I", root / "native/app", "-I", root / "patches/stable-diffusion/overlay/src",
                "-I", sd / "src", "-I", sd / "include", "-I", sd / "ggml/include",
                "-I", sd / "ggml/src", "-isystem", sd / "thirdparty"]
    common = [llvm / "clang++.exe", "-std=c++20", "-DGGML_MAX_NAME=160", "-static"]
    run("exact-conversion-oracles", [sys.executable, "-B", root / "tests/test_moss_lora_conversion.py"])
    contract = out / "moss-lora-contract.exe"
    run("contract-compile", [*common, "-O2", "-Wall", "-Wextra", "-Werror", *includes,
                             root / "tests/moss_lora_contract.cpp", "-o", contract])
    run("contract", [contract])
    names = out / "moss-lora-name-conversion.exe"
    run("pinned-name-conversion-compile", [*common, "-O2", *includes,
                                           root / "tests/moss_lora_name_conversion.cpp",
                                           sd / "src/name_conversion.cpp", "-o", names])
    run("pinned-name-conversion", [names])
    run("strict-adapter-syntax", [*common, *includes, "-fsyntax-only", "-x", "c++",
                                   root / "patches/stable-diffusion/overlay/src/model/adapter/lora.hpp"])
    config = out / "moss-engine-config.exe"
    run("config-compile", [*common, "-O2", "-Wall", "-Wextra", "-Werror", *includes,
                           root / "tests/moss_engine_config.cpp", "-o", config])
    run("config-examples", [config, root / "config/native.json.example",
                             root / "config/moss-native.json.example"])
    files = ["native/app/engine_config.h", "native/app/engine_sdcpp.cpp",
             "native/app/moss_lora_contract.hpp", "native/app/moss_lora_receipt.h",
             "native/app/moss_runtime_admission.h", "config/models.lock.json", "config/moss-lora.plan.json",
             "tools/download_moss.py", "tools/models/convert_moss_lora_bf16_f32.py",
             "tools/models/prepare_moss_model_stage.py", "tests/test_moss_lora_conversion.py",
             "tests/moss_lora_contract.cpp", "tests/moss_lora_name_conversion.cpp",
             "tests/moss_lora_checks.py", "tests/moss_engine_config.cpp",
             "patches/stable-diffusion/overlay/src/model/adapter/lora.hpp",
             "patches/stable-diffusion/overlay/src/pipeline/diffusion_engine.cpp",
             "patches/stable-diffusion/overlay/src/pipeline/image.cpp"]
    receipt = {"schema": "moss-selected-lora-host-checks-v1", "success": True,
               "PS5_access": False, "learned_model_inference": False, "native_GPU_execution": False,
               "runtime_guard_enabled": False, "results": results,
               "sources": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}
    (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"success": True, "checks": len(results), "no_PS5_or_inference": True}))


if __name__ == "__main__":
    main()
