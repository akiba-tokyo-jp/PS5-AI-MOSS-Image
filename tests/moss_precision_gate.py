#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed-data real quantization references + synthetic admission; no GPU/model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(a.config.read_text())
    out = a.out.resolve()
    if out.is_relative_to(root) or (out / "receipt.json").exists():
        raise ValueError("Select fresh output outside repository")
    out.mkdir(parents=True, exist_ok=True)
    llvm, sd, sdk = Path(config["LLVM"]), Path(config["SD"]), Path(config["SDK"])
    env = dict(os.environ, TEMP=str(out), TMP=str(out))
    env["PATH"] = str(llvm) + os.pathsep + env.get("PATH", "")
    results, commands = [], []

    def run(label, command, expected=0):
        commands.append(list(map(str, command)))
        r = subprocess.run(commands[-1], env=env, capture_output=True, text=True, timeout=120)
        (out / (label + ".log")).write_text(r.stdout + r.stderr, encoding="utf-8")
        if r.returncode != expected:
            raise RuntimeError(label + " failed: " + (r.stdout + r.stderr)[-5000:])
        results.append({"check": label, "exit_code": r.returncode, "stdout": r.stdout.strip()})

    (out / "ggml-version.h").write_text('#define GGML_VERSION "host-fixed-oracle-only"\n#define GGML_COMMIT "host-fixed-oracle-only"\n')
    includes = ["-DGGML_MAX_NAME=160", "-I", out, "-I", root / "native/app", "-I", sd / "ggml/include",
                "-I", sd / "ggml/src", "-isystem", sd / "thirdparty"]
    objects = []
    # These contain metadata/quantization routines only; no CPU graph backend or
    # model runner is linked. Dead functions are removed by section GC.
    for name in ("ggml.c", "ggml-quants.c"):
        obj = out / (name + ".o")
        run(name + "-compile", [llvm / "clang.exe", "-O2", "-ffunction-sections", "-fdata-sections",
                               *includes, "-c", sd / "ggml/src" / name, "-o", obj])
        objects.append(obj)
    obj = out / "ggml-threading.o"
    run("threading-compile", [llvm / "clang++.exe", "-std=c++20", "-O2", *includes,
                              "-c", sd / "ggml/src/ggml-threading.cpp", "-o", obj])
    objects.append(obj)
    exe = out / "moss-precision-gate.exe"
    run("fixture-compile", [llvm / "clang++.exe", "-std=c++20", "-O2", "-static", "-Wall", "-Wextra", "-Werror",
                             "-ffunction-sections", "-fdata-sections", *includes,
                             root / "tests/moss_precision_gate.cpp", *objects, "-Wl,--gc-sections", "-o", exe])
    for scenario in ("oracles", "success", "no-backend", "not-vulkan", "wrong-device", "unsupported", "allocation",
                     "no-dispatch", "no-submit", "no-fence", "unverified", "mismatch", "nan", "log-unhealthy"):
        run(scenario, [exe, scenario])
    for scenario in ("upload-unknown", "result-unknown", "readback-unknown", "cleanup-unknown", "compute-failed", "pending"):
        run(scenario, [exe, scenario], 42)
    for suffix in ("-bf16", "-bf16-matrix", "-q4-matrix-small", "-q4-matrix-large"):
        for scenario in ("unsupported", "allocation", "no-dispatch", "no-submit", "no-fence", "unverified", "mismatch", "nan", "log-unhealthy"):
            run(scenario + suffix, [exe, scenario + suffix])
        for scenario in ("upload-unknown", "result-unknown", "readback-unknown", "cleanup-unknown", "compute-failed", "pending"):
            run(scenario + suffix, [exe, scenario + suffix], 42)
    run("nan-unsampled-q4-matrix-large", [exe, "nan-unsampled-q4-matrix-large"])
    fp16_exe = out / "moss-precision-gate-fp16.exe"
    run("fixture-fp16-compile", [llvm / "clang++.exe", "-std=c++20", "-O2", "-static", "-Wall", "-Wextra", "-Werror",
                                 "-DPS5_AI_ENABLE_FP16=1", "-ffunction-sections", "-fdata-sections", *includes,
                                 root / "tests/moss_precision_gate.cpp", *objects, "-Wl,--gc-sections", "-o", fp16_exe])
    # Synthetic mode/policy/refusal tests do not emulate FP16 GPU accuracy.
    # The unchanged double oracle and tolerances must pass on actual hardware.
    for scenario in ("oracles", "success", "mismatch-bf16", "mismatch-bf16-matrix", "mismatch-q4-matrix-small",
                     "mismatch-q4-matrix-large", "mismatch-flash", "nan-flash", "no-fence-flash"):
        run("fp16-" + scenario, [fp16_exe, scenario])
    for scenario in ("upload-unknown-flash", "result-unknown-flash", "readback-unknown-flash", "cleanup-unknown-flash"):
        run("fp16-" + scenario, [fp16_exe, scenario], 42)
    mixed_flags=["-DPS5_AI_ENABLE_FP16=1","-DPS5_AI_MOSS_F32_ACCUMULATION=1",
                 "-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1","-DPS5_AI_MOSS_PERF_VARIANT=1"]
    mixed_exe=out/"moss-precision-gate-fp16-f32accum.exe"
    run("mixed-compile",[llvm/"clang++.exe","-std=c++20","-O2","-static","-Wall","-Wextra","-Werror",
        *mixed_flags,"-ffunction-sections","-fdata-sections",*includes,
        root/"tests/moss_precision_gate.cpp",*objects,"-Wl,--gc-sections","-o",mixed_exe])
    for scenario in ["success",*[x+y for x in ("missing-accum","f16-accum","invalid-accum","mismatch","nan")
                                for y in ("-q4-matrix-small","-q4-matrix-large")]]:
        run("mixed-"+scenario,[mixed_exe,scenario])
    run("PS5-mixed-syntax",[llvm/"clang++.exe","--no-default-config","-target","x86_64-sie-ps5","-D__PROSPERO__=1",
        *mixed_flags,"-fvisibility-nodllstorageclass=default","-isysroot",sdk,
        "-isystem",sdk/"target/include/c++/v1","-isystem",sdk/"target/include",*includes,
        "-std=c++20","-fexceptions","-fcxx-exceptions","-Wall","-Wextra","-Werror","-fsyntax-only",root/"tests/moss_precision_gate.cpp"])
    run("PS5-syntax", [llvm / "clang++.exe", "--no-default-config", "-target", "x86_64-sie-ps5", "-D__PROSPERO__=1",
                       "-fvisibility-nodllstorageclass=default", "-isysroot", sdk,
                       "-isystem", sdk / "target/include/c++/v1", "-isystem", sdk / "target/include", *includes,
                       "-std=c++20", "-fexceptions", "-fcxx-exceptions", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", root / "tests/moss_precision_gate.cpp"])
    run("PS5-fp16-syntax", [llvm / "clang++.exe", "--no-default-config", "-target", "x86_64-sie-ps5", "-D__PROSPERO__=1",
                            "-DPS5_AI_ENABLE_FP16=1", "-fvisibility-nodllstorageclass=default", "-isysroot", sdk,
                            "-isystem", sdk / "target/include/c++/v1", "-isystem", sdk / "target/include", *includes,
                            "-std=c++20", "-fexceptions", "-fcxx-exceptions", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", root / "tests/moss_precision_gate.cpp"])
    sources = [root / "native/app/moss_precision_gate.inc",root/"native/app/moss_accumulation_policy.h",root / "tests/moss_precision_gate.cpp",
               root / "tests/moss_precision_gate.py", sd / "ggml/src/ggml-quants.c", sd / "ggml/src/ggml-quants.h"]
    receipt = {"schema": "moss-precision-fixed-oracle-host-v1", "success": True, "results": results,
               "Q4_K_shape": [3072, 17, 7], "flash_shape": {"head_dim": 128, "heads": 24, "tokens": 8},
               "Q4_K_matrix_shapes": [[3072, 17, 9], [3072, 3072, 304]],
               "Q4_K_large_compared_values": 99, "Q4_K_large_finite_checked_values": 933888,
               "Q4_K_matrix_oracle": "Pinned scalar FP32/no int-dot Q4_K decode and unchanged F32 RHS; CPU double dot; no model weights",
               "BF16_shapes": [[3072, 17, 7], [3072, 17, 9]], "BF16_oracle": "N7 direct F32 RHS; N9 pinned Vulkan BF16 RHS staging; double dot actual rounded values",
               "requested_math_modes_host_checked": ["fp32", "fp16"], "FA_accumulation": "f32 as required by actual MOSS model",
               "FP16_GPU_accuracy_proven": False, "tolerances_relaxed": False,
               "PS5_access": False, "native_GPU_execution": False, "GPU_proof": "not performed; synthetic backend tests only",
               "learned_model_inference": False, "CPU_graph_backend_linked": False,
               "sources": {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}}
    (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"success": True, "checks": len(results), "GPU_execution": False, "model_inference": False}))


if __name__ == "__main__":
    main()
