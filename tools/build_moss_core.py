from moss_kernel_policy import flags as kernel_flags, fields as kernel_fields, validate as validate_kernel, validate_core as validate_kernel_core
from moss_dit_policy import flags as dit_flags, fields as dit_fields, validate as validate_dit
from moss_lora_precision_policy import flags as lora_precision_flags, fields as lora_precision_fields, contract as lora_precision_contract, validate as validate_lora_precision, validate_core as validate_core_lora_precision
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fresh 75-member PS5 SD core build; host-only, no weights, deployment or inference.

The portable recipes were decoded with Windows CommandLineToArgvW.  --check-only
validates/materializes the plan in a fresh output folder without modifying SD.
Coordinate a frozen private SD checkout/overlay before running the real build.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from moss_feature_policy import feature_compile_flags, feature_manifest_fields, ffn_compile_flags, ffn_manifest_fields, validate_ffn_parent
import re
import shutil
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "build-profiles/moss-core.json"
MEMBER_LIST_SHA256 = "687a417c039a1c1fc610b404c2d7407214a263571a84f93cd2a87dfc4001d467"
BASE_SD = "8cd731e7b4667eb8f40a153315b9a7125d09a58d"
GGML = "d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93"
TLS_SHA256 = "ed2e06353a820c05dc3c2a6053d6d2d724af326749fd23e6905122740f1d57ee"
CODE_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".inc", ".inl", ".tcc", ".def", ".c", ".cc", ".cpp", ".cxx", ".metal", ".comp"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("Expected regular input: " + str(path))
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def inside(path, directory):
    return path == directory or directory in path.parents


def d_path(path, label):
    path = Path(path).resolve()
    if path.drive.upper() != "D:":
        raise ValueError(label + " must be on D:; no C: output/source mutation")
    return path


def moss_accumulation_compile_flag(enabled):
    if type(enabled) is not bool:raise ValueError("F32 accumulation policy must be an explicit bool")
    return "-DPS5_AI_MOSS_F32_ACCUMULATION="+str(int(enabled))


def materialize_compile_commands(rows,expand,moss_f32_accumulation=False,moss_spectrum_trial=False,moss_bounded_weight_upload_trial=False,moss_turbo_trial=False,moss_ffn_token_chunk_trial=False,moss_ffn_chunk_tokens=0,moss_lora_storage=None,moss_dit_optimizations=False,moss_dit_kernels=False):
    validate_dit(moss_dit_optimizations,accumulation=moss_f32_accumulation,turbo=moss_turbo_trial,upload=moss_bounded_weight_upload_trial,spectrum=moss_spectrum_trial,ffn=moss_ffn_token_chunk_trial,lora=moss_lora_storage)
    validate_kernel(moss_dit_kernels,dit=moss_dit_optimizations,accumulation=moss_f32_accumulation,turbo=moss_turbo_trial,upload=moss_bounded_weight_upload_trial)
    flag=moss_accumulation_compile_flag(moss_f32_accumulation)
    features=feature_compile_flags(moss_spectrum_trial,moss_bounded_weight_upload_trial,moss_turbo_trial)
    validate_ffn_parent(moss_ffn_token_chunk_trial,moss_ffn_chunk_tokens,spectrum=moss_spectrum_trial,bounded_upload=moss_bounded_weight_upload_trial,turbo=moss_turbo_trial,accumulation=moss_f32_accumulation)
    ffn=ffn_compile_flags(moss_ffn_token_chunk_trial,moss_ffn_chunk_tokens)
    validate_lora_precision(moss_lora_storage,turbo=moss_turbo_trial,upload=moss_bounded_weight_upload_trial,
        spectrum=moss_spectrum_trial,ffn=moss_ffn_token_chunk_trial,accumulation=moss_f32_accumulation,
        variant=1,precision='fp16',experimental=True,trial_10g=True)
    lora=lora_precision_flags(moss_lora_storage,core=True)
    if (moss_spectrum_trial or moss_bounded_weight_upload_trial or moss_turbo_trial) and not moss_f32_accumulation:
        raise ValueError('Isolated feature core requires explicit F32 accumulation')
    for row in rows:
        if any(any(m in arg for m in ("PS5_AI_MOSS_F32_ACCUMULATION","PS5_AI_MOSS_SPECTRUM_TRIAL","PS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL","PS5_AI_MOSS_TURBO_TRIAL","PS5_AI_MOSS_FFN_TOKEN_CHUNK_TRIAL","PS5_AI_MOSS_FFN_CHUNK_TOKENS","PS5_AI_MOSS_LORA_","PS5_AI_MOSS_DIT_")) for arg in row["argv"]):
            raise ValueError("Core recipe must not override the explicit accumulation policy")
    return [[expand(row["argv"][0]),flag,*features,*ffn,*lora,*dit_flags(moss_dit_optimizations),*kernel_flags(moss_dit_kernels),*[expand(arg) for arg in row["argv"][1:]]] for row in rows]


def validate_profile(profile):
    if profile.get("schema") != "moss-core-profile-v1" or profile.get("parser") != "Windows CommandLineToArgvW":
        raise ValueError("Unexpected core recipe schema/parser")
    rows = profile.get("members", [])
    names = [row["member"] for row in rows]
    actual = digest(("\n".join(names) + "\n").encode())
    if len(rows) != 75 or len(set(names)) != 75 or actual != MEMBER_LIST_SHA256 or profile.get("member_list_sha256") != actual:
        raise ValueError("Exact 75-member archive identity/order required")
    if sum(row["language"] == "c++" for row in rows) != 60 or sum(row["language"] == "c" for row in rows) != 15:
        raise ValueError("Expected 60 C++ and 15 third-party C recipes")
    if profile.get("stable_diffusion_base_commit") != BASE_SD or profile.get("ggml_commit_prefix") != GGML[:7] or profile.get("forced_tls_sha256") != TLS_SHA256:
        raise ValueError("Unexpected ABI/TLS profile identity")
    for row in rows:
        args = row["argv"]
        cpp = row["language"] == "c++"
        if row["member"] != Path(row["source"]).name + ".o" or not row["source"].startswith("${SD}/"):
            raise ValueError("Unsafe member/source mapping")
        expected_compiler = "${LLVM}/clang++.exe" if cpp else "${LLVM}/clang.exe"
        if args[0] != expected_compiler or "--target=x86_64-sie-ps5" not in args or "--no-default-config" not in args:
            raise ValueError("Expected PS5 cross-compiler recipe")
        if args.count("-c") != 1 or args[args.index("-c") + 1] != row["source"] or args.count("-o") != 1 or args[args.index("-o") + 1] != "${OUT}/obj/core/" + row["member"]:
            raise ValueError("Unexpected compiler input/output")
        if cpp and any(flag not in args for flag in ["-I${ROOT}/native/app", "-include${ROOT}/native/link/sd-constant-tls.hpp", "-DPS5_AI_CONTROLNET_BATCH_PAIR=0", "-DPS5_AI_MOSS_RUNTIME_ADMITTED=0"]):
            raise ValueError("Required native header/TLS/disabled-runtime flags missing")
        for arg in args:
            if not isinstance(arg, str) or re.search(r"[A-Za-z]:[/\\]", arg) or "\x00" in arg or "\n" in arg:
                raise ValueError("Nonportable/unsafe recipe argument")
            if any(key not in {"ROOT", "SD", "SDK", "LLVM", "OUT"} for key in re.findall(r"\$\{([^}]+)\}", arg)):
                raise ValueError("Unknown profile substitution")
            if "PS5_AI_MOSS_RUNTIME_ADMITTED" in arg and arg != "-DPS5_AI_MOSS_RUNTIME_ADMITTED=0":
                raise ValueError("Runtime admission must stay disabled")
            if "PS5_AI_CONTROLNET_BATCH_PAIR" in arg and arg != "-DPS5_AI_CONTROLNET_BATCH_PAIR=0":
                raise ValueError("ControlNet pair mode must stay disabled")
    headers = profile.get("generated_headers", [])
    if len(headers) != 1 or headers[0]["relative_path"] != "generated/oniguruma/config.h" or digest(headers[0]["text"].encode()) != headers[0]["sha256"]:
        raise ValueError("Expected pinned generated oniguruma header")
    return rows


def checkout_head(folder):
    """Read checkout identity only: never invoke git or change its index/refs."""
    git = folder / ".git"
    if not git.is_dir():
        raise ValueError("Expected private standalone checkout metadata: " + str(folder))
    value = (git / "HEAD").read_text().strip()
    if value.startswith("ref: "):
        ref = value[5:]
        if ".." in ref or not ref.startswith("refs/"):
            raise ValueError("Invalid checkout ref")
        if (git / ref).is_file():
            value = (git / ref).read_text().strip()
        else:
            matches = [line.split()[0] for line in (git / "packed-refs").read_text().splitlines() if line.endswith(" " + ref)]
            if len(matches) != 1:
                raise ValueError("Checkout ref is unresolved")
            value = matches[0]
    if not re.fullmatch("[0-9a-f]{40}", value):
        raise ValueError("Invalid checkout commit")
    return value


def materialize_overlay(sd):
    overlay = ROOT / "patches/stable-diffusion/overlay"
    source_records = []
    for src in sorted(overlay.rglob("*")):
        if not src.is_file():
            continue
        if src.is_symlink():
            raise ValueError("Overlay links are not accepted")
        target = sd / src.relative_to(overlay)
        if not inside(target.resolve(), sd) or target.is_symlink():
            raise ValueError("Overlay destination escaped private SD")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        original = record(src)
        if record(target)["sha256"] != original["sha256"]:
            raise ValueError("Materialized overlay mismatch")
        source_records.append(original)
    if not source_records:
        raise ValueError("Overlay is empty")
    return source_records


def header_snapshot(sd, sdk, llvm, out, sources):
    inputs = set()
    # Superset of actual dependencies: included .cpp/.inc and all system headers
    # are covered even if they are not among the 75 compilation units.
    for folder in [sd / "src", sd / "include", sd / "thirdparty", sd / "ggml/include", sd / "ggml/src", ROOT / "native"]:
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in CODE_SUFFIXES:
                inputs.add(path.resolve())
    resources = list((llvm.parent / "lib/clang").glob("*/include"))
    if len(resources) != 1:
        raise ValueError("Expected one LLVM resource-header tree")
    for folder in [sdk / "target/include", resources[0], out / "generated"]:
        if not folder.is_dir():
            raise ValueError("Missing header tree: " + str(folder))
        inputs.update(path.resolve() for path in folder.rglob("*") if path.is_file())
    inputs.difference_update(Path(row["path"]) for row in sources)
    return [record(path) for path in sorted(inputs)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--source-freeze", type=Path, help="engine source-freeze receipt; required for actual compilation")
    parser.add_argument("--check-only", action="store_true", help="plan/guards only; no vendor mutation or compilation")
    parser.add_argument("--moss-f32-accumulation",action="store_true",help="Explicit MOSS Linear F32 accumulation; preserved in all75 actual compile commands")
    parser.add_argument('--moss-spectrum-trial',action='store_true')
    parser.add_argument('--moss-bounded-weight-upload-trial',action='store_true')
    parser.add_argument('--moss-turbo-trial',action='store_true')
    parser.add_argument('--moss-ffn-token-chunk-trial',action='store_true')
    parser.add_argument('--moss-ffn-chunk-tokens',type=int,choices=(0,2048),default=0)
    parser.add_argument('--moss-lora-storage',choices=('f32','f16'))
    parser.add_argument('--moss-dit-optimizations',action='store_true')
    parser.add_argument('--moss-dit-kernels',action='store_true')
    args = parser.parse_args()
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    rows = validate_profile(profile)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sd = d_path(config["SD"], "Private SD vendor")
    out = d_path(args.out, "Build output")
    sdk = Path(config["SDK"]).resolve(strict=True)
    llvm = Path(config["LLVM"]).resolve(strict=True)
    if inside(sd, ROOT) or any(inside(out, directory) or inside(directory, out) for directory in [ROOT, sd, sdk, llvm]) or any(part.lower() in {"anima", "ps5-ai-anima", "ps5-ai-wai"} for part in out.parts):
        raise ValueError("Use isolated private MOSS vendor and separate fresh build output")
    if out.exists():
        raise ValueError("Output already exists; completed/partial evidence is preserved")
    actual_sd = checkout_head(sd)
    actual_ggml = checkout_head(sd / "ggml")
    if actual_sd != BASE_SD or actual_ggml != GGML:
        raise ValueError("Wrong SD/GGML baseline: a scoped MOSS backport on the pinned ABI is required")
    if record(ROOT / "native/link/sd-constant-tls.hpp")["sha256"] != TLS_SHA256:
        raise ValueError("Native constant TLS header no longer matches the original recipe")
    admission = (ROOT / "native/app/moss_runtime_admission.h").read_text()
    if not re.search(r"#define\s+PS5_AI_MOSS_RUNTIME_ADMITTED\s+0\b", admission):
        raise ValueError("MOSS runtime admission default must remain 0")
    replacements = {"ROOT": ROOT.as_posix(), "SD": sd.as_posix(), "SDK": sdk.as_posix(), "LLVM": llvm.as_posix(), "OUT": out.as_posix()}
    def expand(arg):
        return re.sub(r"\$\{([^}]+)\}", lambda match: replacements[match[1]], arg)
    commands = materialize_compile_commands(rows,expand,args.moss_f32_accumulation,args.moss_spectrum_trial,args.moss_bounded_weight_upload_trial,args.moss_turbo_trial,args.moss_ffn_token_chunk_trial,args.moss_ffn_chunk_tokens,args.moss_lora_storage,args.moss_dit_optimizations,args.moss_dit_kernels)
    for row, command in zip(rows, commands):
        source = Path(expand(row["source"])).resolve(strict=True)
        if not inside(source, sd) or source.drive.upper() != "D:":
            raise ValueError("All 75 source inputs must come from the private D: vendor")
        if not Path(command[0]).is_file():
            raise ValueError("Missing compiler")
    out.mkdir(parents=True)
    for child in ["obj/core", "deps", "logs", "tmp"]:
        (out / child).mkdir(parents=True)
    for header in profile["generated_headers"]:
        target = out / header["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(header["text"], encoding="utf-8", newline="\n")
    manifest = {"schema": "moss-core-build-v1", "state": "checked_plan_only" if args.check_only else "building", "local_build_only": True, "console_executed": False, "PS5_connections": 0, "model_weights_loaded": False, "inference_performed": False, "GPU_used": False, "PNG_generated": False, "MOSS_RUNTIME_ADMITTED": 0, "jobs": args.jobs, "profile": record(PROFILE), "builder": record(__file__), "source_pins": {"stable_diffusion_base": actual_sd, "ggml": actual_ggml, "moss_backport_reference": profile["moss_upstream_commit"]}, "config_paths": {key: replacements[key] for key in ["SD", "SDK", "LLVM"]}, "commands": commands, "member_names": [row["member"] for row in rows], "members": [], "sources": [], "header_inputs": [], "archive": None, "warnings": [], "hash_inputs_stable": False}
    manifest.update({"moss_f32_accumulation":args.moss_f32_accumulation,
        "moss_accumulation_policy":"force_f32" if args.moss_f32_accumulation else "model_default",
        "moss_accumulation_compile_flag":moss_accumulation_compile_flag(args.moss_f32_accumulation)})
    manifest.update(dit_fields(args.moss_dit_optimizations))
    manifest.update(kernel_fields(args.moss_dit_kernels))
    manifest.update(feature_manifest_fields(args.moss_spectrum_trial,args.moss_bounded_weight_upload_trial,args.moss_turbo_trial))
    manifest.update(ffn_manifest_fields(args.moss_ffn_token_chunk_trial,args.moss_ffn_chunk_tokens))
    manifest['policy_inputs']=[record(ROOT/'tools/moss_feature_policy.py')]
    manifest['kernel_policy_input']=record(ROOT/'tools/moss_kernel_policy.py')
    manifest['dit_policy_input']=record(ROOT/'tools/moss_dit_policy.py')
    manifest.update(lora_precision_fields(args.moss_lora_storage,core=True))
    manifest['lora_precision_policy_inputs']=[record(ROOT/'tools/moss_lora_precision_policy.py')]
    target_manifest = out / "manifest.private.json"
    write_json(target_manifest, manifest)
    if args.check_only:
        print(json.dumps({"state": manifest["state"], "members": 75, "C": 15, "CXX": 60, "vendor_modified": False, "compiled": False, "out": out.as_posix()}))
        return
    if not args.source_freeze:
        raise ValueError("Actual compilation requires the coordinated engine --source-freeze receipt")
    freeze = json.loads(args.source_freeze.read_text(encoding="utf-8"))
    if freeze.get("schema") != "ps5-ai-moss-engine-source-freeze-v1" or freeze.get("frozen") is not True or freeze.get("runtime_admitted") is not False or freeze.get("upstream_base_revision") != BASE_SD or freeze.get("ggml_revision") != GGML or Path(freeze["repository_root"]).resolve() != ROOT or Path(freeze["private_vendor_root"]).resolve() != sd:
        raise ValueError("Source-freeze receipt does not bind this MOSS repository/vendor/ABI")
    for key,value in feature_manifest_fields(args.moss_spectrum_trial,args.moss_bounded_weight_upload_trial,args.moss_turbo_trial).items():
        if key in freeze and (type(freeze[key])is not type(value) or freeze[key]!=value):
            raise ValueError('Source freeze feature policy mismatch: '+key)
        if (args.moss_spectrum_trial or args.moss_bounded_weight_upload_trial or args.moss_turbo_trial) and key not in freeze:
            raise ValueError('Feature source freeze lacks explicit policy: '+key)
    if (args.moss_spectrum_trial or args.moss_bounded_weight_upload_trial or args.moss_turbo_trial) and (
        freeze.get('new_core_compilation_required')is not True or freeze.get('old_core_reuse_allowed')is not False):
        raise ValueError('Feature source freeze must require fresh75 and forbid old core reuse')
    for key,value in ffn_manifest_fields(args.moss_ffn_token_chunk_trial,args.moss_ffn_chunk_tokens).items():
        if (args.moss_ffn_token_chunk_trial or key in freeze)and(type(freeze.get(key))is not type(value)or freeze.get(key)!=value):raise ValueError('Source-freeze FFN identity mismatch: '+key)
    for key,value in lora_precision_fields(args.moss_lora_storage,core=True).items():
        if (args.moss_lora_storage is not None or key in freeze) and (type(freeze.get(key))is not type(value) or freeze[key]!=value):
            raise ValueError('Source-freeze private LoRA precision identity mismatch: '+key)
    for key,value in dit_fields(args.moss_dit_optimizations).items():
        if (args.moss_dit_optimizations or key in freeze) and (type(freeze.get(key))is not type(value)or freeze[key]!=value):raise ValueError('DiT source-freeze policy mismatch: '+key)
    for key,value in kernel_fields(args.moss_dit_kernels).items():
        if (args.moss_dit_kernels or key in freeze) and (type(freeze.get(key))is not type(value)or freeze[key]!=value):raise ValueError("Kernel source-freeze mismatch: "+key)
    frozen_inputs = []
    for relative, expected in freeze["sources"].items():
        path = (ROOT / relative).resolve(strict=True)
        if not inside(path, ROOT):
            raise ValueError("Frozen source escaped repository")
        observed = record(path)
        if observed["bytes"] != expected["bytes"] or observed["sha256"] != expected["sha256"]:
            raise ValueError("Frozen engine source mismatch: " + relative)
        frozen_inputs.append(observed)
    manifest["source_freeze"] = record(args.source_freeze)
    manifest["frozen_inputs"] = frozen_inputs
    started = time.monotonic()
    env = dict(os.environ)
    env["TEMP"] = env["TMP"] = (out / "tmp").as_posix()
    env["PATH"] = str(llvm) + os.pathsep + env.get("PATH", "")
    try:
        manifest["overlay_inputs"] = materialize_overlay(sd)
        manifest["sources"] = [record(expand(row["source"])) for row in rows]
        manifest["header_inputs"] = header_snapshot(sd, sdk, llvm, out, manifest["sources"])
        manifest["tool_inputs"] = [record(llvm / name) for name in ["clang.exe", "clang++.exe", "llvm-ar.exe"]]
        write_json(target_manifest, manifest)
        def compile_one(index):
            row = rows[index]
            before = time.monotonic()
            process = subprocess.run(commands[index], cwd=sd, env=env, capture_output=True)
            log = out / "logs" / (row["member"] + ".log")
            log.write_bytes(process.stdout + process.stderr)
            if process.returncode:
                raise RuntimeError(row["member"] + " compiler exit " + str(process.returncode) + "; see " + str(log))
            obj = out / "obj/core" / row["member"]
            with obj.open("rb") as stream:
                elf = stream.read(20)
            if len(elf) != 20 or elf[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<HH", elf, 16) != (1, 62):
                raise ValueError("Compiler did not produce an x86_64 ELF relocatable: " + row["member"])
            dependency = out / "deps" / (row["member"] + ".d")
            if not dependency.is_file() or not dependency.stat().st_size:
                raise ValueError("Missing compiler dependency record")
            return {**record(obj), "member": row["member"], "source": manifest["sources"][index]["path"], "command_index": index, "compile_seconds": round(time.monotonic() - before, 3), "log": record(log), "dependencies": record(dependency)}, process.stderr.decode("utf-8", errors="replace")
        completed = {}
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(compile_one, i): i for i in range(len(rows))}
            for future in as_completed(futures):
                index = futures[future]
                result, stderr = future.result()
                completed[index] = result
                if stderr.strip():
                    manifest["warnings"].append({"member": result["member"], "log": result["log"]})
                print(json.dumps({"compiled": len(completed), "total": 75, "member": result["member"]}), flush=True)
        manifest["members"] = [completed[i] for i in range(75)]
        after_sources = [record(row["path"]) for row in manifest["sources"]]
        after_headers = header_snapshot(sd, sdk, llvm, out, after_sources)
        after_overlay = [record(row["path"]) for row in manifest["overlay_inputs"]]
        if after_sources != manifest["sources"] or after_headers != manifest["header_inputs"] or after_overlay != manifest["overlay_inputs"] or record(PROFILE) != manifest["profile"] or record(__file__) != manifest["builder"] or record(args.source_freeze) != manifest["source_freeze"] or [record(item["path"]) for item in frozen_inputs] != frozen_inputs or [record(item["path"]) for item in manifest["tool_inputs"]] != manifest["tool_inputs"] or [record(item["path"]) for item in manifest["policy_inputs"]] != manifest["policy_inputs"] or [record(item['path']) for item in manifest['lora_precision_policy_inputs']] != manifest['lora_precision_policy_inputs']:
            raise ValueError("Source/header/overlay/profile changed during compilation; archive not admitted")
        if record(manifest["dit_policy_input"]["path"])!=manifest["dit_policy_input"]:raise ValueError("DiT policy changed during compilation")
        if record(manifest['kernel_policy_input']['path'])!=manifest['kernel_policy_input']:raise ValueError('Kernel policy changed during compilation')
        manifest["hash_inputs_stable"] = True
        archive = out / "libstable-diffusion.moss-core.a"
        archive_command = [str(llvm / "llvm-ar.exe"), "rcsD", str(archive), *[row["path"] for row in manifest["members"]]]
        manifest["archive_command"] = archive_command
        subprocess.run(archive_command, cwd=out, env=env, check=True, capture_output=True)
        list_command = [str(llvm / "llvm-ar.exe"), "t", str(archive)]
        members = subprocess.check_output(list_command, cwd=out, env=env, text=True).splitlines()
        if members != manifest["member_names"]:
            raise ValueError("Final archive member names/order differ from the pinned 75-member oracle")
        manifest["archive"] = record(archive)
        manifest["state"] = "crossbuilt_not_PS5_executed"
        manifest["freshly_compiled_members"] = 75
        manifest["old_archive_members_reused"] = 0
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(target_manifest, manifest)
        print(json.dumps({"state": manifest["state"], "archive": manifest["archive"], "members": 75, "warnings": len(manifest["warnings"]), "seconds": manifest["elapsed_seconds"], "PS5_connections": 0}))
    except Exception as error:
        manifest["state"] = "failed_not_admitted"
        manifest["error"] = str(error)
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(target_manifest, manifest)
        raise


if __name__ == "__main__":
    main()
