# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed GPU-profile build policy and actual recipe emission; no compile/GPU."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MACRO = "PS5_AI_MOSS_GPU_PROFILE"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_command_emissions(builder_source, profile, trial_flags,
                             accumulation_flags, gpu_profile_flags, features=(False,False,False)):
    """Execute only actual main's list/loop expressions with a recording run().

    No subprocess, filesystem build operation, inference or modified model is
    involved. This uses the real command-construction AST rather than duplicating
    the native source mapping in a second implementation.
    """
    from moss_feature_policy import feature_compile_flags, ffn_compile_flags
    from moss_lora_precision_policy import flags as lora_flags
    tree = ast.parse(builder_source)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    nodes = list(ast.walk(main))
    common = [node for node in nodes if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "common"
                      for target in node.targets)]
    native_loop = [node for node in nodes if isinstance(node, ast.For)
                   and isinstance(node.target, ast.Name) and node.target.id == "name"
                   and "objects" in ast.dump(node.iter)]
    ggml_loop = [node for node in nodes if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == "recipe"
                 and isinstance(node.iter, ast.Name) and node.iter.id == "ggml_recipes"]
    platform = [node for node in nodes if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "run"
                and "platform_compile" in ast.dump(node)]
    if tuple(map(len, (common, native_loop, ggml_loop, platform))) != (1, 1, 1, 1):
        raise AssertionError("Actual native command construction changed; review extraction")
    selected = ast.Module(body=[copy.deepcopy(common[0]),
                               copy.deepcopy(native_loop[0]),
                               ast.Expr(value=copy.deepcopy(platform[0])),
                               copy.deepcopy(ggml_loop[0])], type_ignores=[])
    ast.fix_missing_locations(selected)
    commands = []
    scope = dict(Path=Path, profile=copy.deepcopy(profile),
                 trial_flags=trial_flags, accumulation_flags=accumulation_flags,
                 gpu_profile_flags=gpu_profile_flags,
                 q4_candidate_flags=["-DPS5_AI_MOSS_Q4_BK64_RAW64=0"],
                 sampler_flags=["-DPS5_AI_MOSS_SAMPLER_TRIAL=0"],
                 feature_flags=feature_compile_flags(*features),
                 ffn_flags=ffn_compile_flags(False,0),lora_flags=lora_flags(None),
                 ggml_recipes=copy.deepcopy(profile["ggml_compile"]),
                 ggml_objects=[], a=SimpleNamespace(precision="fp16"),
                 run=lambda command: commands.append(list(command)),
                 print=lambda *args, **kwargs: None)
    exec(compile(selected, "<record-only-native-command-expressions>", "exec"), scope)
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.drive.upper() != "D:" or out.exists() or out.is_relative_to(ROOT):
        raise RuntimeError("Fresh D output outside repository required")
    out.mkdir(parents=True)
    sys.path.insert(0, str(ROOT / "tools"))
    builder = load_module("moss_gpu_profile_native_build", ROOT / "tools/build.py")
    core = load_module("moss_gpu_profile_core_build", ROOT / "tools/build_moss_core.py")
    paths = [ROOT / "tools/build.py", ROOT / "tools/build_moss_core.py",
             ROOT / "tests/moss_gpu_profile_build.py", ROOT / "build-profiles/native.json",
             ROOT / "build-profiles/moss-core.json"]
    profile_paths = [ROOT / ("build-profiles/moss-performance-" + name + ".json")
                     for name in ("baseline-fp32", "fp16", "submit-fusion-fp32",
                                  "fp16-f32accum", "fp16-f32accum-gpu-profile")]
    paths += profile_paths
    before = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    checks = []

    def check(ok, label):
        if not ok:
            raise AssertionError(label)
        checks.append(label)

    def reject(function, label):
        try:
            function()
        except (RuntimeError, ValueError):
            checks.append("reject " + label)
            return
        raise AssertionError("Accepted " + label)

    profile = json.loads(profile_paths[-1].read_text(encoding="utf-8"))
    mixed = json.loads(profile_paths[-2].read_text(encoding="utf-8"))
    expected = builder.moss_performance_contract(1, True, True)
    check(set(profile) == set(expected)
          and all(builder.same_typed_contract_value(profile[key], value)
                  for key, value in expected.items()), "exact typed GPU-profile contract")
    check(set(profile) - set(mixed) == {"moss_gpu_profile"}
          and all(profile[key] == value for key, value in mixed.items()),
          "new contract adds only explicit GPU profiler opt-in")
    builder.validate_moss_gpu_profile_mode(True, True, 1, "fp16", True, True)
    check(True, "explicit variant1 FP16 F32-accumulation trial10GiB accepted")
    trial_flags = builder.moss_trial_flags(True, True, profile, False, False, 1, True, True)
    check(trial_flags == builder.moss_trial_flags(True, True, mixed, False, False, 1, True),
          "profiling flag remains separate from unchanged trial flags")
    for enabled in (False, True):
        check(builder.moss_gpu_profile_compile_flags(enabled)
              == ["-D" + MACRO + "=" + str(int(enabled))],
              "single exact native profiler macro " + str(enabled))
    for invalid in (0, 1, None, 0.0, 1.0, "0", "1", [], {}):
        reject(lambda invalid=invalid: builder.moss_gpu_profile_compile_flags(invalid),
               "non-bool profiler macro " + repr(invalid))
        reject(lambda invalid=invalid: builder.moss_performance_contract(1, True, invalid),
               "non-bool profile contract " + repr(invalid))
        reject(lambda invalid=invalid: builder.validate_moss_gpu_profile_mode(
            invalid, True, 1, "fp16", True, True), "non-bool enabled mode " + repr(invalid))
    invalid_modes = [
        (True, False, 1, "fp16", True, True), (True, 1, 1, "fp16", True, True),
        (True, None, 1, "fp16", True, True), (True, True, 0, "fp32", True, True),
        (True, True, 2, "fp32", True, True), (True, True, None, "fp16", True, True),
        (True, True, True, "fp16", True, True), (True, True, 1.0, "fp16", True, True),
        (True, True, "1", "fp16", True, True), (True, True, 1, "fp32", True, True),
        (True, True, 1, "FP16", True, True), (True, True, 1, "fp16", False, True),
        (True, True, 1, "fp16", 1, True), (True, True, 1, "fp16", True, False),
        (True, True, 1, "fp16", True, 1),
    ]
    for mode in invalid_modes:
        reject(lambda mode=mode: builder.validate_moss_gpu_profile_mode(*mode),
               "invalid mode " + repr(mode))
    for variant in (0, 2, -1, 3, True, 1.0, "1", None):
        reject(lambda variant=variant: builder.moss_performance_contract(variant, True, True),
               "noncandidate contract variant " + repr(variant))
    for key in profile:
        changed = copy.deepcopy(profile)
        del changed[key]
        reject(lambda changed=changed: builder.moss_trial_flags(
            True, True, changed, False, False, 1, True, True), "missing contract field " + key)
    changes = [
        ("moss_gpu_profile", False), ("moss_gpu_profile", 1), ("moss_gpu_profile", "true"),
        ("moss_f32_accumulation", False), ("moss_f32_accumulation", 1),
        ("performance_variant", True), ("performance_variant", 1.0),
        ("requested_precision", "fp32"), ("max_generation_attempts", 5),
        ("max_generation_attempts", 4.0), ("steps", 49), ("steps", 50.0),
        ("gpu_pool_bytes", 12884901888), ("production_runtime_admitted", True),
        ("allowed_sizes", [[256, 256], [512, 512]]),
        ("allowed_sizes", [[256.0, 256], [1024, 1024]]),
        ("automatic_POST_retries", 0), ("unknown", True),
    ]
    for key, value in changes:
        changed = copy.deepcopy(profile)
        changed[key] = value
        reject(lambda changed=changed: builder.moss_trial_flags(
            True, True, changed, False, False, 1, True, True),
            "changed contract field " + key + " " + repr(value))
    for enabled, trial, quality, benchmark, variant, accumulation, gpu in [
        (False, True, False, False, 1, True, True),
        (True, False, False, False, 1, True, True),
        (True, True, True, False, 1, True, True),
        (True, True, False, True, 1, True, True),
        (True, True, False, False, 1, False, True),
        (True, True, False, False, 1, 1, True),
        (True, True, False, False, 1, True, 1),
        (True, True, False, False, 1, True, False),
    ]:
        reject(lambda enabled=enabled, trial=trial, quality=quality, benchmark=benchmark,
                      variant=variant, accumulation=accumulation, gpu=gpu:
               builder.moss_trial_flags(enabled, trial, profile, quality, benchmark,
                                        variant, accumulation, gpu),
               "incompatible typed trial flags " + repr((enabled, trial, quality, benchmark,
                                                        variant, accumulation, gpu)))
    for variant, (name, precision) in builder.MOSS_PERFORMANCE_VARIANTS.items():
        old = json.loads((ROOT / ("build-profiles/moss-performance-" + name + ".json")).read_text())
        check(old == builder.moss_performance_contract(variant)
              and "moss_gpu_profile" not in old, "old profile unchanged " + name)
        builder.moss_trial_flags(True, True, old, False, False, variant)
        check(True, "old profile still admitted with profiler default0 " + name)
        reject(lambda old=old, variant=variant: builder.moss_trial_flags(
            True, True, old, False, False, variant, variant == 1, True),
            "old profile cannot silently opt in " + name)
    check(mixed == builder.moss_performance_contract(1, True)
          and "moss_gpu_profile" not in mixed, "validated mixed contract remains unchanged")
    reject(lambda: builder.moss_trial_flags(True, True, mixed, False, False, 1, True, True),
           "mixed winner cannot impersonate profiler candidate")
    check(builder.moss_trial_flags(False, False)
          == ["-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=0", "-DPS5_AI_MOSS_TRIAL_10G=0"],
          "production trial admission remains closed")

    native_profile = json.loads((ROOT / "build-profiles/native.json").read_text())
    builder.validate_moss_gpu_profile_recipes(native_profile)
    check(True, "actual native platform/GGML recipes contain no policy override")
    for target in ("platform", "ggml"):
        for override in ("-D" + MACRO + "=1", "-D" + MACRO + "=0", "-U" + MACRO,
                         "-D", MACRO + "=1", "incidental-path/" + MACRO + "/header"):
            changed = copy.deepcopy(native_profile)
            recipe = changed["platform_compile"] if target == "platform" else changed["ggml_compile"][0]
            recipe.append(override)
            if override == "-D":
                recipe.append(MACRO + "=1")
            reject(lambda changed=changed: builder.validate_moss_gpu_profile_recipes(changed),
                   "recipe cannot override profiler " + target + " " + override)
    for key in ("platform_compile", "ggml_compile"):
        for value in (None, [], "not-a-recipe", 1, False):
            changed = copy.deepcopy(native_profile)
            changed[key] = value
            reject(lambda changed=changed: builder.validate_moss_gpu_profile_recipes(changed),
                   "missing/malformed compiler recipes " + key + " " + repr(value))
        changed = copy.deepcopy(native_profile)
        del changed[key]
        reject(lambda changed=changed: builder.validate_moss_gpu_profile_recipes(changed),
               "missing compiler recipe key " + key)
    for target in ("platform", "ggml"):
        for bad_arg in (None, 1, True, []):
            changed = copy.deepcopy(native_profile)
            recipe = changed["platform_compile"] if target == "platform" else changed["ggml_compile"][0]
            recipe.append(bad_arg)
            reject(lambda changed=changed: builder.validate_moss_gpu_profile_recipes(changed),
                   "nonstr compiler argument " + target + " " + repr(bad_arg))
    for bad_recipe in (None, [], 1, "not-a-recipe"):
        changed = copy.deepcopy(native_profile)
        changed["ggml_compile"] = [bad_recipe]
        reject(lambda changed=changed: builder.validate_moss_gpu_profile_recipes(changed),
               "malformed individual GGML recipe " + repr(bad_recipe))
    main_ast = next(node for node in ast.parse((ROOT / "tools/build.py").read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
    recipe_guards = [node for node in ast.walk(main_ast) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "validate_moss_gpu_profile_recipes"]
    output_mkdirs = [node for node in ast.walk(main_ast) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "mkdir"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "out"]
    check(len(recipe_guards) == 1 and bool(output_mkdirs)
          and recipe_guards[0].lineno < min(node.lineno for node in output_mkdirs),
          "recipe override admission precedes output creation and compiler invocation")
    emissions = {}
    for enabled, contract in ((False, mixed), (True, profile)):
        commands = native_command_emissions((ROOT / "tools/build.py").read_text(), native_profile,
            builder.moss_trial_flags(True, True, contract, False, False, 1, True, enabled),
            builder.moss_accumulation_compile_flags(True), builder.moss_gpu_profile_compile_flags(enabled))
        objects = {Path(command[command.index("-o") + 1]).name for command in commands}
        check(objects == set(native_profile["objects"]) | {
            "radv_ps5_platform_native.o", "ggml-vulkan.cpp.o"},
            "actual AST emits every native/platform/GGML recipe object " + str(enabled))
        check(len(commands) == len(objects), "actual command outputs unique " + str(enabled))
        for command in commands:
            check([arg for arg in command if MACRO in arg]
                  == builder.moss_gpu_profile_compile_flags(enabled),
                  "single actual emitted macro " + str(enabled) + " "
                  + Path(command[command.index("-o") + 1]).name)
        emissions[str(enabled)] = commands
    combined=builder.moss_performance_contract(1,True,moss_turbo_trial=True,moss_bounded_weight_upload_trial=True)
    diagnostic=builder.validate_dit_profile(True,mask=15,legacy_gpu_profile=False,accumulation=True,turbo=True,upload=True)
    commands=native_command_emissions((ROOT/'tools/build.py').read_text(),native_profile,
        builder.moss_trial_flags(True,True,combined,False,False,1,True,False,False,False,False,True,True)
        +builder.dit_flags(True)+['-DPS5_AI_MOSS_DIT_OPT_MASK=15'],
        builder.moss_accumulation_compile_flags(True),builder.moss_gpu_profile_compile_flags(True)+diagnostic,
        features=(False,True,True))
    for command in commands:
        check(command.count('-DPS5_AI_MOSS_GPU_PROFILE=1')==command.count('-DPS5_AI_MOSS_DIT_GPU_PROFILE=1')==1
              and command.count('-DPS5_AI_MOSS_DIT_OPT_MASK=15')==1
              and '-DPS5_AI_MOSS_TURBO_TRIAL=1'in command
              and '-DPS5_AI_MOSS_BOUNDED_WEIGHT_UPLOAD_TRIAL=1'in command,
              'actual native/platform/backend command records exact mask15 diagnostic parent '+Path(command[command.index('-o')+1]).name)
    emissions['mask15_diagnostic']=commands
    rows = core.validate_profile(json.loads((ROOT / "build-profiles/moss-core.json").read_text()))
    parameters = dict(ROOT=ROOT.as_posix(), SD="D:/metadata-only/moss", SDK="D:/metadata-only/sdk",
                      LLVM="D:/metadata-only/llvm", OUT=out.as_posix())

    def expand(value):
        for key, replacement in parameters.items():
            value = value.replace("${" + key + "}", replacement)
        return value

    core_commands = core.materialize_compile_commands(rows, expand, True)
    check(len(core_commands) == 75, "all75 actual core recipe commands materialized")
    for index, command in enumerate(core_commands):
        check(not any(MACRO in arg for arg in command)
              and command.count("-DPS5_AI_MOSS_F32_ACCUMULATION=1") == 1,
              "core command stays F32acc1 with no native profiler macro " + str(index))
    help_results = []
    for path in (ROOT / "tools/build.py", ROOT / "tools/build_moss_core.py"):
        result = subprocess.run([sys.executable, "-B", str(path), "--help"],
                                capture_output=True, text=True)
        check(result.returncode == 0, "actual CLI help only " + path.name)
        check(("--moss-gpu-profile" in result.stdout) == (path.name == "build.py"),
              "profiler CLI exposed only on native builder " + path.name)
        (out / (path.name + ".help.log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        help_results.append(dict(path=str(path), returncode=result.returncode))
    after = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    check(before == after, "all inspected policy/profile/test sources unchanged during run")
    receipt = dict(schema="moss-gpu-profile-build-host-v1", success=True,
                   count=len(checks), checks=checks,
                   native_commands_recorded_from_actual_main_AST=emissions,
                   all75_core_recipe_commands_materialized=core_commands,
                   source_sha256=after, CLI_help_only=help_results,
                   compiler_executed=False, PS5_connections=0, GPU_calls=0,
                   model_inference=False, native_or_model_build_claimed=False)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in (
        "success", "count", "compiler_executed", "PS5_connections", "GPU_calls", "model_inference")}))


if __name__ == "__main__":
    main()
