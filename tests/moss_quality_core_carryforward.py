# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate unchanged compiled core dependencies for a native-only quality trial.

Reads the original core manifest/archive and pinned compiler .d files. Synthetic
negative manifests/dependency copies go only to fresh output; no original build
file, model, PS5 state or old receipt is changed.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--benchmark-1024', action='store_true', help='Permit only the reviewed gateway wait change, still requiring all actual core dependencies unchanged')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = a.out.resolve()
    if out.exists() or out.is_relative_to(root) or out.drive.upper() != 'D:':
        raise RuntimeError('Choose fresh output on D: outside the repository')
    out.mkdir(parents=True)
    sys.path.insert(0, str(root/'tools'))
    spec = importlib.util.spec_from_file_location('moss_build', root/'tools/build.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    config = json.loads(a.config.read_text())
    original = json.loads(a.manifest.read_text())
    original_hash = build.digest(a.manifest)
    baseline = build.archive_members(Path(config['DEPS'])/'libstable-diffusion.graph-admission-1024-v3.a')
    checks = []

    def refuse(path, mode, label):
        try:
            build.validate_moss_core(path, baseline, config, mode, benchmark_gateway_only=a.benchmark_1024)
        except RuntimeError:
            checks.append(label)
            return
        raise AssertionError('Unexpected acceptance: '+label)

    refuse(a.manifest, False, 'strict original superset detects changed native inputs')
    refuse(a.manifest, 1, 'integer cannot impersonate explicit quality admission')
    carry = {}
    build.validate_moss_core(a.manifest, baseline, config, True, carry, a.benchmark_1024)
    assert carry['actual_dependency_files_verified'] == 75
    assert carry['original_manifest_sha256'] == original_hash
    assert carry['original_manifest_unchanged'] and not carry['new_core_compilation']
    checks.append('all75 pinned compiler dependencies exclude the changed native-only inputs')
    if a.benchmark_1024:
        assert carry['reviewed_gateway_wait_only_change']['all_other_bytes_identical']
        checks.append('gateway wait literal restoration exactly matches original core superset bytes')
        gateway_record=next(row for row in original['header_inputs'] if row['path'].endswith('/native/gateway/openai.inc'))
        altered=out/'gateway-unreviewed.inc'
        altered.write_bytes((root/'native/gateway/openai.inc').read_bytes()+b'\n/* unreviewed */\n')
        try:build.validate_moss_benchmark_gateway_wait_change(altered,gateway_record)
        except RuntimeError:checks.append('any extra gateway byte change rejected')
        else:raise AssertionError('Unreviewed gateway byte change accepted')

    # Copy only small dependency text, never archive/object/model payloads.
    clone = copy.deepcopy(original)
    deps = out/'deps'
    deps.mkdir()
    for member in clone['members']:
        old = Path(member['dependencies']['path'])
        new = deps/old.name
        shutil.copyfile(old, new)
        member['dependencies']['path'] = str(new)
    clone_path = out/'clone.json'

    def save(value):
        clone_path.write_text(json.dumps(value, indent=2)+'\n')

    bad = copy.deepcopy(clone)
    bad['members'] = bad['members'][:-1]
    save(bad)
    refuse(clone_path, True, 'missing actual dependency record rejected')

    bad = copy.deepcopy(clone)
    bad['members'][0]['dependencies']['sha256'] = '0'*64
    save(bad)
    refuse(clone_path, True, 'changed compiler dependency text rejected')

    bad = copy.deepcopy(clone)
    record = next(row for row in bad['header_inputs'] if row['path'].endswith('/native/app/phase_profile.h'))
    record['sha256'] = '0'*64
    save(bad)
    refuse(clone_path, True, 'changed consumed native header rejected')

    first = clone['members'][0]
    dep = Path(first['dependencies']['path'])
    original_dep = dep.read_bytes()
    dep.write_bytes(original_dep+b' '+str(root/'native/app/moss_trial_contract.hpp').replace('\\','/').encode()+b'\n')
    first['dependencies']['bytes'] = dep.stat().st_size
    first['dependencies']['sha256'] = build.digest(dep)
    save(clone)
    refuse(clone_path, True, 'even allowed native file is refused if a compiler actually consumed it')

    if build.digest(a.manifest) != original_hash:
        raise AssertionError('Original manifest changed')
    receipt = {'success': True, 'checks': checks, 'count': len(checks),
               'carryforward': carry, 'old_manifest_rewritten': False,
               'GPU_calls': 0, 'PS5_connections': 0, 'model_inference': False,
               'source_sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [root/'tools/build.py', Path(__file__).resolve()]}}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'success': True, 'count': len(checks),
                      'actual_dependency_files_verified': 75, 'old_manifest_rewritten': False}))


if __name__ == '__main__':
    main()
