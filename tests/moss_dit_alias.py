# SPDX-License-Identifier: GPL-3.0-or-later
"""Test resolved-storage alias guards, including equal Vulkan sentinel addresses."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError('Use a fresh output outside the repository')
    config = json.loads(args.config.read_text())
    llvm, sd = Path(config['LLVM']), Path(config['SD'])
    out.mkdir(parents=True)
    inputs = [Path(__file__), root/'tests/moss_dit_alias.cpp',
              root/'patches/stable-diffusion/overlay/src/model/diffusion/moss_dit_ops.hpp']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    command = [str(llvm/'clang++.exe'), '-std=c++20', '-O2', '-Wall', '-Wextra',
               '-Werror', '-Wno-unused-function', '-DPS5_AI_MOSS_DIT_OPT_TRIAL=1',
               '-DGGML_MAX_NAME=160', '-ffunction-sections', '-fdata-sections',
               '-I', str(root/'patches/stable-diffusion/overlay/src'),
               '-I', str(sd/'ggml/include'), '-I', str(sd/'ggml/src'),
               str(root/'tests/moss_dit_alias.cpp'), '-static', '-Wl,--gc-sections',
               '-o', str(out/'alias.exe')]
    env = dict(os.environ, PATH=str(llvm)+os.pathsep+os.environ.get('PATH', ''),
               TMP=str(out), TEMP=str(out))
    built = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
    (out/'build.log').write_text(built.stdout+built.stderr, encoding='utf-8')
    if built.returncode:
        raise RuntimeError(built.stderr[-4000:])
    result = subprocess.run([str(out/'alias.exe')], env=env, capture_output=True,
                            text=True, timeout=30, check=True)
    report = json.loads(result.stdout)
    assert report['success'] and report['checks'] == 200007
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    report.update(input_hashes_stable=True, sources=hashes, command=command)
    (out/'receipt.private.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('checks', 'success', 'PS5_connections')}))


if __name__ == '__main__':
    main()
