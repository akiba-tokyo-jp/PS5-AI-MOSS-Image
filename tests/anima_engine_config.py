#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Host configuration/CPU-loan checks; never performs model inference/network I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--sd', type=Path, required=True, help='Pinned stable-diffusion.cpp source checkout')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--host-bin', type=Path, help='Additional compiler runtime DLL directory')
    parser.add_argument('--suite', choices=('config', 'cpu-pool', 'operators'), default='config')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = {'config': 'anima_engine_config', 'cpu-pool': 'anima_cpu_text_pool',
            'operators': 'anima_operator_bounds'}[args.suite]
    executable = output / (stem + ('.exe' if os.name == 'nt' else ''))
    source = root / 'tests' / (stem + '.cpp')
    command = [str(args.compiler.resolve()), '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
               '-I', str(root / 'native/app'), '-I', str(args.sd.resolve() / 'include'),
               '-I', str(root / 'patches/stable-diffusion/overlay/src/core'),
               '-I', str(root / 'patches/stable-diffusion/overlay/ggml/src'),
               '-I', str(args.sd.resolve() / 'ggml/include'), '-isystem', str(args.sd.resolve() / 'ggml/src'),
               '-isystem', str(args.sd.resolve() / 'thirdparty'),
               str(source), '-o', str(executable)]
    env = dict(os.environ)
    directories = [str(args.compiler.resolve().parent)]
    if args.host_bin:
        directories.append(str(args.host_bin.resolve()))
    env['PATH'] = os.pathsep.join(directories + [env.get('PATH', '')])
    env['TEMP'] = env['TMP'] = str(output)
    build = subprocess.run(command, env=env, capture_output=True, text=True)
    (output / 'compile.log').write_text(build.stdout + build.stderr, encoding='utf-8')
    build.check_returncode()
    test = subprocess.run([str(executable)], env=env, capture_output=True, text=True)
    (output / 'test.log').write_text(test.stdout + test.stderr, encoding='utf-8')
    header = root / {'config': 'native/app/engine_config.h',
                     'cpu-pool': 'native/app/cpu_clip_pool_buffer.hpp',
                     'operators': 'patches/stable-diffusion/overlay/src/core/vulkan_scratch_bounds.hpp'}[args.suite]
    result = {'success': test.returncode == 0, 'suite': args.suite, 'model_inference_executed': False,
              'PS5_contacted': False, 'test_stdout': test.stdout.strip(),
              'sources': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (header, source)}}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(test.stdout, end='')
    test.check_returncode()


if __name__ == '__main__':
    main()
