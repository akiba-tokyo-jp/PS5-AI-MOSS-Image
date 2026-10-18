# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU raster and backend-health parser checks; no network or inference."""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    host = Path(config['HOST_BIN'])
    llvm = Path(config['LLVM'])
    env = dict(os.environ)
    env['PATH'] = str(host) + os.pathsep + str(llvm) + os.pathsep + env.get('PATH', '')

    def run(command):
        result = subprocess.run(list(map(str, command)), cwd=out, env=env,
                                capture_output=True, text=True, errors='replace')
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:] + result.stdout[-1000:])
        return result.stdout

    preview = out / 'dashboard-preview.exe'
    run([host / 'clang.exe', '-std=c11', '-O2', '-D_POSIX_C_SOURCE=200809L',
         '-Wall', '-Wextra', '-Werror', '-I', ROOT / 'native/ui',
         ROOT / 'tests/dashboard_preview.c', ROOT / 'native/ui/format.c', '-o', preview])
    raster = json.loads(run([preview]))
    health = out / 'backend-health.exe'
    run([llvm / 'clang.exe', '--target=x86_64-w64-mingw32', '-std=gnu11', '-O2',
         '-Wall', '-Wextra', '-Werror', '-I', ROOT / 'native/gateway',
         '-I', ROOT / 'native/sha256', ROOT / 'tests/backend_health.c',
         ROOT / 'native/gateway/cpu_hooks.c', ROOT / 'native/sha256/sha256.c',
         '-lws2_32', '-o', health])
    parser_checks = json.loads(run([health]))
    result = {'CPU_raster': raster, 'backend_health': parser_checks,
              'network_calls': 0, 'PS5_actions': 0, 'inference_performed': False}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
