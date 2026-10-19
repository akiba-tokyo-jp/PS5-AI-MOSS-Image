# SPDX-License-Identifier: GPL-3.0-or-later
"""Fresh host-only MOSS graph metadata regression; no models, PS5 or network.

Uses the small tracked GGUF-header fixture, a pinned pre-fix Git header and
exact metadata helper bodies from the prepared SD source checkout. Does not
link a CPU graph backend, initialize Vulkan or execute learned model math.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASELINE_COMMIT = '189632386f782fc209e7f3ae1ea6e47138f6475d'
BASELINE_SHA256 = '95fae94bf8035e175dbad4011a843d1fd4c049179a081a87a751c91db9b4ae73'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True, help='Existing local build paths; LLVM and SD are used')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = a.out.resolve()
    if out.exists() or out.is_relative_to(root) or out.drive.upper() != 'D:':
        raise RuntimeError('Choose a fresh output directory on D: outside the repository')
    config = json.loads(a.config.read_text())
    llvm, sd = Path(config['LLVM']), Path(config['SD'])
    fixture = root/'tests/fixtures/moss-selected-dit-header.json'
    metadata = json.loads(fixture.read_text())
    assert metadata['payload_bytes_read'] == 0 and metadata['header_bytes_read'] == 23320
    assert metadata['selected_locked_full_SHA256'] == '902dbb0364d8b313cd91792b3cfe7c87b987386f0c5414986f3f1027b82b2225'
    assert len(metadata['tensors']) == 379
    baseline = subprocess.check_output(['git', 'show', BASELINE_COMMIT+':patches/stable-diffusion/overlay/src/core/vulkan_scratch_bounds.hpp'], cwd=root)
    if hashlib.sha256(baseline).hexdigest() != BASELINE_SHA256:
        raise RuntimeError('Pinned pre-fix baseline header differs')
    out.mkdir(parents=True)
    (out/'baseline').mkdir()
    (out/'baseline/vulkan_scratch_bounds.hpp').write_bytes(baseline)
    (out/'ggml-version.h').write_text('#pragma once\n#define GGML_VERSION "0.25.3"\n#define GGML_COMMIT "d25b121"\n')
    env = dict(os.environ, TMP=str(out), TEMP=str(out))
    env['PATH'] = str(llvm)+os.pathsep+env.get('PATH', '')
    commands = []

    def run(command, label):
        command = list(map(str, command))
        commands.append(command)
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        (out/(label+'.log')).write_text(result.stdout+result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(label+': '+(result.stdout+result.stderr)[-6000:])
        return result.stdout

    def function(path, anchor):
        text = path.read_text()
        start = text.index(anchor)
        end = text.index('{', start)+1
        depth = 1
        while depth:
            if text[end] == '{': depth += 1
            elif text[end] == '}': depth -= 1
            end += 1
        return text[start:end]

    sources = [sd/'ggml/src'/name for name in ('ggml.c', 'ggml-quants.c', 'ggml-threading.cpp')]
    sources.append(sd/'src/core/ggml_extend.cpp')
    helper_sources = [sd/'src/core'/name for name in ('ggml_runner.cpp', 'util.cpp', 'ggml_graph_cut.cpp')]
    tracked = [Path(__file__).resolve(), root/'tests/moss_actual_graph_scratch.cpp', fixture,
               root/'patches/stable-diffusion/overlay/src/core/vulkan_scratch_bounds.hpp',
               root/'patches/stable-diffusion/overlay/src/core/runner_admission.hpp',
               root/'patches/stable-diffusion/overlay/src/model/diffusion/moss_image.hpp', *sources, *helper_sources]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    wrapper = out/'exact-metadata-wrappers.cpp'
    body = '#include "core/ggml_runner.h"\n#include "core/ggml_extend.h"\n#include "core/ggml_graph_cut.h"\n'
    body += function(helper_sources[0], 'ggml_tensor* ggml_ext_attention_ext(GGMLRunnerContext* ctx,')+'\n'
    body += function(helper_sources[1], 'bool starts_with(const std::string& str,')+'\n'
    body += 'namespace sd::ggml_graph_cut {\n'
    body += function(helper_sources[2], 'std::string make_graph_cut_name(')+'\n'
    body += function(helper_sources[2], 'void mark_graph_cut(')+'\n}\n'
    wrapper.write_text(body)
    includes = []
    for path in [root/'native/app', root/'patches/stable-diffusion/overlay/src',
                 root/'patches/stable-diffusion/overlay/src/core', root/'patches/stable-diffusion/overlay/ggml/src',
                 sd/'include', sd/'src', sd/'ggml/include', sd/'ggml/src', out]:
        includes.extend(['-I', path])
    includes.extend(['-isystem', sd/'thirdparty'])
    common = ['-O0', '-ffunction-sections', '-fdata-sections', '-DGGML_SCHED_MAX_COPIES=4', '-DGGML_MAX_NAME=160']
    objects = []
    for source in [*sources, wrapper]:
        obj = out/(source.name+'.o')
        cpp = source.suffix == '.cpp'
        run([llvm/('clang++.exe' if cpp else 'clang.exe'), *common,
             *(['-std=c++20'] if cpp else []), *includes, '-c', source, '-o', obj], source.name+'-compile')
        objects.append(obj)
    exe = out/'actual-graph.exe'
    run([llvm/'clang++.exe', '-std=c++20', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
         '-Wno-unused-parameter', '-Wno-sign-compare', '-Wno-delete-non-abstract-non-virtual-dtor',
         '-static', *common, *includes, root/'tests/moss_actual_graph_scratch.cpp', *objects,
         '-Wl,--gc-sections', '-o', exe], 'test-compile')
    result = json.loads(run([exe, fixture], 'test'))
    assert result['checks'] == 1079 and result['actual_model_forward']
    assert not any(result[key] for key in ('tensor_data_allocated', 'graph_executed', 'backend_initialized', 'PS5_access'))
    assert before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    receipt = {'success': True, 'checks': result['checks'], 'input_hashes_stable': True,
               'baseline_commit': BASELINE_COMMIT, 'baseline_sha256': BASELINE_SHA256,
               'fixture_contains_weight_values': False, 'model_files_opened': 0,
               'selected_LoRA_graph_included': False, 'GPU_calls': 0, 'PS5_connections': 0,
               'graph_executed': False, 'sources': before, 'result': result}
    (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'success': True, 'checks': result['checks'], 'graph_executed': False, 'model_files_opened': 0}))


if __name__ == '__main__':
    main()
