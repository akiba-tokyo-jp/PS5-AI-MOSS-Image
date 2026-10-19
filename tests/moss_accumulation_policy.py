# SPDX-License-Identifier: GPL-3.0-or-later
"""Actual MOSS/LoRA accumulator metadata policy; no model values or graph math."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llvm', type=Path, required=True)
    p.add_argument('--sd', type=Path, required=True)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root, out = Path(__file__).resolve().parents[1], a.out.resolve()
    if out.exists() or out.is_relative_to(root) or out.drive.upper() != 'D:':
        raise ValueError('Use a fresh D: output directory outside the repository')
    llvm, sd, sdk = a.llvm.resolve(), a.sd.resolve(), a.sdk.resolve()
    overlay = root / 'patches/stable-diffusion/overlay'
    model = overlay / 'src/model/diffusion/moss_image.hpp'
    mirror = sd / 'src/model/diffusion/moss_image.hpp'
    if model.read_bytes() != mirror.read_bytes():
        raise RuntimeError('MOSS overlay and prepared vendor must be byte identical')
    fixture = root / 'tests/fixtures/moss-selected-dit-header.json'
    lora = overlay / 'src/model/adapter/lora.hpp'
    sources = [sd / 'ggml/src' / s for s in ('ggml.c', 'ggml-quants.c', 'ggml-threading.cpp')]
    sources.append(sd / 'src/core/ggml_extend.cpp')
    helpers = [sd / 'src/core' / s for s in ('ggml_runner.cpp', 'util.cpp', 'ggml_graph_cut.cpp')]
    tracked = [Path(__file__).resolve(), root / 'tests/moss_accumulation_policy.cpp', model, mirror,
               fixture, lora, sd / 'src/model/common/ggml_block.hpp', *sources, *helpers]
    before = {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in tracked}
    out.mkdir(parents=True)
    env = dict(os.environ, TMP=str(out), TEMP=str(out))
    env['PATH'] = str(llvm) + os.pathsep + env.get('PATH', '')
    commands, checks, extracted = [], [], []

    def run(label, command, reject=False):
        command = list(map(str, command)); commands.append(command)
        r = subprocess.run(command, capture_output=True, text=True, env=env, timeout=180)
        (out / (label + '.log')).write_text(r.stdout + r.stderr, encoding='utf-8')
        if (r.returncode == 0) == reject:
            raise RuntimeError(label + ': unexpected exit ' + str(r.returncode) + ': ' + (r.stdout + r.stderr)[-6000:])
        checks.append({'check': label, 'exit_code': r.returncode, 'expected_compile_rejection': reject})
        return r.stdout

    def method(path, anchor):
        text = path.read_text(encoding='utf-8')
        begin = text.index(anchor)
        end = text.index('{', begin) + 1; depth = 1
        while depth:
            if text[end] == '{': depth += 1
            elif text[end] == '}': depth -= 1
            end += 1
        body = text[begin:end]
        extracted.append({'source': str(path), 'anchor': anchor, 'first_line': text.count('\n', 0, begin) + 1,
                          'bytes_utf8': len(body.encode()), 'sha256': hashlib.sha256(body.encode()).hexdigest()})
        return body

    (out / 'ggml-version.h').write_text('#pragma once\n#define GGML_VERSION "metadata-only"\n#define GGML_COMMIT "metadata-only"\n')
    wrapper = out / 'exact-metadata-wrappers.cpp'
    body = '#include "core/ggml_runner.h"\n#include "core/ggml_extend.h"\n#include "core/ggml_graph_cut.h"\n'
    body += method(helpers[0], 'ggml_tensor* ggml_ext_attention_ext(GGMLRunnerContext* ctx,') + '\n'
    body += method(helpers[1], 'bool starts_with(const std::string& str,') + '\n'
    body += 'namespace sd::ggml_graph_cut {\n' + method(helpers[2], 'std::string make_graph_cut_name(') + '\n'
    body += method(helpers[2], 'void mark_graph_cut(') + '\n}\n'
    wrapper.write_text(body, encoding='utf-8', newline='\n')
    get_diff = method(lora, 'ggml_tensor* get_out_diff(ggml_context* ctx,')
    forward = method(lora, 'ggml_tensor* forward_with_lora(ggml_context* ctx,')
    # Keep the selected normal-LoRA methods byte-identical. The surrounding
    # fixture bypasses file loading/runner lifetime/raw-diff/LoKR, which this
    # metadata precision test intentionally does not claim to verify.
    body = '''struct MetadataLora {
    float multiplier = 0.7f;
    std::unordered_map<std::string, ggml_tensor*> lora_tensors;
    std::set<std::string> applied_lora_tensors, skipped_incompatible_lora_tensors, warned_incompatible_model_tensors;
    float scalar_value(ggml_tensor*) { std::abort(); }
'''
    body += get_diff + '\n};\n'
    body += '''struct MetadataAdapter : WeightAdapter {
    std::vector<std::shared_ptr<MetadataLora>> lora_models;
    explicit MetadataAdapter(std::vector<std::shared_ptr<MetadataLora>> rows) : lora_models(std::move(rows)) {}
    ggml_tensor* patch_weight(ggml_context*, ggml_backend_t, ggml_tensor* w, const std::string&, bool) { return w; }
    ggml_tensor* patch_weight(ggml_context* c, ggml_backend_t b, ggml_tensor* w, const std::string& s) override { return patch_weight(c,b,w,s,false); }
    ggml_tensor* add_lora_to_output(ggml_context*,ggml_backend_t,ggml_tensor*,ggml_tensor*,ggml_tensor*,const std::string&,ForwardParams) override { std::abort(); }
    size_t get_extra_graph_size() override { return 0; }
'''
    body += forward + '\n};\n'
    (out / 'exact-lora-methods.hpp').write_text(body, encoding='utf-8', newline='\n')
    includes = []
    for path in (root/'native/app', overlay/'src', overlay/'src/core', overlay/'ggml/src',
                 sd/'include', sd/'src', sd/'ggml/include', sd/'ggml/src', out):
        includes += ['-I', path]
    includes += ['-isystem', sd/'thirdparty']
    common = ['-O0', '-ffunction-sections', '-fdata-sections', '-DGGML_MAX_NAME=160', '-DGGML_SCHED_MAX_COPIES=4']
    objects = []
    for source in [*sources, wrapper]:
        obj = out / (source.name + '.o'); cpp = source.suffix == '.cpp'
        run(source.name+'-compile', [llvm/('clang++.exe' if cpp else 'clang.exe'), *common,
             *(['-std=c++20'] if cpp else []), *includes, '-c', source, '-o', obj])
        objects.append(obj)
    results = {}
    test = root / 'tests/moss_accumulation_policy.cpp'
    warnings = ['-Wall', '-Wextra', '-Werror', '-Wno-unused-function', '-Wno-unused-parameter',
                '-Wno-sign-compare', '-Wno-delete-non-abstract-non-virtual-dtor']
    for label, flags in (('default', []), ('explicit0', ['-DPS5_AI_MOSS_F32_ACCUMULATION=0']),
                         ('explicit1', ['-DPS5_AI_MOSS_F32_ACCUMULATION=1'])):
        exe = out / (label + '.exe')
        run(label+'-compile', [llvm/'clang++.exe', '-std=c++20', '-static', *common, *warnings,
            *flags, *includes, test, *objects, '-Wl,--gc-sections', '-o', exe])
        result = json.loads(run(label, [exe, fixture]))
        assert len(result['cases']) == 8 and result['actual_MOSS_model_forward'] and result['exact_LoRA_method_bodies']
        assert not any(result[k] for k in ('tensor_payloads_allocated','graph_executed','backend_initialized','model_files_opened','PS5_connections'))
        expected = 'f32' if label == 'explicit1' else 'default'
        assert all(c['accumulation'] == expected for c in result['cases'])
        results[label] = result
    assert results['default']['parameter_dtypes'] == results['explicit0']['parameter_dtypes'] == results['explicit1']['parameter_dtypes']
    syntax = [llvm/'clang++.exe', '--no-default-config', '--target=x86_64-sie-ps5', '-D__PROSPERO__=1',
              '-fvisibility-nodllstorageclass=default', '-isysroot', sdk,
              '-isystem', sdk/'target/include/c++/v1', '-isystem', sdk/'target/include',
              '-std=c++20', '-fexceptions', '-fcxx-exceptions', '-frtti', *common, *includes, '-fsyntax-only', '-x', 'c++']
    for flag in (0, 1):
        run('PS5-header-mode'+str(flag), [*syntax, '-DPS5_AI_MOSS_F32_ACCUMULATION='+str(flag), model])
    for value in ('2', '-1', 'unknown_identifier'):
        label = 'invalid-'+value
        run(label, [llvm/'clang++.exe', '-std=c++20', *common, *includes,
            '-DPS5_AI_MOSS_F32_ACCUMULATION='+value, '-fsyntax-only', '-x', 'c++', model], reject=True)
        diagnostic = (out/(label+'.log')).read_text(encoding='utf-8')
        assert ('PS5_AI_MOSS_F32_ACCUMULATION must be 0 or 1' in diagnostic if value != 'unknown_identifier'
                else "use of undeclared identifier 'unknown_identifier'" in diagnostic)
    assert before == {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in tracked}
    receipt = {'schema': 'moss-f32-accumulation-metadata-host-v1', 'success': True,
        'commands': checks, 'result': results, 'sources_before_after_stable': before, 'exact_extracted_methods': extracted,
        'host_assertions': sum(r['checks'] for r in results.values()),
        'selected_payload_values_read': False, 'learned_model_inference': False, 'GPU_execution': False,
        'PS5_connections': 0, 'credential_reads': 0, 'CPU_graph_backend_linked': False,
        'LoRA_scope': 'Actual Linear and exact normal-LoRA get_out_diff/forward_with_lora metadata bodies; synthetic504tensor rank16/F32 shapes. Loader, strict runtime504 receipts, rawdiff, LoKR, scalar alpha and runner lifetime are not tested.',
        'GPU_accuracy_or_speed_proven': False, 'tolerances_changed': False}
    (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n', encoding='utf-8')
    (out/'receipt.private.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'success': True, 'host_assertions': receipt['host_assertions'], 'checks': len(checks),
                      'model_graphs': 24, 'graph_executed': False, 'PS5_connections': 0}))


if __name__ == '__main__':
    main()
