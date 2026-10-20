#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Actual GPU-profile metadata/guard tests. No model values, Vulkan or PS5 calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(a.config.read_text(encoding='utf-8-sig'))
    sd, llvm = Path(config['SD']).resolve(), Path(config['LLVM']).resolve()
    out = a.out.resolve()
    if out.drive.upper() != 'D:' or out.exists() or out.is_relative_to(repo):
        raise ValueError('Use fresh D output outside repository')
    out.mkdir(parents=True)
    backend = repo / 'patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan.cpp'
    common = backend.with_name('ggml-vulkan-common.h')
    platform = repo / 'native/app/radv_ps5_platform_native.c'
    tracked = [Path(__file__), repo / 'tests/moss_gpu_profile.cpp', repo / 'native/app/moss_gpu_profile.h',
               repo / 'native/app/moss_lora_contract.hpp', sd / 'ggml/include/ggml.h', sd / 'ggml/src/ggml-impl.h', sd / 'ggml/src/ggml.c', backend, common, platform]

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    before = {str(x): digest(x) for x in tracked}
    extracted = []

    def method(path, anchor):
        text = path.read_text(encoding='utf-8')
        start = text.index(anchor)
        end = text.index('{', start) + 1
        depth = 1
        while depth:
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
            end += 1
        body = text[start:end]
        extracted.append({'source': str(path), 'anchor': anchor,
                          'first_line': text.count('\n', 0, start) + 1,
                          'bytes_utf8': len(body.encode()), 'sha256': hashlib.sha256(body.encode()).hexdigest()})
        return body

    # Compile the actual name writer and backend classifier, not test substitutes.
    name_writer = out / 'exact-ggml-set-name.cpp'
    name_writer.write_text('#include "ggml.h"\n' + method(sd / 'ggml/src/ggml.c',
        'struct ggml_tensor * ggml_set_name(struct ggml_tensor * tensor, const char * name) {') + '\n', encoding='utf-8')
    classifier = out / 'exact-backend-node-classifier.cpp'
    classifier.write_text('#include "ggml.h"\n#pragma clang diagnostic push\n#pragma clang diagnostic ignored "-Wunused-function"\n#include "ggml-impl.h"\n#pragma clang diagnostic pop\n#include "moss_gpu_profile.h"\n' + method(backend,
        'static unsigned ps5_ai_moss_gpu_actual_node_class(const ggml_tensor *node) {') + '\n'
        'extern "C" unsigned ps5_ai_test_actual_node_class(const ggml_tensor *node) { return ps5_ai_moss_gpu_actual_node_class(node); }\n' + method(backend,
        'static ps5_ai_moss_gpu_coverage moss_gpu_profile_actual_graph_coverage(const ggml_cgraph *graph) {') + '\n'
        'extern "C" ps5_ai_moss_gpu_coverage ps5_ai_test_actual_graph_coverage(const ggml_cgraph *graph) { return moss_gpu_profile_actual_graph_coverage(graph); }\n', encoding='utf-8')
    # The CPU-safe metadata unit tests cannot execute the PS5 driver. Bind
    # independently inspected ordering invariants to actual integrated sources.
    source_checks = []

    def require(ok, description):
        if not ok:
            raise RuntimeError('Source ordering invariant failed: ' + description)
        source_checks.append(description)

    body = backend.read_text(encoding='utf-8')
    require('s.report.schema=2;' in body, 'actual runtime report uses complete-graph schema2')
    graph_begin = method(backend, 'static void moss_gpu_profile_graph_begin(ggml_backend_vk_context *ctx,const ggml_cgraph *graph) {')
    require(graph_begin.index('moss_gpu_profile_actual_graph_coverage(graph)') < graph_begin.index('ps5_ai_moss_gpu_coverage_complete(&coverage)')
            < graph_begin.index('++s.report.full_graphs_seen;') < graph_begin.index('s.collecting=true;'),
            'actual complete graph metadata qualification precedes timestamp sample collection')
    require('++s.report.skipped_partial_views;++s.report.unprofiled_graphs;return;' in graph_begin,
            'partial observer-cut graph views are counted and skipped before sampling')
    require('if(s.mode==2)' in graph_begin and 'g.complete_main_graph=1;' in graph_begin
            and 'g.lora_down_nodes=coverage.lora_down_nodes;' in graph_begin and 'g.lora_up_nodes=coverage.lora_up_nodes;' in graph_begin,
            'generation-only qualification retains actual complete graph and LoRA node diagnostics; startup gate bypasses qualification')
    start = body.index('static ggml_status ggml_backend_vk_graph_compute(')
    graph_compute = body[start:body.index('\nvoid ggml_vk_graph_optimize(', start)]
    require(graph_compute.index('moss_gpu_profile_graph_begin(ctx,cgraph);') < graph_compute.index('ggml_vk_submit_transfer_ctx(ctx);'),
            'profile graph arms before existing graph submission work')
    require(graph_compute.rindex('ggml_vk_synchronize(ctx);') < graph_compute.rindex('moss_gpu_profile_graph_end(ctx);'),
            'profile results collected AFTER existing final synchronize, not before it')
    require(graph_compute.index('device.waitForFences({ ctx->fence }') < graph_compute.index('ctx->submit_pending = false;'),
            'existing serialized submit helper proves fence completion before clearing pending')
    require(graph_compute.rindex('submit_after(submit_node_idx,') < graph_compute.rindex('moss_gpu_profile_graph_end(ctx);'),
            'all existing final serial submit/fence processing precedes profile result collection')
    graph_end = method(backend, 'static void moss_gpu_profile_graph_end(ggml_backend_vk_context *ctx) {')
    require(graph_end.index('ps5_ai_ggml_dispatch_snapshot(&completed);') < graph_end.index('ps5_ai_moss_gpu_profile_query_cache_readback(')
            < graph_end.index('getQueryPoolResults('), 'actual completion snapshot precedes typed query cache drop and result read')
    require('ctx->submit_pending||!ctx->compute_ctx.expired()' in graph_end,
            'profile result path refuses pending or live command context')
    require('vk::QueryResultFlagBits::e64|vk::QueryResultFlagBits::eWithAvailability' in graph_end
            and 'QueryResultFlagBits::eWait' not in graph_end and 'waitForFences' not in graph_end,
            'profile result read requests64bit availability without WAIT or extra CPU fence wait')
    require('s.values[2*i].ticks<s.values[2*i-1].ticks' in graph_end,
            'actual result processing rejects out of order intervals between successive dispatches')
    dispatch = method(common, 'inline void ggml_vk_dispatch_pipeline(')
    require(dispatch.index('ps5_ai_moss_gpu_profile_dispatch_begin(profile_command)') < dispatch.index('.dispatch(wg0, wg1, wg2)')
            < dispatch.index('ps5_ai_moss_gpu_profile_dispatch_end(profile_command, profile_cookie)'),
            'same captured command buffer brackets each real dispatch command')
    require('waitForFences' not in dispatch and 'queue.submit' not in dispatch and 'resetQueryPool' not in dispatch,
            'generic dispatch wrapper adds no CPU wait, submission or query pool reset of its own')
    cache = method(platform, 'int\nps5_ai_moss_gpu_profile_query_cache_readback(')
    require(cache.index('ps5_ai_ggml_dispatch_snapshot(&completed);') < cache.index('completed.fenced_completions <= before_fences')
            < cache.index('radv_ps5_cpu_flush('), 'typed query cache helper requires actual hardware and advanced completed fence before flushing')
    require('!completed.hardware_verified || completed.work_pending' in cache and 'pool->stride != 8' in cache
            and 'bytes > pool->size - offset' in cache, 'query cache helper enforces idle hardware, timestamp stride and overflow-safe storage bounds')
    require('memset(' not in cache and 'ResetQueryPool' not in cache, 'query readback helper never writes/reset query storage on CPU')
    (out / 'tested-moss_gpu_profile.h').write_bytes(tracked[2].read_bytes())
    env = dict(os.environ, TMP=str(out), TEMP=str(out))
    env['PATH'] = str(llvm) + os.pathsep + env.get('PATH', '')
    commands, results = [], []

    def run(label, args, expected_error=None):
        args = list(map(str, args))
        commands.append(args)
        r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=60)
        combined = r.stdout + r.stderr
        (out / (label + '.log')).write_text(combined, encoding='utf-8')
        if expected_error is None:
            if r.returncode != 0:
                raise RuntimeError(label + ' unexpected exit ' + str(r.returncode) + ': ' + combined[-6000:])
        elif r.returncode == 0 or expected_error not in combined:
            raise RuntimeError(label + ' expected specific compile refusal missing: ' + combined[-6000:])
        results.append({'label': label, 'exit_code': r.returncode, 'expected_failure': expected_error is not None,
                        'expected_diagnostic': expected_error,
                        'output': r.stdout.strip() if expected_error is None else 'expected compile refusal'})

    base = [llvm / 'clang++.exe', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-static',
            '-I', repo / 'native/app', '-I', sd / 'ggml/include', '-I', sd / 'ggml/src',
            repo / 'tests/moss_gpu_profile.cpp', name_writer, classifier]
    policy = ['-DPS5_AI_MOSS_F32_ACCUMULATION=1', '-DPS5_AI_MOSS_PERFORMANCE_TRIAL=1',
              '-DPS5_AI_MOSS_PERF_VARIANT=1', '-DPS5_AI_MOSS_TRIAL_10G=1',
              '-DPS5_AI_MOSS_EXPERIMENTAL_SMALL_TRIAL=1']
    prerequisite_error = 'MOSS_GPU_profile_requires_explicit_FP16_F32acc_10GiB_performance_trial'
    for value in (0, 1):
        exe = out / f'profile{value}.exe'
        run(f'compile-profile{value}', [*base, '-DGGML_MAX_NAME=160', f'-DPS5_AI_MOSS_GPU_PROFILE={value}',
            *(policy if value else []), '-o', exe])
        run(f'run-profile{value}', [exe])
    for value in (2, -1):
        run(f'refuse-profile{value}', [*base, '-DGGML_MAX_NAME=160', f'-DPS5_AI_MOSS_GPU_PROFILE={value}',
            '-o', out / f'invalid{value}.exe'], 'PS5_AI_MOSS_GPU_PROFILE_must_be_zero_or_one')
    run('refuse-default64-test-ABI', [*base, '-DPS5_AI_MOSS_GPU_PROFILE=0', '-o', out / 'invalid64.exe'],
        'Match actual75core/backend name ABI')
    run('refuse-no-prerequisites', [*base, '-DGGML_MAX_NAME=160', '-DPS5_AI_MOSS_GPU_PROFILE=1',
        '-o', out / 'invalid-none.exe'], prerequisite_error)
    for index, flag in enumerate(policy):
        run(f'refuse-missing-prerequisite{index}', [*base, '-DGGML_MAX_NAME=160', '-DPS5_AI_MOSS_GPU_PROFILE=1',
            *policy[:index], *policy[index + 1:], '-o', out / f'invalid-missing{index}.exe'], prerequisite_error)
        changed = [*policy[:index], flag.replace('=1', '=0'), *policy[index + 1:]]
        run(f'refuse-false-prerequisite{index}', [*base, '-DGGML_MAX_NAME=160', '-DPS5_AI_MOSS_GPU_PROFILE=1',
            *changed, '-o', out / f'invalid-false{index}.exe'], prerequisite_error)
    changed = [flag if not flag.startswith('-DPS5_AI_MOSS_PERF_VARIANT=') else '-DPS5_AI_MOSS_PERF_VARIANT=2' for flag in policy]
    run('refuse-other-performance-variant', [*base, '-DGGML_MAX_NAME=160', '-DPS5_AI_MOSS_GPU_PROFILE=1',
        *changed, '-o', out / 'invalid-variant2.exe'], prerequisite_error)
    after = {str(x): digest(x) for x in tracked}
    if after != before:
        raise RuntimeError('Source changed during actual header tests; use fresh output against final source')
    receipt = {'schema': 'moss-gpu-profile-actual-header-host-v3', 'success': True,
               'offline_only': True, 'source_stable': True, 'source_sha256': before,
               'extracted_actual_functions': extracted, 'source_ordering_checks': source_checks, 'commands': commands, 'results': results,
               'model_weight_reads': 0, 'model_inference': False, 'GPU_calls': 0, 'PS5_connections': 0,
               'scope': 'Actual inline metadata helpers, actual backend node classifier and actual name writer. Does not establish GPU timer execution/coherence/fences.'}
    (out / 'receipt.private.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'success': True, 'commands': len(commands), 'outputs': [r['output'] for r in results if r['label'].startswith('run-')],
                     'expected_compile_refusals': sum(r['expected_failure'] for r in results),
                     'actual_source_ordering_checks': len(source_checks),
                     'GPU_calls': 0, 'model_inference': False, 'PS5_connections': 0}))


if __name__ == '__main__':
    main()
