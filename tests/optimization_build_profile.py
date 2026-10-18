"""Check optimization export/build wiring; no model, GPU or console use."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--expected-controlnet-batch-pair',type=int,choices=[0,1],default=0)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    profile=json.loads((ROOT/'build-profiles/native.json').read_text())
    config=json.loads(args.config.read_text());config.update(ROOT=str(ROOT),OUT=str(out))
    def expand(value):
        for key,replacement in config.items():value=value.replace('${'+key+'}',str(replacement).replace('\\','/'))
        assert '${' not in value,value
        return value
    recipes=[profile['runner_compile'],*profile['reference_compile'],*profile['ggml_compile']]
    outputs=[];sources=[]
    for recipe in recipes:
        command=list(map(expand,recipe));source=Path(command[command.index('-c')+1])
        assert source.is_file(),source
        outputs.append(Path(command[command.index('-o')+1]).name);sources.append(str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else 'pinned-upstream/'+source.name)
    assert len(outputs)==8 and len(set(outputs))==8
    assert 'ggml_graph_cut.cpp.o' in outputs and 'ggml-vulkan.cpp.o' in outputs
    assert profile['controlnet_batch_pair']==args.expected_controlnet_batch_pair,'Unexpected ControlNet batch-pair profile.'
    header=Path(expand(profile['generated_shader_header']['path']))
    assert hashlib.sha256(header.read_bytes()).hexdigest()==profile['generated_shader_header']['sha256']
    base=ROOT/'patches/stable-diffusion/overlay'
    assert not (base/'src/model/model_manager.cpp').exists()
    assert (base/'src/model_manager.cpp').is_file()
    canonical=(ROOT/'native/app/pool.h').read_bytes()
    for suffix in ['src/core/pool.h','ggml/src/ggml-vulkan/pool.h']:
        assert (base/suffix).read_bytes()==canonical,suffix
    assert 'ps5_ai_flex_largest_free_span_bytes' in canonical.decode()
    providers={
        'ps5_ai_cpu_clip_shared_direct_live_bytes':ROOT/'native/app/engine_sdcpp.cpp',
        'ps5_ai_ggml_release_idle_scratch':base/'ggml/src/ggml-vulkan/ggml-vulkan.cpp',
        'ps5_ai_flex_largest_free_span_bytes':ROOT/'native/app/pool.c'}
    for symbol,path in providers.items():
        assert len(re.findall(r'\b'+symbol+r'\([^;{}]*\)\s*\{',path.read_text()))==1,(symbol,path)
    manager=(base/'src/model_manager.cpp').read_text()
    assert 'extern "C" int ps5_ai_ggml_release_idle_scratch(ggml_backend_t);' in manager
    assert 'extern "C" uint64_t ps5_ai_cpu_clip_shared_direct_live_bytes(void);' in manager
    assert 'PS5_AI_CONTROLNET_BATCH_PAIR 0' in (base/'src/pipeline/diffusion_engine.cpp').read_text()
    ast.parse((ROOT/'tools/build.py').read_text())
    spec=importlib.util.spec_from_file_location('public_wai_build',ROOT/'tools/build.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def member(name,data):
        header=f'{name:<16}{0:<12}{0:<6}{0:<6}{"100644":<8}{len(data):<10}`\n'.encode()
        assert len(header)==60
        return header+data+(b'\n' if len(data)&1 else b'')
    def archive(data):
        return b'!<arch>\n'+member('//',b'ggml-vulkan.cpp.o/\n')+member('/0',data)+member('other.o/',b'fixed')
    (out/'before.a').write_bytes(archive(b'previous'))
    (out/'after.a').write_bytes(archive(b'new-provider'))
    before=module.archive_members(out/'before.a');after=module.archive_members(out/'after.a')
    assert module.check_archive_update(before,after,['ggml-vulkan.cpp.o'])==['ggml-vulkan.cpp.o']
    try:module.check_archive_update(before,after,['other.o'])
    except RuntimeError:pass
    else:raise AssertionError('Unreviewed member replacement was accepted')
    snapshot=json.loads((ROOT/'docs/source-snapshot.json').read_text())
    for row in snapshot['files']:
        assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()==row['published_sha256'],row['file']
    result=dict(state='passed',model_inference=False,PS5_connections=0,core_units=7,GGML_units=1,
        controlnet_batch_pair=profile['controlnet_batch_pair'],archive_non_target_preservation_checked=True,
        canonical_model_manager=True,shared_pool_headers_identical=True,
        native_API_provider_definitions=list(providers),sources=sources,
        source_snapshot_hashes_checked=len(snapshot['files']),generated_shader_header_hash_checked=True,
        native_cross_build_performed=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
