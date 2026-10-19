# SPDX-License-Identifier: GPL-3.0-or-later
"""Build output must not be an input; actual source/tool changes must fail."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]/'tools/build_input_snapshot.py'
    spec = importlib.util.spec_from_file_location('snapshot_subject', source)
    subject = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subject)
    checks = []
    with tempfile.TemporaryDirectory(dir=args.out) as temp:
        root = Path(temp)
        def file(path, value='original'):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding='utf-8')
            return path
        project = root/'project'
        header = file(project/'native/test.h')
        config = file(root/'build.json', '{}')
        values = {key:str(root/key) for key in ('DEPS','SD','SDK','PLATFORM','VK','VULKAN_HEADERS','LLVM')}
        values['OUT'] = str(root/'output')
        values['NATIVE_TOOL'] = str(file(root/'native-tool.exe'))
        for name in ('clang.exe','ld.lld.exe','llvm-ar.exe','llvm-strip.exe','llvm-readelf.exe','llvm-nm.exe'):
            file(Path(values['LLVM'])/name)
        shader = file(root/'generated/shader.h')
        dep = file(Path(values['DEPS'])/'dep.a')
        external = file(Path(values['SD'])/'src/header.hpp')
        result_file = Path(values['OUT'])/'compiled.o'
        profile = {'dependencies':[{'name':'dep.a'}],
                   'generated_shader_header':{'path':str(shader)},
                   'compile':[str(header),'-o','${OUT}/compiled.o']}
        def expand(text):
            for key,value in values.items(): text=text.replace('${'+key+'}',value)
            return text
        def snapshot(): return subject.collect(project,config,values,profile,expand)
        before = snapshot()
        file(result_file, 'new compilation result')
        subject.require_unchanged(before, snapshot())
        assert str(result_file.resolve()) not in snapshot()
        checks.append('generated output excluded before and after compilation')
        for changed in (header, external, shader, dep, Path(values['LLVM'])/'clang.exe'):
            old = changed.read_text(encoding='utf-8')
            file(changed, 'changed')
            try:
                subject.require_unchanged(before, snapshot())
            except RuntimeError:
                checks.append('detected change '+changed.name)
            else:
                raise AssertionError('Input change accepted: '+str(changed))
            file(changed, old)
        extra = file(Path(values['SD'])/'include/new.hpp')
        try:
            subject.require_unchanged(before, snapshot())
        except RuntimeError:
            checks.append('detected new transitive header')
        else:
            raise AssertionError('New header ignored')
        extra.unlink()
        header.unlink()
        try:
            subject.require_unchanged(before, snapshot())
        except RuntimeError:
            checks.append('detected deleted source header')
        else:
            raise AssertionError('Missing input ignored')
    (args.out/'result.json').write_text(json.dumps({'success':True,'checks':checks},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'success':True,'checks':len(checks),'PS5_connections':0}))

if __name__ == '__main__':
    main()
