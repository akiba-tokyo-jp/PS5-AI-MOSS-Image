# SPDX-License-Identifier: GPL-3.0-or-later
"""Before/after build-input identity checks; no network or credentials."""
import hashlib
from pathlib import Path

def _digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def collect(root, config_path, parameters, profile, expand):
    root = Path(root)
    paths = set()
    for folder in ('native', 'patches/stable-diffusion/overlay', 'assets', 'licenses', 'build-profiles'):
        paths.update(p.resolve() for p in (root/folder).rglob('*') if p.is_file())
    paths.update(p.resolve() for p in (root/'config').glob('*.example') if p.is_file())
    for name in ('tools/build.py', 'tools/build_gateway.py', 'tools/build_input_snapshot.py', 'tools/build_moss_core.py', 'tools/moss_feature_policy.py', 'tools/moss_lora_precision_policy.py', 'tools/moss_dit_policy.py', 'tools/moss_kernel_policy.py'):
        path = root/name
        if path.is_file(): paths.add(path.resolve())
    paths.add(Path(config_path).resolve())
    for row in profile['dependencies']:
        paths.add((Path(parameters['DEPS'])/row['name']).resolve())
    paths.add(Path(expand(profile['generated_shader_header']['path'])).resolve())
    for row in profile.get('compiler_headers', []):
        paths.add(Path(expand(row['path'])).resolve())
    # Include possible transitive headers before compilation, not just .d files
    # discovered afterward. Exclude libraries/models/temporary outputs here.
    suffixes = {'.h', '.hpp', '.hh', '.hxx', '.inc', '.c', '.cpp', '.cc', '.s', '.ld'}
    for key, parts in (
        ('SD', ('src', 'include', 'ggml/src', 'ggml/include', 'thirdparty')),
        ('SDK', ('target/include',)), ('PLATFORM', ('src', 'include')),
        ('VK', ('tooling/native',)), ('VULKAN_HEADERS', ('include',))):
        for part in parts:
            base = Path(parameters[key])/part
            paths.update(p.resolve() for p in base.rglob('*') if p.is_file() and p.suffix.lower() in suffixes)
    for name in ('clang.exe', 'ld.lld.exe', 'llvm-ar.exe', 'llvm-strip.exe', 'llvm-readelf.exe', 'llvm-nm.exe'):
        paths.add((Path(parameters['LLVM'])/name).resolve())
    for base in (Path(parameters['SDK'])/'target/lib', Path(parameters['SDK'])/'ldscripts'):
        paths.update(p.resolve() for p in base.rglob('*') if p.is_file() and p.suffix.lower() in {'.a', '.so', '.x', '.ld'})
    def tokens(value):
        if isinstance(value, dict):
            for item in value.values(): yield from tokens(item)
        elif isinstance(value, list):
            for item in value: yield from tokens(item)
        elif isinstance(value, str): yield value
    for token in tokens(profile):
        if '${' in token and not token.startswith('-'):
            path = Path(expand(token))
            if path.is_file(): paths.add(path.resolve())
    paths.add(Path(parameters['NATIVE_TOOL']).resolve())
    output_root = Path(parameters['OUT']).resolve()
    paths = {p for p in paths if not p.is_relative_to(output_root)}
    return {str(p): {'bytes': p.stat().st_size, 'sha256': _digest(p)} for p in sorted(paths)}

def require_unchanged(before, after):
    changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    if changed:
        raise RuntimeError('Build inputs changed while compiling; artifact not admitted: '+str(changed[:12]))
