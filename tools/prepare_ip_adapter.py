"""Pinned IP-Adapter downloads and bounded F16 storage conversion on D:.

No inference, PS5 connections, or changes to existing deployed models.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import urllib.request

import numpy as np

COMMIT = '018e402774aeeddd60609b4ecdb7e298259dc729'
OUT = Path('models/ip-adapter')
FILES = {
    'ip-adapter-plus_sdxl_vit-h.safetensors': (
        'sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors',
        '3f5062b8400c94b7159665b21ba5c62acdcd7682262743d7f2aefedef00e6581'),
    'clip-vision-vit-h-original-f32.safetensors': (
        'models/image_encoder/model.safetensors',
        '6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030'),
}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(name, remote, expected):
    path = OUT / name
    if path.exists():
        assert sha(path) == expected, 'Existing model SHA mismatch: ' + name
        return path
    partial = path.with_suffix('.download.part')
    assert not partial.exists(), 'Inspect incomplete pinned download before retry'
    url = f'https://huggingface.co/h94/IP-Adapter/resolve/{COMMIT}/{remote}'
    count = 0
    with urllib.request.urlopen(url, timeout=60) as response, partial.open('xb') as f:
        expected_bytes = int(response.headers['Content-Length'])
        while chunk := response.read(8 * 1024**2):
            f.write(chunk)
            count += len(chunk)
            if count // (256 * 1024**2) != (count - len(chunk)) // (256 * 1024**2):
                print(f'{name}: {count}/{expected_bytes} bytes', flush=True)
        f.flush()
        os.fsync(f.fileno())
    assert count == expected_bytes and sha(partial) == expected
    partial.rename(path)
    return path


def convert_vision(source):
    target = OUT / 'clip-vision-vit-h-f16.safetensors'
    receipt_path = OUT / 'vision-storage-conversion.json'
    if target.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['source_sha256'] == sha(source) and receipt['output_sha256'] == sha(target)
        return receipt
    with source.open('rb') as f:
        n, = struct.unpack('<Q', f.read(8))
        assert 0 < n < 8 * 1024**2
        old = json.loads(f.read(n))
    rows = [(k, v) for k, v in old.items() if k != '__metadata__']
    offsets, new, next_offset = [], {}, 0
    for name, row in rows:
        dtype = row['dtype']
        assert dtype in ('F32', 'F16', 'I64')
        old_bytes = math.prod(row['shape']) * {'F32': 4, 'F16': 2, 'I64': 8}[dtype]
        a, b = row['data_offsets']
        assert b - a == old_bytes and n + 8 + b <= source.stat().st_size
        offsets.append((a, b))
        size = old_bytes // 2 if dtype == 'F32' else old_bytes
        new[name] = dict(dtype='F16' if dtype == 'F32' else dtype, shape=row['shape'],
                         data_offsets=[next_offset, next_offset + size])
        next_offset += size
    ordered = sorted(offsets)
    assert ordered[0][0] == 0 and ordered[-1][1] + n + 8 == source.stat().st_size
    assert all(a[1] == b[0] for a, b in zip(ordered, ordered[1:]))
    new['__metadata__'] = {'source_sha256': sha(source), 'source_commit': COMMIT,
                           'conversion': 'F32 floating weights to F16 storage; no inference'}
    header = json.dumps(new, separators=(',', ':')).encode()
    header += b' ' * (-len(header) % 8)
    part = target.with_suffix('.convert.part')
    nonfinite = 0
    max_error = 0.0
    with source.open('rb') as f, part.open('xb') as out:
        out.write(struct.pack('<Q', len(header)))
        out.write(header)
        for name, row in rows:
            a, b = row['data_offsets']
            f.seek(n + 8 + a)
            remaining = b - a
            while remaining:
                chunk = f.read(min(8 * 1024**2, remaining))
                assert chunk
                remaining -= len(chunk)
                if row['dtype'] == 'F32':
                    original = np.frombuffer(chunk, dtype='<f4')
                    with np.errstate(over='ignore', invalid='ignore'):
                        converted = original.astype('<f2')
                    nonfinite += int(np.count_nonzero(~np.isfinite(original)))
                    nonfinite += int(np.count_nonzero(~np.isfinite(converted)))
                    assert not nonfinite, 'Nonfinite/overflow in vision storage conversion'
                    max_error = max(max_error, float(np.max(np.abs(converted.astype('<f4') - original))))
                    out.write(converted.tobytes())
                else:
                    out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    assert part.stat().st_size == 8 + len(header) + next_offset
    receipt = {'source_sha256': sha(source), 'output_sha256': sha(part),
               'output_bytes': part.stat().st_size, 'tensor_count': len(rows),
               'nonfinite_values': nonfinite, 'maximum_absolute_storage_rounding_error': max_error,
               'FP16_storage_only': True, 'PS5_executed': False, 'Windows_inference': False}
    part.rename(target)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    global OUT
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    OUT=parser.parse_args().out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {name: download(name, *data) for name, data in FILES.items()}
    license_url = 'https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/LICENSE'
    with urllib.request.urlopen(license_url, timeout=30) as response:
        license_bytes = response.read(65536)
    assert b'Apache License' in license_bytes and b'Version 2.0' in license_bytes
    (OUT / 'LICENSE-IP-Adapter.txt').write_bytes(license_bytes)
    receipt = {'commit': COMMIT, 'files': {name: {'sha256': sha(path), 'bytes': path.stat().st_size,
                'source_path': FILES[name][0]} for name, path in paths.items()},
               'vision_conversion': convert_vision(paths['clip-vision-vit-h-original-f32.safetensors']),
               'license_url': license_url, 'license_sha256': sha(OUT / 'LICENSE-IP-Adapter.txt'),
               'PS5_changes': 0, 'Windows_inference': False}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
