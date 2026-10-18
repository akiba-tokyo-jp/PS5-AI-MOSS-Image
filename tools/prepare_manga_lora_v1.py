# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert the selected old GLoRA to an equivalent WAI-specific linear LoRA.

This performs model-format conversion on Windows, never image inference.
For this file all convolutions are 1x1. No SVD, rank truncation, or FP16
rounding is used: U=[W@a2,b2], D=[a1;b1], alpha'=2*alpha, rank'=2*rank.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import urllib.request
import numpy as np

SOURCE_NAME = 'Mnga-illustriousXL_v01_V1-CAME.safetensors'
SOURCE_SHA = '19c0156cd6b8d22524fedb4fd80c943810f476b4c3f0b1b604740ea12563d7d1'
SOURCE_URL = ('https://huggingface.co/limlimlimlim/illustrousManga/resolve/'
              '392a3528a095fea966274cf8eb3019d3ba79429c/' + SOURCE_NAME)
SOURCE_BYTES = 338887980


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class TensorFile:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as stream:
            n, = struct.unpack('<Q', stream.read(8))
            if not 0 < n <= 8 * 1024**2:
                raise ValueError('Invalid bounded safetensors header')
            raw = stream.read(n)
        self.header = json.loads(raw)
        self.metadata = self.header.pop('__metadata__', {})
        self.start = n + 8
        self.mapping = np.memmap(self.path, dtype=np.uint8, mode='r')
        intervals = []
        for name, row in self.header.items():
            if row['dtype'] not in ('F16', 'BF16', 'F32'):
                raise ValueError(f'Unsupported dtype: {name}')
            a, b = row['data_offsets']
            expected = math.prod(row['shape']) * (4 if row['dtype'] == 'F32' else 2)
            if a < 0 or b - a != expected or self.start + b > self.path.stat().st_size:
                raise ValueError(f'Invalid tensor extent: {name}')
            intervals.append((a, b))
        previous = 0
        for a, b in sorted(intervals):
            if a != previous:
                raise ValueError('Non-contiguous or overlapping tensor extents')
            previous = b
        if self.start + previous != self.path.stat().st_size:
            raise ValueError('Trailing or missing tensor data')

    def read(self, name):
        row = self.header[name]
        offset = self.start + row['data_offsets'][0]
        count = math.prod(row['shape'])
        if row['dtype'] == 'BF16':
            bits = np.frombuffer(self.mapping, dtype='<u2', count=count, offset=offset)
            value = (bits.astype('<u4') << np.uint32(16)).view('<f4')
        else:
            dtype = '<f2' if row['dtype'] == 'F16' else '<f4'
            value = np.frombuffer(self.mapping, dtype=dtype, count=count, offset=offset).astype('<f4')
        value = value.reshape(row['shape'])
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite source tensor: {name}')
        return value


def base_targets(base):
    result = {}
    for key, row in base.header.items():
        if not key.endswith('.weight') or len(row['shape']) not in (2, 4):
            continue
        if key.startswith('model.diffusion_model.'):
            name = 'lora_unet_' + key[len('model.diffusion_model.'):-7].replace('.', '_')
        elif key.startswith('conditioner.embedders.0.transformer.'):
            name = 'lora_te1_' + key[len('conditioner.embedders.0.transformer.'):-7].replace('.', '_')
        else:
            continue
        if name in result:
            raise ValueError('Ambiguous base tensor mapping')
        result[name] = (key, None)
    prefix = 'conditioner.embedders.1.model.transformer.resblocks.'
    for layer in range(32):
        native = prefix + str(layer) + '.'
        target = f'lora_te2_text_model_encoder_layers_{layer}_'
        for suffix, source in [('mlp_fc1', 'mlp.c_fc.weight'), ('mlp_fc2', 'mlp.c_proj.weight'),
                               ('self_attn_out_proj', 'attn.out_proj.weight')]:
            if native + source in base.header:
                result[target + suffix] = (native + source, None)
        fused = native + 'attn.in_proj_weight'
        if fused in base.header:
            shape = base.header[fused]['shape']
            if len(shape) != 2 or shape[0] != 3 * shape[1]:
                raise ValueError('Unexpected CLIP-G fused QKV dimensions')
            for index, letter in enumerate(('q', 'k', 'v')):
                result[target + f'self_attn_{letter}_proj'] = (fused, index)
    return result


def factors(weight, a1, a2, b1, b2, alpha):
    """Old LyCORIS GLoRA; retain the correction as a rank-2r adapter."""
    conv = weight.ndim == 4
    if weight.ndim not in (2, 4) or (conv and weight.shape[2:] != (1, 1)):
        raise ValueError('Only linear and 1x1 convolutions are supported')
    if not all(x.ndim == weight.ndim and (not conv or x.shape[2:] == (1, 1))
               for x in (a1, a2, b1, b2)):
        raise ValueError('GLoRA factors have unsupported dimensions')
    w = weight.reshape(weight.shape[:2])
    a1, a2, b1, b2 = (x.reshape(x.shape[:2]) for x in (a1, a2, b1, b2))
    rank, input_dim = a1.shape
    output_dim = w.shape[0]
    if rank <= 0 or w.shape[1] != input_dim or a2.shape != (input_dim, rank) or \
            b1.shape != (rank, input_dim) or b2.shape != (output_dim, rank):
        raise ValueError('Factor layout is not the selected old GLoRA convention')
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Invalid GLoRA alpha')
    down = np.concatenate((a1, b1), axis=0).astype('<f4')
    up = np.concatenate((w @ a2, b2), axis=1).astype('<f4')
    if not np.isfinite(up).all() or not np.isfinite(down).all():
        raise ValueError('Non-finite converted factors')
    if conv:
        down = down[:, :, None, None]
        up = up[:, :, None, None]
    return down, up, np.array(2 * alpha, dtype='<f4')


def acquire(destination):
    if destination.exists():
        if destination.stat().st_size != SOURCE_BYTES or sha(destination) != SOURCE_SHA:
            raise ValueError('Existing source does not match original distribution hash')
        return
    temporary = destination.with_suffix('.download.part')
    if temporary.exists():
        raise RuntimeError('Inspect previous partial download before another attempt')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response, temporary.open('xb') as stream:
        digest = hashlib.sha256()
        total = 0
        while chunk := response.read(1024**2):
            total += len(chunk)
            if total > SOURCE_BYTES:
                raise ValueError('Download exceeds the verified distribution size')
            digest.update(chunk)
            stream.write(chunk)
    if total != SOURCE_BYTES or digest.hexdigest() != SOURCE_SHA:
        raise ValueError('Downloaded weight hash or size differs from original')
    temporary.rename(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path('D:/PS5-AI/models/waiIllustriousSDXL_v170.safetensors'))
    parser.add_argument('--directory', type=Path, default=Path('D:/PS5-AI/models/loras'))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    original = args.directory / SOURCE_NAME
    acquire(original)
    source, base = TensorFile(original), TensorFile(args.base)
    if json.loads(source.metadata.get('ss_network_args', '{}')).get('algo') != 'glora':
        raise ValueError('Expected selected GLoRA source')
    expected = {'.a1.weight', '.a2.weight', '.b1.weight', '.b2.weight', '.alpha'}
    all_groups = sorted(k[:-10] for k in source.header if k.endswith('.a1.weight'))
    if len(all_groups) != 1014 or {g + s for g in all_groups for s in expected} != set(source.header):
        raise ValueError('Unexpected or incomplete source tensors')
    # WAI's CLIP-L layer 11 is deliberately unused with SDXL clip_skip=2.
    # Its base matrices are all non-finite; never replace them with zeroes or
    # inject their unused corrections. This app fixes the same clip_skip=2.
    excluded = [g for g in all_groups if g.startswith('lora_te1_text_model_encoder_layers_11_')]
    if len(excluded) != 6:
        raise ValueError('Unexpected CLIP-L final-layer target list')
    groups = [g for g in all_groups if g not in excluded]
    mapping = base_targets(base)
    if any(g not in mapping for g in groups):
        raise ValueError('Some GLoRA layers do not map to this WAI checkpoint')
    print('Hash WAI base; all output is specific to this exact checkpoint', flush=True)
    base_sha = sha(args.base)
    target = args.directory / 'manga-illustrious-wai-v17-r48-f32.safetensors'
    receipt = target.with_suffix('.conversion.json')
    if target.exists() or receipt.exists():
        raise RuntimeError('Preserve completed conversion; choose a new output version')
    header = {'__metadata__': {
        'modelspec.title': 'manga | Illustrious Style LoRA / WAI v17 conversion',
        'modelspec.architecture': 'stable-diffusion-xl-v1-base/lora',
        'ss_network_module': 'networks.lora', 'ss_network_dim': '48',
        'ps5_ai_source_author': 'imge',
        'ps5_ai_source_url': 'https://civitai.com/models/1745962?modelVersionId=1976014',
        'ps5_ai_source_sha256': SOURCE_SHA, 'ps5_ai_base_sha256': base_sha,
        'ps5_ai_conversion': 'old GLoRA: U=[W@a2,b2], D=[a1;b1], alpha=2*original alpha; F32; no SVD',
        'ps5_ai_required_clip_skip': '2',
        'ps5_ai_omitted_unused_layers': json.dumps(excluded),
    }}
    offset = 0
    for g in groups:
        shape = source.header[g + '.a1.weight']['shape']
        up_shape = source.header[g + '.b2.weight']['shape']
        for suffix, dims in [('lora_down.weight', [2 * shape[0], *shape[1:]]),
                             ('lora_up.weight', [up_shape[0], 2 * up_shape[1], *up_shape[2:]]),
                             ('alpha', [])]:
            size = math.prod(dims) * 4
            header[g + '.' + suffix] = {'dtype': 'F32', 'shape': dims, 'data_offsets': [offset, offset + size]}
            offset += size
    raw = json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    raw += b' ' * (-len(raw) % 8)
    temporary = target.with_suffix('.conversion.part')
    max_error = 0.0
    module_bytes = collections.Counter()
    with temporary.open('xb') as stream:
        stream.write(struct.pack('<Q', len(raw)))
        stream.write(raw)
        for index, g in enumerate(groups):
            key, qkv = mapping[g]
            w = base.read(key)
            if qkv is not None:
                unit = w.shape[0] // 3
                w = w[qkv * unit:(qkv + 1) * unit]
            a1, a2, b1, b2 = (source.read(g + '.' + n + '.weight') for n in ('a1', 'a2', 'b1', 'b2'))
            alpha = float(source.read(g + '.alpha'))
            down, up, new_alpha = factors(w, a1, a2, b1, b2, alpha)
            # Independent float64 selected-row/column reconstruction per actual layer.
            rows = np.unique(np.linspace(0, w.shape[0] - 1, min(7, w.shape[0]), dtype=int))
            cols = np.unique(np.linspace(0, w.shape[1] - 1, min(11, w.shape[1]), dtype=int))
            wf = w.reshape(w.shape[:2]).astype(np.float64)
            af, au, bf, bu = (x.reshape(x.shape[:2]).astype(np.float64) for x in (a1, a2, b1, b2))
            reference = ((wf[rows] @ au) @ af[:, cols] + bu[rows] @ bf[:, cols]) * (alpha / af.shape[0])
            actual = (up.reshape(up.shape[:2])[rows].astype(np.float64) @
                      down.reshape(down.shape[:2])[:, cols].astype(np.float64)) * (float(new_alpha) / down.shape[0])
            np.testing.assert_allclose(actual, reference, rtol=2e-4, atol=2e-6)
            max_error = max(max_error, float(np.max(np.abs(actual - reference))))
            for value in (down, up, new_alpha):
                stream.write(value.tobytes(order='C'))
                module_bytes['UNet' if g.startswith('lora_unet_') else 'TextEncoder'] += value.nbytes
            if index % 100 == 0:
                print(f'Converted and checked {index + 1}/{len(groups)} layers', flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    if temporary.stat().st_size != 8 + len(raw) + offset:
        raise ValueError('Converted output size mismatch')
    temporary.rename(target)
    check = TensorFile(target)
    if len(check.header) != 3 * len(groups):
        raise ValueError('Converted tensor count mismatch')
    result = {'state': 'converted_not_PS5_executed', 'source_path': str(original),
              'source_sha256': SOURCE_SHA, 'source_bytes': SOURCE_BYTES,
              'base_path': str(args.base), 'base_sha256': base_sha,
              'output_path': str(target), 'output_sha256': sha(target),
              'output_bytes': target.stat().st_size, 'output_dtype': 'F32',
              'layers': len(groups), 'output_tensors': len(check.header),
              'source_layers': len(all_groups), 'excluded_unused_clip_L_layer11': excluded,
              'required_clip_skip': 2,
              'module_weight_bytes': dict(module_bytes),
              'max_sampled_correction_abs_error_vs_float64': max_error,
              'all_layers_sampled_reconstruction_passed': True,
              'approximate_SVD_or_rank_truncation': False,
              'Windows_image_inference': False, 'PS5_actions': 0}
    receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
