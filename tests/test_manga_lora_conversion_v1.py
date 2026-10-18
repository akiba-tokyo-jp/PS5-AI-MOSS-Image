"""Small independent algebra checks; no model inference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import prepare_manga_lora_v1 as converter
import numpy as np

rng = np.random.default_rng(20261018)
checks = 0
for out_dim, in_dim, rank in [(7, 9, 2), (9, 7, 3), (8, 8, 2)]:
    for conv in (False, True):
        w = rng.normal(size=(out_dim, in_dim)).astype('float32')
        a1 = rng.normal(size=(rank, in_dim)).astype('float32')
        a2 = rng.normal(size=(in_dim, rank)).astype('float32')
        b1 = rng.normal(size=(rank, in_dim)).astype('float32')
        b2 = rng.normal(size=(out_dim, rank)).astype('float32')
        alpha = rank * 0.7
        arrays = [x[:, :, None, None] for x in (w, a1, a2, b1, b2)] if conv else [w, a1, a2, b1, b2]
        down, up, new_alpha = converter.factors(*arrays, alpha)
        down, up = down.reshape(2 * rank, in_dim), up.reshape(out_dim, 2 * rank)
        x = rng.normal(size=(in_dim, 5))
        for strength in (0, 0.25, 1, 1.5):
            reference = w @ x + strength * (alpha / rank) * (w @ (a2 @ (a1 @ x)) + b2 @ (b1 @ x))
            actual = w @ x + strength * (float(new_alpha) / (2 * rank)) * (up @ (down @ x))
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=1e-5)
            checks += 1
try:
    converter.factors(np.zeros((7, 9, 3, 3)), *[np.zeros((2, 2, 1, 1))] * 4, 2)
except ValueError:
    checks += 1
else:
    raise AssertionError('Unsupported convolution must be rejected')
print({'algebra_checks_passed': checks, 'model_inference': False})
