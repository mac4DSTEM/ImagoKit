#!/usr/bin/env python3
"""Generate the multichannel band-pass pins: skimage.filters.butterworth and difference_of_gaussians with channel_axis.
Run from the package root:  python3 -I -B tools/bandpass-channels/pins.py
(a venv with only scikit-image==0.26.0, which brings NumPy 2.5.4 and SciPy 1.18.1). Light: a few small images, seconds.

Writes Tests/ImagoFiltersTests/Fixtures/bandpass-channel-pins.json.

Reference: scikit-image 0.26.0 `filters.butterworth(..., channel_axis=-1)` (the mask on the spatial axes, broadcast over the
channels; `npad` pads every axis, the channel axis too, and the crop takes it back) and `filters.difference_of_gaussians(...,
channel_axis=-1)` (sigma 0 on the channel axis). Images are channel-last (rows, columns, channels), float32 normal draws; the
float32 case is the library's own float32 output, the float64 case the same pixels widened. Also recorded: each channel
filtered alone, which the multichannel answer must equal, and channel_axis=0 on the transposed image. Tolerance as the
single-channel filter pins (assert_allclose dtype defaults; the wheel ships no tests). Taken on arm64.
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import skimage
from skimage import filters

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert skimage.__version__ == "0.26.0", skimage.__version__
assert platform.machine() == "arm64", platform.machine()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests/ImagoFiltersTests/Fixtures/bandpass-channel-pins.json"
rng = np.random.Generator(np.random.PCG64(20261031))


def out(a):
    a = np.asarray(a).ravel()
    return [float(f"{float(v):.17g}") for v in a]


def img(h, w, ch):
    return rng.normal(50, 20, (h, w, ch)).astype(np.float32)


def same(multi, single_fn, x):
    """The multichannel answer equals each channel filtered alone, to float32 rounding of the transform (the Swift test's
    float32 gate: 1e-5 of the largest magnitude)."""
    alone = np.stack([single_fn(x[..., k]) for k in range(x.shape[-1])], axis=-1)
    np.testing.assert_allclose(multi, alone, rtol=0, atol=1e-5 * max(1.0, float(np.abs(alone).max())))


butter = []
for h, w, ch in [(16, 16, 3), (17, 23, 2), (7, 12, 4), (9, 9, 1)]:
    x = img(h, w, ch)
    for cutoff, high_pass, order, sq, npad in [(0.07, True, 2.0, True, 0), (0.2, False, 3.5, False, 0),
                                              (0.07, True, 2.0, True, 5), (0.01, False, 2.0, True, 3)]:
        kw = dict(cutoff_frequency_ratio=cutoff, high_pass=high_pass, order=order, squared_butterworth=sq, npad=npad)
        f32 = filters.butterworth(x, channel_axis=-1, **kw)
        f64 = filters.butterworth(x.astype(np.float64), channel_axis=-1, **kw)
        assert f32.dtype == np.float32 and f64.dtype == np.float64
        same(f32, lambda c: filters.butterworth(c, **kw), x)
        first = filters.butterworth(np.moveaxis(x, -1, 0).astype(np.float64), channel_axis=0, **kw)
        np.testing.assert_allclose(np.moveaxis(first, 0, -1), f64, rtol=1e-12, atol=1e-12)
        butter.append(dict(h=h, w=w, channels=ch, cutoff=cutoff, high=high_pass, order=order, squared=sq, npad=npad,
                           x=out(x), f32=out(f32), f64=out(f64)))
dog = []
for h, w, ch in [(17, 23, 3), (8, 8, 2), (20, 11, 1)]:
    x = img(h, w, ch)
    for low, high in [((1.0, 1.0), None), ((2.0, 3.0), (4.0, 5.0)), ((1.2, 1.2), (3.3, 2.2))]:
        f32 = filters.difference_of_gaussians(x, low, high, channel_axis=-1)
        f64 = filters.difference_of_gaussians(x.astype(np.float64), low, high, channel_axis=-1)
        assert f32.dtype == np.float32 and f64.dtype == np.float64
        same(f32, lambda c: filters.difference_of_gaussians(c, low, high), x)
        dog.append(dict(h=h, w=w, channels=ch, low=low, high=high, x=out(x), f32=out(f32), f64=out(f64)))
OUT.write_text(json.dumps(dict(numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
                               machine=platform.machine(), seed=20261031, butter=butter, dog=dog)))
print(len(butter), "butterworth,", len(dog), "difference_of_gaussians ->", OUT.relative_to(ROOT))
