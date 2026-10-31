#!/usr/bin/env python3
"""Denoising pins for multichannel and integer images: skimage.restoration.denoise_tv_chambolle / denoise_tv_bregman /
denoise_nl_means with channel_axis=-1, and with integer input (converted by img_as_float, or kept by preserve_range).
Run from the package root:  python3 -I -B tools/denoise-channels/pins.py
Writes Tests/ImagoFiltersTests/Fixtures/denoise-channel-pins.json.

Float images are channel-last float32 draws; the float64 cases are the same pixels widened. Integer images hold random
values over the whole range of their type plus its minimum, maximum and zero; their samples are written as decimal
strings (exact for 64-bit types) and every integer case returns float64. Floats are written as 17 significant digits.
"""
import json
from pathlib import Path

import numpy as np
import scipy
import skimage
from skimage.restoration import denoise_nl_means, denoise_tv_bregman, denoise_tv_chambolle

assert np.__version__ == "2.5.4" and scipy.__version__ == "1.18.1" and skimage.__version__ == "0.26.0"
rng = np.random.Generator(np.random.PCG64(20261031))
ROOT = Path(__file__).resolve().parents[2]


def flat(a):
    return [float(f"{float(v):.17g}") for v in np.asarray(a).ravel()]


def colour(h, w, c, noise):
    yy, xx = np.mgrid[0:h, 0:w]
    planes = [0.2 + 0.5 * (xx / max(w - 1, 1)) * (k + 1) / c + 0.3 * (yy / max(h - 1, 1)) * (c - k) / c for k in range(c)]
    base = np.stack(planes, axis=-1)
    return (base + noise * rng.normal(size=(h, w, c))).astype(np.float32)


FLOAT_IMAGES = [
    ("colour 12x10x3", colour(12, 10, 3, 0.08)),
    ("colour 9x7x2", colour(9, 7, 2, 0.12)),
    ("colour 2x2x3", colour(2, 2, 3, 0.2)),
    ("colour 1x9x2", colour(1, 9, 2, 0.1)),
    ("colour 6x5x4", (rng.normal(0.0, 1.0, (6, 5, 4)) * 3.0 + 10.0).astype(np.float32)),
]

INT_TYPES = [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]


def integer_image(dtype, h, w, c):
    info = np.iinfo(dtype)
    a = rng.integers(info.min, info.max, size=(h, w, c), dtype=dtype, endpoint=True)
    flatview = a.reshape(-1)
    flatview[0], flatview[1], flatview[2] = info.min, info.max, 0
    if c == 1:
        a = a[..., 0]
    return a


CHAMBOLLE = [(0.1, 2.0e-4, 200), (0.5, 1.0e-4, 50)]
BREGMAN = [(5.0, 100, 1.0e-3, True), (2.0, 30, 1.0e-6, False)]
NLM = [(5, 3, 0.1, True, 0.0), (5, 3, 0.1, False, 0.0), (3, 2, 0.2, True, 0.05), (4, 2, 0.15, False, 0.02)]

images, cases = [], []


def add_image(name, x, channels, dtype_name, values):
    h, w = x.shape[0], x.shape[1]
    images.append(dict(id=len(images), name=name, h=h, w=w, channels=channels, dtype=dtype_name, x=values))
    return len(images) - 1


def add(kind, image_id, dtype, params, y):
    cases.append(dict(kind=kind, image=image_id, dtype=dtype, params=params, y=flat(y)))


def run_all(x, image_id, channel_axis, out_dtype, integer):
    h, w = x.shape[0], x.shape[1]
    for weight, eps, it in CHAMBOLLE:
        y = denoise_tv_chambolle(x, weight=weight, eps=eps, max_num_iter=it, channel_axis=channel_axis)
        assert y.dtype == out_dtype and y.shape == x.shape
        add("chambolle", image_id, out_dtype.name, dict(weight=weight, eps=eps, max_num_iter=it), y)
    if h >= 2 and w >= 2:
        for weight, it, eps, iso in BREGMAN:
            y = denoise_tv_bregman(x, weight=weight, max_num_iter=it, eps=eps, isotropic=iso, channel_axis=channel_axis)
            assert y.dtype == out_dtype and y.shape == x.shape
            add("bregman", image_id, out_dtype.name, dict(weight=weight, max_num_iter=it, eps=eps, isotropic=iso), y)
    for s, d, hh, fast, sigma in NLM:
        for preserve in ((False, True) if integer else (False,)):
            y = denoise_nl_means(x, patch_size=s, patch_distance=d, h=hh, fast_mode=fast, sigma=sigma,
                                 preserve_range=preserve, channel_axis=channel_axis)
            # the compiled kernel squeezes an axis of length 1 away; the samples keep their order
            assert y.dtype == out_dtype and y.size == x.size and np.all(np.isfinite(y))
            add("nlmeans", image_id, out_dtype.name,
                dict(patch_size=s, patch_distance=d, h=hh, fast_mode=fast, sigma=sigma, preserve_range=preserve), y)


for name, x32 in FLOAT_IMAGES:
    for dtype in (np.float32, np.float64):
        x = x32.astype(dtype)
        image_id = add_image(name, x, x.shape[2], np.dtype(dtype).name, flat(x))
        run_all(x, image_id, -1, np.dtype(dtype), integer=False)

for dtype in INT_TYPES:
    for (h, w, c) in ((7, 6, 1), (5, 4, 3)):
        x = integer_image(dtype, h, w, c)
        name = f"{np.dtype(dtype).name} {h}x{w}x{c}"
        image_id = add_image(name, x, c, np.dtype(dtype).name, [str(int(v)) for v in x.ravel()])
        run_all(x, image_id, None if c == 1 else -1, np.dtype(np.float64), integer=True)

out = ROOT / "Tests" / "ImagoFiltersTests" / "Fixtures" / "denoise-channel-pins.json"
out.write_text(json.dumps(dict(reference="scikit-image 0.26.0 restoration.denoise_* channel_axis=-1 and integer input",
                               images=images, cases=cases)))
print(f"wrote {out} with {len(images)} images and {len(cases)} cases")
