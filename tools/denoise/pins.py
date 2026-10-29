#!/usr/bin/env python3
"""Denoising pins: skimage.restoration.denoise_tv_chambolle / denoise_tv_bregman / denoise_nl_means (2D, one channel).
Run from the package root:  python3 -I -B tools/denoise/pins.py
Writes Tests/ImagoFiltersTests/Fixtures/denoise-pins.json.

Images are float32 draws (a smooth ramp plus noise, a step edge plus noise, tiny and odd sizes, a constant image); the
float64 cases are the same pixels widened, so every image runs in both dtypes and the float32 case is the library's own
float32 output. Floats are written as 17 significant digits (round trip). Chambolle is pure NumPy in 0.26.0; Bregman and
NL-means are compiled Cython kernels, so the compiled package is the oracle (NL-means weighs patches with a
Schraudolph-style exp approximation).
"""
import json
from pathlib import Path

import numpy as np
import scipy
import skimage
from skimage.restoration import denoise_nl_means, denoise_tv_bregman, denoise_tv_chambolle

assert np.__version__ == "2.5.4" and scipy.__version__ == "1.18.1" and skimage.__version__ == "0.26.0"
rng = np.random.Generator(np.random.PCG64(20261023))
ROOT = Path(__file__).resolve().parents[2]


def flat(a):
    a = np.asarray(a).ravel()
    return [float(f"{float(v):.17g}") for v in a]


def ramp(h, w, noise):
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.2 + 0.6 * (xx / max(w - 1, 1)) * 0.7 + 0.3 * (yy / max(h - 1, 1))
    return (base + noise * rng.normal(size=(h, w))).astype(np.float32)


def step(h, w, noise):
    base = np.where(np.arange(w)[None, :] < w // 2, 0.25, 0.75) * np.ones((h, 1))
    return (base + noise * rng.normal(size=(h, w))).astype(np.float32)


# (name, float32 image); the float64 twin is the same pixels widened
IMAGES = [
    ("ramp 16x18", ramp(16, 18, 0.08)),
    ("step 15x20", step(15, 20, 0.1)),
    ("ramp 13x11", ramp(13, 11, 0.05)),
    ("step 9x7", step(9, 7, 0.15)),
    ("normal 1x1", rng.normal(0.5, 0.2, (1, 1)).astype(np.float32)),
    ("normal 1x9", rng.normal(0.5, 0.2, (1, 9)).astype(np.float32)),
    ("normal 8x1", rng.normal(0.5, 0.2, (8, 1)).astype(np.float32)),
    ("normal 2x2", rng.normal(0.5, 0.2, (2, 2)).astype(np.float32)),
    ("normal 3x5", rng.normal(0.5, 0.2, (3, 5)).astype(np.float32)),
    ("constant 9x8", np.full((9, 8), 0.37, np.float32)),
    ("mixed 10x12", (rng.normal(0.0, 1.0, (10, 12)) * 3.0 + 20.0).astype(np.float32)),
    # large values against h: distances far above the cutoff of 5 reach the weight function (saturation and wrap-around)
    ("large 8x9", (rng.normal(0.0, 1.0, (8, 9)) * 1000.0).astype(np.float32)),
]
images = []
cases = []

CHAMBOLLE = [  # weight, eps, max_num_iter
    (0.1, 2.0e-4, 200), (0.5, 1.0e-4, 50), (0.3, 1.0e-6, 100), (1.0, 0.1, 200), (0.05, 2.0e-4, 5), (2.0, 1.0e-3, 300),
]
BREGMAN = [  # weight, max_num_iter, eps, isotropic
    (5.0, 100, 1.0e-3, True), (5.0, 100, 1.0e-3, False), (1.0, 50, 1.0e-6, True), (10.0, 200, 0.1, False),
    (0.5, 30, 1.0e-4, True), (2.0, 7, 1.0e-12, False),
]
NLM = [  # patch_size, patch_distance, h, fast_mode, sigma
    (7, 11, 0.1, True, 0.0), (7, 11, 0.1, False, 0.0), (5, 6, 0.08, True, 0.05), (5, 6, 0.08, False, 0.05),
    (3, 4, 0.15, True, 0.0), (3, 4, 0.15, False, 0.0), (4, 3, 0.2, True, 0.1), (4, 3, 0.2, False, 0.1),
    (9, 5, 0.3, True, 0.02), (9, 5, 0.3, False, 0.02), (6, 2, 0.5, False, 0.0), (6, 2, 0.5, True, 0.0),
]
LARGE = {"ramp 16x18", "step 15x20"}  # the big NL-means grids run on a subset of the parameter sets


def add(kind, image_id, dtype, params, y):
    cases.append(dict(kind=kind, image=image_id, dtype=dtype, params=params, y=flat(y)))


for name, x32 in IMAGES:
    h, w = x32.shape
    for dtype in (np.float32, np.float64):
        x = x32.astype(dtype)
        image_id = len(images)
        images.append(dict(id=image_id, name=name, h=h, w=w, dtype=np.dtype(dtype).name, x=flat(x)))
        for weight, eps, it in CHAMBOLLE:
            y = denoise_tv_chambolle(x, weight=weight, eps=eps, max_num_iter=it)
            assert y.dtype == x.dtype and y.shape == x.shape
            add("chambolle", image_id, dtype.__name__, dict(weight=weight, eps=eps, max_num_iter=it), y)
        if h >= 2 and w >= 2:  # a 1-pixel axis reads outside the image in the Cython kernel; the kit refuses it
            for weight, it, eps, iso in BREGMAN:
                y = denoise_tv_bregman(x, weight=weight, max_num_iter=it, eps=eps, isotropic=iso)
                assert y.dtype == x.dtype and y.shape == x.shape
                add("bregman", image_id, dtype.__name__, dict(weight=weight, max_num_iter=it, eps=eps, isotropic=iso), y)
        for k, (s, d, hh, fast, sigma) in enumerate(NLM):
            if name in LARGE and k % 2 == 1 and s == 7:
                continue  # skip the slow default on the two big grids (11 x 11 search at 7 x 7 patches)
            y = denoise_nl_means(x, patch_size=s, patch_distance=d, h=hh, fast_mode=fast, sigma=sigma)
            assert y.dtype == x.dtype and y.size == x.size and np.all(np.isfinite(y)), (name, s, d, hh, fast, sigma)
            add("nlmeans", image_id, dtype.__name__,
                dict(patch_size=s, patch_distance=d, h=hh, fast_mode=fast, sigma=sigma), y)

out = ROOT / "Tests" / "ImagoFiltersTests" / "Fixtures" / "denoise-pins.json"
out.write_text(json.dumps(dict(reference="scikit-image 0.26.0 restoration.denoise_*", images=images, cases=cases)))
print(f"wrote {out} with {len(images)} images and {len(cases)} cases")
