#!/usr/bin/env python3
"""skimage.feature.peak_local_max pins. python -I -B tools/peaks/pins.py
Writes Tests/ImagoFiltersTests/Fixtures/peak-pins.json: smooth noisy lattices (float32 and float64), tied integer images and a
constant image, over min_distance, thresholds, exclude_border, num_peaks and p_norm.

Each source image is stored once, in float64 (`images`); a case names its image by index and its dtype. A float32 case runs
on `image.astype(np.float32)`, which the Swift test rebuilds as `Float(value)` (both round to nearest, ties to even)."""
import json
from pathlib import Path

import numpy as np
import skimage
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

assert skimage.__version__ == "0.26.0"
rng = np.random.Generator(np.random.PCG64(20261021))


def out(a):
    return [float(f"{float(v):.17g}") for v in np.asarray(a).ravel()]


images = []
cases = []


def image(img):
    images.append(dict(h=img.shape[0], w=img.shape[1], x=out(img.astype(np.float64))))
    return len(images) - 1


def add(index, img, **kw):
    for dt in (np.float32, np.float64):
        x = img.astype(dt)
        ek = kw.get("exclude_border", True)
        res = peak_local_max(x, **kw)
        cases.append(dict(image=index, dtype=dt.__name__, min_distance=kw.get("min_distance", 1),
                          threshold_abs=kw.get("threshold_abs"), threshold_rel=kw.get("threshold_rel"),
                          border=(None if ek is True else (ek if isinstance(ek, tuple) else (int(ek), int(ek)))),
                          num_peaks=(None if "num_peaks" not in kw else kw["num_peaks"]), p_norm=kw.get("p_norm"),
                          peaks=res.tolist()))


for h, w in [(40, 50), (33, 33), (64, 48)]:
    img = ndi.gaussian_filter(rng.normal(size=(h, w)), 1.2) * 10
    i = image(img)
    for md in (1, 2, 3, 5):
        add(i, img, min_distance=md)
        add(i, img, min_distance=md, threshold_rel=0.3)
        add(i, img, min_distance=md, threshold_abs=0.5, exclude_border=False)
        add(i, img, min_distance=md, num_peaks=5)
        add(i, img, min_distance=md, p_norm=2)
        add(i, img, min_distance=md, exclude_border=(1, 3))
    add(i, img, min_distance=0, threshold_abs=1.0, exclude_border=0)
ties = rng.integers(0, 4, (30, 36)).astype(float)
i = image(ties)
for md in (1, 2, 4):
    add(i, ties, min_distance=md)
    add(i, ties, min_distance=md, num_peaks=7, p_norm=1)
const = np.full((12, 15), 3.0)
add(image(const), const, min_distance=2)
zeros = np.zeros((9, 9))
add(image(zeros), zeros, min_distance=1, exclude_border=False)
(Path(__file__).resolve().parents[2] / "Tests/ImagoFiltersTests/Fixtures/peak-pins.json").write_text(
    json.dumps(dict(skimage=skimage.__version__, images=images, cases=cases), separators=(",", ":")))
print(len(images), "images,", len(cases), "cases")
