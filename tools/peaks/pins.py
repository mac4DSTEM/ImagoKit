#!/usr/bin/env python3
"""skimage.feature.peak_local_max pins. python -I -B tools/peaks/pins.py
Writes Tests/ImagoFiltersTests/Fixtures/peak-pins.json: smooth noisy lattices (float32 and float64), tied integer images and a
constant image, over min_distance, thresholds, exclude_border, num_peaks and p_norm."""
import json
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

assert skimage.__version__ == "0.26.0"
rng = np.random.Generator(np.random.PCG64(20261021))


def out(a):
    return [float(f"{float(v):.17g}") for v in np.asarray(a).ravel()]


cases = []


def add(img, **kw):
    for dt in (np.float32, np.float64):
        x = img.astype(dt)
        ek = kw.get("exclude_border", True)
        args = dict(kw)
        res = peak_local_max(x, **args)
        cases.append(dict(h=x.shape[0], w=x.shape[1], dtype=dt.__name__, x=out(x), min_distance=kw.get("min_distance", 1),
                          threshold_abs=kw.get("threshold_abs"), threshold_rel=kw.get("threshold_rel"),
                          border=(None if ek is True else (ek if isinstance(ek, tuple) else (int(ek), int(ek)))),
                          num_peaks=(None if "num_peaks" not in kw else kw["num_peaks"]), p_norm=kw.get("p_norm"),
                          peaks=res.tolist()))


for h, w in [(40, 50), (33, 33), (64, 48)]:
    img = ndi.gaussian_filter(rng.normal(size=(h, w)), 1.2) * 10
    for md in (1, 2, 3, 5):
        add(img, min_distance=md)
        add(img, min_distance=md, threshold_rel=0.3)
        add(img, min_distance=md, threshold_abs=0.5, exclude_border=False)
        add(img, min_distance=md, num_peaks=5)
        add(img, min_distance=md, p_norm=2)
        add(img, min_distance=md, exclude_border=(1, 3))
    add(img, min_distance=0, threshold_abs=1.0, exclude_border=0)
ties = rng.integers(0, 4, (30, 36)).astype(float)
for md in (1, 2, 4):
    add(ties, min_distance=md)
    add(ties, min_distance=md, num_peaks=7, p_norm=1)
add(np.full((12, 15), 3.0), min_distance=2)
add(np.zeros((9, 9)), min_distance=1, exclude_border=False)
Path("Tests/ImagoFiltersTests/Fixtures/peak-pins.json").write_text(json.dumps(dict(skimage=skimage.__version__, cases=cases)))
print(len(cases))
