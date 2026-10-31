#!/usr/bin/env python3
"""Generate the peak_local_max footprint and labels pins.
Run from the package root:  python3 -I -B tools/peak-footprint-labels/pins.py
(a venv with only scikit-image==0.26.0, which brings NumPy 2.5.4 and SciPy 1.18.1). Light: a few small images, seconds.

Writes Tests/ImagoFiltersTests/Fixtures/peak-footprint-labels-pins.json.

Reference: scikit-image 0.26.0 `feature.peak_local_max` with `footprint=` (the maximum filter over that footprint, SciPy's
origin at size // 2, mode 'nearest') and with `labels=` (each label in its bounding box after the border is cut, pixels
of other labels set to the dtype's lowest finite value, the trivial-image rule with the isolated pixels of the label
mask, `num_peaks_per_label`, then a global re-sort and re-spacing when more than `num_peaks` remain). Coordinates are
integers: exact, in order. Each source image is stored once in float64; a float32 case runs on `image.astype(np.float32)`.
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert skimage.__version__ == "0.26.0", skimage.__version__
assert platform.machine() == "arm64", platform.machine()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests/ImagoFiltersTests/Fixtures/peak-footprint-labels-pins.json"
rng = np.random.Generator(np.random.PCG64(20261031))


def out(a):
    return [float(f"{float(v):.17g}") for v in np.asarray(a).ravel()]


images, label_images, footprints, cases = [], [], [], []


def add_image(img):
    images.append(dict(h=img.shape[0], w=img.shape[1], x=out(img.astype(np.float64))))
    return len(images) - 1


def add_labels(lab):
    label_images.append(dict(h=lab.shape[0], w=lab.shape[1], labels=lab.astype(int).ravel().tolist()))
    return len(label_images) - 1


def add_footprint(fp):
    fp = np.asarray(fp, dtype=bool)
    footprints.append(dict(h=fp.shape[0], w=fp.shape[1], mask=fp.ravel().tolist()))
    return len(footprints) - 1


def add(i, img, fp=None, lab=None, **kw):
    for dt in (np.float32, np.float64):
        x = img.astype(dt)
        call = dict(kw)
        if fp is not None:
            call["footprint"] = np.asarray(footprints[fp]["mask"], dtype=bool).reshape(footprints[fp]["h"], footprints[fp]["w"])
        if lab is not None:
            call["labels"] = np.asarray(label_images[lab]["labels"]).reshape(label_images[lab]["h"], label_images[lab]["w"])
        res = peak_local_max(x, **call)
        ek = kw.get("exclude_border", True)
        cases.append(dict(image=i, dtype=dt.__name__, footprint=fp, labels=lab, min_distance=kw.get("min_distance", 1),
                          threshold_abs=kw.get("threshold_abs"), threshold_rel=kw.get("threshold_rel"),
                          border=(None if ek is True else (ek if isinstance(ek, tuple) else (int(ek), int(ek)))),
                          num_peaks=kw.get("num_peaks"), num_peaks_per_label=kw.get("num_peaks_per_label"),
                          p_norm=kw.get("p_norm"), peaks=res.tolist()))


FP = [
    add_footprint(np.ones((1, 1))),
    add_footprint([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    add_footprint([[1, 1, 0], [0, 1, 1]]),           # asymmetric, even rows: the origin at size // 2
    add_footprint(np.ones((1, 5))),
    add_footprint(np.ones((4, 4))),
    add_footprint([[1, 0, 0, 0, 1], [0, 0, 1, 0, 0], [1, 0, 0, 0, 1]]),
]

# Smooth noisy images.
for h, w in [(31, 37), (24, 24)]:
    img = ndi.gaussian_filter(rng.normal(size=(h, w)), 1.3) * 10
    i = add_image(img)
    for fp in FP:
        for md in (1, 3):
            add(i, img, fp=fp, min_distance=md)
        add(i, img, fp=fp, min_distance=2, threshold_rel=0.2, exclude_border=False)
        add(i, img, fp=fp, min_distance=1, exclude_border=(2, 0), num_peaks=5, p_norm=2.0)
    # Labels: a grid of regions, a constant region (the trivial rule), a single-pixel label and a missing label number.
    lab = np.zeros((h, w), dtype=int)
    lab[2:12, 2:15] = 1
    lab[2:12, 16:w - 1] = 2
    lab[14:h - 2, 3:10] = 4                           # label 3 absent
    lab[14:h - 2, 12:w - 4] = 5
    lab[13, 11] = 6                                   # a single pixel
    lab[0:3, 0:3] = 7                                 # a label in the corner, cut by the border
    flat = img.copy()
    flat[lab == 5] = 2.5                              # constant inside label 5
    j = add_image(flat)
    L = add_labels(lab)
    for md in (1, 2, 4):
        add(j, flat, lab=L, min_distance=md)
        add(j, flat, lab=L, min_distance=md, exclude_border=False)
    add(j, flat, lab=L, min_distance=1, num_peaks_per_label=2)
    add(j, flat, lab=L, min_distance=2, num_peaks=4)
    add(j, flat, lab=L, min_distance=3, num_peaks=3, num_peaks_per_label=2, exclude_border=(1, 3))
    add(j, flat, lab=L, min_distance=1, threshold_rel=0.4, exclude_border=False)
    add(j, flat, lab=L, min_distance=2, threshold_abs=-1.0, p_norm=1.0, exclude_border=False)
    for fp in FP[1:4]:
        add(j, flat, fp=fp, lab=L, min_distance=1)
        add(j, flat, fp=fp, lab=L, min_distance=2, num_peaks=6, exclude_border=False)

# Tied integer image with labels touching each other.
tied = rng.integers(0, 4, (16, 18)).astype(np.float64)
k = add_image(tied)
lab2 = (np.arange(16)[:, None] // 4) * 10 + (np.arange(18)[None, :] // 6) + 1
L2 = add_labels(lab2)
for md in (1, 2):
    add(k, tied, lab=L2, min_distance=md, exclude_border=False)
    add(k, tied, fp=FP[1], lab=L2, min_distance=md, exclude_border=False, num_peaks=7)
    add(k, tied, fp=FP[2], min_distance=md, exclude_border=False)

# Footprints that a flip or a wrong origin would change, and label edge cases.
ODD = [add_footprint([[1, 0], [0, 0]]), add_footprint([[1, 1, 0], [0, 0, 0], [0, 0, 1]]), add_footprint([[1, 0, 1]])]
base = np.asarray(images[0]["x"]).reshape(31, 37)
for fp in ODD:
    add(0, base, fp=fp, min_distance=1, exclude_border=False)
    add(k, tied, fp=fp, min_distance=2, exclude_border=False)
lab3 = np.zeros((31, 37), dtype=int)
lab3[3:14, 3:20] = -2                                 # a negative label: background
lab3[3:14, 21:34] = 1000                              # a large gap in the label numbers
lab3[15:28, 3:16] = 1
lab3[15:28, 17:34] = 2                                # beside label 1: the global re-spacing crosses labels
for t in range(10):
    lab3[16 + t, 18 + t] = 3                          # a one-pixel-wide diagonal label
img3 = base.copy()
img3[lab3 == 3] = 1.25                                # constant on the diagonal: the opening removes every pixel
L3 = add_labels(lab3)
m = add_image(img3)
add(m, img3, lab=L3, min_distance=0, exclude_border=False)
add(m, img3, fp=FP[1], lab=L3, min_distance=0, exclude_border=False)
add(m, img3, lab=L3, min_distance=1, num_peaks_per_label=0)
add(m, img3, lab=L3, min_distance=3, num_peaks=5, exclude_border=False)
add(m, img3, fp=ODD[1], lab=L3, min_distance=2, exclude_border=False)

OUT.write_text(json.dumps(dict(numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
                               machine=platform.machine(), seed=20261031, images=images, labels=label_images,
                               footprints=footprints, cases=cases)))
print(len(cases), "cases ->", OUT.relative_to(ROOT))
