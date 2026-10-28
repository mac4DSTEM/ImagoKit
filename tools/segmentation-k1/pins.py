#!/usr/bin/env python3
"""Generate the K1 pins: distance_transform_edt, segmentation.watershed, restoration.rolling_ball.
Run from the package root:  python3 -I -B tools/segmentation-k1/pins.py
(a venv with only scikit-image==0.26.0, which brings NumPy 2.5.4 and SciPy 1.18.1).

Writes Tests/ImagoCoreTests/Fixtures/watershed-edt-rollingball-pins.json.

Reference: SciPy 1.18.1 `ndimage.distance_transform_edt` (decimal 6 in its tests; the values are sqrt of exact squared distances, so
the pins are held to 1e-12), scikit-image 0.26.0 `segmentation.watershed` (labels exact) and `restoration.rolling_ball`
(`assert_allclose` dtype defaults: rtol 1e-7 Float64, 1e-5 Float32; the scikit-image wheel ships no tests).
Inputs are deterministic (PCG64 seed 20261022); floats are written as repr (round trip); no NaN/inf except where stated.
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage import feature, restoration, segmentation

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert skimage.__version__ == "0.26.0", skimage.__version__
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests/ImagoCoreTests/Fixtures/watershed-edt-rollingball-pins.json"
rng = np.random.Generator(np.random.PCG64(20261022))
CMD = "python3 -I -B tools/segmentation-k1/pins.py"


def fl(a):
    v = [float(x) for x in np.asarray(a).flat]
    return v


def it(a):
    return [int(x) for x in np.asarray(a).flat]


def bits(a):
    return "".join("1" if v else "0" for v in np.asarray(a).flat)


# ---------- distance_transform_edt ----------
edt_cases = []


def add_edt(name, mask, sampling=None):
    mask = np.asarray(mask, bool)
    d = ndi.distance_transform_edt(mask) if sampling is None else ndi.distance_transform_edt(mask, sampling=sampling)
    edt_cases.append({"name": name, "height": mask.shape[0], "width": mask.shape[1], "mask": bits(mask),
                      "sampling": None if sampling is None else list(sampling), "distance": fl(d),
                      "hasBackground": bool((~mask).any())})


add_edt("single-pixel-foreground", np.ones((1, 1), bool))
add_edt("single-pixel-background", np.zeros((1, 1), bool))
add_edt("all-background", np.zeros((4, 6), bool))
add_edt("all-foreground", np.ones((4, 6), bool))
m = np.ones((9, 11), bool); m[4, 5] = False
add_edt("one-hole", m)
add_edt("one-hole-anisotropic", m, (0.5, 2.0))
for k, (h, w, p) in enumerate([(16, 16, 0.5), (23, 31, 0.8), (40, 25, 0.3), (7, 64, 0.6)]):
    mk = rng.random((h, w)) < p
    add_edt(f"random-{k}", mk)
    add_edt(f"random-{k}-sampling", mk, (1.5, 0.75))
blob = np.zeros((40, 40), bool)
yy, xx = np.mgrid[:40, :40]
blob |= (yy - 14) ** 2 + (xx - 14) ** 2 < 100
blob |= (yy - 24) ** 2 + (xx - 28) ** 2 < 81
add_edt("two-discs", blob)

# ---------- watershed ----------
ws_cases = []


def add_ws(name, image, markers, mask=None, connectivity=1):
    image = np.asarray(image, float)
    markers = np.asarray(markers, np.int32)
    out = segmentation.watershed(image, markers, mask=mask, connectivity=connectivity)
    ws_cases.append({"name": name, "height": image.shape[0], "width": image.shape[1], "image": fl(image), "markers": it(markers),
                     "mask": None if mask is None else bits(mask), "connectivity": connectivity, "labels": it(out)})


def peaks_to_markers(img, **kw):
    coords = feature.peak_local_max(img, **kw)
    mk = np.zeros(img.shape, np.int32)
    for i, (r, c) in enumerate(coords):
        mk[r, c] = i + 1
    return mk, coords


smooth = ndi.gaussian_filter(rng.random((48, 53)), 2.5)
mk, coords = peaks_to_markers(-smooth, min_distance=3)
add_ws("smooth-peaks-4", smooth, mk, connectivity=1)
add_ws("smooth-peaks-8", smooth, mk, connectivity=2)
msk = rng.random((48, 53)) < 0.85
msk = ndi.binary_opening(msk, iterations=1) | (mk > 0)
add_ws("smooth-peaks-masked", smooth, mk, mask=msk, connectivity=1)
# integer-valued images: many exact ties (the heap's age rule decides)
ints = rng.integers(0, 4, size=(30, 34)).astype(float)
imk = np.zeros((30, 34), np.int32)
pts = rng.choice(30 * 34, size=9, replace=False)
for i, p in enumerate(pts):
    imk[p // 34, p % 34] = i + 1
add_ws("integer-ties-4", ints, imk, connectivity=1)
add_ws("integer-ties-8", ints, imk, connectivity=2)
flat = np.zeros((20, 20)); fmk = np.zeros((20, 20), np.int32)
fmk[3, 3] = 1; fmk[3, 16] = 2; fmk[16, 9] = 3; fmk[10, 10] = 4
add_ws("flat-four-seeds", flat, fmk, connectivity=1)
add_ws("flat-four-seeds-8", flat, fmk, connectivity=2)
# tie-heavy stress: many markers on images of two or three levels, 4- and 8-connected, with and without a mask (the heap decides)
for k, (h, w, levels, nm) in enumerate([(17, 19, 2, 20), (25, 25, 3, 40), (33, 21, 2, 60), (40, 40, 3, 15)]):
    img = rng.integers(0, levels, size=(h, w)).astype(float)
    mkr = np.zeros((h, w), np.int32)
    for i, p in enumerate(rng.choice(h * w, size=nm, replace=False)):
        mkr[p // w, p % w] = i + 1
    mk2 = rng.random((h, w)) < 0.9
    add_ws(f"stress-{k}-4", img, mkr, connectivity=1)
    add_ws(f"stress-{k}-8", img, mkr, connectivity=2)
    add_ws(f"stress-{k}-masked-8", img, mkr, mask=mk2, connectivity=2)
# the classic: two touching discs, markers from the distance transform's peaks
d = ndi.distance_transform_edt(blob)
dmk, _ = peaks_to_markers(d, min_distance=5, labels=blob.astype(int))
add_ws("two-discs-distance", -d, dmk, mask=blob, connectivity=1)
add_ws("two-discs-distance-8", -d, dmk, mask=blob, connectivity=2)
# a marker outside the mask is dropped (markers * mask); markers with non-consecutive labels
sparse = np.zeros((15, 15), np.int32); sparse[2, 2] = 7; sparse[12, 12] = 3
add_ws("non-consecutive-labels", ndi.gaussian_filter(rng.random((15, 15)), 1.5), sparse)
# one-pixel image and a single marker
add_ws("one-pixel", np.array([[0.5]]), np.array([[1]], np.int32))

# ---------- rolling_ball ----------
rb_cases = []


def add_rb(name, image, radius=None, kernel=None):
    image = np.asarray(image)
    bg = restoration.rolling_ball(image, radius=radius) if kernel is None else restoration.rolling_ball(image, kernel=kernel)
    rb_cases.append({"name": name, "height": image.shape[0], "width": image.shape[1], "dtype": str(image.dtype),
                     "image": fl(image), "radius": radius,
                     "kernel": None if kernel is None else {"height": kernel.shape[0], "width": kernel.shape[1], "values": [(None if np.isinf(x) else float(x)) for x in kernel.flat]},
                     "background": fl(bg)})


base = ndi.gaussian_filter(rng.random((26, 31)) * 100, 4)
feat = np.zeros((26, 31)); feat[9:13, 12:17] = 60
add_rb("hat-r3", base + feat, 3)
add_rb("hat-r5", base + feat, 5)
add_rb("hat-r2.5", base + feat, 2.5)
add_rb("noise-r4", rng.random((19, 22)) * 50, 4)
add_rb("float32-r3", (base + feat).astype(np.float32), 3)
add_rb("ellipsoid", base + feat, kernel=restoration.ellipsoid_kernel((7, 11), 6))
add_rb("flat", np.full((9, 9), 5.0), 3)
add_rb("larger-than-image", base[:8, :9], 12)

OUT.write_text(json.dumps({
    "meta": {"skimage": skimage.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
             "python": platform.python_version(), "generator": CMD},
    "edt": edt_cases, "watershed": ws_cases, "rollingBall": rb_cases}, indent=0) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}: {len(edt_cases)} edt, {len(ws_cases)} watershed, {len(rb_cases)} rolling_ball")
