#!/usr/bin/env python3
"""Generate the compact-watershed and watershed-line pins: segmentation.watershed(..., compactness=, watershed_line=), 2D.
Run from the package root:  python3 -I -B tools/watershed-compact/pins.py
(a venv with only scikit-image==0.26.0, which brings NumPy 2.5.4 and SciPy 1.18.1). Light: small images, seconds.

Writes Tests/ImagoCoreTests/Fixtures/watershed-compact-pins.json.

Reference: scikit-image 0.26.0 `segmentation.watershed` with `compactness` > 0 and/or `watershed_line=True`; labels exact (the
wheel ships no tests). Tie-heavy inputs (integer levels, flat plateaus) check the heap order. Inputs are deterministic (PCG64 seed
20261031); floats are written as repr (round trip).
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage import feature, segmentation

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert skimage.__version__ == "0.26.0", skimage.__version__
# The compiled flood fuses value + compactness * distance on the macOS arm64 wheel; another build can break exact ties otherwise.
assert platform.machine() == "arm64", platform.machine()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests/ImagoCoreTests/Fixtures/watershed-compact-pins.json"
rng = np.random.Generator(np.random.PCG64(20261031))
CMD = "python3 -I -B tools/watershed-compact/pins.py"


def bits(a):
    return "".join("1" if v else "0" for v in np.asarray(a).flat)


cases = []


def add(name, image, markers, mask=None, connectivity=1, compactness=0.0, line=False):
    image = np.asarray(image, float)
    markers = np.asarray(markers, np.int32)
    out = segmentation.watershed(image, markers, mask=mask, connectivity=connectivity, compactness=compactness,
                                 watershed_line=line)
    cases.append({"name": name, "height": image.shape[0], "width": image.shape[1], "image": [float(x) for x in image.flat],
                  "markers": [int(x) for x in markers.flat], "mask": None if mask is None else bits(mask),
                  "connectivity": connectivity, "compactness": float(compactness), "watershedLine": bool(line),
                  "labels": [int(x) for x in out.flat]})


def random_markers(h, w, n):
    mk = np.zeros((h, w), np.int32)
    for i, p in enumerate(rng.choice(h * w, size=n, replace=False)):
        mk[p // w, p % w] = i + 1
    return mk


smooth = ndi.gaussian_filter(rng.random((40, 45)), 2.5)
coords = feature.peak_local_max(-smooth, min_distance=3)
smk = np.zeros(smooth.shape, np.int32)
for i, (r, c) in enumerate(coords):
    smk[r, c] = i + 1
flat = np.zeros((20, 20)); fmk = np.zeros((20, 20), np.int32)
fmk[3, 3] = 1; fmk[3, 16] = 2; fmk[16, 9] = 3; fmk[10, 10] = 4
ints = rng.integers(0, 4, size=(30, 34)).astype(float)
imk = random_markers(30, 34, 9)
blob = np.zeros((40, 40), bool)
yy, xx = np.mgrid[:40, :40]
blob |= (yy - 14) ** 2 + (xx - 14) ** 2 < 100
blob |= (yy - 24) ** 2 + (xx - 28) ** 2 < 81
d = ndi.distance_transform_edt(blob)
dmk = np.zeros(blob.shape, np.int32); dmk[14, 14] = 1; dmk[24, 28] = 2
msk = (rng.random((40, 45)) < 0.85) | (smk > 0)

for conn in (1, 2):
    for comp in (0.01, 0.1, 1.0, 10.0):
        add(f"smooth-c{comp}-{conn}", smooth, smk, connectivity=conn, compactness=comp)
        add(f"flat-c{comp}-{conn}", flat, fmk, connectivity=conn, compactness=comp)
        add(f"integer-c{comp}-{conn}", ints, imk, connectivity=conn, compactness=comp)
    add(f"smooth-masked-c0.1-{conn}", smooth, smk, mask=msk, connectivity=conn, compactness=0.1)
    add(f"smooth-line-{conn}", smooth, smk, connectivity=conn, line=True)
    add(f"flat-line-{conn}", flat, fmk, connectivity=conn, line=True)
    add(f"integer-line-{conn}", ints, imk, connectivity=conn, line=True)
    add(f"smooth-masked-line-{conn}", smooth, smk, mask=msk, connectivity=conn, line=True)
    add(f"two-discs-line-{conn}", -d, dmk, mask=blob, connectivity=conn, line=True)
    add(f"smooth-line-c0.1-{conn}", smooth, smk, connectivity=conn, compactness=0.1, line=True)
    add(f"flat-line-c1-{conn}", flat, fmk, connectivity=conn, compactness=1.0, line=True)
    # tie-heavy stress: many markers on two or three levels
    for k, (h, w, levels, nm) in enumerate([(17, 19, 2, 20), (25, 25, 3, 40)]):
        img = rng.integers(0, levels, size=(h, w)).astype(float)
        mkr = random_markers(h, w, nm)
        add(f"stress-{k}-c0.5-{conn}", img, mkr, connectivity=conn, compactness=0.5)
        add(f"stress-{k}-line-{conn}", img, mkr, connectivity=conn, line=True)
        add(f"stress-{k}-line-c0.5-{conn}", img, mkr, connectivity=conn, compactness=0.5, line=True)
# a tie that only the fused multiply-add of the arm64 build breaks (value + compactness * distance contracts)
fimg = np.array([[0, 0, 2, 0, 1, 2, 1], [0, 1, 1, 2, 0, 2, 0], [0, 2, 0, 1, 1, 1, 1], [0, 2, 0, 1, 0, 2, 0],
                 [0, 1, 2, 2, 1, 2, 0], [0, 1, 0, 1, 0, 0, 2]]) * 0.1
fmkr = np.zeros((6, 7), np.int32); fmkr[0, 1] = 2; fmkr[0, 2] = 3; fmkr[1, 6] = 1
add("fma-tie-c0.1", fimg, fmkr, compactness=0.1)
# adjacent markers (the documented limit of the line), and a one-pixel image
adj = np.zeros((9, 9), np.int32); adj[4, 3] = 1; adj[4, 4] = 2
add("adjacent-markers-line", np.zeros((9, 9)), adj, line=True)
add("one-pixel-line-c1", np.array([[0.5]]), np.array([[1]], np.int32), compactness=1.0, line=True)
# markers outside the mask (dropped as markers * mask), and a negative label
msk2 = rng.random((40, 45)) < 0.85
add("smooth-masked-markers-out-line-2", smooth, smk, mask=msk2, connectivity=2, line=True)
add("smooth-masked-markers-out-c0.1-line-1", smooth, smk, mask=msk2, connectivity=1, compactness=0.1, line=True)
nimg = rng.integers(0, 2, size=(17, 19)).astype(float)
nmk = random_markers(17, 19, 20); nmk[nmk == 1] = -1
add("negative-label-line-1", nimg, nmk, connectivity=1, line=True)
add("negative-label-c0.5-2", nimg, nmk, connectivity=2, compactness=0.5)

OUT.write_text(json.dumps({
    "meta": {"skimage": skimage.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
             "python": platform.python_version(), "machine": platform.machine(), "generator": CMD},
    "cases": cases}, separators=(",", ":")) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}: {len(cases)} watershed cases")
