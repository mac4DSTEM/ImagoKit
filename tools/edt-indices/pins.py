#!/usr/bin/env python3
"""Generate the feature-transform pins: ndimage.distance_transform_edt(..., return_indices=True), 2D.
Run from the package root:  python3 -I -B tools/edt-indices/pins.py
(a venv with only scikit-image==0.26.0, which brings NumPy 2.5.4 and SciPy 1.18.1). Light: small masks, seconds.

Writes Tests/ImagoCoreTests/Fixtures/edt-indices-pins.json.

Reference: SciPy 1.18.1 `ndimage.distance_transform_edt` with `return_indices=True`. The indices are integers and held exact (the
nearest background pixel among equally near ones is the one SciPy's Voronoi sweep keeps, so many cases are tie-heavy); the
distances are computed from them as SciPy does and held to 1e-12 (its tests use decimal 6). Inputs are deterministic (PCG64 seed
20261031). Masks with no background pixel are not pinned (the kit refuses them).
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert skimage.__version__ == "0.26.0", skimage.__version__
# The compiled sweep uses fused multiply-add on the macOS arm64 wheel; a build without it can break exact ties the other way.
assert platform.machine() == "arm64", platform.machine()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests/ImagoCoreTests/Fixtures/edt-indices-pins.json"
rng = np.random.Generator(np.random.PCG64(20261031))
CMD = "python3 -I -B tools/edt-indices/pins.py"


def bits(a):
    return "".join("1" if v else "0" for v in np.asarray(a).flat)


cases = []


def add(name, mask, sampling=None):
    mask = np.asarray(mask, bool)
    assert (~mask).any()
    kw = {} if sampling is None else {"sampling": sampling}
    d, ind = ndi.distance_transform_edt(mask, return_indices=True, **kw)
    cases.append({"name": name, "height": mask.shape[0], "width": mask.shape[1], "mask": bits(mask),
                  "sampling": None if sampling is None else [float(s) for s in sampling],
                  "distance": [float(x) for x in d.flat],
                  "nearestRows": [int(x) for x in ind[0].flat], "nearestColumns": [int(x) for x in ind[1].flat]})


# the docstring example
add("docstring", np.array([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 0, 0]]))
add("docstring-sampling", np.array([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 0, 0]]), (2, 1))
add("single-background", np.zeros((1, 1), bool))
add("all-background", np.zeros((4, 6), bool))
m = np.ones((9, 11), bool); m[4, 5] = False
add("one-hole", m)
add("one-hole-anisotropic", m, (0.5, 2.0))
# symmetric holes: every foreground pixel on a bisector is a tie
s = np.ones((13, 13), bool); s[3, 3] = s[3, 9] = s[9, 3] = s[9, 9] = False
add("four-symmetric-holes", s)
s2 = np.ones((10, 17), bool); s2[:, 0] = False; s2[:, 16] = False; s2[0, 8] = False
add("two-walls-and-a-dot", s2)
# columns with no background (pass 1 leaves them unset)
c = np.ones((8, 12), bool); c[2, 1] = False; c[6, 10] = False
add("mostly-empty-columns", c)
c2 = np.ones((1, 9), bool); c2[0, 4] = False
add("one-row", c2)
c3 = np.ones((9, 1), bool); c3[7, 0] = False
add("one-column", c3)
# random masks, sparse and dense background, with and without sampling (dyadic and not)
for k, (h, w, p) in enumerate([(16, 16, 0.5), (23, 31, 0.8), (40, 25, 0.95), (7, 64, 0.6), (33, 33, 0.98), (50, 47, 0.9)]):
    mk = rng.random((h, w)) < p
    if not (~mk).any():
        mk[0, 0] = False
    add(f"random-{k}", mk)
    add(f"random-{k}-sampling", mk, (1.5, 0.75))
    add(f"random-{k}-sampling-irrational", mk, (0.3, 1.1))
# a lattice of background points: ties everywhere
lat = np.ones((21, 26), bool); lat[::4, ::5] = False
add("lattice", lat)
add("lattice-sampling", lat, (1.25, 1.0))
blob = np.zeros((40, 40), bool)
yy, xx = np.mgrid[:40, :40]
blob |= (yy - 14) ** 2 + (xx - 14) ** 2 < 100
blob |= (yy - 24) ** 2 + (xx - 28) ** 2 < 81
add("two-discs", blob)
# an exact tie that only SciPy's fused multiply-add breaks (the arm64 build contracts the sweep's distance sum)
fm = np.ones((12, 10), bool); fm[11, 0] = False; fm[0, 9] = False
add("fma-tie", fm, (0.3, 1.1))

OUT.write_text(json.dumps({
    "meta": {"skimage": skimage.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
             "python": platform.python_version(), "machine": platform.machine(), "generator": CMD},
    "cases": cases}, separators=(",", ":")) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}: {len(cases)} feature-transform cases")
