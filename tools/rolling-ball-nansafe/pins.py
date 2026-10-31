#!/usr/bin/env python3
"""Generate the rolling_ball nansafe pins: restoration.rolling_ball(..., nansafe=True).
Run from the package root:  python3 -I -B tools/rolling-ball-nansafe/pins.py
(a venv with only scikit-image==0.26.0, which brings NumPy 2.5.4 and SciPy 1.18.1). Light: a few small images, seconds.

Writes Tests/ImagoCoreTests/Fixtures/rolling-ball-nansafe-pins.json.

Reference: scikit-image 0.26.0 `restoration.rolling_ball` with `nansafe=True` (`apply_kernel_nan`): a NaN anywhere under the kernel,
including the cells outside the ball whose offset is +infinity, makes the background NaN; so does -infinity under such a cell
(-inf + inf). Floats are held to `assert_allclose` dtype defaults (rtol 1e-7 Float64, 1e-5 Float32; the wheel ships no tests); NaN
must be NaN. Inputs are deterministic (PCG64 seed 20261031); NaN is written as null, +-infinity as the strings "inf" and "-inf".
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage import restoration

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert skimage.__version__ == "0.26.0", skimage.__version__
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests/ImagoCoreTests/Fixtures/rolling-ball-nansafe-pins.json"
rng = np.random.Generator(np.random.PCG64(20261031))
CMD = "python3 -I -B tools/rolling-ball-nansafe/pins.py"


def enc(x):
    x = float(x)
    if np.isnan(x):
        return None
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def fl(a):
    return [enc(x) for x in np.asarray(a).flat]


cases = []


def add(name, image, radius=None, kernel=None):
    image = np.asarray(image)
    if kernel is None:
        bg = restoration.rolling_ball(image, radius=radius, nansafe=True)
    else:
        bg = restoration.rolling_ball(image, kernel=kernel, nansafe=True)
    assert bg.dtype == image.dtype
    cases.append({"name": name, "height": image.shape[0], "width": image.shape[1], "dtype": str(image.dtype),
                  "image": fl(image), "radius": radius,
                  "kernel": None if kernel is None else {"height": kernel.shape[0], "width": kernel.shape[1], "values": fl(kernel)},
                  "background": fl(bg), "nanCount": int(np.isnan(bg).sum())})


base = ndi.gaussian_filter(rng.random((26, 31)) * 100, 4)
feat = np.zeros((26, 31)); feat[9:13, 12:17] = 60
hat = base + feat

# no NaN: nansafe=True equals the default path
add("no-nan-r3", hat, 3)
# sparse NaNs, radius 1 (cross kernel: the diagonal cells are outside the ball and still see NaN), 3, 2.5 and 5
sparse = hat.copy(); sparse[rng.random(sparse.shape) < 0.02] = np.nan
add("sparse-r1", sparse, 1)
add("sparse-r2.5", sparse, 2.5)
add("sparse-r3", sparse, 3)
add("sparse-r5", sparse, 5)
# one NaN in a corner and one in the middle
corner = hat.copy(); corner[0, 0] = np.nan; corner[13, 20] = np.nan
add("two-nans-r4", corner, 4)
# a NaN block
block = hat.copy(); block[3:6, 22:27] = np.nan
add("block-r3", block, 3)
# all NaN, and a single finite pixel among NaNs
add("all-nan", np.full((5, 6), np.nan), 2)
lone = np.full((5, 6), np.nan); lone[2, 3] = 4.0
add("lone-finite", lone, 1)
# -infinity under a cell outside the ball gives NaN (-inf + inf); +infinity does not
infs = hat[:9, :10].copy(); infs[4, 4] = -np.inf; infs[1, 8] = np.inf
add("minus-inf-r1", infs, 1)
add("minus-inf-r2", infs, 2)
# Float32
add("sparse-float32-r3", sparse.astype(np.float32), 3)
add("block-float32-r2", block.astype(np.float32), 2)
# an ellipsoid kernel, and a hand-written kernel
add("sparse-ellipsoid", sparse, kernel=restoration.ellipsoid_kernel((7, 11), 6))
k = np.array([[np.inf, 1.0, np.inf], [0.5, 2.0, 0.25], [np.inf, 1.5, np.inf]])
add("sparse-custom-kernel", sparse[:12, :14], kernel=k)
# kernel values: a NaN cell (all NaN), a -inf cell (D = +inf), an even-size kernel, a Float32 overflow (1e300 becomes +inf)
kn = k.copy(); kn[1, 2] = np.nan
add("nan-kernel-cell", hat[:6, :7], kernel=kn)
km = k.copy(); km[1, 2] = -np.inf
add("minus-inf-kernel-cell", sparse[:12, :14], kernel=km)
even = hat[:7, :8].copy(); even[3, 3] = np.nan
add("even-kernel", even, kernel=np.array([[1.0, 1.0], [0.5, 2.0]]))
kf = k.copy(); kf[1, 2] = 1e300
add("float32-overflowing-kernel-cell", sparse[:12, :14].astype(np.float32), kernel=kf)
# a ball larger than the image: every pixel sees every NaN
small = base[:8, :9].copy(); small[6, 1] = np.nan
add("larger-than-image", small, 12)
smallok = base[:8, :9].copy()
add("larger-than-image-no-nan", smallok, 12)

OUT.write_text(json.dumps({
    "meta": {"skimage": skimage.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
             "python": platform.python_version(), "generator": CMD},
    "cases": cases}, indent=0) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}: {len(cases)} rolling_ball nansafe cases")
