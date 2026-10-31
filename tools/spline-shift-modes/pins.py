#!/usr/bin/env python3
"""Generate the spline-shift pins for the modes "constant" and "wrap": scipy.ndimage.shift at orders 2 to 5.
Run from the package root:  python3 -I -B tools/spline-shift-modes/pins.py
(a venv with only scikit-image==0.26.0, which brings NumPy 2.5.4 and SciPy 1.18.1). Light: a few small images, seconds.

Writes Tests/ImagoAlignmentTests/Fixtures/spline-shift-mode-pins.json, in the format of spline-shift-pins.json.

Reference: SciPy 1.18.1 `ndimage.shift(x, s, order, mode, cval)` with `prefilter=True` on float32 images (float32 output).
Measured rule (this script also checks it in float64 against SciPy before writing): both modes prefilter as "mirror" and
fold a tap beyond the edge as "mirror"; "constant" gives cval for a coordinate outside [0, n - 1] on either axis, "wrap"
maps the coordinate with period n - 1 first. Tolerance: decimal 6 (scipy/ndimage/tests/test_interpolation.py). Taken on arm64.
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
assert platform.machine() == "arm64", platform.machine()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests/ImagoAlignmentTests/Fixtures/spline-shift-mode-pins.json"
rng = np.random.Generator(np.random.PCG64(20261031))


def out(a):
    return [float(f"{float(v):.17g}") for v in np.asarray(a).ravel()]


def basis(d, t):
    if d == 0:
        return 1.0 if -0.5 <= t < 0.5 else 0.0
    h = (d + 1) / 2
    return ((t + h) * basis(d - 1, t + 0.5) + (h - t) * basis(d - 1, t - 0.5)) / d


def mirror(i, n):
    if n == 1:
        return 0
    p = 2 * n - 2
    i = abs(i) % p
    return p - i if i >= n else i


def wrap_coordinate(x, n):
    if n == 1:
        return 0.0
    sz = n - 1
    if x < 0:
        x += sz * (int(-x / sz) + 1)
    elif x > n - 1:
        x -= sz * int(x / sz)
    return x


def taps(x, n, order, mode):
    """The (indices, weights) of one axis, or None for cval."""
    if mode == "constant":
        if x < 0 or x > n - 1:
            return None
    else:
        x = wrap_coordinate(x, n)
    if order & 1:
        start, f = int(np.floor(x)) - order // 2, x - np.floor(x)
    else:
        start, f = int(np.floor(x + 0.5)) - order // 2, x - np.floor(x + 0.5)
    return ([mirror(start + k, n) for k in range(order + 1)], [basis(order, (k - order // 2) - f) for k in range(order + 1)])


def rule(x, s, order, mode, cval):
    """The measured rule in float64, to check it against SciPy before the pins are written."""
    h, w = x.shape
    c = ndi.spline_filter(x.astype(np.float64), order, mode="mirror")
    y = np.full((h, w), cval)
    for r in range(h):
        a = taps(r - s[0], h, order, mode)
        if a is None:
            continue
        for k in range(w):
            b = taps(k - s[1], w, order, mode)
            if b is None:
                continue
            y[r, k] = sum(wi * wj * c[i, j] for i, wi in zip(*a) for j, wj in zip(*b))
    return y


cases = []
for h, w in [(17, 23), (6, 8), (3, 40), (1, 9), (2, 2)]:
    x = rng.normal(50, 20, (h, w)).astype(np.float32)
    for order in (2, 3, 4, 5):
        for mode in ("constant", "wrap"):
            # No half-pixel coordinate with the other axis integral: on a 2-pixel axis it is the mean of two float32 pixels,
            # an exact float32 midpoint that SciPy rounds by the last bits of its spline arithmetic (a tie, not a rule).
            for s in [(0.3, -0.7), (-2.25, 1.5), (4.6, -11.3), (0.0, 0.45), (-31.4, 47.9), (1.0, -2.0)]:
                cval = 7.5 if mode == "constant" else 0.0
                y = ndi.shift(x, s, order=order, mode=mode, cval=cval)
                assert y.dtype == np.float32
                y64 = ndi.shift(x.astype(np.float64), s, order=order, mode=mode, cval=cval)
                assert np.max(np.abs(rule(x, s, order, mode, cval) - y64)) < 1e-9, (h, w, order, mode, s)
                cases.append(dict(h=h, w=w, order=order, mode=mode, shift=list(s), cval=cval, x=out(x), y=out(y)))
OUT.write_text(json.dumps(dict(numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
                               machine=platform.machine(), seed=20261031, cases=cases)))
print(len(cases), "cases ->", OUT.relative_to(ROOT))
