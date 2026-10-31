#!/usr/bin/env python3
"""Generate the multichannel line-profile pins: skimage.measure.profile_line on a (rows, columns, channels) image.
Run from the package root:  python3 -I -B tools/profile-multichannel/pins.py
(a venv with only scikit-image==0.26.0, which brings NumPy 2.5.4 and SciPy 1.18.1). Light: one small image, seconds.

Writes Tests/ImagoCoreTests/Fixtures/profile-multichannel-pins.json.

Reference: scikit-image 0.26.0 `measure.profile_line` with a 3D image: one `ndimage.map_coordinates` per channel on the same
perpendicular lines, stacked channel-last, flipped along the width and reduced along it (`reduce_func(pixels, axis=1)`), or
returned unreduced (`reduce_func=None`, shape (n, w, channels)). On the float32 image (float32 output) and on the same
pixels widened to float64; orders 0, 1, 3 and 5, the five modes, widths 1 to 16, the five reductions. Tolerance:
assert_allclose dtype defaults per element (rtol 1e-5 float32, 1e-7 float64, atol 0; the wheel ships no tests). Taken on
arm64 (the float32 sums follow the build's arithmetic).
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import skimage
from skimage.measure import profile_line

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert skimage.__version__ == "0.26.0", skimage.__version__
assert platform.machine() == "arm64", platform.machine()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests/ImagoCoreTests/Fixtures/profile-multichannel-pins.json"
rng = np.random.Generator(np.random.PCG64(20261031))
H, W, C = 13, 17, 3
img32 = rng.normal(40, 15, (H, W, C)).astype(np.float32)
img64 = img32.astype(np.float64)
REDUCE = dict(mean=np.mean, sum=np.sum, min=np.min, max=np.max, median=np.median)


def out(a):
    return [float(f"{float(v):.17g}") for v in np.asarray(a).ravel()]


cases = []
lines = [((2.0, 1.5), (10.0, 14.0)), ((6.3, -2.0), (6.3, 19.5)), ((4.0, 4.0), (4.0, 4.0))]
turn = 0
for (src, dst) in lines:
    for order in (0, 1, 3, 5):
        for mode, cval in (("reflect", 0.0), ("constant", 7.5), ("nearest", 0.0), ("mirror", 0.0), ("wrap", 0.0)):
            for name in ("mean", "sum", "min", "max", "median", None):
                # The width rotates once per six reductions, so every reduction meets every width (2 for the even
                # median, 8 and up for NumPy's pairwise sum) while the fixture stays small.
                for lw in ((1, 2, 3, 8, 9, 16)[(turn // 6 + turn) % 6],):
                    turn += 1
                    kw = dict(linewidth=lw, order=order, mode=mode, cval=cval, reduce_func=REDUCE[name] if name else None)
                    p32 = profile_line(img32, src, dst, **kw)
                    p64 = profile_line(img64, src, dst, **kw)
                    assert p32.dtype == np.float32 and p64.dtype == np.float64
                    assert p32.shape[-1] == C
                    cases.append(dict(src=src, dst=dst, order=order, mode=mode, cval=cval, lw=lw, reduce=name,
                                      shape=list(p32.shape), f32=out(p32), f64=out(p64)))
OUT.write_text(json.dumps(dict(numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
                               machine=platform.machine(), seed=20261031, height=H, width=W, channels=C,
                               image=out(img32), cases=cases)))
print(len(cases), "cases ->", OUT.relative_to(ROOT))
