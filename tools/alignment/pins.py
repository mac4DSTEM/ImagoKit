#!/usr/bin/env python3
"""Spline-order shifts and masked registration pins. Run from the package root:
  python -I -B tools/alignment/pins.py
Writes Tests/ImagoAlignmentTests/Fixtures/spline-shift-pins.json (scipy.ndimage.shift, orders 2 to 5) and
masked-registration-pins.json (skimage.registration.phase_cross_correlation with masks; cross_correlate_masked).
"""
import json
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked

assert np.__version__ == "2.5.4" and scipy.__version__ == "1.18.1" and skimage.__version__ == "0.26.0"
rng = np.random.Generator(np.random.PCG64(20261021))


def out(a):
    return [float(f"{float(v):.17g}") for v in np.asarray(a).ravel()]


# --- ndimage.shift, orders 2 to 5 --------------------------------------------------------------------------------
SCIPY_MODE = {"grid-constant": "grid-constant", "nearest": "nearest", "reflect": "reflect", "mirror": "mirror", "grid-wrap": "grid-wrap"}
spline = []
for h, w in [(17, 23), (6, 8), (3, 40), (1, 9), (30, 20)]:
    x = rng.normal(50, 20, (h, w)).astype(np.float32)
    for order in (2, 3, 4, 5):
        for mode in SCIPY_MODE:
            for s in [(0.0, 0.0), (0.4, -1.3), (-2.75, 3.1), (5.5, 0.25), (-0.5, 0.5)]:
                cval = 12.5 if mode == "grid-constant" else 0.0
                spline.append(dict(h=h, w=w, order=order, mode=mode, shift=s, cval=cval, x=out(x),
                                   y=out(ndi.shift(x, s, order=order, mode=SCIPY_MODE[mode], cval=cval, output=np.float32))))
Path("Tests/ImagoAlignmentTests/Fixtures/spline-shift-pins.json").write_text(json.dumps(dict(scipy=scipy.__version__, seed=20261021, cases=spline)))

# --- masked registration -----------------------------------------------------------------------------------------
def blobs(h, w, n):
    m = np.zeros((h, w), bool)
    yy, xx = np.mgrid[:h, :w]
    for _ in range(n):
        cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(max(2, min(h, w) // 6), max(3, min(h, w) // 3))
        m |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    return m


masked = []
for dtype in (np.float32, np.float64):
    for (rh, rw), (mh, mw) in [((24, 30), (24, 30)), ((32, 32), (32, 32)), ((20, 28), (16, 22)), ((25, 17), (25, 17))]:
        for k in range(3):
            base = rng.normal(0, 1, (rh + 12, rw + 12))
            base = ndi.gaussian_filter(base, 1.5)
            ref = base[6:6 + rh, 6:6 + rw].astype(dtype)
            dy, dx = int(rng.integers(-4, 5)), int(rng.integers(-4, 5))
            mov = base[6 + dy:6 + dy + mh, 6 + dx:6 + dx + mw].astype(dtype) + 0.05 * rng.normal(size=(mh, mw)).astype(dtype)
            rmask = ~blobs(rh, rw, 1) if k else np.ones((rh, rw), bool)
            mmask = ~blobs(mh, mw, 1) if k == 2 else None
            explicit = (mh, mw) != (rh, rw) or mmask is not None
            ratio = [0.3, 0.5, 0.1][k]
            kw = dict(reference_mask=rmask, overlap_ratio=ratio)
            if explicit:
                kw["moving_mask"] = mmask if mmask is not None else np.ones((mh, mw), bool)
            shift, e, p = phase_cross_correlation(ref, mov, **kw)
            cc = cross_correlate_masked(mov, ref, kw.get("moving_mask", rmask), rmask, mode="full", axes=(0, 1), overlap_ratio=ratio)
            unique = int((cc == cc.max()).sum()) == 1
            masked.append(dict(dtype=dtype.__name__, rh=rh, rw=rw, mh=mh, mw=mw, ratio=ratio, ref=out(ref), mov=out(mov),
                               rmask=rmask.ravel().astype(int).tolist(),
                               mmask=(kw["moving_mask"].ravel().astype(int).tolist() if explicit else None),
                               shift=[float(shift[0]), float(shift[1])], cc=out(cc), ccShape=list(cc.shape), unique=unique))
Path("Tests/ImagoAlignmentTests/Fixtures/masked-registration-pins.json").write_text(json.dumps(dict(skimage=skimage.__version__, seed=20261021, cases=masked)))
print(len(spline), len(masked), sum(c["unique"] for c in masked))
