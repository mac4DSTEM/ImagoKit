#!/usr/bin/env python3
"""Generate the spline-order (2 to 5) line-profile pins; run from the package root:
../tools/memcap.sh -- References/oracle-venv/bin/python -I -B tools/measure/spline_pins.py

Writes Tests/ImagoCoreTests/Fixtures/spline-profile-pins.json.
  - profile: skimage.measure.profile_line at orders 2 to 5, the five modes, several widths and reductions, on float32
    images of several shapes (including ones shorter than the prefilter's truncation length and a single row), as float32
    output and on the same pixels widened to float64; segments inside, across and far outside the image.
A separate generator from pins.py, so measure-pins.json is untouched.
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage.measure import profile_line

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert skimage.__version__ == "0.26.0", skimage.__version__
assert platform.python_version() == "3.14.7", platform.python_version()
SEED = 20261019
rng = np.random.Generator(np.random.PCG64(SEED))


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).tolist()


SHAPES = [(17, 23), (5, 6), (3, 40), (1, 9), (40, 2), (60, 70)]
images = []
for h, w in SHAPES:
    images.append(rng.normal(50, 20, (h, w)).astype(np.float32))

REDUCERS = {"mean": np.mean, "sum": np.sum, "min": np.min, "max": np.max, "median": np.median}
SCIPY_MODE = {"constant": "grid-constant", "nearest": "nearest", "reflect": "reflect", "mirror": "mirror", "wrap": "grid-wrap"}
profile_cases = []
for ii, img in enumerate(images):
    h, w = img.shape
    img64 = img.astype(np.float64)
    segs = []
    for _ in range(5):
        segs.append((rng.uniform(-0.3 * h, 1.3 * h), rng.uniform(-0.3 * w, 1.3 * w),
                     rng.uniform(-0.3 * h, 1.3 * h), rng.uniform(-0.3 * w, 1.3 * w)))
    segs.append((0, 0, h - 1, w - 1))
    segs.append((h / 2, 0, h / 2, w - 1))
    segs.append((-40.5, -33.25, h + 30.5, w + 28.75))              # far outside both ends
    segs.append((h / 3, w / 3, h / 3, w / 3))                      # zero length
    for si, (r0, c0, r1, c1) in enumerate(segs):
        for order in (2, 3, 4, 5):
            for mode in ("constant", "nearest", "reflect", "mirror", "wrap"):
                lw = (1, 3, 4, 7)[(si + order + len(mode)) % 4]
                name = list(REDUCERS)[(si + order + ii) % 5]
                cval = 7.5 if mode == "constant" else 0.0
                src, dst = (float(r0), float(c0)), (float(r1), float(c1))
                kw = dict(linewidth=lw, order=order, mode=mode, cval=cval, reduce_func=REDUCERS[name])
                out32 = profile_line(img, src, dst, **kw)
                out64 = profile_line(img64, src, dst, **kw)
                assert out32.dtype == np.float32 and out64.dtype == np.float64
                profile_cases.append(dict(image=ii, src=src, dst=dst, lw=lw, order=order, mode=mode, cval=cval,
                                          reduce=name, f32=out32.tolist(), f64=out64.tolist()))

out = Path(__file__).resolve().parents[2] / "Tests/ImagoCoreTests/Fixtures/spline-profile-pins.json"
out.write_text(json.dumps(dict(
    versions=dict(numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
                  python=platform.python_version()), seed=SEED,
    images=[dict(height=a.shape[0], width=a.shape[1], bits=bits(a.ravel())) for a in images],
    profile=profile_cases), separators=(",", ":")) + "\n")
print(f"Wrote {out.name}: {len(profile_cases)} profiles, {out.stat().st_size} bytes")
