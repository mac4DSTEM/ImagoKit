#!/usr/bin/env python3
"""Generate the window-family pins: the 14 `scipy.signal.get_window` families beyond the first 11, through
skimage.filters.window for 2D shapes. Run from the package root:  python3 -I -B tools/window-families/pins.py
(a venv with only scikit-image==0.26.0, which brings NumPy 2.5.4 and SciPy 1.18.1). Light: a few small windows, seconds.

Writes Tests/ImagoFiltersTests/Fixtures/window-family-pins.json.

Reference: SciPy 1.18.1 `get_window(spec, max(shape), fftbins=False)` (the symmetric window) and scikit-image 0.26.0
`filters.window(spec, shape)`. Families: triang, parzen, bohman, barthann, cosine, exponential, taylor, lanczos,
kaiser_bessel_derived, general_cosine, general_gaussian, general_hamming, dpss, chebwin. The 1D tolerances are the ones
scipy/signal/tests/test_windows.py uses per family (written out in the Swift test); the 2D window has no upstream test (the
wheel ships none), so it takes assert_allclose float64 defaults widened by the 1D family tolerance it interpolates.
Also records the refusals: kaiser_bessel_derived of odd length, dpss with NW >= M/2 or NW <= 0.
The 2D coordinates are float32, so the pins are taken on arm64 (the machine the fixtures are checked on).
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy.signal import get_window
from skimage import filters

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert skimage.__version__ == "0.26.0", skimage.__version__
assert platform.machine() == "arm64", platform.machine()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests/ImagoFiltersTests/Fixtures/window-family-pins.json"

# (family, params as written to the fixture, get_window spec)
SPECS = [
    ("triang", [], "triang"),
    ("parzen", [], "parzen"),
    ("bohman", [], "bohman"),
    ("barthann", [], "barthann"),
    ("cosine", [], "cosine"),
    ("exponential", [1.0], ("exponential", None, 1.0)),
    ("exponential", [3.5], ("exponential", None, 3.5)),
    ("taylor", [4, 30, 1], "taylor"),
    ("taylor", [5, 35, 0], ("taylor", 5, 35, False)),
    ("taylor", [1, 20, 1], ("taylor", 1, 20, True)),
    ("lanczos", [], "lanczos"),
    ("kaiser_bessel_derived", [4.0], ("kaiser_bessel_derived", 4.0)),
    ("kaiser_bessel_derived", [np.pi / 2], ("kaiser_bessel_derived", np.pi / 2)),
    ("general_cosine", [0.5, 0.3, 0.2], ("general_cosine", np.array([0.5, 0.3, 0.2]))),
    ("general_cosine", [1, 1.942604, 1.340318, 0.440811, 0.043097],
     ("general_cosine", np.array([1, 1.942604, 1.340318, 0.440811, 0.043097]))),
    ("general_gaussian", [1.5, 7.0], ("general_gaussian", 1.5, 7.0)),
    ("general_gaussian", [1.0, 3.0], ("general_gaussian", 1.0, 3.0)),
    ("general_hamming", [0.7], ("general_hamming", 0.7)),
    ("dpss", [2.5], ("dpss", 2.5)),
    ("dpss", [4.0], ("dpss", 4.0)),
    ("chebwin", [50.0], ("chebwin", 50.0)),
    ("chebwin", [100.0], ("chebwin", 100.0)),
]
SHAPES = [(8, 8), (16, 9), (31, 31), (20, 33), (5, 64), (1, 1), (2, 3), (7, 7)]


def out(a):
    return [float(f"{float(v):.17g}") for v in np.asarray(a, dtype=np.float64).ravel()]


cases, refusals = [], []
for family, params, spec in SPECS:
    for h, w in SHAPES:
        m = max(h, w)
        if family == "kaiser_bessel_derived" and m % 2:
            continue
        if family == "dpss" and m > 1 and params[0] >= m / 2:  # length 0 and 1 answer ones before NW is checked
            continue
        w1 = get_window(spec, m, fftbins=False)
        w2 = filters.window(spec, (h, w))
        cases.append(dict(family=family, params=params, h=h, w=w, window1d=out(w1), window2d=out(w2)))
# 1D only: length 2 for every family (the edge of bohman, lanczos, chebwin order 1, dpss NW < 1), and the image-sized
# lengths where the dpss eigensolver and the chebwin DFT do their real work.
long = []
for family, params, spec in SPECS:
    if family == "dpss":
        spec, params = ("dpss", 0.75), [0.75]
    long.append(dict(family=family, params=params, count=2, window1d=out(get_window(spec, 2, fftbins=False))))
for family, params, spec in [("dpss", [2.5], ("dpss", 2.5)), ("dpss", [16.0], ("dpss", 16.0)),
                             ("chebwin", [50.0], ("chebwin", 50.0)), ("chebwin", [100.0], ("chebwin", 100.0))]:
    for m in (1024, 1023, 4096):
        long.append(dict(family=family, params=params, count=m, window1d=out(get_window(spec, m, fftbins=False))))
for family, params, spec, m in [("kaiser_bessel_derived", [4.0], ("kaiser_bessel_derived", 4.0), 7),
                                ("kaiser_bessel_derived", [4.0], ("kaiser_bessel_derived", 4.0), 1),
                                ("dpss", [2.5], ("dpss", 2.5), 5),
                                ("dpss", [0.0], ("dpss", 0.0), 8)]:
    try:
        get_window(spec, m, fftbins=False)
    except ValueError:
        refusals.append(dict(family=family, params=params, count=m))
    else:
        raise AssertionError(f"{family} {params} {m} was not refused")
OUT.write_text(json.dumps(dict(numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
                               machine=platform.machine(), cases=cases, long=long, refusals=refusals)))
print(len(cases), "cases,", len(long), "1D cases,", len(refusals), "refusals ->", OUT.relative_to(ROOT))
