#!/usr/bin/env python3
"""Generate the rolling_ball Float16 pins: restoration.rolling_ball on a float16 image, which scikit-image promotes to float32
(`_supported_float_type`), computes in float32 (the kernel converted to float32 too) and casts back to float16
(`background.astype(image.dtype)`). The script checks that the float16 answer is the float32 one cast back.
Run from the package root:  python3 -I -B tools/rolling-ball-float16/pins.py
(a venv with only scikit-image==0.26.0, which brings NumPy 2.5.4 and SciPy 1.18.1). Light: four tiny images, under a second.

Writes Tests/ImagoCoreTests/Fixtures/rolling-ball-float16-pins.json.

The cases: a kernel value that is finite in float32 but +infinity in float16 (70000), a ball whose offsets do not fit Float16
exactly, an ellipsoid kernel, and nansafe with a NaN. The kit computes in Float and rounds to Float16 once, as scikit-image does,
so the backgrounds are held exact (NaN where the pin is NaN). Inputs are deterministic (PCG64 seed 20261031) and float16-exact; +-infinity and NaN are written as the
strings "inf", "-inf" and "nan".
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import skimage
from skimage import restoration

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert skimage.__version__ == "0.26.0", skimage.__version__
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests/ImagoCoreTests/Fixtures/rolling-ball-float16-pins.json"
rng = np.random.Generator(np.random.PCG64(20261031))
CMD = "python3 -I -B tools/rolling-ball-float16/pins.py"


def enc(x):
    x = float(x)
    if np.isnan(x):
        return "nan"
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def fl(a):
    return [enc(x) for x in np.asarray(a).flat]


cases = []


def add(name, image, radius=None, kernel=None, nansafe=False):
    assert image.dtype == np.float16
    if kernel is None:
        bg = restoration.rolling_ball(image, radius=radius, nansafe=nansafe)
    else:
        bg = restoration.rolling_ball(image, kernel=kernel, nansafe=nansafe)
    assert bg.dtype == np.float16, bg.dtype
    # The float32 computation, cast back: what scikit-image returns, restated.
    img32 = image.astype(np.float32)
    bg32 = (restoration.rolling_ball(img32, radius=radius, nansafe=nansafe) if kernel is None
            else restoration.rolling_ball(img32, kernel=kernel, nansafe=nansafe))
    assert np.array_equal(bg32.astype(np.float16), bg, equal_nan=True), name
    case = {"name": name, "height": image.shape[0], "width": image.shape[1], "image": fl(image), "nansafe": nansafe,
            "background": fl(bg)}
    if kernel is None:
        case["radius"] = radius
    else:
        case["kernel"] = {"height": kernel.shape[0], "width": kernel.shape[1], "values": fl(kernel)}
    cases.append(case)


image = (rng.random((5, 6)) * 100).astype(np.float16)
k = np.array([[np.inf, 1.0, np.inf], [0.5, 70000.0, 3.0], [np.inf, 1.5, np.inf]])
with np.errstate(over="ignore"):
    assert np.isinf(np.float16(70000.0)) and np.isfinite(np.float32(70000.0))
add("kernel-overflows-float16", image, kernel=k)
add("ball-radius-2.7", (rng.random((7, 8)) * 3).astype(np.float16), radius=2.7)
add("ellipsoid-1.3", (rng.random((6, 7)) * 10).astype(np.float16), kernel=restoration.ellipsoid_kernel((3, 5), 1.3))
withnan = (rng.random((5, 5)) * 10).astype(np.float16)
withnan[0, 0] = np.nan
add("nansafe-radius-1.5", withnan, radius=1.5, nansafe=True)

OUT.write_text(json.dumps({
    "meta": {"skimage": skimage.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
             "python": platform.python_version(), "generator": CMD},
    "cases": cases}, indent=0) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}: {len(cases)} rolling_ball float16 cases")
