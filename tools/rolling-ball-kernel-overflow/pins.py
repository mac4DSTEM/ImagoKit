#!/usr/bin/env python3
"""Generate the rolling_ball kernel-overflow pins: restoration.rolling_ball(image, kernel=k) with nansafe=False, where a kernel
value is finite in Float64 but overflows Float32.
Run from the package root:  python3 -I -B tools/rolling-ball-kernel-overflow/pins.py
(a venv with only scikit-image==0.26.0, which brings NumPy 2.5.4 and SciPy 1.18.1). Light: two tiny images, under a second.

Writes Tests/ImagoCoreTests/Fixtures/rolling-ball-kernel-overflow-pins.json.

Reference: scikit-image 0.26.0 converts the kernel to the image type first (`kernel.astype(float_type)`), then sets the offset to
+infinity where the converted kernel is +infinity, so on a Float32 image 1e300 is a cell outside the kernel and -1e300 gives an
offset of +infinity; `apply_kernel` takes the minimum, so such cells never win. Floats are held to `assert_allclose` dtype defaults
(rtol 1e-7 Float64, 1e-5 Float32; the wheel ships no tests). Inputs are deterministic (PCG64 seed 20261031); +-infinity is written
as the strings "inf" and "-inf".
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
OUT = ROOT / "Tests/ImagoCoreTests/Fixtures/rolling-ball-kernel-overflow-pins.json"
rng = np.random.Generator(np.random.PCG64(20261031))
CMD = "python3 -I -B tools/rolling-ball-kernel-overflow/pins.py"


def enc(x):
    x = float(x)
    assert not np.isnan(x)
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def fl(a):
    return [enc(x) for x in np.asarray(a).flat]


cases = []


def add(name, image, kernel):
    bg = restoration.rolling_ball(image, kernel=kernel)
    assert bg.dtype == image.dtype
    assert np.isfinite(bg).all(), name
    cases.append({"name": name, "height": image.shape[0], "width": image.shape[1], "dtype": str(image.dtype),
                  "image": fl(image), "kernel": {"height": kernel.shape[0], "width": kernel.shape[1], "values": fl(kernel)},
                  "background": fl(bg)})


image = rng.random((5, 6)) * 100
k = np.array([[np.inf, 1.0, np.inf], [0.5, 2.0, 1e300], [np.inf, 1.5, np.inf]])
add("overflow-float32", image.astype(np.float32), k)
add("overflow-float64", image, k)
kn = k.copy(); kn[1, 0] = -1e300
add("both-overflows-float32", image.astype(np.float32), kn)

OUT.write_text(json.dumps({
    "meta": {"skimage": skimage.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
             "python": platform.python_version(), "generator": CMD},
    "cases": cases}, indent=0) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}: {len(cases)} rolling_ball kernel-overflow cases")
