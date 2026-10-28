#!/usr/bin/env python3
"""Generate the zoom and rotate pins; run from the package root:
python3 -I -B tools/resampling/pins.py     (a venv with only scikit-image==0.26.0: NumPy 2.5.4, SciPy 1.18.1)

Writes Tests/ImagoAlignmentTests/Fixtures/resampling-pins.json.
Reference: SciPy 1.18.1 `ndimage.zoom` and `ndimage.rotate` (decimal 6 in scipy/ndimage/tests/test_interpolation.py), float32 input
in [0, 1) so the float32 output is held to 1.5e-6 as the kit's shift pins are. Inputs deterministic (PCG64 seed 20261023); floats as repr.
Coverage: orders 0 to 5, the modes the kit supports (constant, grid-constant, nearest, reflect, mirror, grid-wrap; wrap at orders 0 and 1),
zoom factors above and below one and unequal per axis, grid_mode both ways, odd and even sizes, one-row and one-column images;
rotate by quarter turns and by other angles (also negative and over 360), reshape both ways, cval nonzero.
"""
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage as ndi

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests/ImagoAlignmentTests/Fixtures/resampling-pins.json"
rng = np.random.Generator(np.random.PCG64(20261023))
CMD = "python3 -I -B tools/resampling/pins.py"
warnings.simplefilter("ignore")

images = {
    "a7x9": rng.random((7, 9)).astype(np.float32),
    "b12x5": rng.random((12, 5)).astype(np.float32),
    "c1x8": rng.random((1, 8)).astype(np.float32),
    "d6x1": rng.random((6, 1)).astype(np.float32),
    "e16x16": ndi.gaussian_filter(rng.random((16, 16)), 1.5).astype(np.float32),
}
MODES = ["constant", "grid-constant", "nearest", "reflect", "mirror", "grid-wrap"]
cases = []


def fl(a):
    """float32 values as their shortest round-trip decimal (the inputs and SciPy's outputs are float32)."""
    return [float(np.format_float_positional(np.float32(x), unique=True)) for x in np.asarray(a).flat]


def add_zoom(name, order, mode, zoom, grid_mode, cval=0.25):
    a = images[name]
    out = ndi.zoom(a, zoom, order=order, mode=mode, cval=cval, grid_mode=grid_mode)
    cases.append({"kind": "zoom", "image": name, "height": a.shape[0], "width": a.shape[1], "pixels": fl(a), "order": order, "mode": mode,
                  "cval": cval, "zoomRows": zoom[0], "zoomColumns": zoom[1], "gridMode": grid_mode,
                  "outHeight": out.shape[0], "outWidth": out.shape[1], "result": fl(out)})


def add_rotate(name, order, mode, angle, reshape, cval=0.25):
    a = images[name]
    out = ndi.rotate(a, angle, reshape=reshape, order=order, mode=mode, cval=cval)
    cases.append({"kind": "rotate", "image": name, "height": a.shape[0], "width": a.shape[1], "pixels": fl(a), "order": order, "mode": mode,
                  "cval": cval, "angle": angle, "reshape": reshape,
                  "outHeight": out.shape[0], "outWidth": out.shape[1], "result": fl(out)})


for order in range(6):
    for mode in MODES:
        for name, zoom, grid in [("a7x9", (1.7, 1.7), False), ("b12x5", (2.3, 0.6), False), ("a7x9", (0.5, 0.5), True),
                                 ("e16x16", (1.0, 1.5), True), ("a7x9", (3.0, 2.0), False)]:
            add_zoom(name, order, mode, zoom, grid)
    for name, zoom in [("c1x8", (3.0, 1.6)), ("d6x1", (1.6, 4.0))]:
        add_zoom(name, order, "constant", zoom, False)
        add_zoom(name, order, "mirror", zoom, True)
for order in (0, 1):
    add_zoom("a7x9", order, "wrap", (1.7, 1.7), False)
    add_zoom("a7x9", order, "wrap", (2.0, 0.7), True)

for order in range(6):
    for mode in MODES:
        for angle, reshape in [(30, True), (-37.5, True), (90, True), (45, False), (200.25, True)]:
            add_rotate("a7x9", order, mode, angle, reshape)
    for angle in (0, 180, 270, 360, 12.3, 100, -90):
        add_rotate("b12x5", order, "constant", angle, True)
        add_rotate("e16x16", order, "nearest", angle, False)
for order in (0, 1):
    add_rotate("a7x9", order, "wrap", 33, True)
add_rotate("c1x8", 1, "constant", 30, True)
add_rotate("d6x1", 3, "mirror", 60, True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"meta": {"numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version(), "generator": CMD},
                           "cases": cases}) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}: {len(cases)} cases, {OUT.stat().st_size // 1024} KiB")
