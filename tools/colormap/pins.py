#!/usr/bin/env python3
"""Generate the colour-table pins; run from the package root:
python3 -I -B tools/colormap/pins.py [APP_LUT_JSON]

Writes Tests/ImagoCoreTests/Fixtures/colormap-pins.json.

The tables are piecewise-linear anchors (not matplotlib's maps; matplotlib is not in the pinned environment), so the pin is an
independent NumPy reading of the DEFINITION in Sources/ImagoCore/Colormap.swift: np.interp per channel at t = i / (N - 1),
bytes by round-half-away of channel * 255. With APP_LUT_JSON (the application's own table bytes, produced by compiling its
Colormaps.swift offline, see the CHANGELOG), the script asserts the NumPy tables equal the application's byte for byte and the
application's colour-table sampler equals the NumPy reading of the sampler, then records both in the fixture.
"""
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4", np.__version__
assert platform.python_version() == "3.14.7", platform.python_version()
ROOT = Path(__file__).resolve().parents[2]

ANCHORS = {
    "gray": [(0, 0, 0, 0), (1, 1, 1, 1)],
    "viridis": [(0.00, 0.267, 0.005, 0.329), (0.25, 0.254, 0.265, 0.530), (0.50, 0.128, 0.567, 0.551),
                (0.75, 0.369, 0.789, 0.383), (1.00, 0.993, 0.906, 0.144)],
    "inferno": [(0.00, 0.001, 0.000, 0.014), (0.25, 0.258, 0.039, 0.406), (0.50, 0.578, 0.148, 0.404),
                (0.75, 0.865, 0.316, 0.226), (0.90, 0.988, 0.645, 0.040), (1.00, 0.988, 0.998, 0.645)],
    "rdbu": [(0.00, 0.129, 0.400, 0.674), (0.50, 0.969, 0.969, 0.969), (1.00, 0.698, 0.094, 0.168)],
}


def rnd(x):
    """Swift's .rounded(): half away from zero."""
    f = math.floor(abs(x))
    r = f + (1 if abs(x) - f >= 0.5 else 0)
    return r if x >= 0 else -r


def lut(name, n):
    a = np.array(ANCHORS[name], dtype=np.float64)
    t = np.arange(n, dtype=np.float64) / (n - 1)
    out = []
    for i in range(n):
        for c in (1, 2, 3):
            v = float(np.interp(t[i], a[:, 0], a[:, c]))
            out.append(min(max(rnd(v * 255), 0), 255))
        out.append(255)
    return out


def sample(table, v):
    p = min(max(v * 256 - 0.5, 0.0), 255.0)
    i0 = math.floor(p)
    i1 = min(i0 + 1, 255)
    t = p - i0
    res = []
    for c in range(3):
        a, b = table[i0 * 4 + c], table[i1 * 4 + c]
        res.append(min(max(rnd(a + (b - a) * t), 0), 255))
    return res


def main():
    app = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else None
    luts = {}
    for name in ANCHORS:
        for n in (256, 44, 2, 3):
            luts[f"{name}-{n}"] = lut(name, n)
    grid = [-0.1 + 1.2 * i / 2000 for i in range(2001)]
    samples = [c for v in grid for c in sample(luts["inferno-256"], v)]
    if app is not None:
        for k, v in luts.items():
            assert app[k] == v, k
        assert app["sample"] == samples
        print("numpy tables and sampler equal the application's byte for byte")
    out = {"numpy": np.__version__, "luts": luts, "sampleLut": "inferno-256", "sampleGrid": grid, "sample": samples,
           "appChecked": app is not None}
    (ROOT / "Tests/ImagoCoreTests/Fixtures/colormap-pins.json").write_text(json.dumps(out))


main()
