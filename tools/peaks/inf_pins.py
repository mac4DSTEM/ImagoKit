#!/usr/bin/env python3
"""skimage.feature.peak_local_max on images holding +inf and -inf (0.6.1: infinite pixels are accepted and match; NaN stays refused).
python -I -B tools/peaks/inf_pins.py
Writes Tests/ImagoFiltersTests/Fixtures/peak-inf-pins.json; pixels are numbers, "inf" or "-inf"."""
import json
from pathlib import Path

import numpy as np
import skimage
from skimage.feature import peak_local_max

assert skimage.__version__ == "0.26.0"
rng = np.random.Generator(np.random.PCG64(20261022))


def enc(v):
    return "inf" if v == np.inf else "-inf" if v == -np.inf else float(f"{float(v):.17g}")


cases = []
for h, w, d, border, thr, nump, dtype in [
    (9, 11, 1, True, None, None, "float64"), (12, 10, 2, True, None, None, "float64"), (8, 8, 1, False, None, None, "float64"),
    (10, 10, 3, False, None, None, "float64"), (9, 9, 1, False, 0.5, None, "float32"), (11, 13, 2, True, None, 3, "float32"),
    (7, 7, 0, False, None, None, "float64"),
]:
    x = rng.random((h, w)).round(3)
    idx = rng.choice(h * w, 4, replace=False)
    x.flat[idx[:2]] = np.inf
    x.flat[idx[2:]] = -np.inf
    x = x.astype(dtype)
    kw = dict(min_distance=d, exclude_border=(d if border else False), threshold_abs=thr, num_peaks=nump if nump else np.inf)
    p = peak_local_max(x, **kw)
    cases.append({"h": h, "w": w, "dtype": dtype, "x": [enc(v) for v in x.ravel()], "min_distance": d, "border": border,
                  "threshold_abs": thr, "num_peaks": nump, "peaks": p.tolist()})
out = Path(__file__).resolve().parents[2] / "Tests/ImagoFiltersTests/Fixtures/peak-inf-pins.json"
out.write_text(json.dumps({"skimage": skimage.__version__, "cases": cases}))
print("wrote", out, len(cases))
