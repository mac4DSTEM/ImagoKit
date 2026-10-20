#!/usr/bin/env python3
"""Generate the radial-chart sample pins; run from the package root:
../tools/memcap.sh -- ../References/ImagoKit/oracle-venv/bin/python -I -B tools/radialchart/pins.py

Writes Tests/ImagoFFTTests/Fixtures/radial-chart-pins.json.

No package implements this chart preparation, so each pin is an independent NumPy reading of the definition in
Sources/ImagoFFT/RadialChartSamples.swift: the drawn bins are k >= 1 with count > 0 at (k + 1/2) * width; |q| = centre * factor
(one float64 multiplication); a sample is drawable when mean > 0 and finite; nearest is np.argmin of |q - x| (the first of
equal distances); the power axis marks are the decades of the smallest and largest drawn mean (decadeMarks' definition).
Doubles are stored as uint64 bit patterns so the comparison is bit for bit.
"""
import json
import math
import platform
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4", np.__version__
assert platform.python_version() == "3.14.7", platform.python_version()
ROOT = Path(__file__).resolve().parents[2]
rng = np.random.Generator(np.random.PCG64(20261020))


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64).tolist()


def samples(width, mean, count, factor):
    out = []
    for k in range(1, len(mean)):
        if count[k] > 0:
            q = (np.float64(k) + np.float64(0.5)) * np.float64(width) * np.float64(factor)
            m = mean[k]
            if m > 0 and math.isfinite(m):
                out.append((k, float(q), float(m), int(count[k])))
    return out


def marks(lo, hi, plot_h, spacing):
    if not (lo > 0 and hi >= lo and math.isfinite(lo) and math.isfinite(hi) and plot_h > 0 and spacing > 0):
        return []
    first = math.floor(math.log10(lo))
    span = max(math.ceil(math.log10(hi)) - first, 1)
    step = max(math.ceil(spacing * span / plot_h), 1)
    return [float(10.0 ** e) for e in range(first, first + span + 1, step)]


def main():
    cases = []
    for n, width, factor, kind in [(2, 0.5, 1.0, "plain"), (1, 1.0, 1.0, "plain"), (9, 0.3125, 1e9, "plain"),
                                   (40, 0.07, 1 / 1.0e-9, "zeros"), (40, 1.3, 1.0, "special"), (200, 0.0123, 3.7, "decades"),
                                   (64, 2.0, 0.1, "zeros")]:
        mean = 10.0 ** rng.uniform(-3, 8, n) if kind != "decades" else 10.0 ** rng.uniform(-9, 9, n)
        count = rng.integers(0, 40, n)
        mean[0] = 1e12
        if kind == "zeros":
            mean[rng.random(n) < 0.3] = 0.0
            count[rng.random(n) < 0.2] = 0
        if kind == "special" and n > 8:
            mean[3] = np.nan; mean[5] = np.inf; mean[6] = -np.inf; mean[7] = -2.0; mean[8] = 5e-324
        s = samples(width, mean, count, factor)
        qs = [t[1] for t in s]
        queries = [-1.0, 0.0] + [float(x) for x in rng.uniform(0, max(qs, default=1.0) * 1.2, 12)] + qs[:3]
        near = []
        for x in queries:
            if not s:
                near.append(-1)
            else:
                near.append(int(np.argmin([abs(t[1] - x) for t in s])))
        means = [t[2] for t in s]
        cases.append({
            "width": width, "factor": factor, "units": "1/nm",
            "meanBits": bits(mean), "count": [int(c) for c in count],
            "binsOut": [t[0] for t in s], "qBits": bits([t[1] for t in s]), "meanOutBits": bits(means),
            "countOut": [t[3] for t in s],
            "queries": queries, "nearestIndex": near,
            "marks": bits(marks(min(means), max(means), 48.0, 16.0)) if means else [],
        })
    (ROOT / "Tests/ImagoFFTTests/Fixtures/radial-chart-pins.json").write_text(json.dumps({"numpy": np.__version__, "cases": cases}))
    print(len(cases), "cases")


main()
