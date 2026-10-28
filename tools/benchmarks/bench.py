#!/usr/bin/env python3
"""The Python side of docs/benchmarks.md: the same functions, sizes and dtypes as Tests/ImagoKitParityTests/Benchmarks/Benchmarks.swift.
  python -I -B tools/benchmarks/bench.py /path/python.json [--large]
One warm-up call, five timed calls, milliseconds. Single-threaded NumPy FFT and SciPy filters, as installed in the pinned environment.
"""
import json
import sys
import time

import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage import filters, measure, registration

out_path = sys.argv[1]
large = "--large" in sys.argv
sizes = [128, 512] + ([2048, 4096, 8192] if large else [])
rng = np.random.Generator(np.random.PCG64(1))
results = []


def record(name, dtype, n, fn):
    fn()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    results.append(dict(name=name, dtype=dtype, size=n, ms=times))
    print(f"BENCH {name} {dtype} {n} best {min(times):.3f} ms")


def img(n, dt):
    return (rng.random((n, n)) * 40 + 30).astype(dt)


for n in sizes:
    for dt, label in ((np.float32, "float32"), (np.float64, "float64")):
        x = img(n, dt)
        record("fft2", label, n, lambda: np.fft.fft2(x))
        record("rfft2", label, n, lambda: np.fft.rfft2(x))
        spec = np.fft.rfft2(x)
        record("irfft2", label, n, lambda: np.fft.irfft2(spec, s=(n, n)))
        if n == 512:
            y = img(500, dt)
            record("fft2", label, 500, lambda: np.fft.fft2(y))
            if dt is np.float64:
                for m in (360, 480, 504):
                    ym = img(m, dt)
                    record("fft2", label, m, lambda: np.fft.fft2(ym))
        record("gaussian_filter sigma=3", label, n, lambda: ndi.gaussian_filter(x, 3, mode="reflect"))
        record("uniform_filter size=5", label, n, lambda: ndi.uniform_filter(x, 5))
        record("butterworth", label, n, lambda: filters.butterworth(x, 0.1))
    x = img(n, np.float32)
    o = img(n, np.float32)
    record("difference_of_gaussians", "float32", n, lambda: filters.difference_of_gaussians(x, 2))
    for order in (1, 3):
        record(f"shift order={order}", "float32", n, lambda: ndi.shift(x, (2.3, -1.7), order=order, mode="reflect"))
    if n <= 2048:
        record("phase_cross_correlation", "float32", n, lambda: registration.phase_cross_correlation(x, o))
        record("phase_cross_correlation upsample=10", "float32", n, lambda: registration.phase_cross_correlation(x, o, upsample_factor=10))
        x64, o64 = x.astype(np.float64), o.astype(np.float64)
        record("phase_cross_correlation", "float64", n, lambda: registration.phase_cross_correlation(x64, o64))
    src, dst = (5, 5), (n - 6, n - 6)
    record("profile_line order=1 width=5", "float32", n, lambda: measure.profile_line(x, src, dst, linewidth=5, order=1, mode="reflect"))
    record("profile_line order=3 width=5", "float32", n, lambda: measure.profile_line(x, src, dst, linewidth=5, order=3, mode="reflect"))
    mask = x > 50
    record("label connectivity=2", "bool", n, lambda: measure.label(mask, connectivity=2))
for n in (64, 256):
    for dt, label in ((np.float32, "float32"), (np.float64, "float64")):
        a, b = img(n, dt), img(n, dt)
        m = np.ones((n, n), bool)
        record("masked phase_cross_correlation", label, n, lambda: registration.phase_cross_correlation(a, b, reference_mask=m))
json.dump(dict(results=results, numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__), open(out_path, "w"))
