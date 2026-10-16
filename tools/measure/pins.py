#!/usr/bin/env python3
"""Generate the measurement, histogram and line-profile pins; run from the package root:
../tools/memcap.sh -- ../References/ImagoKit/oracle-venv/bin/python -I -B tools/measure/pins.py

Writes Tests/ImagoCoreTests/Fixtures/measure-pins.json.
  - histogram: numpy.histogram(f[isfinite(f)], bins, range) on float32 arrays; pixels, edges and extremes as float32 bit
    patterns (uint32), counts as integers, the float64 mean of the finite pixels, or the exception numpy raises.
  - profile: skimage.measure.profile_line on the float32 image (float32 output) and on the same pixels widened to float64,
    for orders 0 and 1, the five modes, several widths and reductions; plus reduce_func=None rows.
  - measure: distance and angle of random point triples with anisotropic scales, from numpy float64.
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import skimage
from skimage.measure import profile_line

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert skimage.__version__ == "0.26.0", skimage.__version__
assert platform.python_version() == "3.14.7", platform.python_version()
SEED = 20261017
rng = np.random.Generator(np.random.PCG64(SEED))


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).tolist()


# ---------------------------------------------------------------- histogram
hist_cases = []


def add_hist(name, f, bins, rng_=None):
    f = np.asarray(f, dtype=np.float32)
    g = f[np.isfinite(f)]
    case = dict(name=name, bins=bins, pixels=bits(f),
                range=None if rng_ is None else bits(np.array(rng_, dtype=np.float32)))
    try:
        r = None if rng_ is None else (np.float32(rng_[0]), np.float32(rng_[1]))
        counts, edges = np.histogram(g, bins=bins, range=r)
        assert edges.dtype == np.float32, edges.dtype
        case.update(counts=counts.tolist(), edges=bits(edges))
    except ValueError as e:
        case.update(error=str(e))
    if g.size:
        case.update(min=bits(np.array([g.min()]))[0], max=bits(np.array([g.max()]))[0],
                    finite=int(g.size), mean=float(np.mean(g, dtype=np.float64)),
                    meanAbs=float(np.mean(np.abs(g), dtype=np.float64)))
    hist_cases.append(case)


for bins in (1, 3, 10, 96, 257):
    add_hist(f"gauss-{bins}", rng.normal(100, 25, 600), bins)
    add_hist(f"uniform-neg-{bins}", rng.uniform(-1e-3, 5e-3, 400), bins)
    add_hist(f"integers-{bins}", rng.integers(0, 4096, 500), bins)
    add_hist(f"explicit-range-{bins}", rng.normal(0, 3, 400), bins, (-2.0, 2.5))
    add_hist(f"large-{bins}", rng.normal(3e7, 4e6, 300), bins)
add_hist("constant", np.full(50, 7.25), 96)
add_hist("constant-zero", np.zeros(20), 10)
add_hist("constant-2e24", np.full(20, 2.0 ** 24), 10)           # +-0.5 is lost: numpy raises
add_hist("signed-zeros", [0.0, -0.0, 1.0, -0.0, 0.0, -1.0], 4)
add_hist("only-negative-zero", [-0.0, -0.0], 4)
add_hist("nan-inf", [1, 2, np.nan, np.inf, -np.inf, 3, 4.5, 9, np.nan], 5)
add_hist("all-nan", [np.nan, np.inf], 5)
add_hist("one-pixel", [3.5], 96)
add_hist("one-ulp-apart", [1.0, np.nextafter(np.float32(1.0), np.float32(2.0))], 96)
add_hist("denormal-range", np.linspace(0, 200, 201) * np.float32(1e-45), 96)
add_hist("range-point", rng.normal(0, 1, 100), 8, (0.5, 0.5))
add_hist("range-reversed", rng.normal(0, 1, 100), 8, (1.0, -1.0))
add_hist("range-inf", rng.normal(0, 1, 100), 8, (-np.inf, 1.0))
# Pixels planted on the edges of a 20-bin histogram and one ulp either side.
edges20 = np.histogram(np.array([0.0, 1.0], dtype=np.float32), bins=20)[1]
planted = np.concatenate([edges20, np.nextafter(edges20, np.float32(2)), np.nextafter(edges20, np.float32(-1))])
add_hist("on-edges", planted, 20, (0.0, 1.0))
edges7 = np.histogram(np.array([-3.3, 11.1], dtype=np.float32), bins=7)[1]
planted7 = np.concatenate([edges7, np.nextafter(edges7, np.float32(99)), np.nextafter(edges7, np.float32(-99))])
add_hist("on-edges-7", planted7, 7)

# ---------------------------------------------------------------- profile_line
H, W = 17, 23
img32 = rng.normal(50, 20, (H, W)).astype(np.float32)
img64 = img32.astype(np.float64)
profile_cases = []
lines = [((2, 3), (12, 19)), ((0, 0), (16, 22)), ((5, 5), (5, 5)), ((8, 1), (8, 21)), ((1, 10), (15, 10)),
         ((-4.5, -3.25), (9.75, 30.5)), ((3.3, 2.2), (25.1, 40.7)), ((14, 20), (2, 4)), ((6.5, 0), (6.5, 22.6)),
         ((0.4, 0.9), (0.4, 1.3)), ((-30.7, 12.3), (40.1, 11.9)), ((10, 30), (10, -12))]
for li, (src, dst) in enumerate(lines):
    for order in (0, 1):
        for mode in ("constant", "nearest", "reflect", "mirror", "wrap"):
            widths = (1, 3, 8, 9) if (li + order) % 2 == 0 else (2, 4, 5, 25)
            widths = widths[(li + len(mode)) % 2::2]
            for lw in widths:
                for name, fn in (("mean", np.mean), ("sum", np.sum), ("min", np.min), ("max", np.max), ("median", np.median)):
                    if name != "mean" and (li + lw + order + len(mode)) % 7:
                        continue
                    cval = 7.5 if mode == "constant" else 0.0
                    kw = dict(linewidth=lw, order=order, mode=mode, cval=cval, reduce_func=fn)
                    out32 = profile_line(img32, src, dst, **kw)
                    out64 = profile_line(img64, src, dst, **kw)
                    assert out32.dtype == np.float32 and out64.dtype == np.float64
                    profile_cases.append(dict(src=src, dst=dst, lw=lw, order=order, mode=mode, cval=cval, reduce=name,
                                              f32=out32.tolist(), f64=out64.tolist()))
unreduced = []
for src, dst, lw, order, mode in [((2, 3), (12, 19), 3, 1, "reflect"), ((1, 2), (4, 2), 3, 0, "constant"),
                                  ((-3.5, 4), (6.2, 17.9), 4, 1, "mirror")]:
    kw = dict(linewidth=lw, order=order, mode=mode, cval=0.0, reduce_func=None)
    unreduced.append(dict(src=src, dst=dst, lw=lw, order=order, mode=mode,
                          f32=profile_line(img32, src, dst, **kw).tolist(),
                          f64=profile_line(img64, src, dst, **kw).tolist()))

# ---------------------------------------------------------------- distance and angle
measure = []
for _ in range(100):
    p = rng.uniform(-50, 50, (3, 2))
    sx, sy = rng.uniform(0.01, 3, 2)
    a, b, c = p
    dc, dr = b[0] - a[0], b[1] - a[1]
    u = np.array([(a[0] - b[0]) * sx, (a[1] - b[1]) * sy])
    v = np.array([(c[0] - b[0]) * sx, (c[1] - b[1]) * sy])
    deg = float(np.degrees(np.arctan2(abs(u[0] * v[1] - u[1] * v[0]), u @ v)))
    measure.append(dict(a=a.tolist(), b=b.tolist(), c=c.tolist(), sx=float(sx), sy=float(sy),
                        distancePx=float(np.hypot(dc, dr)), distancePhys=float(np.hypot(dc * sx, dr * sy)),
                        angleDegrees=deg))

out = Path(__file__).resolve().parents[2] / "Tests/ImagoCoreTests/Fixtures/measure-pins.json"
out.write_text(json.dumps(dict(
    versions=dict(numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
                  python=platform.python_version()), seed=SEED,
    histogram=hist_cases, image=dict(height=H, width=W, bits=bits(img32.ravel())), profile=profile_cases,
    unreduced=unreduced, measure=measure), separators=(",", ":")) + "\n")
print(f"Wrote {out.name}: {len(hist_cases)} histograms, {len(profile_cases)} profiles, {len(measure)} measures, "
      f"{out.stat().st_size} bytes")
