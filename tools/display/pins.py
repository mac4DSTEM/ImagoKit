#!/usr/bin/env python3
"""Generate the display-numerics pins; run from the package root:
../References/python/skimage/bin/python -I -B tools/display/pins.py

Writes Tests/ImagoCoreTests/Fixtures/display-core-pins.json and Tests/ImagoFFTTests/Fixtures/display-fft-pins.json.

The numbers here are display conventions with no Python package behind them, so each pin is an independent NumPy / Python
reading of the DEFINITION recorded in the Swift file's header (box averages by integer division, the 1-2-5 ladder, the
colour-table interpolation, the square-region rules, ...), written without reference to the Swift code. Float32 arrays are
stored as uint32 bit patterns (thumbnails: base64 of little-endian float32), so the quick check (D12) can be bit for bit.
"""
import json
import math
import platform
import unicodedata
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4", np.__version__
assert platform.python_version() == "3.14.7", platform.python_version()
SEED = 20261018
rng = np.random.Generator(np.random.PCG64(SEED))
ROOT = Path(__file__).resolve().parents[2]


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).tolist()


def b64(a):
    import base64
    return base64.b64encode(np.ascontiguousarray(a, dtype='<f4').tobytes()).decode()


def rnd(x):
    """Swift's .rounded(): half away from zero."""
    f = math.floor(abs(x))
    r = f + (1 if abs(x) - f >= 0.5 else 0)
    return r if x >= 0 else -r


def normalized_unit(u):
    u = unicodedata.normalize("NFC", u or "").strip().lower().replace(" ", "")
    return u.replace("ångström", "angstrom")


NM = {}
for names, v in [(("m", "meter", "meters", "metre", "metres"), 1e9),
                 (("mm", "millimeter", "millimeters", "millimetre", "millimetres"), 1e6),
                 (("µm", "μm", "um", "micron", "microns", "micrometer", "micrometers", "micrometre", "micrometres"), 1e3),
                 (("nm", "nanometer", "nanometers", "nanometre", "nanometres"), 1.0),
                 (("å", "a", "angstrom", "angstroms", "ang"), 0.1),
                 (("pm", "picometer", "picometers", "picometre", "picometres"), 1e-3)]:
    for n in names:
        NM[n] = v


def nm_per(u):
    return NM.get(normalized_unit(u))


# ================================================================= core: normalisation
norm_cases = []
for i, (n, kind) in enumerate([(50, "gauss"), (200, "wide"), (1, "one"), (30, "const"), (40, "neg"), (60, "zero"), (80, "nonfinite"),
                               (20, "allnan"), (25, "huge")]):
    if kind == "gauss":
        a = rng.normal(1000, 300, n)
    elif kind == "wide":
        a = rng.lognormal(3, 3, n)
    elif kind == "one":
        a = np.array([5.0])
    elif kind == "const":
        a = np.full(n, 3.25)
    elif kind == "neg":
        a = rng.normal(-5, 20, n)
    elif kind == "zero":
        a = np.zeros(n)
    elif kind == "nonfinite":
        a = rng.normal(0, 7, n)
        a[[3, 9, 17]] = [np.nan, np.inf, -np.inf]
    elif kind == "allnan":
        a = np.full(n, np.nan)
    else:
        a = np.array([-3e38, 3e38, 1.0, 2.0])
    a = a.astype(np.float32)
    fin = a[np.isfinite(a)]
    lo, hi = (np.float32(0), np.float32(1)) if fin.size == 0 else (fin.min(), fin.max())
    out = {}
    for sym in (False, True):
        with np.errstate(all="ignore"):
            if sym:
                m = max(abs(lo), abs(hi))
                if not m > 0:
                    r = np.where(np.isfinite(a), np.float32(0.5), np.float32(-1))
                else:
                    r = np.where(np.isfinite(a), np.float32(0.5) + np.float32(0.5) * (a / m), np.float32(-1))
            else:
                span = np.float32(hi - lo)
                if not span > 0:
                    r = np.where(np.isfinite(a), np.float32(0), np.float32(-1))
                else:
                    r = np.where(np.isfinite(a), (a - lo) / span, np.float32(-1))
        out["symmetric" if sym else "linear"] = bits(r.astype(np.float32))
    with np.errstate(all="ignore"):
        lg = np.where(np.isfinite(a), np.log10(1.0 + np.maximum(a, 0).astype(np.float64)), 0.0).astype(np.float32)
    norm_cases.append(dict(kind=kind, pixels=bits(a), lo=bits([lo])[0], hi=bits([hi])[0], log=bits(lg), **out))

# ================================================================= core: thumbnails
thumb_cases = []
for (W, H) in [(120, 100), (97, 150), (400, 10), (5, 5), (96, 96), (97, 97), (123, 1)]:
    a = rng.uniform(0, 1, (H, W)).astype(np.float32)
    mask = rng.uniform(0, 1, (H, W)) < 0.05
    a[mask] = -1
    if W > 100 and H > 8:
        a[0:3, 0:200] = -1                # a band of invalid pixels: some boxes are all invalid
        a[5, 7] = np.nan
    big = max(W, H)
    if big <= 96:
        w, h, out = W, H, a
    else:
        sc = lambda s: 96 if s == big else max(1, rnd(s * 96 / big))
        w, h = sc(W), sc(H)
        xs = np.arange(W)
        cb = np.searchsorted((np.arange(1, w + 1) * W) // w, xs, side="right")
        ys = np.arange(H)
        rb = np.searchsorted((np.arange(1, h + 1) * H) // h, ys, side="right")
        out = np.full((h, w), np.float32(-1), dtype=np.float32)
        for j in range(h):
            for i in range(w):
                box = a[np.ix_(rb == j, cb == i)].astype(np.float64)
                ok = np.isfinite(box) & (box != -1)
                if ok.any():
                    out[j, i] = np.float32(box[ok].sum() / ok.sum())
    thumb_cases.append(dict(width=W, height=H, w=w, h=h, pixels=b64(a.ravel()), out=b64(out.ravel())))

# ================================================================= core: scale bar
UNITS = ["nm", "µm", "m", "mm", "Å", "pm", "1/nm", "1/Å", "px", "", "mrad", "A^-1", "um", "Nanometres", "metre", "kg"]
SIZES = [(512, 512), (1024, 768), (4096, 4096), (200, 100), (1000, 1000), (2048, 2048), (64, 64), (300, 3000), (8192, 512)]
PX_LABELS = ["px", "pix", "[pix]", "[px]", "pixel", "pixels"]
REAL = {}
for names, c in [(("a", "å", "angstrom", "ang"), "Å"), (("nm", "nanometer", "nanometers", "nanometre", "nanometres"), "nm"),
                 (("pm", "picometer", "picometers", "picometre", "picometres"), "pm")]:
    for n in names:
        REAL[n] = c
RECIP = {}
for names, c in [(("1/a", "1/å", "a^-1", "å^-1", "a⁻¹", "å⁻¹", "angstrom^-1", "angstrom⁻¹", "1/angstrom", "ang^-1", "ang⁻¹"), "Å⁻¹"),
                 (("1/nm", "nm^-1", "nm⁻¹", "1/nanometer", "1/nanometre", "nanometer^-1", "nanometer⁻¹", "nanometre^-1", "nanometre⁻¹"), "nm⁻¹"),
                 (("mrad",), "mrad")]:
    for n in names:
        RECIP[n] = c


def display_label(u):
    n = normalized_unit(u)
    if n in PX_LABELS:
        return "px"
    return RECIP.get(n) or REAL.get(n) or u


def calibrated(ax):
    return ax is not None and ax[1] != "" and math.isfinite(ax[0]) and ax[0] > 0


def shown(ax, reference):
    scale, unit = ax
    isM = normalized_unit(unit) in ("m", "meter", "meters", "metre", "metres")
    if not (isM and calibrated(ax) and math.isfinite(reference) and reference > 0):
        return ax
    length = abs(reference)
    per, lab = (1e9, "nm") if length < 1e-6 else ((1e6, "µm") if length < 1e-3 else (1e3, "mm"))
    return (scale * per, lab)


def nice125(v):
    if not (v > 0 and math.isfinite(v)):
        return 1.0
    e = math.floor(math.log10(v))
    base = 10.0 ** e
    m = v / base
    nice = 1.0 if m < 1.5 else (2.0 if m < 3.5 else (5.0 if m < 7.5 else 10.0))
    return nice * base


def ladder(v, unit):
    rungs = ["nm", "µm", "mm"]
    canon = {"nm": "nm", "nanometre": "nm", "nanometres": "nm", "nanometer": "nm", "nanometers": "nm",
             "µm": "µm", "μm": "µm", "um": "µm", "mm": "mm"}.get(unit)
    if canon is None or not (v > 0 and math.isfinite(v)):
        return v, unit
    idx = start = rungs.index(canon)
    s = v
    while s >= 1000 and idx < 2:
        s /= 1000
        idx += 1
    while s < 1 and idx > 0:
        s *= 1000
        idx -= 1
    return (v if idx == start else nice125(s)), rungs[idx]


def fmt(v):
    if v >= 10:
        return "%.0f" % v
    if v >= 1:
        return "%g" % v
    return "%.3g" % v


def plan(x, y, W, H):
    sx = shown(x, x[0] * W) if x else None
    sy = shown(y, y[0] * H) if y else None
    along = sx[0] if sx else (sy[0] if sy else None)
    if sx and sx[1] != "" and along is not None and math.isfinite(along) and along > 0:
        per, unit = along, display_label(sx[1])
    else:
        per, unit = 1.0, "px"
    u = max(1, rnd(min(W, H) / 512))
    target = min(70.0 * u, W / 4)
    nice = nice125(per * target)
    length = nice / per
    if not (math.isfinite(length) and length >= 0.5 and length < W):
        return None
    val, lu = ladder(nice, unit)
    return dict(perPixel=per, unit=unit, value=val, labelUnit=lu, label=f"{fmt(val)} {lu}", lengthPixels=length,
                drawnPixels=rnd(length), overlayScale=u)


bar_cases = []
for _ in range(250):
    def axis():
        if rng.uniform() < 0.12:
            return None
        return (float(10.0 ** rng.uniform(-12, 3)) if rng.uniform() < 0.9 else float(rng.choice([0.0, -1.0, 1.0])),
                str(rng.choice(UNITS)))
    x, y = axis(), axis()
    if rng.uniform() < 0.3 and x:
        y = x
    W, H = SIZES[int(rng.integers(len(SIZES)))]
    bar_cases.append(dict(x=x, y=y, width=W, height=H, plan=plan(x, y, W, H)))
for W, H in SIZES:                                    # round numbers, the usual calibrations
    for sc, un in [(0.0105, "nm"), (0.25, "nm"), (1.0, "nm"), (0.5, "µm"), (2.5e-10, "m"), (3e-9, "m"), (4e-5, "m"), (0.05, "Å"),
                   (0.002, "1/nm"), (0.0125, "1/Å"), (0.1, "mrad"), (1, "px")]:
        bar_cases.append(dict(x=(sc, un), y=(sc, un), width=W, height=H, plan=plan((sc, un), (sc, un), W, H)))
nice_cases, ladder_cases = [], []
for e in range(-10, 10):
    for m in (1.0, 1.4999999999999998, 1.5, 1.5000000000000002, 2.9, 3.4999999999999996, 3.5, 3.5000000000000004, 7.4999999999999991,
              7.5, 9.99, 0.99999999999):
        v = m * 10.0 ** e
        nice_cases.append(dict(v=v, nice=nice125(v), format=fmt(v)))
        for u in ("nm", "µm", "mm", "um", "μm", "Å", "1/nm", "px", "pm", "nanometre"):
            lv, lu = ladder(v, u)
            ladder_cases.append(dict(v=v, unit=u, value=lv, outUnit=lu))
nice_cases += [dict(v=v, nice=nice125(v), format=fmt(v)) for v in (0.0, -3.0)]

# ================================================================= core: display mapping
map_cases = []
CLAMP = lambda x, lo, hi: lo if not (x > lo) else (x if x < hi else hi)
for _ in range(8):
    lut = rng.integers(0, 256, 1024).tolist()
    lo = float(np.float32(rng.uniform(0, 0.5)))
    hi = float(np.float32(lo + rng.uniform(-0.1, 0.6))) if rng.uniform() < 0.9 else lo
    gamma = float(rng.choice([1.0, 0.5, 2.2, 0.01, 3.0, float(rng.uniform(0.2, 4))]))
    ns = [-1.0, 0.0, 1.0, 0.5, float(np.float32(lo)), float(np.float32(hi)), 1.5, -0.25] + \
         [float(np.float32(v)) for v in rng.uniform(0, 1, 40)]
    outs = []
    for n in ns:
        if not (n >= 0):
            outs.append([82, 82, 87])
            continue
        v = CLAMP(n, 0.0, 1.0)
        span = (hi - lo) if (hi - lo) > 1e-6 else 1e-6
        v = CLAMP((v - lo) / span, 0.0, 1.0)
        v = v ** (1 / (gamma if gamma > 0.05 else 0.05))
        p = CLAMP(v * 256 - 0.5, 0.0, 255.0)
        i0 = math.floor(p)
        i1 = min(i0 + 1, 255)
        t = p - i0
        outs.append([int(CLAMP(float(rnd((lut[i0 * 4 + c] + (lut[i1 * 4 + c] - lut[i0 * 4 + c]) * t))), 0.0, 255.0)) for c in range(3)])
    map_cases.append(dict(lut=lut, low=bits([lo])[0], high=bits([hi])[0], gamma=gamma, n=bits(ns), rgb=outs))

core = dict(numpy=np.__version__, python=platform.python_version(), seed=SEED, normalization=norm_cases, thumbnails=thumb_cases,
            bars=bar_cases, nice=nice_cases, ladder=ladder_cases, mapping=map_cases)
p = ROOT / "Tests/ImagoCoreTests/Fixtures/display-core-pins.json"
p.write_text(json.dumps(core, separators=(",", ":"), allow_nan=False) + "\n")
print(f"Wrote {p.name}: {p.stat().st_size} bytes")

# ================================================================= fft: regions, bins, decades, units
def maximum_side(w, h):
    s = min(w, h)
    ev = s - (s & 1)
    return ev if ev >= 32 else None


def edge(x, count):
    return int(min(max(rnd(x + 0.5), 0), count))


def shifted(row, col, h, w, iw, ih):
    return dict(row=min(max(row, 0), max(ih - h, 0)), col=min(max(col, 0), max(iw - w, 0)), height=h, width=w)


def drawn(p0, p1, W, H):
    L = maximum_side(W, H)
    if L is None or not all(math.isfinite(v) for v in (*p0, *p1)):
        return None
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ext = max(abs(dx), abs(dy))
    side = int(min(max(2 * math.floor(ext / 2), 32), L))
    ex, ey = edge(p0[0], W), edge(p0[1], H)
    col = ex if dx >= 0 else ex - side
    row = ey if dy >= 0 else ey - side
    return shifted(row, col, side, side, W, H)


def typed(left, top, size, W, H):
    L = maximum_side(W, H)
    if L is None:
        return None
    side = min(max(size - (size & 1), 32), L)
    return shifted(top, left, side, side, W, H)


def whole(d):
    return int(min(max(rnd(d), -1e9), 1e9))


region = []
for _ in range(300):
    W, H = int(rng.integers(20, 700)), int(rng.integers(20, 700))
    p0 = [float(rng.uniform(-50, W + 50)), float(rng.uniform(-50, H + 50))]
    p1 = [float(rng.uniform(-80, W + 80)), float(rng.uniform(-80, H + 80))]
    if rng.uniform() < 0.1:
        p1 = [p0[0] + float(rng.integers(-3, 4)) * 0.5, p0[1]]
    crop = drawn(p0, p1, W, H)
    c0 = typed(int(rng.integers(-20, 300)), int(rng.integers(-20, 300)), int(rng.integers(-5, 800)), W, H)
    mv = None
    if c0 is not None:
        dx, dy = float(rng.uniform(-300, 300)), float(rng.uniform(-300, 300))
        mv = dict(dx=dx, dy=dy, crop=shifted(c0["row"] + whole(dy), c0["col"] + whole(dx), c0["height"], c0["width"], W, H))
    cs = None if maximum_side(W, H) is None else dict(row=(H - maximum_side(W, H)) // 2, col=(W - maximum_side(W, H)) // 2,
                                                      height=maximum_side(W, H), width=maximum_side(W, H))
    region.append(dict(width=W, height=H, p0=p0, p1=p1, drawn=crop, typedArgs=None, typed=c0, moved=mv, centred=cs,
                       maximum=maximum_side(W, H)))

bins = []
for _ in range(200):
    W, H = int(rng.integers(1, 400)), int(rng.integers(1, 400))
    qx, qy = float(rng.uniform(-30, W + 30)), float(rng.uniform(-30, H + 30))
    r = (min(max(rnd(qy), 0), H - 1), min(max(rnd(qx), 0), W - 1))
    kx, ky = int(rng.integers(-W, W)), int(rng.integers(-H, H))
    ok = -(W // 2) <= kx < W - W // 2 and -(H // 2) <= ky < H - H // 2
    bins.append(dict(width=W, height=H, qx=qx, qy=qy, row=r[0], col=r[1], kx=kx, ky=ky,
                     k=None if not ok else [H // 2 + ky, W // 2 + kx]))


def decades(lo, hi, ph, ms):
    if not (lo > 0 and hi >= lo and math.isfinite(lo) and math.isfinite(hi) and ph > 0 and math.isfinite(ph) and ms > 0 and math.isfinite(ms)):
        return []
    first = math.floor(math.log10(lo))
    span = max(math.ceil(math.log10(hi)) - first, 1)
    step = max(math.ceil(ms * span / ph), 1)
    return [10.0 ** k for k in range(first, first + span + 1, step)]


dec = []
for _ in range(100):
    lo = float(10.0 ** rng.uniform(-8, 4))
    hi = lo * float(10.0 ** rng.uniform(0, 9))
    ph, ms = float(rng.uniform(40, 600)), float(rng.uniform(10, 60))
    dec.append(dict(lo=lo, hi=hi, plotHeight=ph, minimumSpacing=ms, marks=decades(lo, hi, ph, ms)))
dec.append(dict(lo=-1.0, hi=3.0, plotHeight=100.0, minimumSpacing=10.0, marks=[]))

FUNITS = ["nm", "µm", "m", "mm", "Å", "pm", "px", "", "1/nm", "1/Å", "1/m", "1/px", "1/µm", "kg", "1/kg", "Nanometres", "A", " nm ", "mrad"]
RPX = "px⁻¹"
RNM = "nm⁻¹"


def uncal(u):
    n = normalized_unit(u)
    return n == "" or n == "px"


def f_length(d, u):
    nm = nm_per(u)
    if nm is not None:
        return d * nm, "nm"
    if uncal(u):
        return d, "px"
    return d, u


def spectrum_unit(u):
    n = normalized_unit(u)
    if n == "" or n == "1/px":
        return 1.0, RPX
    if n.startswith("1/") and nm_per(n[2:]) is not None:
        return 1 / nm_per(n[2:]), RNM
    nm = nm_per(u)
    if nm is not None:
        return nm, "nm"
    return 1.0, u


def f_reciprocal(q, u):
    nm = nm_per(u)
    if nm is not None:
        return q / nm, RNM
    if uncal(u):
        return q, RPX
    t = u.strip()
    if t.startswith("1/") and len(t) > 2:
        f, lab = spectrum_unit(t[2:])
        return q * f, lab
    return q, "1/" + u


def aspect(w, h, sx, ux, sy, uy):
    if normalized_unit(ux) == normalized_unit(uy):
        a, b = sx, sy
    else:
        x, y = spectrum_unit(ux), spectrum_unit(uy)
        if x[1] != y[1]:
            return 1.0
        a, b = sx * x[0], sy * y[0]
    try:
        r = (w * a) / (h * b)
    except ZeroDivisionError:
        return 1.0
    return r if math.isfinite(r) and r > 0 else 1.0


units = []
for u in FUNITS:
    q = float(rng.uniform(0.1, 9))
    units.append(dict(unit=u, d=q, length=list(f_length(q, u)), reciprocal=list(f_reciprocal(q, u)), spectrum=list(spectrum_unit(u))))
aspects = []
for ux in FUNITS:
    for uy in FUNITS:
        w, h = int(rng.integers(8, 300)), int(rng.integers(8, 300))
        sx, sy = float(10.0 ** rng.uniform(-3, 1)), float(10.0 ** rng.uniform(-3, 1))
        if rng.uniform() < 0.05:
            sy = 0.0
        aspects.append(dict(width=w, height=h, sx=sx, ux=ux, sy=sy, uy=uy, aspect=aspect(w, h, sx, ux, sy, uy)))

# ================================================================= fft: the log-power display
disp = []
for (w, h) in [(24, 18), (33, 33), (16, 16), (7, 5), (64, 48), (3, 3), (1, 1), (40, 7)]:
    pw = (10.0 ** rng.uniform(-3, 9, w * h)).astype(np.float32)
    pw[rng.uniform(0, 1, w * h) < 0.05] = 0
    if w * h > 30:
        pw[3] = np.inf
        pw[5] = np.nan
    pw[(h // 2) * w + w // 2] = np.float32(1e12)           # the DC is large and must stay NaN
    scale = (float(w) * float(h)) ** 2
    with np.errstate(all="ignore"):
        v = np.where((pw > 0) & np.isfinite(pw), np.log10(pw.astype(np.float64) / scale), np.nan).astype(np.float32)
    v[(h // 2) * w + w // 2] = np.nan
    fin = np.isfinite(v)
    n_fin = int(fin.sum())
    window = None
    sorted_median = None
    if fin.any():
        rr, cc = np.divmod(np.arange(w * h), w)
        far = fin & ((cc - w // 2) ** 2 + (rr - h // 2) ** 2 > 9)
        if far.any():
            white = v[far].max()
            lo, hi = v[fin].min(), v[fin].max()
            n = int(fin.sum())
            if lo < hi:
                bw = (np.float64(hi) - np.float64(lo)) / 2048
                idx = np.minimum(((v[fin].astype(np.float64) - np.float64(lo)) / bw).astype(np.int64), 2047)
                counts = np.bincount(idx, minlength=2048)
                half = n / 2
                below = 0
                med = np.float32(hi)
                for b in range(2048):
                    if below + counts[b] >= half:
                        m = float(lo) + (b + (half - below) / counts[b]) * bw
                        med = np.float32(min(max(m, float(lo)), float(hi)))
                        break
                    below += counts[b]
            else:
                med = lo
            window = [bits([med])[0], bits([white])[0]]
            sorted_median = float(np.sort(v[fin].astype(np.float64))[math.ceil(n_fin / 2) - 1])   # the (ceil(n/2))-th smallest
            binwidth = float((np.float64(hi) - np.float64(lo)) / 2048)
    disp.append(dict(width=w, height=h, power=bits(pw), values=bits(v), window=window, sortedMedian=sorted_median,
                     binWidth=binwidth if window else None))

fft = dict(numpy=np.__version__, python=platform.python_version(), seed=SEED, region=region, bins=bins, decades=dec, units=units,
           aspects=aspects, display=disp)
p = ROOT / "Tests/ImagoFFTTests/Fixtures/display-fft-pins.json"
p.write_text(json.dumps(fft, separators=(",", ":"), allow_nan=False) + "\n")
print(f"Wrote {p.name}: {p.stat().st_size} bytes")
