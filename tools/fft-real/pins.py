#!/usr/bin/env python3
"""Real-input FFT pins and the mixed-radix lengths: numpy.fft.rfft2 / irfft2 at larger shapes, rfft / irfft (n=, norm=), and
fft / ifft at lengths of small prime factors (2, 3, 5, 7) that the Float64 path serves by its mixed-radix plan.
Run from the package root:
  ../References/python/skimage/bin/python -I -B tools/fft-real/pins.py
Writes Tests/ImagoFFTTests/Fixtures/fft-real-pins.json (read by Tests/ImagoKitParityTests/FFT/FFTRealTests.swift).
Deterministic: PCG64 seed 20261023, floats written as Python reprs. Inputs are float64 normal draws rounded to float32, so the
Float32 path sees the same values; the truth is NumPy's float64 result on those values.
"""
import json
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4", np.__version__
SEED = 20261023
rng = np.random.Generator(np.random.PCG64(SEED))
NORMS = ("backward", "forward", "ortho")


def f32(a):
    return a.astype(np.float32).astype(np.float64)


def cnormal(*shape):
    return f32(rng.normal(size=shape)) + 1j * f32(rng.normal(size=shape))


def flat(a):
    return a.ravel().tolist()


def pair(a):
    return [flat(a.real), flat(a.imag)]


# rfft2 / irfft2 at larger and odd shapes. `trunc` is a half spectrum with more columns than s needs, `pad` one with fewer.
rfft2 = []
for h, w in [(64, 64), (50, 45), (63, 40), (33, 34), (2, 100), (100, 2), (37, 41), (1, 64), (64, 1), (3, 5)]:
    x = f32(rng.normal(size=(h, w)))
    for norm in NORMS:
        if norm != "backward" and (h, w) not in [(50, 45), (33, 34), (3, 5)]:
            continue
        X = np.fft.rfft2(x, norm=norm)
        zc = {"default": w // 2 + 1, "trunc": w // 2 + 3, "pad": max(1, w // 2 - 1)}
        irfft = {}
        for key, cols in zc.items():
            if key != "default" and (h, w) not in [(33, 34), (3, 5), (37, 41), (64, 1)]:
                continue
            Z = cnormal(h, cols)
            if key == "default" and 2 * (cols - 1) == 0:
                continue
            kw = dict(s=(h, w)) if key != "default" else dict(s=(h, w))
            irfft[key] = dict(cols=cols, z=pair(Z), out=flat(np.fft.irfft2(Z, norm=norm, **kw)))
        # numpy's default output width 2 * (cols - 1), only where that is positive
        Zd = cnormal(h, max(2, w // 2 + 1))
        rfft2.append(dict(h=h, w=w, norm=norm, x=flat(x), X=pair(X), irfft=irfft,
                          zdefault=dict(cols=Zd.shape[1], z=pair(Zd), out=flat(np.fft.irfft2(Zd, norm=norm)))))

# rfft(x, n, norm): x of length `length` is cropped or zero padded to n. n is a power of two (the plan lengths).
rfft1 = []
for n in (1, 2, 4, 8, 16, 64, 256, 1024):
    for length in sorted({n, max(1, n // 2), 2 * n} if n <= 64 else {n}):
        x = f32(rng.normal(size=length))
        for norm in NORMS:
            if n > 64 and norm != "backward":
                continue
            X = np.fft.rfft(x, n=n, norm=norm)
            rfft1.append(dict(n=n, length=length, norm=norm, x=flat(x), X=pair(X)))

# irfft(X, n, norm): X of `m` bins is cropped or zero padded to n // 2 + 1; the imaginary parts of bins 0 and n/2 are dropped.
irfft1 = []
for m, n in [(2, None), (3, None), (5, None), (9, None), (33, None), (129, None), (513, None), (1, 1), (3, 8), (17, 8), (5, 4),
             (5, 16), (2, 4), (9, 2), (600, 1024)]:
    if m > 130 and False:
        continue
    Z = cnormal(m)
    for norm in NORMS:
        if m > 130 and norm != "backward":
            continue
        out = np.fft.irfft(Z, n=n, norm=norm)
        irfft1.append(dict(m=m, n=n if n is not None else 2 * (m - 1), explicit=n is not None, norm=norm, z=pair(Z), out=flat(out)))

# Complex fft / ifft (norm="backward") of one axis at lengths whose prime factors are 2, 3, 5, 7: 1 x N images, axis 1.
LENGTHS = [3, 5, 6, 7, 9, 10, 12, 14, 15, 18, 20, 21, 24, 25, 27, 28, 30, 35, 36, 42, 45, 49, 60, 63, 70, 75, 84, 90, 100, 105,
           120, 125, 135, 150, 180, 210, 243, 343, 360, 480, 500, 504, 625]
lines = []
for n in LENGTHS:
    z = cnormal(n)
    lines.append(dict(n=n, z=pair(z), fft=pair(np.fft.fft(z)), ifft=pair(np.fft.ifft(z))))

# 2D complex fft2 / ifft2 at mixed lengths on both axes.
planes = []
for h, w in [(30, 28), (45, 42), (21, 50), (25, 18), (49, 9)]:
    z = cnormal(h, w)
    planes.append(dict(h=h, w=w, z=pair(z), fft2=pair(np.fft.fft2(z)), ifft2=pair(np.fft.ifft2(z))))

out = Path(__file__).resolve().parents[2] / "Tests/ImagoFFTTests/Fixtures/fft-real-pins.json"
out.write_text(json.dumps(dict(numpy=np.__version__, seed=SEED, rfft2=rfft2, rfft1=rfft1, irfft1=irfft1, lines=lines,
                               planes=planes), separators=(",", ":")) + "\n")
print(f"Wrote {out.name}: {len(rfft2)} rfft2, {len(rfft1)} rfft, {len(irfft1)} irfft, {len(lines)} lines, {len(planes)} planes, "
      f"{out.stat().st_size} bytes")
