#!/usr/bin/env python3
"""FFT-surface pins: numpy.fft.fft2 / ifft2 (axes, norm), rfft2 / irfft2, fftshift / ifftshift.
Run from the package root:
  python -I -B tools/fftsurface/pins.py
Writes Tests/ImagoCoreTests/Fixtures/fft-surface-pins.json. Inputs are float64 normal draws, rounded to float32 so the
Float32 path sees the same pixels; the truth is NumPy's float64 result on those pixels.
"""
import json
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4", np.__version__
rng = np.random.Generator(np.random.PCG64(20261021))
SHAPES = [(1, 7), (7, 1), (4, 5), (5, 6), (30, 20), (7, 9), (16, 16), (12, 15), (25, 10)]
fft_cases = []
for h, w in SHAPES:
    x = (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))).astype(np.complex64).astype(np.complex128)
    for axes_name, axes in (("both", (0, 1)), ("axis0", (0,)), ("axis1", (1,))):
        for norm in ("backward", "forward", "ortho"):
            fft_cases.append(dict(h=h, w=w, axes=axes_name, norm=norm, re=x.real.ravel().tolist(), im=x.imag.ravel().tolist(),
                                  forward=[np.fft.fft2(x, axes=axes, norm=norm).real.ravel().tolist(), np.fft.fft2(x, axes=axes, norm=norm).imag.ravel().tolist()],
                                  inverse=[np.fft.ifft2(x, axes=axes, norm=norm).real.ravel().tolist(), np.fft.ifft2(x, axes=axes, norm=norm).imag.ravel().tolist()]))
rfft_cases = []
for h, w in SHAPES:
    if w < 2:
        continue
    x = rng.normal(size=(h, w)).astype(np.float32).astype(np.float64)
    for norm in ("backward", "forward", "ortho"):
        X = np.fft.rfft2(x, norm=norm)
        # an irfft2 input that is NOT Hermitian-clean (random imaginary DC / Nyquist column): NumPy discards those imaginary parts
        Z = (rng.normal(size=X.shape) + 1j * rng.normal(size=X.shape)).astype(np.complex64).astype(np.complex128)
        ncol = [None] + ([w + 1] if True else [])
        rfft_cases.append(dict(h=h, w=w, norm=norm, x=x.ravel().tolist(), X=[X.real.ravel().tolist(), X.imag.ravel().tolist()],
                               zre=Z.real.ravel().tolist(), zim=Z.imag.ravel().tolist(), zcols=Z.shape[1],
                               z_default=np.fft.irfft2(Z, norm=norm).ravel().tolist(), z_w=np.fft.irfft2(Z, s=(h, w), norm=norm).ravel().tolist(),
                               round_trip=np.fft.irfft2(X, s=(h, w), norm=norm).ravel().tolist()))
shift_cases = []
for h, w in [(1, 1), (4, 5), (5, 4), (6, 6), (7, 9), (3, 10)]:
    a = np.arange(h * w).reshape(h, w)
    for name, axes in (("both", None), ("axis0", (0,)), ("axis1", (1,))):
        shift_cases.append(dict(h=h, w=w, axes=name, fftshift=np.fft.fftshift(a, axes=axes).ravel().tolist(),
                                ifftshift=np.fft.ifftshift(a, axes=axes).ravel().tolist()))
Path("Tests/ImagoCoreTests/Fixtures/fft-surface-pins.json").write_text(json.dumps(dict(numpy=np.__version__, seed=20261021, fft=fft_cases, rfft=rfft_cases, shift=shift_cases)))
print(len(fft_cases), len(rfft_cases), len(shift_cases))
