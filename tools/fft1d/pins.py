#!/usr/bin/env python3
"""Generate FFT1D pins; run from the package root:
../References/python/skimage/bin/python -I -B tools/fft1d/pins.py

Inputs are exact binary fractions shared by Float32/Float64; expected outputs
are NumPy complex128, so the oracle does not hide Float32 rounding error.
"""
import json
import platform
from pathlib import Path
import numpy as np

assert np.__version__ == "2.5.4", np.__version__
assert platform.python_version() == "3.14.7", platform.python_version()
rng = np.random.Generator(np.random.PCG64(20261016))
cases = []
for n in (1, 2, 4, 8, 64, 256, 1024):
    re = rng.integers(-4096, 4097, n).astype(np.float64) / 1024
    im = rng.integers(-4096, 4097, n).astype(np.float64) / 1024
    x = re + 1j * im
    forward = np.fft.fft(x)
    inverse = np.fft.ifft(x, norm="forward")  # unnormalised, as vDSP
    cases.append(dict(n=n, re=re.tolist(), im=im.tolist(),
                      forwardRe=forward.real.tolist(), forwardIm=forward.imag.tolist(),
                      inverseRe=inverse.real.tolist(), inverseIm=inverse.imag.tolist()))
output = Path(__file__).resolve().parents[2] / "Tests/ImagoFFTTests/Fixtures/fft1d.json"
output.write_text(json.dumps(dict(numpy=np.__version__, python=platform.python_version(),
                                  seed=20261016, cases=cases), separators=(",", ":")) + "\n")
print(f"Wrote {output.name}: {len(cases)} cases, {output.stat().st_size} bytes")
