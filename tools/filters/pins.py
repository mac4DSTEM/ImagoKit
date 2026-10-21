#!/usr/bin/env python3
"""Filter pins: scipy.ndimage.gaussian_filter / uniform_filter, skimage.filters.difference_of_gaussians / butterworth /
window. Run from the package root:  ../refs/skimage/bin/python -I -B tools/filters/pins.py
Writes Tests/ImagoFiltersTests/Fixtures/filter-pins.json. Images are float32 normal draws; the float32 case is the
library's own float32 output, the float64 case the same pixels widened.
"""
import json
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage import filters

assert np.__version__ == "2.5.4" and scipy.__version__ == "1.18.1" and skimage.__version__ == "0.26.0"
rng = np.random.Generator(np.random.PCG64(20261021))
SHAPES = [(1, 9), (9, 1), (3, 5), (6, 8), (17, 23), (30, 20)]
MODES = ["reflect", "constant", "nearest", "mirror", "wrap"]


def img(h, w):
    return rng.normal(50, 20, (h, w)).astype(np.float32)


def out(a):
    a = np.asarray(a).ravel()
    if a.dtype == np.float32:  # shortest float32 repr: parses back to the same float32
        return [float(f"{float(v):.17g}") for v in a]
    return [float(f"{v:.17g}") for v in a]


gauss, uniform = [], []
for h, w in SHAPES:
    x = img(h, w)
    for sig in [(1.0, 1.0), (2.5, 0.7), (0.0, 1.5), (1.3, 4.0)]:
        for mode in MODES:
            for trunc in ((4.0, 2.0) if (h, w) == (17, 23) else (4.0,)):
                cval = 7.5 if mode == "constant" else 0.0
                gauss.append(dict(h=h, w=w, sigma=sig, mode=mode, truncate=trunc, cval=cval, x=out(x),
                                  f32=out(ndi.gaussian_filter(x, sig, mode=mode, cval=cval, truncate=trunc)),
                                  f64=out(ndi.gaussian_filter(x.astype(np.float64), sig, mode=mode, cval=cval, truncate=trunc))))
    for size in [(3, 3), (2, 5), (4, 1), (7, 6)]:
        for mode in MODES:
            cval = -3.0 if mode == "constant" else 0.0
            uniform.append(dict(h=h, w=w, size=size, mode=mode, cval=cval, x=out(x),
                                f32=out(ndi.uniform_filter(x, size, mode=mode, cval=cval)),
                                f64=out(ndi.uniform_filter(x.astype(np.float64), size, mode=mode, cval=cval))))
dog = []
for h, w in [(17, 23), (40, 31), (8, 8)]:
    x = img(h, w)
    for low, high in [((1.0, 1.0), None), ((2.0, 3.0), (4.0, 5.0)), ((0.5, 1.5), (0.5, 1.5)), ((1.2, 1.2), (3.3, 2.2))]:
        dog.append(dict(h=h, w=w, low=low, high=high, x=out(x),
                        f32=out(filters.difference_of_gaussians(x, low, high)),
                        f64=out(filters.difference_of_gaussians(x.astype(np.float64), low, high))))
butter = []
for h, w in [(16, 16), (17, 23), (30, 20), (7, 12)]:
    x = img(h, w)
    for cutoff in (0.5, 0.07, 0.01):
        for high_pass in (True, False):
            for order in (2.0, 3.5):
                for sq in (True, False):
                    for npad in (0, 5):
                        if npad and (h, w) != (17, 23):
                            continue
                        butter.append(dict(h=h, w=w, cutoff=cutoff, high=high_pass, order=order, squared=sq, npad=npad, x=out(x),
                                           f32=out(filters.butterworth(x, cutoff, high_pass, order, squared_butterworth=sq, npad=npad)),
                                           f64=out(filters.butterworth(x.astype(np.float64), cutoff, high_pass, order, squared_butterworth=sq, npad=npad))))
wins = []
types = [("boxcar", "boxcar"), ("hann", "hann"), ("hamming", "hamming"), ("blackman", "blackman"), ("blackmanharris", "blackmanharris"),
         ("nuttall", "nuttall"), ("flattop", "flattop"), ("bartlett", "bartlett"), ("tukey", ("tukey", 0.3)), ("tukey", ("tukey", 0.5)),
         ("gaussian", ("gaussian", 12.0)), ("kaiser", ("kaiser", 8.6))]
for name, spec in types:
    for shape in [(8, 8), (16, 9), (31, 31), (20, 33), (5, 64), (1, 1), (2, 3)]:
        wins.append(dict(type=name, param=(spec[1] if isinstance(spec, tuple) else None), h=shape[0], w=shape[1],
                         window1d=out(scipy.signal.get_window(spec, max(shape), fftbins=False)),
                         window2d=out(filters.window(spec, shape))))
Path("Tests/ImagoFiltersTests/Fixtures/filter-pins.json").write_text(json.dumps(dict(
    numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__, seed=20261021,
    gauss=gauss, uniform=uniform, dog=dog, butter=butter, windows=wins)))
print(len(gauss), len(uniform), len(dog), len(butter), len(wins))
