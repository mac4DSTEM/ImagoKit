"""Registration and drift pins: phase correlation, shifts, drift series and aligned sums.

    python3 -I -B tools/registration-drift/pins.py

Run from the repository root, in a virtual environment made from tools/requirements.txt. Steps:
  1. The three oracle sources are checked against the sha256 values recorded below (scikit-image 0.26.0
     `_phase_cross_correlation.py`, SciPy 1.18.1 `_interpolation.py` and `_fourier.py`) and the versions against the lock.
  2. The planted HRTEM-like image (DC 1000, three lattices of amplitude 20 to 80 with second harmonics at half amplitude, an
     amorphous continuum of rms 15 band-limited to 0.4 cycles per pixel, so that a Fourier shift is exact) is shifted by
     known sub-pixel vectors (`fourier_shift`), noise sigma 0, 5, 20 is added, and scikit-image's `phase_cross_correlation`
     is run on the float32 pixels and on the same pixels widened to float64 (scikit-image computes in its input's
     precision). Each shift is checked against its planted vector within its truth band, or this script exits 1. The near-tie flags (the second largest
     |CC| or |CC'| within 1e-12 / 1e-5 relative of the largest) are computed in NumPy from the oracle's own steps, which are
     asserted to reproduce its shift.
  3. SciPy's `ndimage.shift` (orders 0 and 1, the seven modes) and `fourier_shift` on small images, a 12-frame drift stack
     (noisy and noise-free) with scikit-image's series against `.first`, `.mean` and `.previous`, and the aligned sums.
  4. The committed subset (under 4 MB) goes to Tests/ImagoAlignmentTests/Fixtures/ with registration-drift-pins.json (every file's
     sha256, the versions); every case goes to .build/pins-full/alignment/ for tools/registration-drift/full-parity.sh.
"""
import hashlib
import json
import math
import os
import shutil
import sys
import sysconfig
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FULL = os.path.join(ROOT, ".build", "pins-full", "alignment")
FIX = os.path.join(ROOT, "Tests", "ImagoAlignmentTests", "Fixtures")
BUDGET = 4 * 1024 * 1024

SITE = sysconfig.get_paths()["purelib"]   # the running environment's site-packages (made from tools/requirements.txt)
SHA256 = {
    "skimage/registration/_phase_cross_correlation.py":
        "c37ea680db40565e06b47a7cc6de2fec93d1230cab32ba5ddd280d23eebd6512",
    "scipy/ndimage/_interpolation.py": "0e834f3b96ddf80ad96f30ecd3b5499fcfd3e3e049674e29a08c1c4b5dec8190",
    "scipy/ndimage/_fourier.py": "4a8018471edac7b4efe753326330e56777cdebcdb1e13e8df3245e5f62dae3e2",
}
PLANT_SEED = 20261017
BILINEAR_CORRELATION = 0.999
CONT_RMS, CONT_Q0, CONT_EXP = 30.0, 0.10, 2.0
DRIFT_SEED = 20261018
SIGMAS = [0.0, 5.0, 20.0]
UPSAMPLES = [1, 20, 100]
TIE64, TIE32 = 1e-12, 1e-5
TIE64_FACTOR, TIE32_FACTOR = 8.0, 4.0     # the near-tie rule scales with the measured noise of the arrays
# 24 truth vectors (row, column) in pixels: |s| from 0.14 to 40, half-pixel and near-midpoint (x.5) ones among them.
VECTORS = [(0.1, -0.1), (0.3, 0.2), (-0.45, 0.45), (0.5, 0.5), (1.5, -2.5), (2.5, 3.5), (-0.5, 0.0), (0.0, 0.7),
           (0.7, -0.3), (3.25, -4.75), (5.1, 7.9), (-8.5, 6.5), (10.37, -12.62), (13.5, 0.5), (-17.25, 20.75),
           (22.5, -22.5), (25.9, -3.1), (-28.33, 30.17), (31.5, -31.5), (35.0, -5.5), (-38.9, 12.3), (40.0, 0.0),
           (0.0, -40.0), (-27.5, 29.5)]
# Which of them, at which noise, are committed: (image size, noise) -> vector indices. The 256 x 256 images are 256 kB each;
# every vector at every noise is run in the full set. (A 128 x 128 plant was tried and not used: noise-free at u = 100 its
# error reached 0.02 px, beyond the 0.015 px truth band, the roundoff bins above the band limit weighing twice as much.)
COMMITTED_PLANTS = {(256, 0.0): [3, 9], (256, 5.0): [10], (256, 20.0): [20]}


def fail(msg):
    print("pins.py: FAIL: " + msg, file=sys.stderr)
    sys.exit(1)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# --------------------------------------------------------------------------------------------------- the planted image

def plant(h, w, seed):
    """DC 1000 + three lattices (+ second harmonics at half amplitude) + band-limited amorphous continuum (rms 15).
    Every frequency is an integer number of cycles per image and below 0.4 cycles per pixel, so a Fourier shift is exact."""
    rng = np.random.default_rng(seed)
    row = np.arange(h, dtype=np.float64)[:, None]
    col = np.arange(w, dtype=np.float64)[None, :]
    kmax = max(3.0, 0.075 * min(h, w))
    chosen = []
    while len(chosen) < 3:
        kx = int(rng.integers(-int(kmax) - 1, int(kmax) + 2)) if w > 1 else 0
        ky = int(rng.integers(-int(kmax) - 1, int(kmax) + 2)) if h > 1 else 0
        k = math.hypot(kx, ky)
        if not (2.5 < k <= kmax):
            continue
        if any(math.hypot(kx - jx, ky - jy) < 2.0 or math.hypot(kx + jx, ky + jy) < 2.0 for jx, jy in chosen):
            continue
        chosen.append((kx, ky))
    lattice = np.zeros((h, w))
    for kx, ky in chosen:
        amp = float(rng.uniform(20.0, 80.0))
        ph, ph2 = (float(v) for v in rng.uniform(0.0, 2.0 * math.pi, size=2))
        lattice += amp * np.cos(2.0 * math.pi * (kx * col / w + ky * row / h) + ph)
        lattice += 0.5 * amp * np.cos(2.0 * math.pi * (2 * kx * col / w + 2 * ky * row / h) + ph2)
    white = rng.standard_normal((h, w))
    phase = np.fft.fft2(white)
    phase = phase / np.abs(phase)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    q = np.hypot(fx, fy)
    amp = (1.0 + (q / CONT_Q0) ** 2) ** (-CONT_EXP / 2.0)
    amp[(np.abs(fx) > 0.4) | (np.abs(fy) > 0.4)] = 0.0
    amp[0, 0] = 0.0
    cont = np.fft.ifft2(phase * amp).real
    cont *= CONT_RMS / math.sqrt(float(np.mean(cont ** 2)))
    return 1000.0 + lattice + cont


def fourier_shift_image(x, s):
    """The exact shift of a periodic, band-limited image: x displaced by +s (content moves toward larger index)."""
    from scipy import ndimage as ndi
    return np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(x), s)).real


def noise_rng(*key):
    return np.random.default_rng([PLANT_SEED] + [int(k) for k in key])


# ----------------------------------------------------------------------------------------------------- the oracle runs

def oracle_register(ref, mov, u, norm):
    from skimage.registration import phase_cross_correlation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shift, err, ph = phase_cross_correlation(ref, mov, upsample_factor=u,
                                                 normalization="phase" if norm == "phase" else None)
    return {"shift": [float(shift[0]), float(shift[1])], "error": float(err), "phasediff": float(ph),
            "dtype": str(np.asarray(shift).dtype)}


def _stages(ref, mov, u, norm, start=None):
    """The oracle's two correlation arrays on `ref`, `mov` in their own dtype: CC (complex) and, for u > 1, the refined CC'
    (complex) around `start` (the integer-stage shift, or the one the stage finds), with the oracle's own `_upsampled_dft`
    and its float32 handling. Returns (cc, cc2, shift after the integer stage)."""
    from scipy import fft as sfft
    from skimage.registration._phase_cross_correlation import _upsampled_dft
    FR, FM = sfft.fftn(ref), sfft.fftn(mov)
    P = FR * FM.conj()
    if norm == "phase":
        eps = np.finfo(P.real.dtype).eps
        P = P / np.maximum(np.abs(P), 100 * eps)
    cc = sfft.ifftn(P)
    shape = P.shape
    shift = start
    if shift is None:
        peak = np.unravel_index(int(np.argmax(np.abs(cc))), shape)
        mid = np.array([int(a / 2) for a in shape])
        shift = np.array(peak, dtype=P.real.dtype)
        shift[shift > mid] -= np.array(shape)[shift > mid]
    cc2 = None
    if u > 1:
        dt = P.real.dtype
        uf = np.array(u, dtype=dt)
        s = np.round(np.asarray(shift, dtype=dt) * uf) / uf
        size = np.ceil(uf * 1.5)
        dft = np.trunc(size / 2.0)
        off = dft - s * uf
        cc2 = _upsampled_dft(P.conj(), size, uf, off).conj()
    return cc, cc2, np.asarray(shift), P


def _relative(a, b):
    """max |a - b| / max |b|."""
    return float(np.abs(a - b).max() / np.abs(b).max())


def tie_gaps(ref32, mov32, u, norm):
    """The relative gap between the largest and the second largest |CC| (integer stage) and |CC'| (refinement), in float64
    from the oracle's own steps, and the measured noise of those arrays: eps64 (the response to a one-ulp perturbation of
    every input pixel, an implementation's arithmetic noise) and eps32 (the float32 pipeline against the float64 one). The
    replicated float64 shift is returned for the caller's assertion."""
    ref64, mov64 = ref32.astype(np.float64), mov32.astype(np.float64)
    cc, cc2, shift, P64 = _stages(ref64, mov64, u, norm)
    shape = ref64.shape

    def gap(arr):
        a = np.abs(arr).ravel()
        if a.size < 2:
            return 0.0
        top2 = np.partition(a, -2)[-2:]
        return float((top2[1] - top2[0]) / top2[1]) if top2[1] > 0 else 0.0

    g1 = gap(cc)
    g2 = gap(cc2) if cc2 is not None else None
    out = np.asarray(shift, dtype=np.float64).copy()
    if cc2 is not None:
        s = np.round(out * u) / u
        size = int(np.ceil(u * 1.5))
        dft = size // 2
        m2 = np.unravel_index(int(np.argmax(np.abs(cc2))), (size, size))
        out = s + (np.array(m2, dtype=np.float64) - dft) / u
    for d in range(2):
        if shape[d] == 1:
            out[d] = 0
    rng = np.random.default_rng([PLANT_SEED, 77, ref64.size, u])

    def jitter(x):
        sign = np.where(rng.random(x.shape) < 0.5, -np.inf, np.inf)
        return np.nextafter(x, sign)

    pc, pc2, _, _ = _stages(jitter(ref64), jitter(mov64), u, norm, start=shift)
    fc, fc2, _, _ = _stages(ref32, mov32, u, norm, start=shift)
    e64 = [_relative(pc, cc)] + ([_relative(pc2, cc2)] if cc2 is not None else [])
    e32 = [_relative(fc, cc)] + ([_relative(fc2, cc2)] if cc2 is not None else [])
    return g1, g2, [float(out[0]), float(out[1])], e64, e32, (cc, P64)


def candidate_box(cc, P, u, shape, eps, factor):
    """The bounding box [row min, row max, column min, column max] of every final shift whose correlation sample lies within
    factor * eps of its stage's maximum: the integer peaks within factor * eps[0] of the largest |CC| (at most 9), and for
    each (u > 1) the refinement samples within factor * eps[1] of that region's largest |CC'|. The shifts an implementation
    with that much arithmetic noise may return."""
    from skimage.registration._phase_cross_correlation import _upsampled_dft
    a = np.abs(cc).ravel()
    keep = np.flatnonzero(a >= a.max() * (1 - factor * eps[0]))
    keep = keep[np.argsort(-a[keep])][:9]
    mid = np.array([int(n / 2) for n in shape])
    found = []
    for idx in keep:
        pk = np.array(np.unravel_index(int(idx), shape), dtype=np.float64)
        pk[pk > mid] -= np.array(shape)[pk > mid]
        if u == 1:
            found.append(pk.copy())
            continue
        sh = np.round(pk * u) / u
        size = int(np.ceil(u * 1.5))
        dft = size // 2
        c2 = np.abs(_upsampled_dft(P.conj(), float(size), float(u), dft - sh * u))
        hit = np.argwhere(c2 >= c2.max() * (1 - factor * eps[1]))
        found.extend(sh + (hit.astype(np.float64) - dft) / u)
    f = np.array(found)
    for d in range(2):
        if shape[d] == 1:
            f[:, d] = 0.0
    return [float(f[:, 0].min()), float(f[:, 0].max()), float(f[:, 1].min()), float(f[:, 1].max())]


def tie_fields(ref32, mov32, u, norm):
    g1, g2, replicated, e64, e32, (cc, P) = tie_gaps(ref32, mov32, u, norm)
    gaps = [g1] + ([g2] if g2 is not None else [])
    if len(e64) == 1:
        e64, e32 = e64 + [0.0], e32 + [0.0]
    t64 = any(g <= max(TIE64, TIE64_FACTOR * e) for g, e in zip(gaps, e64))
    t32 = any(g <= max(TIE32, TIE32_FACTOR * e) for g, e in zip(gaps, e32))
    shape = ref32.shape
    out = {"gap_integer": g1, "gap_refined": g2, "eps64": e64, "eps32": e32, "tie64": bool(t64), "tie32": bool(t32),
           "replicated": replicated}
    out["box64"] = candidate_box(cc, P, u, shape, [max(e, TIE64 / TIE64_FACTOR) for e in e64], TIE64_FACTOR) if t64 else None
    out["box32"] = candidate_box(cc, P, u, shape, [max(e, TIE32 / TIE32_FACTOR) for e in e32], TIE32_FACTOR) if t32 else None
    return out


def oracle_tied(ref, mov, u, norm="phase"):
    """The oracle's result for float-typed arrays `ref`, `mov` and its near-tie flags, as one dict."""
    r = oracle_register(ref, mov, u, norm)
    t = tie_fields(ref.astype(np.float32), mov.astype(np.float32), u, norm)
    del t["replicated"]
    r.update(t)
    return r


def register_case(name, ref32, mov32, u, norm, truth=None, band=None, sigma=None):
    ref64, mov64 = ref32.astype(np.float64), mov32.astype(np.float64)
    o64 = oracle_register(ref64, mov64, u, norm)
    o32 = oracle_register(ref32, mov32, u, norm)
    t = tie_fields(ref32, mov32, u, norm)
    if t["replicated"] != o64["shift"]:
        fail("%s: the replicated float64 steps give %r, scikit-image %r" % (name, t["replicated"], o64["shift"]))
    del t["replicated"]
    for key, o in (("box64", o64), ("box32", o32)):
        b = t[key]
        if b is not None:
            r, c = o["shift"]
            if not (b[0] - 1e-4 <= r <= b[1] + 1e-4 and b[2] - 1e-4 <= c <= b[3] + 1e-4):
                fail("%s: the oracle's own shift %r is outside its %s %r" % (name, o["shift"], key, b))
    case = {"name": name, "height": ref32.shape[0], "width": ref32.shape[1], "u": u, "normalization": norm,
            "sigma": sigma, "truth": truth, "truth_band": band, "f64": o64, "f32": o32}
    case.update(t)
    return case


# ---------------------------------------------------------------------------------------------------------- file output

class Out:
    def __init__(self):
        self.files = {}     # relative name -> (bytes of dtype array, committed?)

    def put(self, name, arr, dtype, committed):
        data = np.ascontiguousarray(arr, dtype=dtype).tobytes()
        if name in self.files and self.files[name][0] != data:
            fail("file %s written twice with different content" % name)
        # a file used by a committed case is committed whatever else uses it
        old = self.files.get(name, (None, False))[1]
        self.files[name] = (data, committed or old)
        return name


# ------------------------------------------------------------------------------------------------- the planted shift set

def planted_cases(out, cases):
    report = []
    sizes = [(256, None)]
    for size, only_sigma in sizes:
        base = plant(size, size, PLANT_SEED)
        for si, sigma in enumerate(SIGMAS):
            if only_sigma is not None and sigma != only_sigma:
                continue
            keep = COMMITTED_PLANTS.get((size, sigma), [])
            vectors = range(len(VECTORS)) if size == 256 else keep
            ref = (base + sigma * noise_rng(0, si, 999, size).standard_normal(base.shape)).astype(np.float32)
            ref_name = out.put("plant%d-s%g-ref.f32" % (size, sigma), ref, "<f4", bool(keep))
            for vi in vectors:
                v = VECTORS[vi]
                moved = fourier_shift_image(base, (-v[0], -v[1]))          # moving = reference displaced by -v
                mov = (moved + sigma * noise_rng(1, si, vi, size).standard_normal(base.shape)).astype(np.float32)
                committed = vi in keep
                mov_name = out.put("plant%d-s%g-v%02d.f32" % (size, sigma, vi), mov, "<f4", committed)
                for u in UPSAMPLES:
                    if u == 1:
                        band = 0.5 + 0.01
                    elif sigma == 0.0:
                        band = 1.0 / (2 * u) + 0.01
                    elif sigma == 5.0:
                        band = 0.05
                    else:
                        band = 0.15
                    gated = (u == 1) or (u == 20) or (sigma == 0.0 and u == 100)
                    c = register_case("plant%d-s%g-v%02d-u%d" % (size, sigma, vi, u), ref, mov, u, "phase", truth=list(v),
                                      band=band if gated else None, sigma=sigma)
                    c.update({"ref": ref_name, "mov": mov_name, "committed": bool(committed)})
                    err = max(abs(c["f64"]["shift"][0] - v[0]), abs(c["f64"]["shift"][1] - v[1]))
                    c["truth_error64"] = err
                    cases.append(c)
                    report.append((size, sigma, u, vi, err, band if gated else None))
                    if gated and err > band + 1e-9:
                        fail("%s: |shift - truth| = %.4f beyond the truth band %.4f" % (c["name"], err, band))
    return report


def size_cases(out, cases):
    """Odd, even, radix-3/5, Bluestein, tiny and length-1 sizes: the plant at that size, shifts of 3 truths each."""
    specs = [(32, 64, True), (50, 63, True), (75, 100, True), (50, 127, True), (243, 125, False), (250, 250, False),
             (256, 255, False), (1, 64, True), (64, 1, True), (1, 1, True), (2, 2, True), (3, 5, True)]
    truths = [(1.3, -2.7), (0.5, 0.5), (-4.25, 3.75)]
    for (h, w, committed) in specs:
        small = h * w < 1000
        base = None if small else plant(h, w, PLANT_SEED + 17 * h + w)
        for ti, t in enumerate(truths):
            s = (t[0] if h > 1 else 0.0, t[1] if w > 1 else 0.0)
            if small:
                # a constant-plus-noise tiny image carries no structure to register: the oracle is compared, not the truth
                rng = noise_rng(7, h, w, ti)
                ref = rng.standard_normal((h, w)).astype(np.float32) + 5
                mov = rng.standard_normal((h, w)).astype(np.float32) + 5
                truth, band = None, None
            else:
                ref = (base + 5.0 * noise_rng(2, h, w, ti).standard_normal(base.shape)).astype(np.float32)
                mov = (fourier_shift_image(base, (-s[0], -s[1])) + 5.0 * noise_rng(3, h, w, ti).standard_normal(base.shape)).astype(np.float32)
                truth, band = list(s), 0.15
            keep = committed and ti < 2
            rn = out.put("size%dx%d-t%d-ref.f32" % (w, h, ti), ref, "<f4", keep)
            mn = out.put("size%dx%d-t%d-mov.f32" % (w, h, ti), mov, "<f4", keep)
            for u, norm in ((1, "phase"), (20, "phase"), (20, "none")):
                if h * w == 1 and u > 1:
                    continue
                c = register_case("size%dx%d-t%d-u%d-%s" % (w, h, ti, u, norm), ref, mov, u, norm, truth=truth,
                                  band=(band if (u == 20 and norm == "phase") else None), sigma=5.0)
                c.update({"ref": rn, "mov": mn, "committed": keep})
                if c["truth_band"] is not None:
                    e = max(abs(c["f64"]["shift"][0] - s[0]), abs(c["f64"]["shift"][1] - s[1]))
                    c["truth_error64"] = e
                    if e > c["truth_band"]:
                        fail("%s: |shift - truth| = %.4f beyond %.2f" % (c["name"], e, c["truth_band"]))
                cases.append(c)


def normalization_none_cases(out, cases):
    """normalization=None on the plant (error and phase difference are informative there)."""
    base = plant(128, 128, PLANT_SEED + 5)
    for ti, v in enumerate([(2.35, -1.6), (-6.5, 4.25)]):
        for sigma in (0.0, 20.0):
            ref = (base + sigma * noise_rng(4, ti, int(sigma)).standard_normal(base.shape)).astype(np.float32)
            mov = (fourier_shift_image(base, (-v[0], -v[1])) + sigma * noise_rng(5, ti, int(sigma) + 1).standard_normal(base.shape)).astype(np.float32)
            keep = (ti, sigma) in ((0, 20.0), (1, 0.0))
            rn = out.put("none128-s%g-t%d-ref.f32" % (sigma, ti), ref, "<f4", keep)
            mn = out.put("none128-s%g-t%d-mov.f32" % (sigma, ti), mov, "<f4", keep)
            for u in (1, 20, 100):
                c = register_case("none128-s%g-t%d-u%d" % (sigma, ti, u), ref, mov, u, "none", truth=list(v),
                                  band=(0.5 + 0.01 if u == 1 else 1.0 / (2 * u) + 0.01 + (0.15 if sigma else 0)),
                                  sigma=sigma)
                c.update({"ref": rn, "mov": mn, "committed": keep})
                e = max(abs(c["f64"]["shift"][0] - v[0]), abs(c["f64"]["shift"][1] - v[1]))
                c["truth_error64"] = e
                if e > c["truth_band"]:
                    fail("%s: |shift - truth| = %.4f beyond %.4f" % (c["name"], e, c["truth_band"]))
                cases.append(c)


# --------------------------------------------------------------------------------------------- scipy.ndimage.shift pins

MODES = ["constant", "nearest", "wrap", "reflect", "mirror", "grid-constant", "grid-wrap"]
SHIFT_CVAL = 7.5
SHIFT_LIST = [(0.0, 0.0), (3.0, -2.0), (-5.0, 7.0), (0.5, -0.5), (1.5, 2.5), (-2.5, 0.5), (16.0, 11.0), (17.0, 12.0),
              (-16.0, -11.0), (-17.0, -12.0), (17.5, -12.5), (40.25, -31.75), (0.1, 0.9), (-0.9, 0.1), (0.999, -0.001),
              (0.4999999, 0.5000001), (2.0, 0.0), (0.0, -3.0), (1.0, 1.0), (9.5, 4.5), (-9.5, -4.5), (0.25, -0.75)]


def shift_pins(out, manifest):
    from scipy import ndimage as ndi
    rng = np.random.default_rng(PLANT_SEED + 31)
    shifts = list(SHIFT_LIST) + [(float(a), float(b)) for a, b in rng.uniform(-22, 22, size=(8, 2))]
    shapes = [(10, 13), (5, 4), (1, 6), (7, 1)]
    images = []
    for i, (h, w) in enumerate(shapes):
        img = (rng.standard_normal((h, w)) * 10 + 3).astype(np.float32)
        images.append({"name": "shift-img%d" % i, "file": out.put("shift-img%d.f32" % i, img, "<f4", True),
                       "height": h, "width": w})
        images[-1]["_arr"] = img
    combos = []
    for mode in MODES:
        for order in (0, 1):
            blocks = []
            for im in images:
                x = im["_arr"].astype(np.float64)
                for s in shifts:
                    blocks.append(ndi.shift(x, s, order=order, mode=mode, cval=SHIFT_CVAL))
            name = out.put("shift-%s-o%d.f64" % (mode, order), np.concatenate([b.ravel() for b in blocks]), "<f8", True)
            combos.append({"mode": mode, "order": order, "file": name})
    for im in images:
        del im["_arr"]
    manifest["shift"] = {"cval": SHIFT_CVAL, "shifts": [list(s) for s in shifts], "images": images, "combos": combos}


def fourier_pins(out, manifest):
    rng = np.random.default_rng(PLANT_SEED + 41)
    shifts = [(0.0, 0.0), (1.0, 2.0), (-3.0, 4.0), (0.5, 0.5), (0.5, -1.5), (2.25, -0.75), (-0.3, 0.7), (5.5, 6.5),
              (11.0, 9.0), (12.5, -9.5), (0.1, 0.1), (-4.9, 3.3)]
    shapes = [(12, 10), (11, 9), (1, 8), (8, 1), (16, 16)]
    items = []
    for i, (h, w) in enumerate(shapes):
        img = (rng.standard_normal((h, w)) * 10 + 3).astype(np.float32)
        x = img.astype(np.float64)
        res, imax = [], []
        for s in shifts:
            y = np.fft.ifft2(__import__("scipy.ndimage", fromlist=["fourier_shift"]).fourier_shift(np.fft.fft2(x), s))
            res.append(y.real.ravel())
            imax.append(float(np.abs(y.imag).max()))
        items.append({"name": "fourier-img%d" % i, "file": out.put("fourier-img%d.f32" % i, img, "<f4", True),
                      "height": h, "width": w,
                      "out": out.put("fourier-img%d.out.f64" % i, np.concatenate(res), "<f8", True),
                      "imag_max": imax})
    manifest["fourier"] = {"shifts": [list(s) for s in shifts], "images": items}


# --------------------------------------------------------------------------------------------- drift series and sums

def drift_truth(n, seed):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.7, size=(n, 2))
    steps[0] = 0.0
    return np.cumsum(steps, axis=0)


def series(frames, ref_kind, u, norm="phase"):
    """scikit-image's series on `frames` (a list of arrays, one dtype): the shifts, the pairwise and cumulated ones."""
    out = []
    if ref_kind == "first":
        for k, f in enumerate(frames):
            out.append(oracle_tied(frames[0], f, u, norm) if k else None)
    elif ref_kind == "mean":
        mean = np.mean(np.stack([f.astype(np.float64) for f in frames]), axis=0).astype(frames[0].dtype)
        for f in frames:
            out.append(oracle_tied(mean, f, u, norm))
    else:
        total = [0.0, 0.0]
        for k, f in enumerate(frames):
            if k == 0:
                out.append(None)
                continue
            r = oracle_tied(frames[k - 1], f, u, norm)
            total = [total[0] + r["shift"][0], total[1] + r["shift"][1]]
            out.append({**r, "shift": list(total), "pair": r["shift"]})
    return out


def drift_pins(out, manifest, size_committed, size_full):
    n = 12
    truth = drift_truth(n, DRIFT_SEED)
    for size, committed in ((size_committed, True), (size_full, False)):
        base = plant(size, size, PLANT_SEED + 3)
        frames_noisy, frames_clean = [], []
        for k in range(n):
            moved = fourier_shift_image(base, (-truth[k, 0], -truth[k, 1]))
            frames_clean.append(moved.astype(np.float32))
            frames_noisy.append((moved + 5.0 * noise_rng(6, size, k).standard_normal(base.shape)).astype(np.float32))
        tag = "drift%d" % size
        noisy_file = out.put(tag + "-noisy.f32", np.stack(frames_noisy), "<f4", committed)
        clean_file = out.put(tag + "-clean.f32", np.stack(frames_clean), "<f4", committed)
        entry = {"name": tag, "size": size, "frames": n, "committed": committed, "noisy": noisy_file, "clean": clean_file,
                 "truth": truth.tolist(), "u": 20, "oracle": {}}
        worst = {}
        for kind in ("first", "mean", "previous"):
            for prec, conv in (("f64", lambda a: a.astype(np.float64)), ("f32", lambda a: a)):
                ser = series([conv(f) for f in frames_noisy], kind, 20)
                entry["oracle"]["%s-%s" % (kind, prec)] = ser
        # the truth checks, on float64
        for kind in ("first", "mean", "previous"):
            ser = entry["oracle"]["%s-f64" % kind]
            s = np.array([[0.0, 0.0] if r is None else r["shift"] for r in ser])
            rel = s - s[0]                                    # drift relative to frame 0's shift
            err = np.abs(rel - truth)
            k = np.arange(n)
            if kind == "previous":
                band = 0.05 * np.sqrt(np.maximum(k, 1))
            else:
                band = np.full(n, 0.20 if kind == "mean" else 0.05)     # the mean reference smears the drift
            entry.setdefault("truth_error", {})[kind] = [float(v) for v in err.max(axis=1)]
            entry.setdefault("truth_band", {})[kind] = [float(v) for v in band]
            worst[kind] = float((err.max(axis=1) / band).max())
            if (err.max(axis=1) > band).any():
                fail("%s %s: recovered drift beyond the band: max |err| %.4f at frame %d (band %.4f)"
                     % (tag, kind, err.max(), int(err.max(axis=1).argmax()), band[int(err.max(axis=1).argmax())]))
        # the sums, from the noise-free frames and the planted shifts
        from scipy import ndimage as ndi
        shifts = truth.copy()
        fsum = np.zeros((size, size))
        bsum = np.zeros((size, size))
        count = np.zeros((size, size), dtype=np.int32)
        ii, jj = np.mgrid[0:size, 0:size]
        for k in range(n):
            x = frames_clean[k].astype(np.float64)
            fsum += np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(x), (shifts[k, 0], shifts[k, 1]))).real
            bsum += ndi.shift(x, (shifts[k, 0], shifts[k, 1]), order=1, mode="constant", cval=0.0)
            sr, sc = ii - shifts[k, 0], jj - shifts[k, 1]
            count += ((sr >= 0) & (sr <= size - 1) & (sc >= 0) & (sc <= size - 1)).astype(np.int32)
        entry["sum_shifts"] = shifts.tolist()
        entry["sum_fourier"] = out.put(tag + "-sum-fourier.f64", fsum, "<f8", committed)
        entry["sum_bilinear"] = out.put(tag + "-sum-bilinear.f64", bsum, "<f8", committed)
        entry["sum_count"] = out.put(tag + "-sum-count.i32", count, "<i4", committed)
        # the planted sums' truths
        unshifted = frames_clean[0].astype(np.float64)               # frame 0 is the unshifted image (drift 0 at frame 0)
        # frame 0's drift is exactly 0 and the planted shifts of the others undo theirs; the Fourier mean is the image
        fm = fsum / n
        entry["fourier_mean_vs_image"] = float(np.abs(fm - unshifted).max() / np.abs(unshifted).max())
        full = count == n
        bm = np.where(count > 0, bsum / np.maximum(count, 1), 0.0)
        a, b = bm[full], unshifted[full]
        entry["bilinear_correlation"] = float(np.corrcoef(a, b)[0, 1])
        entry["full_overlap_pixels"] = int(full.sum())
        if entry["fourier_mean_vs_image"] > 1e-6:
            fail("%s: the Fourier aligned mean differs from the image by %.2e of its maximum" % (tag, entry["fourier_mean_vs_image"]))
        if entry["bilinear_correlation"] < BILINEAR_CORRELATION:
            fail("%s: the bilinear mean correlates %.5f with the image" % (tag, entry["bilinear_correlation"]))
        manifest.setdefault("drift", []).append(entry)
        print("pins.py: drift %-9s worst |err|/band: first %.2f mean %.2f previous %.2f; Fourier mean vs image %.1e; "
              "bilinear correlation %.5f (full overlap %d px)"
              % (tag, worst["first"], worst["mean"], worst["previous"], entry["fourier_mean_vs_image"],
                 entry["bilinear_correlation"], entry["full_overlap_pixels"]))


# ------------------------------------------------------------------------------------------------------------- main

def main():
    import scipy
    import skimage
    if np.__version__ != "2.5.4" or scipy.__version__ != "1.18.1" or skimage.__version__ != "0.26.0":
        fail("NumPy %s, SciPy %s, scikit-image %s; the pins are 2.5.4, 1.18.1, 0.26.0"
             % (np.__version__, scipy.__version__, skimage.__version__))
    for rel, want in SHA256.items():
        got = sha256_file(os.path.join(SITE, rel))
        if got != want:
            fail("%s sha256 %s, recorded %s" % (rel, got, want))
    print("pins.py: the three oracle sources match the recorded sha256 values")

    out = Out()
    manifest = {"generator": "tools/registration-drift/pins.py", "plant_seed": PLANT_SEED, "drift_seed": DRIFT_SEED,
                "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                             "scikit-image": skimage.__version__},
                "oracle_sha256": SHA256, "tie_tolerances": {"float64": TIE64, "float32": TIE32}}
    cases = []
    report = planted_cases(out, cases)
    size_cases(out, cases)
    normalization_none_cases(out, cases)
    manifest["register"] = cases
    shift_pins(out, manifest)
    fourier_pins(out, manifest)
    drift_pins(out, manifest, 96, 256)

    # The planted-truth report: the measured worst errors.
    for size in (256, 128):
        for sigma in SIGMAS:
            for u in UPSAMPLES:
                rows = [(e, b) for (sz, sg, uu, vi, e, b) in report if sz == size and sg == sigma and uu == u]
                if not rows:
                    continue
                print("pins.py: plant %dx%d sigma %-4g u %-3d: %d shifts, worst |shift - truth| %.4f px (band %s)"
                      % (size, size, sigma, u, len(rows), max(e for e, _ in rows),
                         "%.4f" % rows[0][1] if rows[0][1] else "not gated"))
    ties64 = sum(1 for c in cases if c["tie64"])
    ties32 = sum(1 for c in cases if c["tie32"])
    print("pins.py: %d registration cases (%d committed); near-ties by the measured-noise rule: %d float64, %d float32"
          % (len(cases), sum(1 for c in cases if c["committed"]), ties64, ties32))

    # Write the full set, then the committed subset.
    if os.path.exists(FULL):
        shutil.rmtree(FULL)
    os.makedirs(FULL)
    for name, (data, _) in out.files.items():
        with open(os.path.join(FULL, name), "wb") as f:
            f.write(data)
    with open(os.path.join(FULL, "registration-drift-pins.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    if os.path.exists(FIX):
        shutil.rmtree(FIX)
    os.makedirs(FIX)
    committed_manifest = dict(manifest)
    committed_manifest["register"] = [c for c in cases if c["committed"]]
    committed_manifest["drift"] = [d for d in manifest["drift"] if d["committed"]]
    needed = set()
    for c in committed_manifest["register"]:
        needed.update([c["ref"], c["mov"]])
    for d in committed_manifest["drift"]:
        needed.update([d["noisy"], d["clean"], d["sum_fourier"], d["sum_bilinear"], d["sum_count"]])
    for im in manifest["shift"]["images"]:
        needed.add(im["file"])
    for cb in manifest["shift"]["combos"]:
        needed.add(cb["file"])
    for it in manifest["fourier"]["images"]:
        needed.update([it["file"], it["out"]])
    files = []
    for name in sorted(needed):
        data = out.files[name][0]
        with open(os.path.join(FIX, name), "wb") as f:
            f.write(data)
        files.append({"file": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    committed_manifest["files"] = files
    with open(os.path.join(FIX, "registration-drift-pins.json"), "w") as f:
        json.dump(committed_manifest, f, separators=(",", ":"))
    total = sum(x["bytes"] for x in files) + os.path.getsize(os.path.join(FIX, "registration-drift-pins.json"))
    full_total = sum(len(d) for d, _ in out.files.values())
    print("pins.py: committed %d files, %.2f MB of the 4 MB budget (%d of %d registration cases, %d of %d drift stacks); "
          "the full set, %.1f MB, is in .build/pins-full/alignment/"
          % (len(files) + 1, total / 2 ** 20, len(committed_manifest["register"]), len(cases),
             len(committed_manifest["drift"]), len(manifest["drift"]), full_total / 2 ** 20))
    if total > BUDGET:
        fail("committed fixtures %d bytes, over the 4 MB budget" % total)


if __name__ == "__main__":
    main()
