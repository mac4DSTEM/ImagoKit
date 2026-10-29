#!/usr/bin/env python3
"""Generate the precipitation-analysis pins; run from the package root:
python3 -I -B tools/precipitation/pins.py     (a venv with only scikit-image==0.26.0, which brings NumPy 2.5.4 and SciPy 1.18.1)

Writes Tests/ImagoCoreTests/Fixtures/threshold-morphology-pins.json and
Tests/ImagoCoreTests/Fixtures/regionprops-needle-pins.json.

Reference: scikit-image 0.26.0 `filters.threshold_*`, `morphology.binary_*` / `remove_small_*`, `segmentation.clear_border`,
`measure.label` / `regionprops`; SciPy 1.18.1 `ndimage.binary_*`, `binary_fill_holes`, `generate_binary_structure`, `ndimage.sum` etc.
The scikit-image 0.26.0 wheel ships no tests, so its numbers are held to assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32);
masks, labels, bbox, area and every integer are exact. SciPy's ndimage measurement tests use decimal 6.
Facts the pins record that a port must follow:
  * ndimage.binary_erosion/dilation/opening/closing default structure = generate_binary_structure(2, 1) (the cross) and
    border_value 0, so erosion eats the image border; skimage.morphology.binary_* (deprecated in 0.26) default footprint = the
    same cross and mode='ignore' (the border does not erode). Both are pinned, with the border-touching cases that differ.
  * remove_small_objects in 0.26 is called with max_size=N (removes objects of area <= N); the old min_size=M is max_size=M-1.
Inputs are deterministic (PCG64 seed 20261021, rasterised needles); floats are written as repr (round trip); no NaN/inf.
"""
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

assert np.__version__ == "2.5.4", np.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert skimage.__version__ == "0.26.0", skimage.__version__
ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "Tests/ImagoCoreTests/Fixtures"
rng = np.random.Generator(np.random.PCG64(20261021))
CMD = "python3 -I -B tools/precipitation/pins.py"


def meta():
    return {"skimage": skimage.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "python": platform.python_version(), "generator": CMD}


def mask_str(m):
    return "".join("1" if v else "0" for v in np.asarray(m).flat)


def ints(a):
    return [int(v) for v in np.asarray(a).flat]


def floats(a):
    v = [float(x) for x in np.asarray(a).flat]
    assert all(np.isfinite(v)), "non-finite value in pins"
    return v


# ---------- synthetic inputs ----------
def needle(h, w, r0, c0, r1, c1, half_width):
    """Rasterised needle: pixels whose centre is within half_width of the segment (r0,c0)-(r1,c1)."""
    rr, cc = np.indices((h, w), dtype=np.float64)
    d = np.array([r1 - r0, c1 - c0], float)
    L2 = d @ d
    t = np.clip(((rr - r0) * d[0] + (cc - c0) * d[1]) / L2, 0, 1)
    return np.hypot(rr - (r0 + t * d[0]), cc - (c0 + t * d[1])) <= half_width


def ellipse(h, w, r, c, a, b, theta):
    rr, cc = np.indices((h, w), dtype=np.float64)
    y, x = rr - r, cc - c
    u = x * np.cos(theta) + y * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta)
    return (u / a) ** 2 + (v / b) ** 2 <= 1


def binary_cases():
    cs = {}
    h = w = 64
    # diagonal needles at several angles, so feret_diameter_max != axis_major_length
    cs["needles-diagonal"] = (needle(h, w, 6, 6, 56, 20, 2.2) | needle(h, w, 10, 58, 50, 30, 1.6)
                              | needle(h, w, 58, 40, 58, 60, 1.5) | needle(h, w, 4, 30, 24, 28, 1.0))
    cs["needles-45deg-thin"] = needle(h, w, 5, 5, 58, 58, 1.2) | needle(h, w, 40, 8, 8, 40, 1.2)
    cs["needles-touching-border"] = needle(h, w, 0, 10, 30, 40, 2.0) | needle(h, w, 40, 63, 63, 20, 2.0)
    cs["needle-cross-3d-like"] = needle(h, w, 8, 32, 56, 32, 1.8) | needle(h, w, 32, 6, 32, 58, 1.8) | needle(h, w, 12, 12, 52, 52, 1.4)
    cs["ellipses-rotated"] = (ellipse(h, w, 18, 18, 12, 4, 0.6) | ellipse(h, w, 44, 40, 14, 5, -0.9) | ellipse(h, w, 50, 12, 6, 6, 0))
    ring = ellipse(h, w, 32, 32, 20, 20, 0) & ~ellipse(h, w, 32, 32, 11, 11, 0)
    ring[32, 8:20] = False  # a gap-free-looking ring with a cut channel
    cs["ring-with-hole-and-gap"] = ring
    holes = np.zeros((h, w), bool)
    holes[8:56, 8:56] = True
    for r, c in [(20, 20), (20, 40), (40, 30)]:
        holes[r:r + 5, c:c + 4] = False
    holes[30, 30] = False  # a 1-pixel hole
    cs["slab-with-holes"] = holes
    noise = rng.random((h, w)) < 0.35
    cs["salt-noise"] = noise
    cs["salt-on-needles"] = cs["needles-diagonal"] ^ (rng.random((h, w)) < 0.03)
    cs["checkerboard"] = (np.indices((12, 13)).sum(0) % 2) == 0
    cs["single-pixel-and-line"] = np.zeros((9, 11), bool)
    cs["single-pixel-and-line"][2, 2] = True
    cs["single-pixel-and-line"][6, 1:10] = True
    cs["empty"] = np.zeros((6, 7), bool)
    cs["full"] = np.ones((5, 9), bool)
    return cs


def grey_cases():
    cs = {}
    h = w = 64
    base = needle(h, w, 6, 6, 56, 20, 2.2) | needle(h, w, 10, 58, 50, 30, 1.6) | ellipse(h, w, 44, 44, 9, 5, 0.5)
    for name, lo, hi, sig in [("bimodal-f64", 0.2, 0.7, 0.05), ("bimodal-lownoise-f64", 0.1, 0.9, 0.01), ("bimodal-highnoise-f64", 0.3, 0.6, 0.12)]:
        cs[name] = np.where(base, hi, lo) + rng.normal(0, sig, (h, w))
    f = np.where(base, 0.7, 0.2) + rng.normal(0, 0.05, (h, w))
    cs["bimodal-f32"] = f.astype(np.float32)
    cs["bimodal-u8"] = np.clip(np.where(base, 170, 50) + rng.normal(0, 14, (h, w)), 0, 255).round().astype(np.uint8)
    cs["bimodal-u16"] = np.clip(np.where(base, 40000, 12000) + rng.normal(0, 3000, (h, w)), 0, 65535).round().astype(np.uint16)
    cs["bimodal-i16-negative"] = (np.where(base, 400, -300) + rng.normal(0, 60, (h, w))).round().astype(np.int16)
    yy, xx = np.indices((h, w))
    cs["gradient-background"] = np.where(base, 0.55, 0.15) + 0.3 * xx / w + rng.normal(0, 0.03, (h, w))
    cs["skewed-exponential"] = rng.exponential(1.0, (h, w)) + 2.0 * base
    cs["tiny-9x7-f64"] = rng.random((9, 7))
    return cs


def th(fn, img, **kw):
    try:
        return float(fn(img, **kw))
    except Exception as e:  # recorded, never silent
        return {"raises": type(e).__name__, "message": str(e)[:120]}


def task32():
    gr = []
    for name, g in grey_cases().items():
        e = {"name": name, "dtype": str(g.dtype), "width": g.shape[1], "height": g.shape[0], "values": floats(g)}
        e["otsu"] = th(filters.threshold_otsu, g)
        e["otsu_nbins64"] = th(filters.threshold_otsu, g, nbins=64)
        e["li"] = th(filters.threshold_li, g)
        e["yen"] = th(filters.threshold_yen, g)
        e["triangle"] = th(filters.threshold_triangle, g)
        e["mean"] = th(filters.threshold_mean, g)
        e["isodata"] = th(filters.threshold_isodata, g)
        e["local_b15"] = floats(filters.threshold_local(g, 15)) if name in ("bimodal-f64", "gradient-background") else None
        e["local_b15_off"] = floats(filters.threshold_local(g, 15, offset=0.05)) if name in ("bimodal-f64", "gradient-background") else None
        e["local_b15_mean"] = floats(filters.threshold_local(g, 15, method="mean")) if name in ("bimodal-f64", "gradient-background") else None
        o = e["otsu"]
        lo, hi = o - 0.1 * (np.ptp(g.astype(float))), o + 0.1 * (np.ptp(g.astype(float)))
        e["hysteresis"] = {"low": float(lo), "high": float(hi),
                           "mask": mask_str(filters.apply_hysteresis_threshold(g, lo, hi))}
        e["otsu_mask"] = mask_str(g > o)
        gr.append(e)

    se4 = ndi.generate_binary_structure(2, 1)
    se8 = ndi.generate_binary_structure(2, 2)
    mp = []
    for name, m in binary_cases().items():
        h, w = m.shape
        e = {"name": name, "width": w, "height": h, "mask": mask_str(m)}
        for tag, st in [("cross", se4), ("square", se8)]:
            nd = {}
            nd["erosion"] = mask_str(ndi.binary_erosion(m, st))
            nd["erosion_border1"] = mask_str(ndi.binary_erosion(m, st, border_value=1))
            nd["dilation"] = mask_str(ndi.binary_dilation(m, st))
            nd["opening"] = mask_str(ndi.binary_opening(m, st))
            nd["closing"] = mask_str(ndi.binary_closing(m, st))
            nd["erosion_iter2"] = mask_str(ndi.binary_erosion(m, st, iterations=2))
            nd["dilation_iter2"] = mask_str(ndi.binary_dilation(m, st, iterations=2))
            e["ndimage_" + tag] = nd
        e["fill_holes"] = mask_str(ndi.binary_fill_holes(m))
        e["fill_holes_square"] = mask_str(ndi.binary_fill_holes(m, se8))
        with warnings.catch_warnings():  # skimage 0.26 deprecates binary_*; the numbers are still the reference
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            e["skimage_default"] = {
                "erosion": mask_str(morphology.binary_erosion(m)),
                "dilation": mask_str(morphology.binary_dilation(m)),
                "opening": mask_str(morphology.binary_opening(m)),
                "closing": mask_str(morphology.binary_closing(m)),
            }
        # remove_small_objects: max_size=N removes area <= N (old min_size=M == max_size=M-1)
        e["remove_small_objects"] = [
            {"max_size": n, "connectivity": c, "mask": mask_str(morphology.remove_small_objects(m, max_size=n, connectivity=c))}
            for n in (0, 3, 9, 40) for c in (1, 2)]
        e["remove_small_holes"] = [
            {"max_size": n, "connectivity": c, "mask": mask_str(morphology.remove_small_holes(m, max_size=n, connectivity=c))}
            for n in (0, 1, 20) for c in (1, 2)]
        e["clear_border"] = {"buffer0": mask_str(segmentation.clear_border(m)),
                             "buffer2": mask_str(segmentation.clear_border(m, buffer_size=2))}
        mp.append(e)
    structures = {f"rank{r}_conn{c}": {"rank": r, "connectivity": c, "structure": ints(ndi.generate_binary_structure(r, c)),
                                        "shape": [3] * r}
                  for r in (2, 3) for c in range(1, r + 1)}
    out = {**meta(), "task": 32,
           "note": "ndimage defaults: structure rank2/conn1 cross, border_value 0 (erosion eats the border). skimage.morphology.binary_*: cross, mode='ignore'. remove_small_objects max_size=N removes area <= N.",
           "grey": gr, "binary": mp, "structures": structures}
    (FIX / "threshold-morphology-pins.json").write_text(json.dumps(out))
    return len(gr), len(mp)


# ---------- regionprops ----------
def prop_row(r):
    d = {"label": int(r.label), "area": int(r.area), "bbox": [int(v) for v in r.bbox],
         "centroid": floats(r.centroid), "orientation": float(r.orientation),
         "axis_major_length": float(r.axis_major_length), "axis_minor_length": float(r.axis_minor_length),
         "feret_diameter_max": float(r.feret_diameter_max), "perimeter": float(r.perimeter),
         "perimeter_crofton": float(r.perimeter_crofton), "eccentricity": float(r.eccentricity),
         "solidity": float(r.solidity), "equivalent_diameter_area": float(r.equivalent_diameter_area),
         "extent": float(r.extent), "euler_number": int(r.euler_number), "area_filled": int(r.area_filled),
         "area_convex": int(r.area_convex)}
    for k in ("eccentricity", "solidity"):
        assert np.isfinite(d[k])
    return d


def task33():
    ent = []
    for name, m in binary_cases().items():
        if name in ("salt-noise", "empty"):
            pass
        h, w = m.shape
        e = {"name": name, "width": w, "height": h, "mask": mask_str(m)}
        for conn in (1, 2):
            lab, n = measure.label(m, connectivity=conn, return_num=True)
            e[f"labels{conn}"] = ints(lab)
            e[f"count{conn}"] = int(n)
            rows = []
            for r in measure.regionprops(lab):
                rows.append(prop_row(r))
            e[f"props{conn}"] = rows
        ent.append(e)

    # intensity statistics over a label image: scipy.ndimage labelled measurements and regionprops intensity_*
    img = (rng.random((64, 64)) * 100).astype(np.float64)
    m = binary_cases()["needles-diagonal"]
    lab, n = measure.label(m, connectivity=2, return_num=True)
    idx = list(range(1, n + 1))
    fo = ndi.find_objects(lab)
    intensity = {
        "width": 64, "height": 64, "image": floats(img), "labels": ints(lab), "count": int(n),
        "sum": floats(ndi.sum(img, lab, idx)), "mean": floats(ndi.mean(img, lab, idx)),
        "std": floats(ndi.standard_deviation(img, lab, idx)), "min": floats(ndi.minimum(img, lab, idx)),
        "max": floats(ndi.maximum(img, lab, idx)), "center_of_mass": [floats(c) for c in ndi.center_of_mass(img, lab, idx)],
        "min_position": [ints(p) for p in ndi.minimum_position(img, lab, idx)],
        "max_position": [ints(p) for p in ndi.maximum_position(img, lab, idx)],
        "find_objects": [[int(s[0].start), int(s[0].stop), int(s[1].start), int(s[1].stop)] for s in fo],
        "regionprops": [{"label": int(r.label), "intensity_mean": float(r.intensity_mean), "intensity_min": float(r.intensity_min),
                         "intensity_max": float(r.intensity_max), "intensity_std": float(r.intensity_std),
                         "centroid_weighted": floats(r.centroid_weighted)}
                        for r in measure.regionprops(lab, intensity_image=img)],
    }
    out = {**meta(), "task": 33,
           "note": "feret_diameter_max is the needle length (maximum caliper distance over the convex hull); axis_major_length is 4*sqrt(largest inertia eigenvalue), which is about 2/sqrt(3) times the length of a thin straight rod, so the two differ on every needle in the pins. Tolerance: skimage assert_allclose dtype defaults (rtol 1e-7); ints exact; scipy ndimage decimal 6.",
           "cases": ent, "intensity": intensity}
    (FIX / "regionprops-needle-pins.json").write_text(json.dumps(out))
    return len(ent), sum(e["count2"] for e in ent)


a = task32()
b = task33()
print("task 32:", a[0], "grey images,", a[1], "binary cases; task 33:", b[0], "masks,", b[1], "objects (8-connected)")
