#!/usr/bin/env python3
"""Generate the segmentation pins; run from the package root:
../tools/memcap.sh -- ../References/python/skimage/bin/python -I -B tools/segmentation/pins.py [APP_SEG_BINARY]

Writes Tests/ImagoCoreTests/Fixtures/segmentation-pins.json.

Reference: scikit-image 0.26.0 `measure.label` (connectivity 1 and 2, masks and value maps) and `measure.regionprops`
(area, bbox, centroid, inertia_tensor_eigvals, axis_major_length, axis_minor_length, orientation, moments_central).
Exact for labels, area, bbox; floats are written as repr (round trip). The extents along and across the principal axis,
the direction in image coordinates and the edge flag are a convention with no package behind it: they are pinned to the
application's own Float32 routine, compiled offline from its source (APP_SEG_BINARY reads the cases JSON given as argv[1]
and prints each 8-connected component in raster order of its first pixel), as {length, width, orientation, edge}.
"""
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from skimage import measure

assert np.__version__ == "2.5.4", np.__version__
assert platform.python_version() == "3.14.7", platform.python_version()
import skimage
assert skimage.__version__ == "0.26.0", skimage.__version__
ROOT = Path(__file__).resolve().parents[2]
rng = np.random.Generator(np.random.PCG64(20261020))


def blob_mask(h, w, density, smooth):
    a = rng.random((h, w))
    if smooth:
        # box-smoothed noise gives compact, elongated objects
        k = np.ones((3, 5)) / 15
        pad = np.pad(a, ((1, 1), (2, 2)), mode="edge")
        s = np.zeros_like(a)
        for i in range(3):
            for j in range(5):
                s += pad[i:i + h, j:j + w]
        a = s / 15
    return a < np.quantile(a, density)


def cases():
    out = []
    for h, w, d, sm in [(1, 1, 1.0, False), (1, 17, 0.6, False), (23, 1, 0.6, False), (8, 8, 0.4, False),
                        (31, 40, 0.3, False), (31, 40, 0.5, True), (64, 64, 0.2, True), (64, 64, 0.45, True),
                        (50, 33, 0.35, True), (50, 33, 0.7, False), (17, 17, 0.5, True), (100, 80, 0.3, True)]:
        for rep in range(2):
            out.append(blob_mask(h, w, d, sm))
    out.append(np.zeros((6, 7), bool))
    out.append(np.ones((5, 9), bool))
    out.append((np.indices((8, 9)).sum(0) % 2) == 0)          # checkerboard: 4 vs 8 connectivity
    diag = np.eye(9, dtype=bool)
    out.append(diag)                                          # a diagonal line
    out.append(np.pad(np.ones((3, 3), bool), 2))              # a square interior
    return out


def props(label_img, conn):
    res = []
    for r in measure.regionprops(label_img):
        res.append({
            "label": int(r.label), "area": int(r.area), "bbox": [int(v) for v in r.bbox],
            "centroid": [float(v) for v in r.centroid],
            "eigvals": [float(v) for v in r.inertia_tensor_eigvals],
            "inertia": [float(v) for v in r.inertia_tensor.flat],
            "major": float(r.axis_major_length), "minor": float(r.axis_minor_length),
            "orientation": float(r.orientation),
        })
    return res


def main():
    cs = cases()
    entries = []
    for m in cs:
        h, w = m.shape
        e = {"width": w, "height": h, "mask": "".join("1" if v else "0" for v in m.flat)}
        for conn in (1, 2):
            lab, n = measure.label(m, connectivity=conn, return_num=True)
            e[f"labels{conn}"] = [int(v) for v in lab.flat]
            e[f"count{conn}"] = int(n)
            e[f"props{conn}"] = props(lab, conn)
        entries.append(e)
    # value maps: equal-value connected regions, background 0 and background nil
    vals = []
    for h, w, k in [(20, 25, 4), (9, 7, 2), (33, 33, 5)]:
        v = rng.integers(0, k, size=(h, w)).astype(np.int32)
        v = np.where(rng.random((h, w)) < 0.5, np.repeat(np.repeat(v[::2, ::2][: (h + 1) // 2, : (w + 1) // 2], 2, 0), 2, 1)[:h, :w], v)
        d = {"width": w, "height": h, "values": [int(x) for x in v.flat]}
        for conn in (1, 2):
            lab0, n0 = measure.label(v, background=0, connectivity=conn, return_num=True)
            labn, nn = measure.label(v, background=-1, connectivity=conn, return_num=True)
            d[f"bg0_{conn}"] = [int(x) for x in lab0.flat]; d[f"bg0count_{conn}"] = int(n0)
            d[f"all_{conn}"] = [int(x) for x in labn.flat]; d[f"allcount_{conn}"] = int(nn)
        vals.append(d)
    app = None
    if len(sys.argv) > 1:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cases.json"
            p.write_text(json.dumps([{"width": e["width"], "height": e["height"], "mask": [int(c) for c in e["mask"]]}
                                     for e in entries]))
            r = subprocess.run([sys.argv[1], str(p)], capture_output=True, text=True, check=True)
            app = json.loads(r.stdout)
        for e, a in zip(entries, app):
            assert len(a) == e["count2"], "application's component count differs from skimage connectivity 2"
            for o, pr in zip(a, e["props2"]):
                assert o["area"] == pr["area"]
            e["app"] = a
    out = {"skimage": skimage.__version__, "numpy": np.__version__, "masks": entries, "values": vals,
           "appChecked": app is not None}
    (ROOT / "Tests/ImagoCoreTests/Fixtures/segmentation-pins.json").write_text(json.dumps(out))
    print(len(entries), "masks;", sum(e["count2"] for e in entries), "objects (8-connected)")


main()
