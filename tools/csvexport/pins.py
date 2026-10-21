#!/usr/bin/env python3
"""Generate the CSV-writer pins; run from the package root:
../tools/memcap.sh -- ../References/python/skimage/bin/python -I -B tools/csvexport/pins.py CSVGEN_BINARY

Writes Tests/ImagoFFTTests/Fixtures/csv-export-pins.json (radial profile and spots CSV) and
Tests/ImagoCoreTests/Fixtures/csv-tidy-pins.json (the tidy results CSV).

The reference is the application's own writers. CSVGEN_BINARY is a scratch executable that compiles the application's
ProfileCSV (number, headerValue, scaleLine, quoted), FFTExport (radial profile and spots writers) and ResultsCSV sources
unchanged (stubs for its image axis, export source and result row types) against this package, reads the cases JSON given as
argv[1] and prints each file's text. The kit's writers must reproduce those texts byte for byte, given the same title,
validation and lineage sentences (the harness prints the application's own sentences first; the cases use them).
Doubles are stored as uint64 bit patterns. The cases are seeded and cover NaN, infinities, signed zero, subnormals,
control characters, quotes, commas and newlines in text cells, uncalibrated axes, crops, and empty tables.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rng = np.random.Generator(np.random.PCG64(20261020))
BIN = sys.argv[1]


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


SPECIAL = [0.0, -0.0, 1.0, 0.1, 1 / 3, 1e300, 5e-324, 2.2250738585072014e-308, float("nan"), float("inf"), float("-inf"),
           123456789.123456789, 4.2e-9, 0.3125]
NASTY = ["plain.dm4", "with,comma.dm4", 'a "quoted" name', "line\nbreak", "cr\r\nlf", "tab\tand\u0085nel sep ",
         "Ångström µm é́", "", "x" * 40, "ctrl\x01\x7f"]
UNITS = ["nm", "Å", "um", "µm", "m", "pm", "px", "", "1/nm", "1/Å", "1/m", "1/px", "mrad", "°"]


def rnum(scale=1.0):
    r = rng.random()
    if r < 0.25:
        return SPECIAL[int(rng.integers(len(SPECIAL)))]
    return float(rng.normal() * 10 ** rng.uniform(-9, 9) * scale)


def axis():
    r = rng.random()
    if r < 0.2:
        return None
    return {"originBits": bits(rnum()), "scaleBits": bits(abs(rnum()) if rng.random() < 0.8 else rnum()),
            "unit": str(rng.choice(UNITS))}


def header(crop_ok=True):
    notes = [NASTY[int(i)] for i in rng.integers(0, len(NASTY), int(rng.integers(0, 3)))]
    if rng.random() < 0.3:
        notes.append("Unvalidated reader (UNVALIDATED)")
    return {
        "software": NASTY[int(rng.integers(len(NASTY)))], "fileName": NASTY[int(rng.integers(len(NASTY)))],
        "object": NASTY[int(rng.integers(len(NASTY)))],
        "imageWidth": int(rng.integers(1, 5000)), "imageHeight": int(rng.integers(1, 5000)),
        "x": axis(), "y": axis(), "notes": notes,
        "crop": [int(v) for v in rng.integers(0, 3000, 4)] if rng.random() < 0.5 else None,
        "spectrumWidth": int(rng.integers(32, 4096)), "spectrumHeight": int(rng.integers(32, 4096)),
        "spacingXBits": bits(abs(rnum())), "spacingYBits": bits(abs(rnum())),
        "unitsX": str(rng.choice(UNITS)), "unitsY": str(rng.choice(UNITS)),
    }


def radial(i):
    n = int(rng.integers(0, 40)) if i > 2 else [0, 1, 2][i]
    mean = [rnum() for _ in range(n)]
    count = [int(c) for c in rng.integers(0, 50, n)]
    if i % 3 == 0:
        count = [0] * n
    return {"header": header(), "units": str(rng.choice(UNITS)), "widthBits": bits(float(10 ** rng.uniform(-6, 3))),
            "meanBits": [bits(m) for m in mean], "count": count}


def spot(k):
    return {"number": k + 1, "inMask": bool(rng.random() < 0.5), "m": None}


def spots(i, constants):
    n = [0, 1][i] if i < 2 else int(rng.integers(1, 9))
    u = str(rng.choice(["nm", "Å", "um", "px", "m", "1/nm", ""]))
    ru = str(rng.choice(["1/nm", "1/Å", "1/px", "nm", "1/m"]))
    rows = []
    for k in range(n):
        rows.append({"number": k + 1, "inMask": bool(rng.random() < 0.5), "m": {
            "clickedRow": int(rng.integers(0, 4000)), "clickedCol": int(rng.integers(0, 4000)),
            "snappedRow": int(rng.integers(0, 4000)), "snappedCol": int(rng.integers(0, 4000)),
            "centroidRowBits": bits(rnum()), "centroidColBits": bits(rnum()),
            "width": int(rng.integers(32, 4096)), "height": int(rng.integers(32, 4096)),
            "qxBits": bits(rnum()), "qyBits": bits(rnum()), "qBits": bits(rnum()), "dBits": bits(rnum()),
            "units": u, "reciprocalUnits": ru, "neighbour": bool(rng.random() < 0.3)}})
    return {"header": header(), "rows": rows, "omitted": int(rng.integers(0, 6)),
            "maskRadiusBits": bits(rnum()), "maskEdgeBits": bits(rnum()),
            "notes": [NASTY[int(j)] for j in rng.integers(0, len(NASTY), int(rng.integers(0, 3)))]}


PRODUCTS = [("distance", "marks"), ("angle", "marks"), ("profile", "marks"), ("spot", "spots")]
VALIDATIONS = [["distance", "closed form, distance"], ["angle", 'closed form, angle, with "quotes", and commas'],
               ["profile", "samples: scikit-image 0.26.0 (a line\nbreak)"], ["spot", "planted spots"]]


def entry():
    r = rng.random()
    return {"name": NASTY[int(rng.integers(len(NASTY)))] if r < 0.2 else str(rng.choice(["length", "x1", "y1", "d", "q"])),
            "valueBits": None if rng.random() < 0.15 else bits(rnum()),
            "unit": str(rng.choice(UNITS)), "text": NASTY[int(rng.integers(len(NASTY)))] if rng.random() < 0.3 else ""}


def tidy(i):
    n = [0, 1][i] if i < 2 else int(rng.integers(1, 12))
    rows = []
    for k in range(n):
        p, g = PRODUCTS[int(rng.integers(4))]
        rows.append({"group": g, "id": f"{g}:{k}", "label": NASTY[int(rng.integers(len(NASTY)))] if rng.random() < 0.3 else f"Row {k}",
                     "product": p, "entries": [entry() for _ in range(int(rng.integers(0, 6)))],
                     "refusal": NASTY[int(rng.integers(len(NASTY)))] if rng.random() < 0.2 else None,
                     "lineage": ("s%d" % k) if rng.random() < 0.4 else None})
    notes = [NASTY[int(j)] for j in rng.integers(0, len(NASTY), int(rng.integers(0, 3)))]
    if rng.random() < 0.4:
        notes.append("unvalidated")
    return {"source": {"software": "Tool 0.1.0 (1)", "fileName": NASTY[int(rng.integers(len(NASTY)))],
                       "object": NASTY[int(rng.integers(len(NASTY)))], "width": int(rng.integers(1, 5000)),
                       "height": int(rng.integers(1, 5000)), "x": axis(), "y": axis(), "notes": notes},
            "rows": rows, "session": "Saved at 14:03, in \"x.json\"\nnext line", "validations": VALIDATIONS}


def run(inp):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cases.json"
        p.write_text(json.dumps(inp))
        r = subprocess.run([BIN, str(p)], capture_output=True, text=True, check=True)
        return json.loads(r.stdout)


def main():
    constants = run({"radial": [], "spots": [], "tidy": []})["constants"]
    inp = {"radial": [radial(i) for i in range(16)], "spots": [spots(i, constants) for i in range(14)],
           "tidy": [tidy(i) for i in range(16)]}
    res = run(inp)
    assert any(r is None for r in res["radial"]) and any(r is not None for r in res["radial"])
    assert res["spots"][0] is None and res["tidy"][0] is None
    fft = {"constants": constants, "radial": [{"input": i, "expected": e} for i, e in zip(inp["radial"], res["radial"])],
           "spots": [{"input": i, "expected": e} for i, e in zip(inp["spots"], res["spots"])]}
    td = {"title": "macTEM results", "validations": VALIDATIONS,
          "descriptionLines": [
              "# points: x1, y1, x2, y2, ... are image pixels (x column, y row, pixel centres at integers); an angle's vertex "
              "is point 2; click and centroid are bins of the centred spectrum; region is in image pixels",
              "# units: lengths in the file's unit as the file states it (a Velox metre axis stays in metres), px for pixels"],
          "lineageLine": "# lineage: a mark's node id in the session record once saved, empty before and for FFT spots (not recorded)",
          "cases": [{"input": i, "expected": e} for i, e in zip(inp["tidy"], res["tidy"])]}
    (ROOT / "Tests/ImagoFFTTests/Fixtures/csv-export-pins.json").write_text(json.dumps(fft))
    (ROOT / "Tests/ImagoCoreTests/Fixtures/csv-tidy-pins.json").write_text(json.dumps(td))
    print("radial", len(fft["radial"]), "spots", len(fft["spots"]), "tidy", len(td["cases"]))


main()
