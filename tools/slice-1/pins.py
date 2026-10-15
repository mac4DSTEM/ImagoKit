"""Slice 1a pins (docs/registration-slice-1.md section 2): the spectrum, radial profile, spot and angle fixtures.

    ../tools/memcap.sh -- References/oracle-venv/bin/python -I -B tools/slice-1/pins.py

Run from the repository root, before the Swift code (the registration's rule). Steps:
  1. macTEM's fft-parity and spot-parity harness scripts at MACTEM_COMMIT are extracted with `git show` from ../macTEM (never
     its working tree) into .build/apps/, and run with this interpreter: make_fixture.py plants the images, reference.py
     checks NumPy against every planted truth (it exits 1 otherwise) and writes NumPy's references. Their outputs land in
     .build/pins-full/{fft,spot}/.
  2. The angle cases (registration 5.4) are planted here and measured with the spot harness's own NumPy estimator
     (imported from the extracted reference.py, not copied); the plant truth |theta - theta0| <= 0.1 deg (noise-free) or
     0.3 deg (noisy) and the spacing ratio within 0.4 % must hold in NumPy, or this script exits 1.
  3. The committed subset (under 8 MB; the split is printed) is written to Tests/ImagoFFTTests/Fixtures/ with manifests
     whose paths are relative to that folder, every file's sha256, and the versions. .build/pins-full/ keeps every case
     for tools/slice-1/full-parity.sh.
Slice 1b (amendment 7): the fft harness's `ifft` cases are committed with the spectrum and profile cases; macTEM's
hrtem-filter-parity (plants.py, reference.py, SEED0 20261015) runs too, and two of its 41 cases are committed to
Tests/ImagoFiltersTests/Fixtures/ (slice-1b-pins.json); the budget is 10 MB over both folders.
"""
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

MACTEM_COMMIT = "3d17311"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
APPS = os.path.join(ROOT, ".build", "apps", "macTEM-" + MACTEM_COMMIT)
FULL = os.path.join(ROOT, ".build", "pins-full")
FIX = os.path.join(ROOT, "Tests", "ImagoFFTTests", "Fixtures")
HARNESS_FILES = ["tools/fft-parity/make_fixture.py", "tools/fft-parity/reference.py",
                 "tools/spot-parity/make_fixture.py", "tools/spot-parity/reference.py",
                 "tools/hrtem-filter-parity/plants.py", "tools/hrtem-filter-parity/reference.py"]
FILTER_FIX = os.path.join(ROOT, "Tests", "ImagoFiltersTests", "Fixtures")
HRTEM_COMMITTED = ["bin-128", "pct5-128"]          # the defaults with the energy region; the low-frequency taper
HRTEM_KEYS = ["absf", "wiener", "W", "M", "a", "w", "c1", "S", "firstpass", "nearmargins", "p2region"]
SPOT_COMMIT_MAX_PIXELS = 256 * 128          # spot cases up to 256 x 128 are committed; 256 x 256 and 512 x 512 are full only,
SPOT_COMMITTED_EXTRA = ["neighbour-3bins", "noisy256-k10-r0"]   # but for the one neighbour case and one noisy case (Gate B)
ANGLE_COMMITTED = ["a256-iso-t60-clean", "a256-aniso-t109.47-clean", "a256-iso-t145-noisy", "a256-aniso-t30-noisy"]
BUDGET = 10 * 1024 * 1024                         # amendment 7 (was 8 MB before slice 1b)
ANGLE_SEED = 20261016


def fail(msg):
    print("pins.py: FAIL: " + msg, file=sys.stderr)
    sys.exit(1)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def extract():
    for rel in HARNESS_FILES:
        out = os.path.join(APPS, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        src = subprocess.run(["git", "-C", os.path.join(ROOT, "..", "macTEM"), "--no-optional-locks", "show",
                              MACTEM_COMMIT + ":" + rel], check=True, capture_output=True).stdout
        with open(out, "wb") as f:
            f.write(src)


def run_harness(name, folder=None, maker="make_fixture.py"):
    here = os.path.join(APPS, "tools", (folder or name) + "-parity")
    planted, ref = os.path.join(FULL, name, "planted"), os.path.join(FULL, name, "ref")
    for script, args in ((maker, [planted]), ("reference.py", [planted, ref])):
        r = subprocess.run([sys.executable, "-I", "-B", os.path.join(here, script)] + args, capture_output=True, text=True)
        log = os.path.join(FULL, name + "-" + script.replace(".py", ".log"))
        with open(log, "w") as f:
            f.write(r.stdout + r.stderr)
        if r.returncode != 0:
            fail("%s/%s exited %d (log %s)" % (name, script, r.returncode, log))
    with open(os.path.join(ref, "manifest.json")) as f:
        return json.load(f)


def load_spot_estimator():
    path = os.path.join(APPS, "tools", "spot-parity", "reference.py")
    spec = importlib.util.spec_from_file_location("mactem_spot_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.estimate


# The angle plants (registration 5.4).
ANGLES = [30.0, 60.0, 90.0, math.degrees(math.acos(-1.0 / 3.0)), 145.0]
SAMPLINGS = {"iso": (0.0123, 0.0123), "aniso": (0.0123, 0.0246)}
SHAPES = [(256, 256), (384, 512)]                       # (h, w)
RELS = [0.15, 0.25, 0.35]
FIRST_DIRECTIONS = [10.0, 35.0, 70.0]
RATIOS = [1.0, 1.2, 0.8, 1.1, 0.9]
OFFSETS = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
NOISE_SEEDS = [ANGLE_SEED + i for i in range(4)]


def angle_name(h, w, sampling, theta, tier):
    size = "a%d" % w if h == w else "a%dx%d" % (w, h)
    return "%s-%s-t%s-%s" % (size, sampling, ("%.2f" % theta).rstrip("0").rstrip("."), tier)


def plant_angle(h, w, dx, dy, theta, n):
    """Two lattices: spot 1 at an index-space direction and radius with a sub-bin offset; spot 2 at a physical angle theta
    from spot 1's physical q, its magnitude a ratio of |q1|. Returns (k1, k2) in index units (kx, ky) or None."""
    rel = RELS[n % len(RELS)]
    a1 = math.radians(FIRST_DIRECTIONS[n % len(FIRST_DIRECTIONS)])
    r1 = rel * 0.5 * min(w, h)
    ox, oy = OFFSETS[n % len(OFFSETS)]
    k1 = (round(r1 * math.cos(a1)) + ox, round(r1 * math.sin(a1)) + oy)
    q1 = (k1[0] / (w * dx), k1[1] / (h * dy))
    phi = math.atan2(q1[1], q1[0]) + math.radians(theta)
    for ratio in RATIOS[n % len(RATIOS):] + RATIOS[:n % len(RATIOS)]:
        m = ratio * math.hypot(*q1)
        q2 = (m * math.cos(phi), m * math.sin(phi))
        k2 = (q2[0] * w * dx, q2[1] * h * dy)
        inside = abs(k2[0]) < w / 2 - 4 and abs(k2[1]) < h / 2 - 4
        apart = min(math.hypot(k2[0] - k1[0], k2[1] - k1[1]), math.hypot(k2[0] + k1[0], k2[1] + k1[1]),
                    math.hypot(*k2)) >= 6.0
        if inside and apart:
            return k1, k2
    return None


def angle_cases(estimate):
    out = os.path.join(FULL, "angle")
    os.makedirs(out, exist_ok=True)
    cases, n = [], 0
    for (h, w) in SHAPES:
        for sname, (dx, dy) in SAMPLINGS.items():
            for theta in ANGLES:
                tiers = ["clean"] + (["noisy"] if (h, w) == (256, 256) else [])
                for tier in tiers:
                    k = plant_angle(h, w, dx, dy, theta, n)
                    if k is None:
                        fail("no plant for %dx%d %s theta %g" % (w, h, sname, theta))
                    k1, k2 = k
                    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
                    img = 10.0 + np.cos(2 * np.pi * (k1[0] * xx / w + k1[1] * yy / h) + 0.3) \
                        + np.cos(2 * np.pi * (k2[0] * xx / w + k2[1] * yy / h) + 1.1)
                    seed = None
                    if tier == "noisy":
                        seed = NOISE_SEEDS[n % len(NOISE_SEEDS)]
                        img = img + np.random.default_rng(seed).normal(0.0, 0.5, img.shape)
                    x32 = img.astype(np.float32)
                    name = angle_name(h, w, sname, theta, tier)
                    path = os.path.join(out, name + ".in.f32")
                    x32.astype("<f4").tofile(path)
                    spots = []
                    for kk in (k1, k2):
                        click = (int(round(h // 2 + kk[1])), int(round(w // 2 + kk[0])))
                        e = estimate(x32, click, dx, dy)
                        if e["near_dc"] or e["near_edge"]:
                            fail("%s: a planted spot was refused" % name)
                        crow, ccol = e["centroid"]
                        qx, qy = (ccol - w // 2) * (1.0 / (w * dx)), (crow - h // 2) * (1.0 / (h * dy))
                        spots.append({"planted_kx": kk[0], "planted_ky": kk[1], "click_row": click[0],
                                      "click_col": click[1], "snapped_row": int(e["snapped"][0]),
                                      "snapped_col": int(e["snapped"][1]), "centroid_row": crow, "centroid_col": ccol,
                                      "qx": qx, "qy": qy, "d": e["d"]})
                    a, b = spots
                    cross = a["qx"] * b["qy"] - a["qy"] * b["qx"]
                    dot = a["qx"] * b["qx"] + a["qy"] * b["qy"]
                    theta_np = math.degrees(math.atan2(abs(cross), dot))
                    p1 = (k1[0] / (w * dx), k1[1] / (h * dy))
                    p2 = (k2[0] / (w * dx), k2[1] / (h * dy))
                    theta0 = math.degrees(math.atan2(abs(p1[0] * p2[1] - p1[1] * p2[0]), p1[0] * p2[0] + p1[1] * p2[1]))
                    if abs(theta0 - theta) > 1e-9:
                        fail("%s: planted angle %r is not %r" % (name, theta0, theta))
                    ratio0 = math.hypot(*p2) / math.hypot(*p1)          # d1/d2 = |q2|/|q1|
                    ratio_np = a["d"] / b["d"]
                    band = 0.1 if tier == "clean" else 0.3
                    if abs(theta_np - theta0) > band:
                        fail("%s: NumPy angle %.4f deg vs planted %.4f deg, beyond %g" % (name, theta_np, theta0, band))
                    if abs(ratio_np / ratio0 - 1) > 0.004:
                        fail("%s: NumPy spacing ratio %.6f vs planted %.6f, beyond 0.4 %%" % (name, ratio_np, ratio0))
                    cases.append({"name": name, "input": name + ".in.f32", "width": w, "height": h, "scale_x": dx,
                                  "scale_y": dy, "units": "nm", "tier": tier, "seed": seed, "theta0": theta0,
                                  "theta_numpy": theta_np, "ratio0": ratio0, "ratio_numpy": ratio_np, "spots": spots})
                    print("pins.py: angle %-28s theta0 %8.4f  numpy %8.4f  (|d| %.4f deg)  ratio err %.2e"
                          % (name, theta0, theta_np, abs(theta_np - theta0), ratio_np / ratio0 - 1))
                    n += 1
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"numpy": np.__version__, "seed": ANGLE_SEED, "cases": cases}, f, indent=1)
    return cases


def relocate(case, keys, src_dirs, dest):
    """Copy the files a case names into `dest`; the manifest keeps their base names."""
    out = dict(case)
    for key in keys:
        p = case.get(key)
        if not p:
            continue
        src = p if os.path.isabs(p) else next(os.path.join(d, p) for d in src_dirs + [ROOT]
                                              if os.path.exists(os.path.join(d, p)))
        shutil.copyfile(src, os.path.join(dest, os.path.basename(src)))
        out[key] = os.path.basename(src)
    return out


def main():
    if not np.__version__ == "2.5.4":
        fail("NumPy %s, the pin is 2.5.4 (tools/oracle/references-lock.txt)" % np.__version__)
    if os.path.exists(FULL):
        shutil.rmtree(FULL)
    os.makedirs(FULL)
    extract()
    fft = run_harness("fft")
    spot = run_harness("spot")
    hrtem = run_harness("hrtem", folder="hrtem-filter", maker="plants.py")
    angle = angle_cases(load_spot_estimator())

    if os.path.exists(FIX):
        shutil.rmtree(FIX)
    files = []
    sections = {}
    # Spectrum and profile cases (slice 1a); the ifft cases wait for 1b. Magnitude is sqrt(power) and is not committed.
    d = os.path.join(FIX, "fft")
    os.makedirs(d)
    keep = [c for c in fft["cases"] if c["kind"] in ("spectrum", "profile", "ifft")]
    sections["fft"] = [relocate({k: v for k, v in c.items() if k != "magnitude"},
                                ["input", "power", "axis_x", "axis_y", "pmean", "pcount", "ifft"], [], d) for c in keep]
    # Crop cases carry the cropped input too (written by reference.py as <case>.cropped.in.f32).
    for c in sections["fft"]:
        if c.get("crop"):
            src = os.path.join(FULL, "fft", "ref", c["name"] + ".cropped.in.f32")
            shutil.copyfile(src, os.path.join(d, os.path.basename(src)))
            c["cropped_input"] = os.path.basename(src)
    d = os.path.join(FIX, "spot")
    os.makedirs(d)
    keep = [c for c in spot["cases"]
            if c["width"] * c["height"] <= SPOT_COMMIT_MAX_PIXELS or c["name"] in SPOT_COMMITTED_EXTRA]
    sections["spot"] = [relocate(c, ["input"], [], d) for c in keep]
    d = os.path.join(FIX, "angle")
    os.makedirs(d)
    keep = [c for c in angle if c["name"] in ANGLE_COMMITTED]
    if len(keep) != len(ANGLE_COMMITTED):
        fail("committed angle cases missing: %s" % sorted(set(ANGLE_COMMITTED) - {c["name"] for c in keep}))
    sections["angle"] = [relocate(c, ["input"], [os.path.join(FULL, "angle")], d) for c in keep]

    if os.path.exists(FILTER_FIX):
        shutil.rmtree(FILTER_FIX)
    d = os.path.join(FILTER_FIX, "hrtem")
    os.makedirs(d)
    hsection = []
    for c in hrtem["cases"]:
        if c["name"] not in HRTEM_COMMITTED:
            continue
        out = {k: c[k] for k in ("name", "tier", "n", "params", "input", "passes")}
        shutil.copyfile(os.path.join(FULL, "hrtem", "ref", c["input"]), os.path.join(d, c["input"]))
        out["files"] = {}
        for k in HRTEM_KEYS:
            if k in c["files"]:
                shutil.copyfile(os.path.join(FULL, "hrtem", "ref", c["files"][k]), os.path.join(d, c["files"][k]))
                out["files"][k] = c["files"][k]
        hsection.append(out)
    if [c["name"] for c in hsection] != HRTEM_COMMITTED:
        fail("committed hrtem cases missing")
    ffiles = []
    for root, _, names in os.walk(FILTER_FIX):
        for n in sorted(names):
            p = os.path.join(root, n)
            ffiles.append({"file": os.path.relpath(p, FILTER_FIX), "sha256": sha256(p), "bytes": os.path.getsize(p)})
    for root, _, names in os.walk(FIX):
        for n in sorted(names):
            p = os.path.join(root, n)
            files.append({"file": os.path.relpath(p, FIX), "sha256": sha256(p), "bytes": os.path.getsize(p)})
    total = sum(f["bytes"] for f in files) + sum(f["bytes"] for f in ffiles)
    if total > BUDGET:
        fail("committed fixtures %d bytes, over the %d MB budget" % (total, BUDGET // 2**20))
    import scipy
    import skimage
    pins = {"generator": "tools/slice-1/pins.py", "mactem_commit": MACTEM_COMMIT,
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                         "scikit-image": skimage.__version__},
            "fft": sections["fft"], "spot": sections["spot"], "angle": sections["angle"], "files": files}
    with open(os.path.join(FIX, "slice-1a-pins.json"), "w") as f:
        json.dump(pins, f, indent=1)
    with open(os.path.join(FILTER_FIX, "slice-1b-pins.json"), "w") as f:
        json.dump({"generator": "tools/slice-1/pins.py", "mactem_commit": MACTEM_COMMIT, "versions": pins["versions"],
                   "hrtem": hsection, "files": ffiles}, f, indent=1)
    full_counts = {"fft": len(fft["cases"]), "spot": len(spot["cases"]), "angle": len(angle), "hrtem": len(hrtem["cases"])}
    print("pins.py: committed %d files, %.2f MB: fft %d of %d cases, spot %d of %d, angle %d of %d, hrtem %d of %d; "
          "the full set in .build/pins-full/"
          % (len(files) + len(ffiles), total / 2**20, len(sections["fft"]), full_counts["fft"], len(sections["spot"]),
             full_counts["spot"], len(sections["angle"]), full_counts["angle"], len(hsection), full_counts["hrtem"]))


if __name__ == "__main__":
    main()
