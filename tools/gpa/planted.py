#!/usr/bin/env python3
"""Planted-truth lattice images for GPA strain (ImagoKit task 21).

numpy only. Writes, for each case, `<case>.npy` (float64 lattice image, shape (rows, cols)) and one `truth.json`.

Conventions (rows and columns, never x and y):
  * Array axis 0 is the row, axis 1 the column; every vector is (row, col); pixel (i, j) sits at r = (i, j).
  * Reciprocal vectors g are in cycles per pixel, g = (g_row, g_col).
  * The lattice image is  I(r) = sum_{g in {+-g1, +-g2}} 0.5 * cos(2 pi g . (r - u(r)))  with u the planted displacement
    (pixels). The lattice at r sits where the perfect lattice was at r - u(r).
  * Hytch's phase for reflection g is P_g(r) = -2 pi g . u(r), so  grad P_g = -2 pi G D  with G the 2x2 matrix whose rows
    are g1, g2 and D_ij = d u_i / d r_j the distortion tensor. D is what a GPA implementation must return.
  * Strain = symmetric part of D, rotation = antisymmetric part (small-deformation form; second order terms are not
    planted, the cases keep |D| <= 0.02 so they are below 2e-4).

Cases (all periodic-commensurate carriers, so the perfect lattice has no spectral leakage):
  uniform   D constant: e_rr, e_cc, e_rc, rotation. Truth is the same number everywhere.
  bump      D constant background plus a Gaussian displacement bump (an inclusion-like strain field), analytic gradient.
  perfect   u = 0. Truth is exactly zero strain (the null case; any bias shows here).

The script also runs a plain numpy GPA (Gaussian mask in Fourier space, gradient of the phase from Im(conj(H) grad H)/|H|^2)
and records its worst interior error per case under `selfcheck`. That number is a measured, not claimed, tolerance floor:
it shows what a plain implementation with the stated mask reaches on this input (about 6e-4 uniform, 4e-3 bump). It is not an upstream
pin; the Mac pin scripts do not apply to GPA (no Python package ships it), the planted truth is the reference.

Run:  python3 -I planted.py [--out DIR] [--size 256]
"""
import argparse
import json
import os
import sys

import numpy as np

# carriers (cycles per pixel), chosen as integer multiples of 1/size so the perfect image is exactly periodic
K1 = (0, 16)    # g1 = (0, 16/size) horizontal fringes' normal along columns
K2 = (16, 0)    # g2 = (16/size, 0) along rows; perpendicular here, the second case below uses a non-orthogonal pair
K2_OBLIQUE = (14, 8)


def displacement(case, rows, cols):
    """u (2, rows, cols) in pixels and D (2, 2, rows, cols), analytic."""
    i, j = np.meshgrid(np.arange(rows, dtype=np.float64), np.arange(cols, dtype=np.float64), indexing="ij")
    u = np.zeros((2, rows, cols))
    d = np.zeros((2, 2, rows, cols))
    if case == "perfect":
        return u, d
    lin = np.array([[0.010, 0.004], [-0.002, -0.006]])  # D_ij of the background: e_rr 0.010, e_cc -0.006, shear/rotation
    ci, cj = (rows - 1) / 2, (cols - 1) / 2
    for a in range(2):
        u[a] = lin[a, 0] * (i - ci) + lin[a, 1] * (j - cj)
        d[a, 0] = lin[a, 0]
        d[a, 1] = lin[a, 1]
    if case == "bump":
        # u_row and u_col: Gaussian bumps of amplitude 2.0 and -1.5 px, width 40 and 50 px, centred at (0.4, 0.6) of the frame
        for a, (amp, ri, rj, s) in enumerate([(2.0, 0.4, 0.6, 40.0), (-1.5, 0.55, 0.45, 50.0)]):
            ri, rj = ri * rows, rj * cols
            gauss = amp * np.exp(-((i - ri) ** 2 + (j - rj) ** 2) / (2 * s * s))
            u[a] += gauss
            d[a, 0] += gauss * (-(i - ri) / (s * s))
            d[a, 1] += gauss * (-(j - rj) / (s * s))
    return u, d


def lattice(g_pairs, u, rows, cols):
    i, j = np.meshgrid(np.arange(rows, dtype=np.float64), np.arange(cols, dtype=np.float64), indexing="ij")
    img = np.zeros((rows, cols))
    for g in g_pairs:
        phase = 2 * np.pi * (g[0] * (i - u[0]) + g[1] * (j - u[1]))
        img += 2 * 0.5 * np.cos(phase)  # +g and -g terms
    return img


def numpy_gpa(img, gs, mask_sigma):
    """Reference-free numpy GPA: returns the distortion tensor D (2, 2, rows, cols)."""
    rows, cols = img.shape
    f = np.fft.fft2(img)
    fr = np.fft.fftfreq(rows)[:, None]
    fc = np.fft.fftfreq(cols)[None, :]
    grads = []
    for g in gs:
        mask = np.exp(-((fr - g[0]) ** 2 + (fc - g[1]) ** 2) / (2 * mask_sigma ** 2))
        h = np.fft.ifft2(f * mask)
        ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        h = h * np.exp(-2j * np.pi * (g[0] * ii + g[1] * jj))  # remove the carrier: H = 0.5 A exp(-i 2 pi g.u)
        # gradient of the phase without unwrapping: Im(conj(H) dH) / |H|^2, central differences
        gr = np.gradient(h, axis=0)
        gc = np.gradient(h, axis=1)
        den = np.abs(h) ** 2
        grads.append(np.stack([np.imag(np.conj(h) * gr) / den, np.imag(np.conj(h) * gc) / den]))
    gp = np.stack(grads)  # (reflection, derivative axis, rows, cols)
    gmat = np.array(gs)
    inv = np.linalg.inv(gmat)
    return -np.einsum("ab,bjxy->ajxy", inv, gp) / (2 * np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n = args.size
    truth = {
        "generator": "ImagoKit/tools/gpa/planted.py",
        "convention": "rows and columns; I = sum_{+-g} 0.5 cos(2 pi g.(r-u)); P_g = -2 pi g.u; grad P_g = -2 pi G D, "
                      "D_ij = du_i/dr_j; g in cycles per pixel as (g_row, g_col)",
        "upstream_pin": "none; no Python package ships GPA. The planted analytic field is the reference.",
        "cases": {},
    }
    mask_sigma = 3.2 / n  # Gaussian mask sigma in cycles per pixel (0.0125 at size 256): 5 sigma to the nearest other reflection
    for name, k2, case in [("uniform", K2, "uniform"), ("bump", K2, "bump"), ("perfect", K2, "perfect"),
                           ("bump_oblique", K2_OBLIQUE, "bump")]:
        gs = [(K1[0] / n, K1[1] / n), (k2[0] / n, k2[1] / n)]
        u, d = displacement(case, n, n)
        img = lattice(gs, u, n, n)
        np.save(os.path.join(args.out, f"{name}.npy"), img)
        np.save(os.path.join(args.out, f"{name}_u.npy"), u)
        np.save(os.path.join(args.out, f"{name}_distortion.npy"), d)
        strain = 0.5 * (d + d.transpose(1, 0, 2, 3))
        rot = 0.5 * (d[0, 1] - d[1, 0])
        np.save(os.path.join(args.out, f"{name}_strain.npy"), strain)
        np.save(os.path.join(args.out, f"{name}_rotation.npy"), rot)
        # interior = away from the frame by 3 mask correlation lengths (1 / (2 pi sigma)) so edge leakage is excluded
        margin = int(np.ceil(3 / (2 * np.pi * mask_sigma)))
        inner = (slice(margin, n - margin), slice(margin, n - margin))
        est = numpy_gpa(img, gs, mask_sigma)
        err = float(np.max(np.abs(est[(slice(None), slice(None)) + inner] - d[(slice(None), slice(None)) + inner])))
        truth["cases"][name] = {
            "image": f"{name}.npy",
            "shape": [n, n],
            "g1": list(gs[0]), "g2": list(gs[1]),
            "displacement": f"{name}_u.npy (2, rows, cols) pixels",
            "distortion": f"{name}_distortion.npy (2, 2, rows, cols), D_ij = du_i/dr_j",
            "strain": f"{name}_strain.npy (symmetric part of D)",
            "rotation": f"{name}_rotation.npy (antisymmetric part, radians, (D_rc - D_cr)/2)",
            "max_abs_distortion": float(np.max(np.abs(d))),
            "interior_margin_px": margin,
            "mask_sigma_cycles_per_px": mask_sigma,
            "selfcheck": {"max_abs_error_interior": err,
                          "meaning": "worst |D_numpy_gpa - D_truth| in the interior with the stated Gaussian mask"},
            "tolerance_rationale": "Compare in the interior only (margin above). The mask is a low-pass on the strain field: "
                                   "the bump's Gaussian gradient is smoothed, so tolerance is set from selfcheck "
                                   "(use 2x selfcheck, floor 1e-6) rather than from float round-off. 'perfect' "
                                   "is exactly periodic and agrees to round-off; 'uniform' carries a phase ramp that is not periodic, "
                                   "so frame-edge leakage reaches the interior at about 6e-4.",
        }
        print(f"{name:13s} max|D|={np.max(np.abs(d)):.4f} interior numpy-GPA max error={err:.3e}")
    with open(os.path.join(args.out, "truth.json"), "w") as fh:
        json.dump(truth, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
