# GPA planted truth (ImagoKit task 21)

`planted.py` writes lattice images with an analytic displacement field, so a GPA strain implementation can be checked against
a known answer. numpy only; no upstream pin exists (no Python package ships GPA), the planted field is the reference and the
result stays badged `validation: "none"` until it passes on a dataset with truth.

```sh
python3 -I tools/gpa/planted.py --out "$TMPDIR/gpa-planted"     # about 1 s; --size N for another frame (multiple of 16)
```

Output per case (`uniform`, `bump`, `perfect`, `bump_oblique`): `<case>.npy` (the image), `_u.npy` (displacement, pixels),
`_distortion.npy` (D, shape 2x2xrowsxcols), `_strain.npy`, `_rotation.npy`, and one `truth.json` with the reflections, the
conventions, the interior margin and the measured error of a plain numpy GPA (`selfcheck`).

## How the Mac session uses it
1. Generate into a temporary folder; do not commit the `.npy` files (rule: generated files do not live in a repo). If a small
   case is wanted as a fixture, use `--size 128` and commit it in its own commit with the reason.
2. The Swift test builds the same field from the formulas in the `planted.py` docstring (or loads the `.npy`) and compares
   inside the interior (`interior_margin_px`), never at the frame edge.
3. Tolerance: take 2x the `selfcheck.max_abs_error_interior` of the case, floor 1e-6. The measured plain-numpy values are about
   6e-4 (uniform), 4e-3 (bump), round-off (perfect). `perfect` must give zero strain to round-off in any correct implementation.
4. Convention to hold: rows and columns, g as (g_row, g_col) in cycles per pixel, `I = sum 0.5 cos(2 pi g.(r-u))`,
   `P_g = -2 pi g.u`, `D_ij = du_i/dr_j`. If the kit uses another sign, flip it in the wrapper and say so in `docs/parity.md`.

Caveat: small-deformation form only (|D| <= 0.04); second order terms are not planted.
