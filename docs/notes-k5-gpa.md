# Notes k5: GPA strain (for the session that merges CHANGELOG, parity, Package.swift)

## Package.swift (needed, not edited here)
There is no `ImagoFiltersTests` test target yet; the new tests live in `Tests/ImagoFiltersTests/` (which has only `Fixtures/`).
Add before the "Test support" target:

```swift
.testTarget(name: "ImagoFiltersTests", dependencies: ["ImagoCore", "ImagoFilters"], exclude: ["Fixtures"], swiftSettings: settings),
```

`@testable import ImagoFilters` is used; the tests need only ImagoCore and ImagoFilters. Run:
`swift test -c release --parallel --filter GeometricPhaseAnalysisTests`.

## CHANGELOG bullet
- Added `GeometricPhaseAnalysis` (ImagoFilters): geometric phase analysis strain after Hytch, Snoeck and Kilaas (1998) from two reflections: Bragg-filtered phases, unwrapping-free phase gradients, the distortion tensor D, strain (e_rr, e_cc, e_rc), rotation, amplitudes, optional reference window, plus `findReflections`. Float64 on `FFTSurface`. `validation: "none"`: agrees with the planted fields of `tools/gpa/planted.py`, not yet checked on a dataset with truth.

## parity.md row
| Geometric phase analysis (GPA) strain | planted truth, validation none | `GeometricPhaseAnalysis.analyze`, `.findReflections` | planted field of `tools/gpa/planted.py`, no upstream package; 2x the plain-numpy selfcheck error per case (uniform 9.8e-4, bump 3.12e-3, bump_oblique 2.97e-3, perfect 1e-6), floor 1e-6, interior only (planted.py) | P3 | Float64 CPU only; measured errors equal the numpy GPA's (4.89e-4, 1.56e-3, 1.48e-3, 2.5e-11) at size 128 |

## Measured (size 128, interior margin 20 px, max over the four D components)
uniform 4.887e-4 (tol 9.8e-4); bump 1.560e-3 (tol 3.12e-3); bump_oblique 1.485e-3 (tol 2.97e-3); perfect 2.5e-11 (tol 1e-6).
The numbers equal the plain numpy GPA's: same mask, same central differences, so the error is the method's (mask smoothing of the bump, edge leakage for the non-periodic ramp), not round-off. Not beaten, by design.

## Sign conventions (as tools/gpa/README.md, no flip needed)
Rows and columns; g = (g_row, g_col) cycles per pixel; I = sum 0.5 cos(2 pi g.(r-u)); P_g = -2 pi g.u; grad P_g = -2 pi G D (G rows g1, g2); D_ij = du_i/dr_j (`dRC` = du_row/d col). Strain = symmetric part; rotation = (D_rc - D_cr)/2 (positive when u_row grows along columns). Returned phase images are wrapped in (-pi, pi]; D never uses them.

## Decisions and deviations
- Mask sigma: `sigmaInLatticeUnits` (default 0.2) times min(|g1|, |g2|), isotropic Gaussian on +g only. At size 128 with the planted g this is exactly planted.py's 3.2/n.
- Mask distance on the signed (fftfreq) axis, not periodic, as the planted numpy GPA.
- Phase gradient by central differences (one-sided on the edge, like numpy.gradient), not a spectral derivative; keeps the result comparable to the selfcheck. A spectral derivative would remove the finite-difference part of the error.
- Where |H|^2 <= 1e-20 of its maximum, D is NaN (no phase). A constant image gives all NaN.
- Refusals: sizes (count != rows*columns, a side below 8), non-finite pixels, |sin angle(g1, g2)| < 0.02, a g component with |g| >= 0.5 or g = 0, sigma not positive finite, reference window empty or outside. Error enum `GeometricPhaseAnalysisError`.
- `Result.validation` is an instance `let` "none" and `Result.validationNone` a static.
- `findReflections`: Hann window, strongest 3x3-local maxima in the half plane, log-parabola refinement; second peak at least 3 bins away with |sin| >= 0.2. Tested only on the four planted images (exact to 1e-3 bin when unstrained, within 0.3 bin when strained).

## Open items
- Validate on a dataset with truth (then drop the badge).
- Metal path (D16) and Float32 variant; at 4k x 4k the CPU path is untimed.
- Second order (large-deformation) terms and non-orthogonal-lattice correction of g from the refinement are not done.
- Spectral or smoothed gradient option; a mask choice other than Gaussian; reflections beyond two (least-squares over several).
- Apps should show the amplitude maps so the user sees where the phase is unreliable.
- tools/gpa/planted.py needs numpy; none in the system python, used `~/miniconda3/envs/py4dstem/bin/python -I`.
