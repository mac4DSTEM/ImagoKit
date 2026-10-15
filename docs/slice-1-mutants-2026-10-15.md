# Slice 1a mutants (2026-10-15)

Registration section 7, on copies of the frozen commit `eb042aa` (never the working tree), one warm release `.build`, each mutant
applied alone by exact string replacement (a missing target fails the script), built, tested, reverted. Run by a Sonnet lane
(`scratchpad/mutants-1a/`: `run.py`, one log per mutant); the follow-ups by the session on clones of `f51d1f8` and of the tree
before `amendment 6`'s commit. Unmutated baseline at `eb042aa`: 18 test functions, 136 test cases, green.

## The campaign: 35 mutants, all applied, all built

| Mutant | Change | Caught at `eb042aa` by |
|---|---|---|
| M1 | `FFT2DDouble` Bluestein chirp without the mod-2N reduction | **not caught** → caught by `doubleFFTIsAtRoundoffOnOneAxis` (1009 × 1: 3.0e-13 against 1.9e-15 unmutated) |
| M1f | the same in `FFT2D` (Float) | **equivalent** (below) |
| M2 / M2d | radix-3 direction flipped (Float / Double) | the DFT comparison, spectrum, peaks, profile |
| M3 | Double inverse never scaled | `doubleFFTEqualsTheDirectDFT` (round trip) |
| M4 | Float small-power-of-two exception removed | `floatFFTIsWithinItsBandOfTheDirectDFT` (8 × 32, 4 × 8) |
| M4d | the same in `FFT2DDouble` | **equivalent on this machine** (below) |
| M5 | Double radix-5 `c1` off by 7.4e-15 | **not caught** → caught by `doubleFFTIsAtRoundoffOnOneAxis` (5 × 1: 5.3e-15 against 2.5e-16; 125 × 1: 7.6e-15 against 6.2e-16) |
| M6, M7 | no shift; shift (n + 1)/2 | spectrum, peaks, profile |
| M8 | x's scale for both axes | spectrum, profile, the profile refusal |
| M9 | power = magnitude | spectrum, profile |
| M10 | fftfreq over n − 1 | spectrum (axes bit for bit) |
| M11–M15 | profile: pixel bins, Δq_x on both axes, round for floor, DC dropped, sum for mean | `radialProfileMatchesNumPy` |
| M16–M23 | spot: no mean, no Hann, 3 × 3 window, magnitude weights, no snap, no tie tolerance, one spacing, qx only | spot and angle against NumPy and the plant |
| M24 | neighbour threshold 0.1 → 0.2 | **full set only** at `eb042aa`; caught by the committed gate since `f51d1f8` (`neighbour-3bins` committed) |
| M25 | nearDC radius 3 → 2 | `spotMatchesNumPyAndThePlant` (the refusals) |
| M26 | acos for atan2 | `angleClosedForms` (the 1e-8 rad case) |
| M27 | pixel q instead of physical | `angleMatchesNumPyAndThePlant` (the anisotropic cases) |
| M28 | radians reported as degrees | `angleClosedForms`, the angle against NumPy |
| M29, M30 | Å as 1 nm; a reciprocal axis not unwrapped | `lengthUnits` |
| M31 | crop origin not moved | `cropKeepsThePhysicalPosition`, `cropEqualsTheCroppedImage` |
| M32 | the overflow guard removed | **not caught** (the test's `Int.max × 2` wraps to −2 and fails the count anyway) → caught by `micrographRefusesASizeThatWrapsToItsPixelCount` (2³² × 2³² wraps to 0) |

## The Gate B follow-ups (on `f51d1f8`)

| Mutant | Change | Caught by |
|---|---|---|
| A | the neighbour flag's true branch removed | `spotMatchesNumPyAndThePlant` (`neighbour-3bins`) |
| B | `measure` ignores its crop | `spotOnACropIsTheSpotOfTheCroppedImage` |
| C | a case dropped from the committed manifest | `everyPinnedCaseIsLoaded` |
| D | a profile point at the bin edge, not its centre | `spectrumMembersAndRefusals` |
| E | `qPerNanometre` inverted | `spotUnitsInNanometres` |

## Equivalent mutants

- **M1f.** The Float path's chirp angle is computed in Double from n² and only then cast to Float: without the reduction the
  Double angle at n = 1009 is off by about 3e-10 rad, far below Float's 6e-8 rounding (the lane's 1009 × 1 run: 1.73e-7 of the
  maximum mutated, band 1e-5, no change). It could matter only at lengths in the hundreds of thousands; the reduction stays
  (it is the origin's code, and the apps-parity check holds it bit for bit).
- **M4d.** The exception exists because `vDSP_fft2d_zip` (Float) returns wrong values for 8 × n; `vDSP_fft2d_zipD` returned
  correct values (about 1e-15) on every small power-of-two shape tried (8 × 8, 8 × 16, 8 × 32, 8 × 64, 16 × 8, 64 × 8, 4 × 4,
  4 × 8). The Double path keeps the exception for symmetry and safety on other macOS versions; no test can catch its removal
  where the Double call is correct.

Result: every new test of slice 1a is broken by at least one mutant; 33 of 35 campaign mutants and 5 of 5 follow-ups are
caught by the committed gate, the other two argued equivalent above.

## Slice 1b (on `81ab1b7`, then the follow-ups on `120ba8c`)

A Sonnet lane (`scratchpad/mutants-1b/`), 26 mutants of section 7 plus one: unmutated 33 functions, 168 cases, green.

| Mutant | Change | Caught at `81ab1b7` by | After `120ba8c` |
|---|---|---|---|
| M1–M4, M4-d | no Friedel partner; hard edge; partner at −k−1; inverse unscaled (Float32 / Float64) | `maskedInverseMatchesNumPy` (M1 and M4-d also `aMeasuredSpotBecomesAMaskSpot`) | — |
| M5 | union as sum | **not caught** (no pinned spots overlap) | `maskIsTheMaximumOfTheSpotsOnAnOddHeight` |
| M6 | centred row index (h + 1)/2 | **not caught** (every pinned height is even) | `maskIsTheMaximumOfTheSpotsOnAnOddHeight` (h = 63) |
| M7 | `Spot.weight`'s cosine sign | **not caught** (probed only at r + e/2; `compute` had its own inline copy) | the mask now calls `Spot.weight`: `maskWeightClosedForms` (r + e/4, r + 3e/4), `maskedInverseMatchesNumPy` |
| M7b | the same in the inline copy | `maskedInverseMatchesNumPy` | (the copy is gone) |
| M8 | `maskSpot` with the width for ky | `aMeasuredSpotBecomesAMaskSpot` | — |
| H1–H5, H8–H10, H12 | ABSF sign; edge window exponent 12; 5 × 5 median; 256 angles; Butterworth exponent; threshold comparison; DC not added back; W from the windowed image; Wiener from the untapered a | `filtersMatchTheFloat64Reading` (H5 also `stepClosedForms`; H12 only on `pct5-128`) | — |
| H6 | cycle cap `<=` | full set only (`cycles3-*`) | `theCycleCapStopsTheLoop` |
| H7 | taper radius rounded down | full set only (`pct20-*`) | `theTaperRadiusRoundsHalfAwayFromZero` |
| H11 | flag disk `<=` | full set only (26 cases) | `nothingIsFlaggedOnTheDiskEdge` |
| H13 | the bilinear sample's upper clamp removed | **equivalent**: `bilinear` is called only with ρ ≤ n/2 − 1, so row = n/2 + ρ cos θ ≤ n − 1 and the clamp never binds | — |
| H14 | `delta` range to 60 | `refusalsByName` | — |
| H15 | the Float32 path's windowed image not rounded to Float32 | **no test can catch it** (a one-ulp input change, far inside the Float32 bands); its catcher is the apps-parity check, bit for bit against macTEM | — |

Result for 1b: every new test broken at least once; 26 of 27 mutants caught by the committed gate after `120ba8c` (the six
follow-ups verified on a clone), H13 equivalent, H15 caught by apps-parity.
