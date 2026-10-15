# ImagoKit slice 1: the FFT of an image, spot spacings and angles, the masked inverse, the HRTEM filters (registration)

Registered 2026-10-15, before any code, by the ImagoKit session (the coordinator's side session; the owner's decision of
2026-10-15 15:25 to start the sixth kit, image analysis). Tier 1: every number here is one a scientist reads (a lattice
spacing, an angle, a spectrum, a filtered image). What is written here is fixed before the code; a change after code starts
is an amendment, dated, with its reason (section 11).

## 1. What slice 1 makes

Three modules with code, `ImagoCore`, `ImagoFFT`, `ImagoFilters`; `ImagoAlignment` stays a placeholder whose scope section 10
registers. Public API only: nothing an app needs is `package`; no message, comment or doc names an app or the suite.
Dependencies: Foundation and Accelerate; every module builds for macOS 15 and iOS 18 (iPadOS) and the tests run on macOS.
Swift 6 language mode, strict concurrency; every value type is `Sendable`, every computation a pure function of its inputs.

Port order: **1a** (the first code): `ImagoCore` and `ImagoFFT`'s spectrum, radial profile, spot measurement, spacings and
angles. **1b**: the masked inverse FFT and `ImagoFilters`. Each part has its own frozen commit, mutants and refuter; 0.0.1 is
the first green part.

### 1.1 `ImagoCore`

- `AxisCalibration`: `scale` (physical units per pixel), `unit` (the file's string, unnormalised), `origin` (the pixel index
  of the physical zero, default 0); the physical value at index `i` is `(i − origin) · scale`; `isCalibrated` ⇔ `unit` is not
  empty and `scale` is finite and positive (an uncalibrated axis computes in pixels, unit "px").
- `Micrograph`: `width`, `height`, `pixels: [Float]` (row-major, x fastest, row 0 first), `x`, `y: AxisCalibration?`. The
  initialiser throws on `width·height ≠ pixels.count`, a size under 1, or a size whose product overflows. Non-finite pixels
  are accepted by the type and refused by the computations that cannot carry them (the filters).
- `PixelCrop` (`row`, `col`, `height`, `width`, half-open) and `Micrograph.cropped(_:)`: the pixels of the region, each axis's
  scale and unit kept, its origin moved by the offset (so a physical position is the same in the crop); `nil` when the crop
  is empty or leaves the image.
- `LengthUnit`: nanometres per unit for the length spellings the image formats write (m, mm, µm in its three spellings, nm,
  Å in its spellings and "a"/"angstrom", pm; case and spaces folded), `nil` otherwise; the reciprocal-unit naming ("1/nm",
  "1/px"; an axis already reciprocal, "1/nm", transforms to "nm", never "1/1/nm").
- `FFT2D`: the exact N-point 2D DFT of a complex `ny × nx` row-major array, for every `nx, ny ≥ 1`, never padding the data.
  Forward unnormalised; inverse scaled by `1/(nx·ny)` unless the caller asks otherwise (NumPy's `norm="backward"`).
  **Two precisions, both public, chosen by the caller, never by the hardware:**
  - `FFT2D<Float>` (the default for the products): Accelerate's vDSP, the algorithm of mac4DSTEM's `FFT2D` at `7a0f727f`
    unchanged: `vDSP_fft2d_zip` for powers of two ≥ 16 on both axes, else axis by axis: radix-2 `vDSP_fft_zip`, exact radix-3
    and radix-5 plans (Float twiddles), `vDSP_DFT_Execute` where vDSP supports the length (f·2ⁿ, f ∈ {1, 3, 5, 15}), and an
    exact Bluestein (chirp-z) transform otherwise, its chirp built from n² reduced modulo 2N in Double.
  - `FFT2D<Double>`, the **reference path**: the same decomposition in Double (`vDSP_fft_zipD`, `vDSP_DFT_zop_CreateSetupD`,
    the radix-3/5 plans and Bluestein with Double twiddles), plus a direct O(N²) DFT in Double used only by the tests as an
    independent check. The small-power-of-two exception (the 2D call is not used below 16 on either axis: measured to return
    wrong values for 8 × n, mac4DSTEM 2026-09-15) applies to both.
- `Precision`: `.float32` / `.float64`, the parameter every product in `ImagoFFT` and `ImagoFilters` takes (default
  `.float32`), recorded on the product.

### 1.2 `ImagoFFT`

- `PowerSpectrum.compute(_ image:, crop:, precision:)`: `F = fftshift(fft2(x))` of the (cropped) pixels as given — nothing
  windowed, padded or mean-removed; `magnitude = |F|`, `power = |F|²`, centred, the zero frequency at `(h/2, w/2)` (integer
  division, odd and even alike); `axisX[i] = (i − w/2) · (1/(w·dx))` (that order of operations, `fftshift(fftfreq(w, dx))`),
  `axisY` likewise; `spacingX = 1/(w·dx)`, `spacingY = 1/(h·dy)`; units "1/<unit>" (or the axis's own unit for an axis already
  reciprocal), "1/px" uncalibrated (d = 1). The arrays are `[Float]` on the Float32 path and `[Double]` on the Float64 path
  (two stored properties, one `nil`; a `Double` accessor converts either).
- `RadialProfile.compute(_ spectrum:, width:)`: for the signed indices `(i, j) = (c − w/2, r − h/2)`,
  `|q| = √((i·Δq_x)² + (j·Δq_y)²)`; bin `k = ⌊|q| / wb⌋`, `wb` given or `max(Δq_x, Δq_y)`; per bin the mean power
  (accumulated in Double) and the pixel count; the DC in bin 0, counted there; an empty bin has mean 0 and count 0. Refused:
  axes with different units, a width not positive and finite, a width giving more bins than the spectrum has pixels.
  `RadialProfile.points` gives the bins k ≥ 1 with a pixel, each at its centre `(k + ½)·wb`.
- `SpotMeasurement.measure(_ image:, crop:, clickRow:, clickColumn:, precision:)`, the estimator macTEM registered as V1-6
  slice C′, every step fixed: the pixels minus their mean (Double), times `hanning(h) ⊗ hanning(w)` with
  `hanning(n)[k] = 0.5 − 0.5·cos(2πk/(n − 1))` (`[1]` for n = 1), transformed; the centred power (Double); **snap**: the
  largest power within ±2 bins of the click (clipped to the spectrum), every bin within `1e-6` relative of that maximum
  counting as largest, the first of them in row-major order; **refusals**: the snapped bin within Euclidean index distance 3
  of `(h/2, w/2)` (`nearDC`), or within 2 bins of the spectrum's edge (`nearEdge`), a click outside the spectrum, axes in
  different units; **centroid**: the power-weighted mean row and column over the 5 × 5 window around the snapped bin, no
  background; **q**: `qx = (col_c − w/2)/(w·dx)`, `qy = (row_c − h/2)/(h·dy)`, `q = √(qx² + qy²)`, **d = 1/q** in the image's
  unit; **neighbour flag**: another local maximum (above each of its 8 neighbours by more than `1e-6` relative) within
  Euclidean distance 3 of the snapped bin, other than the snapped bin, with power above `0.1 ×` the spot's. The spectrum shown
  (1.2's `PowerSpectrum`) stays unwindowed; the window belongs to the measurement.
- `SpotMeasurement` carries the click, the snapped bin, the centroid, `qx`, `qy`, `q`, `d`, `units`, `reciprocalUnits`,
  `neighbour`, the precision, and **`dNanometres`** / **`qPerNanometre`** when the unit is a length (`LengthUnit`).
- **New in the kit, the angle between two spots**: `SpotMeasurement.angle(_ a:, _ b:)`:
  `θ = atan2(|qa_x·qb_y − qa_y·qb_x|, qa_x·qb_x + qa_y·qb_y)` in degrees, in [0°, 180°] (atan2, not acos: acos loses the small
  angles), from the physical q vectors, so an anisotropic sampling is accounted for; refused when the two spots carry
  different reciprocal units or either q is zero. Also `LatticePair.measure(a, b)`: both spacings, the angle, and the ratio
  `d_a/d_b`, the numbers an indexing step reads.
- `MaskedInverseFFT.compute(_ image:, spots:, includeFriedelPartners: true, precision:)`: each spot a centre `(kx, ky)` in
  frequency-index units from the centre, a radius `r` and a raised-cosine edge `e`, both in index units:
  `m = 1` for `d ≤ r`, `½(1 + cos(π(d − r)/e))` for `e > 0, r < d < r + e`, else 0, `d` the index distance; each spot's
  Friedel partner `(−kx, −ky)` added; the union is the maximum; the result is the real part of
  `ifft2(ifftshift(fftshift(fft2(x)) · M))`, and the largest |imaginary part| is reported. `SpotMeasurement.maskSpot(radius:
  edge:)` makes the mask spot of a measured spot (centroid minus `(w/2, h/2)`).

### 1.3 `ImagoFilters`

`HRTEMFilter.filter(_ image:, parameters:, precision:)`: the average background subtraction filter (ABSF) and the Wiener
filter after Kilaas (1998), as D. R. G. Mitchell's `HRTEM_Filter.s` v4.0 (2019-11-01) describes them, in the steps A1–A6 macTEM
registered as V1-6b to V1-6d (ADR 014–016 there), for a square, even `n × n` image, `n ≥ 32`; `r` a pixel's distance from
`(n/2, n/2)`; every spectrum the magnitude of the fftshifted forward FFT:

- A1 `E = 1/(1 + 0.414 (r/0.4n)²⁴)`; `M = |FFT(I·E)|`; `S` = the 3 × 3 median of `M`, edges replicated
  (`scipy.ndimage.median_filter(size=3, mode="nearest")`).
- A2 `t = 1`; while `d < delta` and `passes < cycles`: `A = rot(M)`; `c = max((S − A)/S · 100, 0)`; the flags gain every pixel
  with `c > 100 − t` and `r < n/2` (never where `S = 0`); `M` takes `A` on every flagged pixel; `d = 100 · flagged / n²`;
  `t += step`. `rot(X)`: X sampled bilinearly at `(n/2 + ρ cos θ, n/2 + ρ sin θ)` (row, column), clamped to the image
  (`map_coordinates(order=1, mode="nearest")`), for 512 angles `θ = 2πk/512` and `ρ = 0 … n/2 − 1`, averaged into `P(ρ)`;
  `rot(X)` at a pixel is `P` linearly interpolated at `r`, held at `P(n/2 − 1)` beyond.
- A3 `M = rot(M)`; `W = |FFT(I)|`.
- A4 `a = max((W − M)/W, 0)` (0 where `W = 0`); `R = round(n/2 · pct/100)` half away from zero; where `r < R`,
  `a = (1 − r/R)·0.1 + 0.05`; `w = max(a (W + M)/W, 0)` from the clamped, tapered `a`.
- A5 with smoothing, `a` and `w` × `1/(1 + 0.414 (r / (Ro · n/2))^(2·order))`.
- A6 `ABSF = real(IFFT(FFT(I)·a))`, `Wiener = real(IFFT(FFT(I)·w))`.
- Parameters and ranges (out of range refused by name, never clamped): delta 5 ∈ [0, 50], step 2 ∈ [1, 10], cycles 99 ∈
  [1, 99], smoothing on, order 4 ∈ [1, 20], Ro 0.3 ∈ [0.01, 3], pct 0 ∈ [0, 20]. Refused: non-square, odd, under 32, a
  non-finite pixel.
- Precision on the Float32 path, as macTEM's V1-6c: the mean is removed before each transform and the DC bin set in Double
  (`|Σ I|`, and `mean · mask(DC)` added back after the inverse); every mask, spectrum, median and the loop in Double. On the
  Float64 path the same algebra in Double throughout.
- The result carries both images, every stage (M before the loop, S, pass 1's c, the final M, W, both masks, the flags with the
  pass each was first flagged, the passes, flags per pass, the final d), the parameters and the precision.

The DC is never replaced (rot(M) at r = 0 samples only the centre), so `a(0) = 1 − |Σ I·E| / |Σ I|` and the ABSF output's mean
is the input's times that: the script's behaviour, kept and documented on the type. Nothing of Mitchell's script is copied (it
states no licence); this is a functional re-implementation of the steps above.

## 2. The oracles and pins

- **NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0** on CPython 3.14.7 (`tools/oracle/references-lock.txt`, wheels with
  `--no-deps`, `pip check` clean; freeze `References/oracle-venv-freeze.txt`): `numpy.fft.fft2`, `ifft2`, `fftshift`,
  `ifftshift`, `fftfreq`, `numpy.hanning`; `scipy.ndimage.median_filter`, `map_coordinates`. scikit-image is slice 2's oracle
  (`registration.phase_cross_correlation`) and is installed now so slice 2 runs on the same environment. These are the releases
  macTEM's harnesses were measured on, so the kit's pins and macTEM's planted references are made by the same code.
- **macTEM's parity harnesses at `3d17311` as planted truth**: `tools/fft-parity` (`make_fixture.py`, `reference.py`: 19 cases
  of slices A, B, D), `tools/spot-parity` (38 cases), `tools/hrtem-filter-parity` (`plants.py`, `reference.py`: 41 cases, SEED0
  20261015). `tools/slice-1/pins.py` extracts them with `git -C ../macTEM show 3d17311:<path>` into `.build/apps/` (never from the
  working tree, never committed here), runs them with the kit's interpreter under `../tools/memcap.sh` (one at a time), and
  converts their planted images, their NumPy references and their plant self-checks into the kit's fixtures. The harness
  scripts' own plant checks (numpy agrees with the plant before anything counts) must pass, or `pins.py` exits 1.
- **The angle** has no harness: `pins.py` plants it (section 5.4) with its own NumPy reference of the estimator.
- **Pins before code.** `pins.py` writes `Tests/ImagoFFTTests/Fixtures/` and `Tests/ImagoFiltersTests/Fixtures/`: per case the
  input (`.f32`, little-endian), the reference arrays (`.f64`), and `pins.json` with the scalars, the versions, and each
  file's sha256. **Committed fixtures are held under 8 MB**: the full case set's references are written for every case whose
  inputs and outputs fit; the cases that do not (the 250 × 250 and 256 × 256 references) are regenerated into `.build/pins-full/`
  by `pins.py --full` and checked by `tools/slice-1/full-parity.sh` (the same Swift tests pointed at that folder by an
  environment variable), run and logged at every frozen commit; CI runs the committed subset. The split is printed by
  `pins.py` and recorded in the amendment that lands the pins.
- **The owner's data (agreement only, no truth)**: one HRTEM image, `$MACEM_DATA/tem/Camera Ceta 1623 1.05 Mx 20260605
  Ceta.tif`, read with `tifffile` (in the venv) into `~/Library/Caches/ImagoKit/real-data/` (outside the repository: the
  suite rule), its centred-square crop's spectrum, radial profile, three spots clicked at the strongest maxima, both
  filters; the Swift test is skipped by name when the export is absent. A manifest line in `../data/README.md` (for the
  coordinator to commit).

## 3. What is bit-identical and what has a tolerance

| Comparison | Claim |
|---|---|
| Float32 path against macTEM `3d17311`'s `PowerSpectrum`, `RadialProfile`, `SpotMeasure`, `MaskedInverseFFT`, `HRTEMFilter` on every planted case (the apps-parity check, section 6) | **bit for bit**: the same code in the same order of operations; any reordering is an amendment before it lands |
| Frequency axes and spacings, both paths, against `fftshift(fftfreq(n, d))` | **bit for bit** |
| Crop: the spectrum of a crop against the spectrum of the cropped image | **bit for bit** |
| Float32 spectrum (magnitude, power) against NumPy float64 | within **1e-5 of its maximum** (macTEM measured 5e-8 to 1e-7); bins below 1e-3 of the maximum measured and printed, not gated (macTEM: up to 1.9e-2 relative) |
| Float64 spectrum against NumPy float64 | within **1e-12 of its maximum**; against the kit's direct Double DFT within 1e-12 on sizes ≤ 64 |
| Radial profile, Float32 | every bin within **1e-5 of the profile's maximum**, and every bin k ≥ 1 within **1e-5 of the largest non-DC bin** (amendment 3); counts **equal** |
| Radial profile, Float64 | every bin k ≥ 1 within **1e-11 of the largest non-DC bin**; counts equal |
| Spot, Float32: snapped bin; centroid; d | snapped bin **equal** to NumPy's; centroid within **1e-6 px**; d follows |
| Spot, Float64 | centroid within **1e-10 px** of NumPy's |
| Masked inverse, Float32 | real image within **1e-5 of its maximum**; imaginary residual below **1e-5** of it |
| Masked inverse, Float64 | within **1e-12** of the maximum; imaginary below 1e-12 |
| HRTEM filters, Float32 (macTEM V1-6d's bands) | both outputs within **1e-5** of the float64 output's maximum; W, M, a, w W-weighted (Σ\|Δ\|·W / Σ\|x\|·W over r > 0) within **1e-4**; the ABSF mean ratio within **1e-6** of the float64 mask at the DC; **flags per bin**: δ = 3 × the case's largest pass-1 \|Δc\| over the disk, and at the first pass where any bin differs every differing bin's float64 margin is at most δ; pass counts printed |
| HRTEM filters, Float64 | both outputs within **1e-10** of the reference's maximum; flags and pass counts **equal** to the reference's on every case whose float64 margins are all above 1e-9 points (cases with a smaller margin named, flags compared with that margin) |
| Angle between spots, both paths, against the NumPy evaluation of the same estimator | within **1e-6 degrees** (Float32) and **1e-9 degrees** (Float64) |

"Within t of its maximum" means `max |got − want| ≤ t · max |want|`. A difference beyond a registered bound is a failure of the
port, never a tolerance to widen. Truth claims (section 5) are separate and do not depend on the oracle.

## 4. The Swift tests

Swift Testing, one test target per module with code (`ImagoCoreTests`, `ImagoFFTTests`, `ImagoFiltersTests`). Parameterised
over the pins' cases (`@Test(arguments:)`), so each case and check is a named result. The Float64 path runs every case the
Float32 path runs. Closed-form unit tests beside the pins: `fftfreq` and `hanning` values, the A1 edge window, the A4 taper and
the A5 Butterworth against their formulas, the mask's raised cosine at `r`, `r + e/2`, `r + e`, the angle of (1, 0) and (0, 1)
(90°), of parallel and antiparallel vectors (0°, 180°), of vectors 1e-8 rad apart (atan2's reason), every refusal by name.

## 5. Planted truth (independent of NumPy)

### 5.1 Spectrum and profile (macTEM fft-parity, slice A and B cases)

Sums of plane waves `DC + A cos(2π(kx·x/w + ky·y/h) + φ)` at bin-centred and off-bin frequencies, scales 0.0123 nm (x) and
0.0246 nm (y), sizes exercising every FFT path: 128 × 96 (radix-2 and vDSP DFT), 125 × 243 (radix-5, radix-3), 127 × 50 and
250 × 250 (Bluestein), seeded noise (N(0, 10), seed 20261012), an uncalibrated 125 × 243, crops odd and even. Truth: the DC bin
equals `DC·w·h`; every bin-centred wave holds `A·w·h/2` at `(h/2 + ky, w/2 + kx)` and its Friedel partner, each a local maximum,
the axis there equal to `k/(n·d)`. The planted ring `100 exp(−r²/2σ²) cos(2π q₀ r)` peaks in the bin holding `q₀`.

### 5.2 Spot spacings (macTEM spot-parity, 38 cases)

Lattices of amplitude 1 on DC 10 with a weak second lattice (0.3, off-bin), sizes 64, 128, 256, 512, 256 × 128, 96 × 128, at 5
bins and 0.1, 0.25, 0.4 Nyquist, directions 20° and 70°, sub-bin offsets 0, 0.25, 0.5, clicks 0–2 bins off; DC 1000 cases;
noisy cases (σ 1, N = 256 and 512, seeds 20261012–20261019); one refusal (`nearDC`) and one neighbour case. Truth: **clean
cases |d − d₀|/d₀ < 0.2 %, noisy < 0.5 %** (macTEM measured ≤ 0.09 % clean), on both paths; the refusal refused and nothing
else; the neighbour flag set exactly where planted.

### 5.3 Masked inverse (macTEM fft-parity, slice D cases)

A strong lattice A (100) and a weak lattice B (20), off-bin, at 128 × 96 and 127 × 50; masks on A (r 2.5, e 2) and on B (r 3,
e 0). Truth: the result correlates above 0.99 with the planted masked lattice and below 0.1 in magnitude with the other.

### 5.4 Angles (new, `pins.py`, seed 20261016)

Two lattices of amplitude 1 on DC 10, at 256 × 256 and 512 × 384 (Bluestein on one axis), isotropic (0.0123 nm both axes) and
anisotropic (0.0123 / 0.0246 nm), with planted angles 30°, 60°, 90°, 109.47° (the {111}/{1-11} angle), 145°, each lattice at
0.15–0.35 Nyquist with sub-bin offsets 0, 0.25 and 0.5; noise-free and σ 0.5 (four seeds). Truth: the planted angle
`θ₀ = atan2(|q₁ × q₂|, q₁·q₂)` of the planted physical q vectors: **|θ − θ₀| ≤ 0.1° noise-free and ≤ 0.3° noisy**, on both paths;
the spacing ratio `d₁/d₂` within 0.4 % of the planted one (twice the spacing band). The bands follow from 5.2's spacing bound
(0.2 % in |q| ⇒ ≈ 0.1° for a spot at 0.15 Nyquist); a case outside them is a failure, not a band to widen. The NumPy evaluation of
the estimator is pinned beside the truth (section 3's row).

### 5.5 HRTEM filters (macTEM hrtem-filter-parity, 41 plants)

Float32 n × n images: DC 1000, a lattice of three cosine waves (amplitudes 20–80) and their second harmonics inside the
Butterworth's `Ro·n/2`, an isotropic random-phase continuum (a Gaussian ring at 0.08 n and a k^−1.5 power law, rms 15), white
noise σ 5; tiers `bin`, `offbin`, `noise`, `ramp`, `nyquist` at 128, 256 and 250 (Bluestein); off-bin offsets 0, 0.25, 0.5;
one case per varied parameter; `lattice-clean` and `constant`. Truth: at the defaults, each output's energy in the region
r ≤ 0.15 n outside ± 2 bins of every planted peak (DC excluded) below the input's (background removed); the constant image
filters to its mean times the DC mask; the closed forms of 4.

## 6. The apps-parity check (the switch's guarantee)

`tools/slice-1/apps-parity/`: a harness compiled from macTEM `3d17311`'s TEMCore sources (`PowerSpectrum.swift`,
`RadialProfile.swift`, `SpotMeasure.swift`, `MaskedInverseFFT.swift`, `HRTEMFilter.swift`, `FFT2D.swift`,
`CalibratedImage.swift`, `FFTRoomMath.swift` for `CalibrationUnitConversion` if it is needed, by `git show` into `.build/`) and
from this kit, run on every planted case, comparing the Float32 path's outputs **bit for bit** (`bitPattern` equality of every
Float and Double). This is what lets macTEM delete its originals at its switch with its own pinned tests as the second proof.
It runs at every frozen commit of slice 1 and is logged in `docs/status.md`; it needs macTEM checked out beside the kit and is
not part of CI.

## 7. Mutants (planned; each new test broken once)

On a copy of the frozen commit (never the working tree), each must turn at least one named test red:
- FFT2D: Bluestein chirp without the mod-2N reduction (caught on 250 × 250 Float64 only: the Float32 path's error stays inside
  1e-5 — the mutant's catcher is the Float64 band, said so here); radix-3 butterfly sign flipped; inverse not scaled; the
  small-power-of-two exception removed (8 × 32 case).
- Spectrum: no shift; shift `(n + 1)/2` on odd n; x's scale for both axes; power = magnitude; fftfreq over `n − 1`; crop origin
  not moved.
- Profile: bins in frequency pixels; Δq_x on both axes; round for floor; DC dropped; sum for mean.
- Spot: no mean removal (the DC 1000 case); no Hann; 3 × 3 window; magnitude weights; no snap; one spacing for both axes;
  qx only; the snap tie tolerance removed; the neighbour threshold 0.1 → 0.2.
- Angle: acos for atan2 (the 1e-8 rad case); pixel instead of physical q (the anisotropic cases); radians reported as degrees.
- Masked inverse: no Friedel partner; hard edge; partner at `n − k − 1`; unscaled inverse.
- Filters: ABSF sign; DC removed; one flag moved; `a` scaled by 1.001; the median window 5 × 5; the rotational average over 256
  angles; the Butterworth exponent `order` for `2·order`; the cycle cap off by one; the taper radius rounded down.

The results go to `docs/slice-1-mutants-<date>.md`; a mutant no test catches is either an equivalent mutant (argued in that
file) or a missing test, added before the freeze.

## 8. Gates and predicted outcomes (registered now)

1. `pins.py` passes every plant self-check of the three macTEM harnesses and of 5.4 (predicted: yes; they passed in macTEM on
   the same NumPy and SciPy).
2. The gate `tools/test.sh` green on the committed fixtures, both paths; `full-parity.sh` green on the full set.
3. Every bound of section 3 holds; every truth of section 5 holds on both paths. Predicted: the Float32 numbers equal macTEM's
   measured ones (the code is the same); the Float64 path within 1e-13 of NumPy on the spectrum; the angle within 0.05°
   noise-free.
4. The apps-parity check bit for bit on every case (predicted: yes).
5. Every new test broken once by a mutant of section 7.
6. One Gate B refuter (a Sonnet subagent, independent, on the frozen commit; a second only on a named doubt); Gate D if a
   discrepancy's cause is unproven.
7. The consumer check names every public symbol of 1.1–1.3 with a value it checks.

## 9. What is not claimed

No number is validated on a microscope. The filters have never been compared with DigitalMicrograph (macTEM's badge "not
compared with Digital Micrograph" travels with them). Spot accuracy is promised only where 5.2 measured it: noise-free
≤ 0.2 %; noisy (σ 1) ≤ 0.5 % from 256 px and 0.1 Nyquist up; smaller or noisier images are measured, not promised (macTEM:
1–5 % at 64 px). A ring fainter than about 1e-6 of the DC bin is not reliable on the Float32 path (use `.float64`). An off-bin
lattice keeps only part of its amplitude through the HRTEM filters (macTEM measured 5–57 % at n = 128 half a bin off): the
method's limit, stated. Non-finite pixels are refused by the filters and propagate through the FFT products.

## 10. Slice 2's scope (registered as scope only; its own registration comes before its code)

`ImagoAlignment`: drift and stack alignment.
- **Phase cross-correlation** with sub-pixel refinement by the upsampled matrix-multiply DFT (Guizar-Sicairos, Thurman and
  Fienup, Opt. Lett. 33 (2008) 156), against `skimage.registration.phase_cross_correlation` (0.26.0: `upsample_factor`,
  `normalization` "phase" and None, `space` "real"); the shift, the error and the phase difference; masked correlation
  (Padfield 2012) only if a consumer asks. mac4DSTEM's `ParallaxAlignment` (py4DSTEM's factor-8 matrix-DFT refinement) is the
  in-suite seed to read; its numbers stay CuboKit's.
- **Sub-pixel shifts**: `scipy.ndimage.shift` (order 1 and 3, the modes "constant", "nearest", "wrap") and
  `scipy.ndimage.fourier_shift`.
- **Drift series and aligned sums**: each frame against a reference (the first, the running mean, or the previous frame,
  cumulated), outliers by a stated rule, the aligned sum and mean with the overlap mask; planted truth: a stack of one image
  shifted by known sub-pixel drifts with noise.
- Float32 (vDSP) and Float64 paths as in slice 1, with the tolerance registered; Metal only later and only with a CPU
  reference and a registered tolerance.

**Later slices, named so their homes are known:** GPA strain (`ImagoStrain`, Hÿtch, Snoeck and Kilaas 1998, planted strain
fields as truth; no Python reference is standard, so the choice of a second implementation is part of that registration);
atom-column finding (`ImagoAtoms`: Atomap is GPL-3.0, so a port is a GPL-3.0 module of its own, or a separate package, decided
there); bandpass and denoising filters (`skimage.filters.butterworth`, `difference_of_gaussians`, `scipy.ndimage`); a shared
Metal numerics layer is a suite question, not this kit's.

## 11. Amendments

### 2026-10-15 15:40 CEST, the slice-1a pins land (before any Swift code)

`tools/slice-1/pins.py` (run under `../tools/memcap.sh`, peak 169 MB) extracted macTEM `3d17311`'s fft-parity and spot-parity
scripts into `.build/apps/`, ran them with the kit's interpreter (NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0, CPython
3.14.7): every plant self-check of both harnesses passed (gate 1). The spot harness at `3d17311` has 41 cases (38 spots and 3
refusals: `nearDC` plus two `nearEdge` cases added by macTEM's slice C′ review), not the 38 its README names; all 41 are
pinned. The angle plants of 5.4: 30 cases (256 × 256 clean and noisy, 512 × 384 clean, iso and aniso, five angles); NumPy's
estimator recovers every planted angle within **0.0021°** noise-free (band 0.1°) and **0.024°** noisy (band 0.3°), spacing
ratios within 0.4 %. The noisy cases use one seed each, cycling over 20261016–20261019 (the registration's "four seeds").

**The committed split** (5.56 MB of the 8 MB budget, `Tests/ImagoFFTTests/Fixtures/slice-1a-pins.json` with every file's
sha256): fft 15 of 19 cases (all spectrum and profile cases; the 4 `ifft` cases are 1b's), power and axes only (the magnitude
is √power, recomputed by the test from the committed power, and compared as such); spot 23 of 41 (every case up to 256 × 128
pixels, the refusals and the DC 1000 cases among them); angle 4 of 30 (`a256-iso-t60-clean`, `a256-aniso-t109.47-clean`,
`a256-iso-t145-noisy`, `a256-aniso-t30-noisy`). The rest is in `.build/pins-full/` for `tools/slice-1/full-parity.sh`.

### 2026-10-15 15:42 CEST, before the first Swift line: the two precisions' types

- `FFT2D<Float>` / `FFT2D<Double>` (1.1) become two classes, **`FFT2D`** (Float32: mac4DSTEM's code at `7a0f727f` with only
  its access modifiers changed) and **`FFT2DDouble`** (the same code translated line by line to Double: `vDSP_*D` calls,
  `DSPDoubleSplitComplex`, Double twiddles). Reason: vDSP's API is not generic, and a generic wrapper would put a layer of
  protocol dispatch into the path that must stay bit-identical to the apps'. Both conform to a public protocol `FFT2DTransform`
  (`nx`, `ny`, the scalar type) for callers that want one name. The numbers and bands of section 3 do not change.
- The products store their arrays in a `SpectrumValues`-style enum (`.float32([Float])` / `.float64([Double])`) with a
  `Double` accessor, not two optional properties (same content, one source of truth).
- `SpotMeasurement` has no public `Options` (macTEM's existed only for its harness's negative controls; the kit's mutants run
  on copies of the frozen commit).

### 2026-10-15 15:45 CEST, at the first test run: the profile's second band is over bins k ≥ 1

Section 3's "within 1e-5 of the largest non-DC bin" is macTEM's `profile-away` check, which compares bins k ≥ 1 only
(`zip(mean, want).dropFirst()` in its harness): the DC bin, about 10⁶ × a ring, carries a Float32 error of about 2e-3 of the
ring and is held by the first band (1e-5 of the maximum). The first test run compared bin 0 too and failed on all five
profile cases on the Float32 path (1.9e-3); the test now reads as the source check does. A clarification of what was
registered, not a band change; the Float64 row says k ≥ 1 likewise.

### 2026-10-15 15:50 CEST: the real-data check waits for a raw HRTEM image

Section 2 named `$MACEM_DATA/tem/Camera Ceta 1623 1.05 Mx 20260605 Ceta.tif`. Read with tifffile 2026.9.20: one page, 2048 ×
2048 × 3 `uint8`, values 0–255: an 8-bit RGB render, not detector counts (the three 630 kx and 1.05 Mx HRTEM images in `tem/`
exist only as such TIFFs). A spectrum of a render is not a check of anything a user measures, so the real-data check is not
run on it. It runs when the owner adds a raw HRTEM image (Velox `.emd` or DM4, counts) to `data/tem/`; it stays agreement only,
outside the repository, skipped by name when absent. Nothing else in slice 1 depends on it.

### 2026-10-15 15:58 CEST, after the Gate B refuter on `eb042aa` (slice 1a)

The refuter (a Sonnet subagent, independent; logs in the session's scratchpad `refuter-1a/`) **upheld slice 1a with
findings, no wrong number**. Its independent numbers on 47 images the pins do not cover (251 × 97, 1999 × 1531, 2048², 1 × N,
N × 1, 1 × 1, DC 1e6, anisotropic Å, crops on the last row and column): Float64 power within 1.3e-15 of the maximum (band
1e-12), Float32 within 3.6e-7 (band 1e-5), axes bit for bit, profile counts equal, snapped bins equal, centroids within 4e-8 px
(Float32) and 1e-10 px (Float64); 64 threads on one `FFT2D` and one `FFT2DDouble` over every plan, 0 bitwise mismatches; the
port's Float32 arithmetic identical to macTEM's (and the apps-parity check, section 6, then showed 580,392 of 580,392 values
identical). Findings and what was done:
1. and 2. (should-fix) The neighbour flag's true branch and the spot's crop path were not exercised by the committed gate (a
   mutant of each passed it). Fixed in `f51d1f8`: `neighbour-3bins` and `noisy256-k10-r0` committed (6.06 MB), and
   `CoverageTests` (the crop path on both precisions, the case counts, `PowerSpectrumError.crop`, `RadialProfile.points`, the
   spacings, the nm accessors); each new test broken once (5 of 5 mutants caught).
3. **The Float32 bands of section 3 for the centroid (1e-6 px) and the angle (1e-6°) are claims on the planted cases**, not
   on every input: a lattice 2000 times stronger than a second one gave a Float32 angle 1.3e-6° from NumPy (centroids 4e-7 to
   7e-7 px); at 200:1, 1.6e-8°. Use `.float64` where that matters. The truth bands (0.1°, 0.3°) catch gross errors (pixel q on
   anisotropic data), not a 0.05° bias; the NumPy band guards the estimator.
4. The "printed, not gated" distribution of bins below 1e-3 of the maximum (section 3) is not implemented; the registered truths
   "DC bin = DC·w·h" and "A·w·h/2" hold through the NumPy match, and the consumer check asserts both on its own lattice. Open.
5. The Float32 1e-5-of-maximum band is blind next to a DC of 1e6 (a lattice peak is 4.5e-10 of the maximum); the k ≥ 1 profile
   error reached 2.3e-5 to 9.9e-4 of the largest non-DC bin there. Section 9's caveat (a ring fainter than about 1e-6 of the DC
   is not reliable on Float32) covers it; `SpotMeasurement` removes the mean and is not affected.
6. Behaviour against macTEM, none numeric: `reciprocalUnits` of an axis already in "1/…" is the unit after the "1/" (macTEM
   wrote "1/1/…"); `RadialProfile.compute` throws where macTEM returned nil; refusal payloads carry the units; a 1 × 1 image's
   default-width profile is refused (as macTEM's); `mixedUnits` compares the unit strings as written (" nm" and "nm" differ),
   while `LengthUnit` folds them for conversion. Recorded in `docs/consumers-0.0.1.md`.
7. Open, for 0.1.0's API pass: `FFT2D.transform` traps on a size mismatch (a `precondition`, now to be documented or made a
   throw); `angle(qa:qb:)` loses range beyond about 1e±150 components (normalise by the larger component); a subnormal scale
   counts as calibrated; a spot on non-finite pixels returns NaN instead of throwing; `fftfreq`, `nextPow2`, `frequencies` and
   `hanning` are public helpers to keep or hide deliberately; a doc line that the angle is unsigned (a spot and its Friedel
   partner give θ and 180° − θ).
8. Full-parity reads the manifests' absolute paths into `.build/pins-full` of the generating checkout; the case counts are now
   asserted.
9. The apps-parity harness did not exist at `eb042aa`; it landed in `3f744ca` and passed.

### 2026-10-15 16:03 CEST, after the slice-1a mutants

35 mutants of section 7 on `eb042aa` (`docs/slice-1-mutants-2026-10-15.md`): 28 caught by the committed gate, M24 by the full
set only (committed since `f51d1f8`), M1 (Double chirp unreduced), M5 (a radix-5 constant off by 7e-15) and M32 (the overflow
guard) caught by no test, M1f and M4d equivalent (argued there). Added: `doubleFFTIsAtRoundoffOnOneAxis` (1009 × 1 at 1e-14,
5 × 1 and 125 × 1 at 2e-15 of the maximum: **test bands at roundoff, tighter than section 3's registered 1e-12**, measured
unmutated at 1.9e-15, 2.5e-16, 6.2e-16) and `micrographRefusesASizeThatWrapsToItsPixelCount`; M1, M5, M32 and M24 are now caught.

### 2026-10-15 16:14 CEST, the slice-1b pins land (before any 1b code)

`tools/slice-1/pins.py` now also runs macTEM `3d17311`'s hrtem-filter-parity (`plants.py`, `reference.py`, SEED0 20261015;
41 cases, every self-check passed, 31 s, peak 534 MB under memcap) and commits the fft harness's four `ifft` cases
(`ifA`/`ifB` at 128 × 96 and 127 × 50). The HRTEM references are 191 MB for the 41 cases, so two are committed to
`Tests/ImagoFiltersTests/Fixtures/` (`slice-1b-pins.json`): **`bin-128`** (the defaults, with the energy region) and
**`pct5-128`** (the low-frequency taper), each with the input, both outputs, W, M, a, w, pass 1's c, S, each bin's first
flagged pass and the near-margin list (plus the energy region for `bin-128`); the other 39 are in `.build/pins-full/hrtem`
for `full-parity.sh`. **The committed budget rises from 8 to 10 MB** over both fixture folders (9.30 MB now): two HRTEM cases
are the least that exercises both the default path and the taper in CI. Existing fixtures regenerated byte-identical.

The HRTEM tests apply macTEM's harness checks as section 3 states them (outputs, W-weighted stages over r > 0 with the
"undefined where W off the DC is roundoff" exception, the DC ratio, the flags rule with δ = 3 × pass 1's largest |Δc| over
the disk at the first pass where any bin differs, the energy check for the noisy tiers at the defaults). On the Float64 path
the flag rule's δ is `max(3 × |Δc|, 1e-9)` points and the pass count must equal the reference's when no bin differs.
