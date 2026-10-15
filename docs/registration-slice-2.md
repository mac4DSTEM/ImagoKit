# ImagoKit slice 2: drift and stack alignment (registration)

Registered 2026-10-15, before any code, by the ImagoKit session (the coordinator's word at 16:08: "register slice 2, drift
and stack alignment"; scope first named in `docs/registration-slice-1.md` section 10). Tier 1: a drift, a shift and an aligned
sum are numbers a scientist reads (and that move every number measured on the sum). Fixed before the code; a change after
code starts is a dated amendment (section 10).

## 1. What slice 2 makes

`ImagoAlignment` (depends on `ImagoCore`; Foundation and Accelerate; macOS 15, iOS 18), Float32 and Float64 paths as in slice 1
(`Precision`, `FFT2D`, `FFT2DDouble`):

- `PhaseCorrelation.register(reference:moving:upsampleFactor:normalization:precision:) -> ImageRegistration`: the translation
  that registers `moving` onto `reference`, scikit-image's `phase_cross_correlation` (section 3.1) without masks and without
  `disambiguate`; `ImageRegistration` holds `shift` (row, column, in pixels, the shift to APPLY to `moving`), `error`,
  `phaseDifference`, the upsample factor, the normalization and the precision. Images must have equal sizes (refused by
  name otherwise); non-finite pixels are refused (scikit-image raises on NaN after the fact; the kit refuses before).
- `ImageShift.shift(_:by:order:mode:cval:precision:) -> Micrograph`: SciPy's `ndimage.shift` with `order` 0 or 1 and `mode`
  "constant", "nearest", "wrap", "reflect", "mirror", "grid-constant", "grid-wrap" (section 3.2); `order` 3 (the cubic
  B-spline with its prefilter) is slice 2b, registered there.
- `ImageShift.fourierShift(_:by:precision:) -> Micrograph`: `real(ifft2(ndimage.fourier_shift(fft2(x), shift)))` (3.3), the
  exact sub-pixel shift of a periodic image.
- `DriftSeries.estimate(_ frames: [Micrograph], reference:upsampleFactor:normalization:precision:) -> [ImageRegistration]`:
  every frame registered against `.first` (frame 0), `.mean` (the mean of the unshifted frames) or `.previous` (frame k − 1,
  the shifts cumulated: s_k = Σ_{i ≤ k} r_i, r_i frame i against frame i − 1); the first frame's shift is (0, 0) for
  `.first` and `.previous`.
- `AlignedSum.sum(_ frames:, shifts:, method:precision:) -> AlignedStack`: each frame shifted by its shift (`.fourier`, or
  `.bilinear` = order 1, mode "constant", cval 0) and summed; for `.bilinear` the per-pixel count of frames whose shifted
  footprint covers the pixel (the overlap mask: a frame covers an output pixel when its source coordinate lies in
  [0, n − 1] on both axes), and the mean over that count (0 where it is 0); for `.fourier` the count is the frame count
  everywhere (stated: periodic, edges wrap).

## 2. The oracles and pins

- **scikit-image 0.26.0** `skimage/registration/_phase_cross_correlation.py` (sha256 c37ea680db40565e…, in the pinned venv):
  `phase_cross_correlation` at line 202 (the algorithm, lines 325–418), `_upsampled_dft` 16, `_compute_phasediff` 93,
  `_compute_error` 106. **SciPy 1.18.1** `scipy/ndimage/_interpolation.py` (sha256 0e834f3b96ddf80a…) `shift` at 659;
  `scipy/ndimage/_fourier.py` (sha256 4a8018471edac7b4…) `fourier_shift` at 253. **NumPy 2.5.4** `fft`. CPython 3.14.7.
  `tools/slice-2/pins.py` asserts the three sha256 values and writes `Tests/ImagoAlignmentTests/Fixtures/` before any Swift
  code (committed under the kit's 10 MB budget together with slice 1's, or a separate 4 MB budget named in the amendment that
  lands them), the rest in `.build/pins-full/alignment`.
- **In-suite seed, read-only:** mac4DSTEM `7a0f727f` `mac4DSTEM/Core/Analysis/ParallaxAlignment.swift` (`correlationShift`,
  line 535: py4DSTEM's factor-8 matrix-DFT refinement of the parallax method) is read for the Swift structure of the
  upsampled DFT; its numbers stay CuboKit's and mac4DSTEM's. mac4DSTEM's `FFT1D` is not needed (the matrix DFT is two
  matrix products).

## 3. The definitions, pinned

### 3.1 Phase correlation (scikit-image's, lines 325–418)

For real `reference` R and `moving` M of one shape (h, w), on the chosen precision's FFT:
1. `F_R = fft2(R)`, `F_M = fft2(M)`; `P = F_R · conj(F_M)`; with normalization "phase", `P /= max(|P|, 100·eps)`, eps the
   precision's machine epsilon (Float32: 2⁻²³; Float64: 2⁻⁵²), as scikit-image takes it from the product's dtype.
2. `CC = ifft2(P)`; the integer peak `(r₀, c₀)` = the first maximum of `|CC|` in row-major order (NumPy's argmax);
   midpoints `trunc(h/2)`, `trunc(w/2)`; a peak index above its midpoint has the axis size subtracted (so the shift is in
   (−n/2, n/2]).
3. `upsampleFactor` u = 1: shift = (r₀, c₀) as wrapped; `src_amp = Σ|F_R|²/(h·w)`, `target_amp = Σ|F_M|²/(h·w)`, `CCmax =
   CC[peak]`.
4. u > 1: `shift = round(shift · u)/u` (NumPy's round, half to even); region size `S = ceil(1.5·u)`; `dftshift = trunc(S/2)`;
   offsets `dftshift − shift·u`; `CC′ = conj(upsampledDFT(conj(P), S, u, offsets))` where `upsampledDFT` applies, axis by axis
   from the last, the kernel `exp(−2πi · (k − offset) · fftfreq(n, u))` for k = 0 … S − 1 (the matrix-multiply DFT of
   Guizar-Sicairos, Thurman and Fienup, Opt. Lett. 33 (2008) 156); the first maximum of `|CC′|` in row-major order, minus
   `dftshift`, divided by u, added to the shift; `src_amp = Σ|F_R|²`, `target_amp = Σ|F_M|²` (**not** divided by the size
   in this branch: scikit-image's quirk, reproduced and named on the type); `CCmax = CC′[peak]`.
5. An axis of length 1 has shift 0. `error = √|1 − |CCmax|²/(src_amp·target_amp)|`; `phaseDifference = atan2(Im CCmax,
   Re CCmax)`.

The sign: the returned shift, applied to `moving` (`ndimage.shift(moving, shift)` or `fourier_shift`), registers it onto
`reference`; equivalently, `moving` is `reference` displaced by −shift.

### 3.2 `ndimage.shift`, orders 0 and 1

Output pixel (i, j) takes the input at `(i − s_r, j − s_c)`; order 1 is bilinear on the four neighbours, order 0 the nearest
(SciPy's rounding of the coordinate, half up after the mode's mapping, read from `_interpolation.py` and the C source's
behaviour at the pins: the pins decide ties and the kit matches them). Modes as SciPy 1.18.1 documents them: "constant"
(points outside [0, n − 1] take cval; order 1 interpolates toward cval only for "grid-constant"; "constant" for order ≤ 1
gives cval outside the input's extent), "nearest", "wrap" (SciPy's "wrap" with its documented period n − 1 behaviour, kept
as SciPy has it, and "grid-wrap" with period n), "reflect" (half-sample symmetric), "mirror" (whole-sample symmetric). Each
mode's mapping is pinned by cases at the edges; a case where the kit and SciPy differ is a failure, not a choice.

### 3.3 Fourier shift

`fourier_shift(F, s)` multiplies `F[k_r, k_c]` by `exp(−2πi (s_r·f_r + s_c·f_c))`, `f = fftfreq(n)` per axis (SciPy's
definition for a full complex input, n = −1); the kit returns `real(ifft2(·))` and reports the largest |imaginary| part
(nonzero at Nyquist for an even size and a non-integer shift, by the definition; stated, not hidden).

### 3.4 Bands

| Comparison | Claim |
|---|---|
| Float64 shift against scikit-image, every case | **equal** (both on the 1/u grid); a case whose two largest |CC| or |CC′| values lie within 1e-12 relative is named in the pins and excluded from equality, its shift then within 1/u |
| Float64 error and phase difference | within **1e-10** absolute |
| Float32 shift against scikit-image | **equal** except at near-ties, which the pins name: every case whose two largest values lie within 1e-5 relative (the Float32 FFT's measured error relative to the maximum, slice 1) is excluded from equality and held within 1/u |
| Float32 error, phase difference | within **1e-5** |
| `ndimage.shift` order 0 and 1, Float64 / Float32, against SciPy | within **1e-12** / **1e-6** of the image's maximum |
| Fourier shift, Float64 / Float32, against NumPy + SciPy | within **1e-12** / **1e-5** of the maximum |
| Drift series and aligned sums | as their parts; the cumulative `.previous` series equals the cumulated pairwise shifts **exactly** (the same additions in the same order) |

## 4. Planted truth (independent of scikit-image)

- **Shifts:** a 256 × 256 HRTEM-like image (three lattices of amplitude 20–80 on DC 1000, the slice-1 HRTEM plant generator,
  seed 20261017) Fourier-shifted by known sub-pixel shifts (|s| from 0.1 to 40 px, 24 vectors including half-pixel and
  near-midpoint ones) — exact for a periodic image — then noise σ = 0, 5 and 20. Truth: |ŝ − s| ≤ 1/(2u) + 0.01 px
  noise-free at u = 20 and 100; ≤ 0.05 px at σ = 5 and ≤ 0.15 px at σ = 20 (u = 20); predictions measured in NumPy at pin
  time and printed, the bands fixed here.
- **Drift:** stacks of 12 frames of the same plant with a random-walk drift (steps N(0, 0.7 px), seed 20261018) and noise σ
  = 5: the recovered drift within 0.05 px of the planted one for `.first` and `.mean`; `.previous` accumulates error, its
  bound stated as 0.05·√k px for frame k.
- **Aligned sum:** the `.fourier` aligned mean of the noise-free drifted stack equals the unshifted image to 1e-6 of its
  maximum (Float64); the `.bilinear` mean correlates above 0.999 with it inside the full-overlap region; the overlap count is
  exactly the number of frames covering each pixel (a closed form from the planted shifts).
- **Sizes:** odd and even, 255 × 256, 250 × 250 (Bluestein), 64 × 32, one axis of length 1 (shift 0 there).

## 5. Mutants (planned; each new test broken once)

The midpoint without trunc; the wrap at `>=` instead of `>`; round half away instead of half to even; the region size
`ceil(1.5u)` as `1.5u`; `dftshift` not subtracted; the kernel's sign; the upsampled branch's amplitudes divided by the size;
the error without √; the phase difference as atan2(re, im); "phase" normalization without the 100·eps floor; the shift
returned with the opposite sign; bilinear weights swapped between axes; "reflect" as "mirror"; "wrap" with period n where
SciPy uses n − 1 (or the converse, whichever the pins show); the Fourier shift's sign; the overlap count off by one at the
edge; `.previous` not cumulated.

## 6. Gates and predictions

1. `pins.py` asserts the sha256 values and the planted truths in NumPy (predicted to hold: phase correlation at u = 20 is
   accurate to about 0.025 px noise-free on a lattice image).
2. The gate green on both paths; full parity green; the consumer check names every public symbol.
3. Every new test broken once; one Gate B refuter on the frozen commit; Gate D if a cause is unproven.

## 7. What is not claimed

Rotation, scale, shear and non-rigid (per-region) drift; masked phase correlation (Padfield 2012) and `disambiguate`; order 2
to 5 interpolation (slice 2b registers order 3); Metal paths. A registration between frames that share no structure returns
a number with a large `error`: the kit reports it, it does not refuse it.

## 8. The apps

Neither app aligns stacks today (macTEM's "drift-corrected stack sums" are a v1.1 table-stakes item in the suite ROADMAP;
mac4DSTEM's parallax alignment is CuboKit's). The first consumer is macTEM's stack sum, when it is built on 0.1.x.

## 9. Order

Slice 2 after slice 1's 0.0.2; slice 2b (order-3 splines) when a consumer needs it.

## 10. Amendments

### 2026-10-15 17:09 CEST (amendment 1), the slice-2 pins land

`tools/slice-2/pins.py` (run under `../tools/memcap.sh`, peak 0.3 GB, 5 s) asserts the three sha256 values of section 2 and the
pinned versions (NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0, CPython 3.14.7), writes the full set to
`.build/pins-full/alignment/` (32.1 MB, 330 registration cases) and a committed subset to
`Tests/ImagoAlignmentTests/Fixtures/` (**3.96 MiB of a separate 4 MiB budget**, 81 files with every file's sha256 in
`slice-2-pins.json`; slice 1's 9.3 MB of 10 is untouched): 68 registration cases, the shift and Fourier-shift pins, and the
96 × 96 drift stack. What the pins showed, and what changes in this registration because of it:

1. **The planted image.** Section 4's plant is the lattices (three, amplitude 20 to 80 on DC 1000, integer frequencies up to
   0.075·min(h, w), each with a second harmonic at half amplitude, seed 20261017) plus an amorphous continuum; the continuum
   is not the slice-1 generator's. Its power-law spectrum (amplitude k^-1.5, rms 15) holds nearly all its power in the first
   few frequency bins, and phase correlation, which gives every frequency one weight, then fails at σ = 5 (worst error 1.1 px
   on the 24 vectors, 19 beyond 0.15 px). The continuum is now random-phase with amplitude (1 + (q/0.10)²)⁻¹, q in cycles per
   pixel, rms 30, band-limited to |f| ≤ 0.4 on both axes (so no Nyquist content and the Fourier shift is exact), and `plant()`
   is the one generator for every size. (0.10 and the exponent 2 are the values at which the bilinear mean's correlation of
   section 4 reaches 0.999 on both stacks; exponent 1 gave 0.9977.) Noise is added independently to the reference and to each
   moving image; the moving image is `fourier_shift(base, −s)`, so the registered shift is +s.
2. **Planted truth, measured in NumPy** (24 vectors, 256 × 256; u = 20 unless stated): noise-free worst |ŝ − s| **0.020** px
   (band 0.035), at u = 100 **0.010** (band 0.015), at u = 1 0.500 (grid); σ = 5: **0.030** (band 0.05); σ = 20: **0.050**
   (band 0.15); u = 100 at σ > 0 is printed, not gated (0.010 and 0.050). A 128 × 128 noise-free plant was tried for the
   committed subset and reached 0.020 at u = 100 (the roundoff bins above the band limit weigh twice as much): the committed
   plant cases are 256 × 256.
3. **Drift stack** (12 frames, random-walk steps N(0, 0.7²) seed 20261018, σ = 5): committed at 96 × 96, full set also 256 × 256.
   Worst recovered |error| over the band: `.first` 0.93 / 0.48 (96² / 256²), `.previous` 0.94 / 0.42, and **`.mean` 0.93 of a
   band that is now 0.20 px, not 0.05**: the mean of a drifting stack is a smeared reference, scikit-image itself returns up
   to 0.185 px error on frames 6 and 8 (96²; 0.147 at 256²; the same centred on the mean error, 0.13 to 0.175), at every size
   tried. Section 4's "`.first` and `.mean` within 0.05 px" becomes: `.first` 0.05, `.mean` 0.20, `.previous` 0.05·√k. The
   sums use the planted drift itself as the shifts: the Fourier aligned mean equals the unshifted image to **6.5e-8** (96²) and
   6.8e-8 (256²) of its maximum, the bilinear mean correlates **0.99938** (96²) and **0.99917** (256²) with it inside the
   full-overlap region (8,463 and 63,503 pixels).
4. **The near-tie rule is measured, not 1e-5.** Section 3.4's premise (the Float32 FFT's error relative to the maximum is
   1e-5) holds for the correlation of images with real noise (ε32 ≤ 1.5e-6 at σ = 5 and 20) and fails for noise-free ones:
   phase normalisation gives unit weight to every bin, including those whose content is below the arithmetic noise of the
   transform, so the float32 correlation of the noise-free plant differs from the float64 one by up to **2.5e-2** of its
   maximum (4.4e-2 in the full set), and SciPy's own float32 run picks a different integer peak than its float64 run on the
   half-pixel vector (`plant256-s0-v03-u1`: (0, 0) against (0, 1), the two maxima 0.6 % apart). So each case carries ε32
   (the float32 pipeline of the oracle's own steps against its float64 pipeline, per stage) and ε64 (the response of the
   float64 pipeline to a one-ulp perturbation of every input pixel, an implementation's arithmetic noise; at most 9.7e-10), and
   is flagged a near-tie when the gap between the two largest samples is at most max(1e-5, 4·ε32) (Float32) or max(1e-12,
   8·ε64) (Float64): 117 and 17 of the 330 cases. A flagged case is held to the **bounding box** of the shifts whose
   correlation sample lies within that noise of the maximum (the integer peaks within it, each refined), not to 1/u; the
   box is in the pins, and `pins.py` asserts that the oracle's own shift lies in it. Every unflagged case is held to equality.
5. **scikit-image on float32 input** returns a float32 shift (5.1 comes back as 5.099999904632568) and a float32 error and
   phase difference. The pins record both oracle runs (the same pixels, float32 and widened to float64); the Float32 kit is
   compared with the float32 run, the Float64 kit with the float64 one. The two oracle runs disagree in the shift in 148 of
   the 330 cases, 117 of them by representation alone, 31 by a grid step or more. The error is `√|1 − x|` with x near 1
   when the images match: its float64 value of an exact copy is ill-conditioned (error 3e-8 comes back with 1.5e-8 of noise),
   and the float32 oracle differs from the float64 one by up to 5.5e-4 in error and 2.7e-4 in phase difference on the
   noise-free plant. Section 3.4's error and phase bands are therefore applied as: the **squared** error within 1e-12 (Float64) /
   1e-5 (Float32), the error itself within 1e-10 / 1e-5 where the oracle's error is at least 0.1, and the phase difference
   (compared on the circle: π and −π are one angle) within 1e-10 / 1e-5, each **plus** 8 (error²) or 4 (phase) times the case's
   measured ε of the last correlation stage; the measured ε is 0 to 1e-6 on every case with real noise.
6. **Section 3.2's mode table, checked against SciPy.** A Python transliteration of the C `map_coordinate`
   and of `NI_ZoomShift`'s order-0 and order-1 branches (the coordinate `k − s` mapped per axis; order 0 reads `floor(c + 1/2)`,
   order 1 `floor(c)` and `floor(c) + 1`; a neighbour beyond the edge is mapped by the mode's integer rule, or is `cval` for
   "grid-constant") reproduces `ndimage.shift` to 1e-12 on 6 shapes (13 × 10, 5 × 4, 2 × 2, 1 × 6, 7 × 1, 3 × 3) × 7 modes × 2
   orders × 44 shifts (integer, half-integer, edge-crossing, 100.3, 1e-9 and random): 0 differences. It shows: "wrap" has
   period n − 1 (n = 5, shift 2: `[30, 40, 10, 20, 30]`); "constant" gives `cval` for every coordinate outside [0, n − 1],
   also in (−1, 0); order 0 rounds half up after the mapping (shift 0.5 reproduces the image, shift −0.5 reads the next
   pixel); "grid-constant" interpolates toward `cval` from the edge pixels. The committed pins hold 4 images (10 × 13, 5 × 4,
   1 × 6, 7 × 1), 30 shifts and cval 7.5 for each of the 14 (mode, order) pairs. `fourier_shift` indexes frequencies as
   `fftfreq` does (the Nyquist index of an even size is negative); the pins hold 5 images (12 × 10, 11 × 9, 1 × 8, 8 × 1,
   16 × 16) × 12 shifts and each case's largest imaginary part.
7. **Not an effect, a correction.** Section 3.1 step 4 calls the amplitudes of the refined branch (`Σ|F|²`, not divided by the
   pixel count) "scikit-image's quirk": the refined correlation is likewise the unnormalised sum, so the two branches give the
   same error for the same pair. The code and its type document this as consistent; the mutant "amplitudes divided by the size"
   stays (it gives √(N² − 1) for an identical pair).
8. **Round half to even** (step 4, `np.round(shift·u)/u`): the shift there is an integer, so `shift·u` is an integer and the
   rounding never meets a half; "round half away" is an equivalent mutant (section 5 lists it).

