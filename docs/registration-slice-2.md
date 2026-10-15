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

None yet.
