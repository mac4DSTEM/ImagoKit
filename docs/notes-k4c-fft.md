# Notes for the merging session: k4c FFT slice (branch worktree-agent-a39bfad414629ec54)

Commits, oldest first: `dfa4db1` rfft2/irfft2 + butterworth, masked and plain phase correlation on them; `9f2b6ca` public 1D
`rfft`/`irfft`; `bcd6863` mixed-radix Float64 plan; `0271eff` butterworth half-spectrum mask; `9a4c839` column pass through a
transpose; `6ca783c` masked correlation at the next fast length (separable, revertable); `e4a6b3c` docs/benchmarks.md; this file.
Files not touched, as told: CHANGELOG.md, docs/parity.md, docs/status.md, docs/open-items.md, README.md, Package.swift, .claude, CI.
Tests: `swift test -c release --parallel --skip ImagoKitParityTests` and the full `swift test -c release --parallel` are green;
`tools/consumer-check/run.sh` passes (138 checks, no warnings).

## CHANGELOG (suggested for the next minor, 0.8.0)

### Added
- **Real-input 2D transform**: `FFTSurface.rfft2`/`irfft2` (Float64 and Float32) now run on a real-to-complex path (two real rows share one complex
  row transform, then the `columns/2 + 1` columns as complex lines; the inverse backwards) instead of truncating the complex transform:
  512x512 Float64 `rfft2` 2.07 -> 0.72 ms (NumPy 0.65), `irfft2` 4.34 -> 0.63 ms. Same `s=`, `norm=`, axes behaviour; pinned at 16 more shapes
  (`tools/fft-real/pins.py`).
- **1D `rfft` / `irfft`** on `FFT1D` and `FFT1DDouble` (`rfft(_:norm:)`, `irfft(re:im:norm:)` on a plan; static `rfft(_:n:norm:)`,
  `irfft(re:im:n:norm:)` that build the plan and return nil for an `n` that is not a power of two): `n=` truncation and zero padding,
  `norm=` backward/forward/ortho, the imaginary parts of bin 0 and of the Nyquist bin ignored, as `numpy.fft`.
- **Mixed-radix Float64 plan** (`MixedRadixPlanD`, internal): lengths of prime factors 2, 3, 5, 7 that no vDSP plan serves (500, 504, 360, 7, 14, 49 ...)
  no longer take Bluestein: `fft2` Float64 500x500 4.64 -> 1.46 ms (NumPy 1.49), 504x504 4.51 -> 1.26, 360x360 3.03 -> 0.60. The `vDSP_DFT_zop`
  path for 2^n x {1,3,5,15} and Bluestein for every other length stay.

### Changed
- `butterworth` (Float32 and Float64), `PhaseCorrelation.register` and the drift series and the masked correlation transform their real images with the
  real-input transform: 512x512 `butterworth` Float64 6.28 -> 1.45 ms (scikit-image 1.61), Float32 2.61 -> 1.02; `phase_cross_correlation` Float64 7.66 -> 3.14,
  Float32 2.65 -> 2.15; masked 256x256 Float64 62.0 -> 13.1, Float32 28.3 -> 9.7. Values agree to rounding (the pins pass unchanged).
- `FFTSurface` keeps its last six plans (a loop of same-size calls pays the setup once).
- The masked correlation transforms at the next length of prime factors 2, 3, 5, 7 and slices back to `M + R - 1`, as scikit-image does with
  `scipy.fft.next_fast_len` (before: the exact length, Bluestein for 127, 255, 511 ...). Same correlation up to rounding.
- `ImageRegistration.phaseDifference` at upsample 1 is now `atan2(0, peak)`: 0 for a positive peak, pi for a negative one (before: the sign
  of 1e-16 rounding noise in the imaginary part of the correlation); the refined (upsample > 1) value is unchanged.
- The axis-wise 2D path (lengths no radix-2 2D call serves) transposes instead of gathering columns: Float64 480x480 2.03 -> 1.53 ms.

### Migration
None: no signature changed. `FFT2DDouble`/`FFT2D` gained `package`-level members only; `FFT1D`/`FFT1DDouble` gained the four public members above.

## docs/parity.md

Replace row 21 (`rfft2`, `irfft2`) with:

| `rfft2`, `irfft2` (with `s`, `norm`) | present (0.6.0) | `FFTSurface.rfft2/irfft2` | numpy/fft/tests/test_pocketfft.py: atol 1e-6 (rtol 1e-7) | study | real-input path (two real rows per complex transform; 0.72 ms Float64 at 512x512, NumPy 0.65); 24 + 16 pinned shape/norm cases incl. `s` truncation and padding; worst scaled difference 9e-14 (Float64), 1.4e-5 (Float32; the test holds it to 1e-5 of the array maximum) |

Row 18 (`fft2`, `ifft2`): append to the last cell `; Float64 lengths of prime factors 2, 3, 5, 7 not served by vDSP run on a mixed-radix plan (43 pinned lengths and 5 shapes, worst scaled difference 4e-14), Bluestein for the rest`.

Row 25 (`fft`, `ifft` 1D) unchanged (still radix-2 positive lengths only).

Replace row 26 (`rfft`, `irfft`, `hfft`, `ihfft`) with two rows:

| `rfft`, `irfft` (1D real; `n=`, `norm=`) | present | `FFT1D.rfft/irfft`, `FFT1DDouble.rfft/irfft` (plan and static forms) | numpy/fft/tests/test_pocketfft.py: atol 1e-6 (rtol 1e-7) | useful | power-of-two lengths only (the 1D plans'); `n=` is the plan length (static forms take `n:`); 94 pinned cases, worst scaled difference 1.9e-14 (Float64), 6.8e-6 (Float32) |
| `hfft`, `ihfft` (1D Hermitian) | missing |  | numpy/fft/tests/test_pocketfft.py: atol 1e-6 (rtol 1e-7) | useful | ACOM (DiffractoKit) needs a 1D FFT of any length |

Totals line: `88 rows ...: 28 present, 11 partial, 49 missing. By priority: 35 study, 33 useful, 20 not planned.` (one row is split, and `rfft`/`irfft` moves from missing to present).

Row 80 (`butterworth`): append `; on the real-input transform (irfftn(mask * rfftn(x)) as scikit-image), the mask built on the half spectrum`.
Row 125 (masked `phase_cross_correlation`): replace `exact-tie maxima are not reproducible across FFT lengths` with `transforms at the next 7-smooth length (SciPy's next_fast_len also allows 11), so an exact tie can still differ from SciPy where the lengths differ`.

## docs/open-items.md

Delete items 1 and 2 of "Slower than Python at the typical size" (done). Keep item 3 (`label`, unchanged: 2.36x). Rewrite item 4 and add one item (renumber the list):

> 2. **`phase_cross_correlation` with `upsample_factor=10` Float32 is 1.15x (128²) and 1.27x (512²)**; the integer-pixel case is now 0.6x. The FFTs
>    are no longer the cost: at 512² the upsampled case takes 4.9 ms against 2.2 ms for the whole integer case, and the difference is the Hermitian
>    completion of P plus the matrix-multiply DFT (`upsampledDFT`, scalar loops); not profiled stage by stage. What would refute it: a profile showing another stage.
>    Fix: the two small complex matrix products on Accelerate (`cblas`/`vDSP_mmul`).
> 3. **`fft2` Float64 on the vDSP lengths (2^n x {3,5,15}) is up to 1.12x**, e.g. 480² 1.53 ms against 1.37 (NumPy). A batch of 480 lines takes the same time on
>    `vDSP_DFT_zop` and on the mixed-radix plan (0.49 ms), so the remainder is the transposes around the vDSP calls; the mixed-radix plan needs none for the columns.
>    `CLAUDE.md` keeps the vDSP path.

In "Work items from the audit": the line "Masked registration: scikit-image's `xcorr == xcorr.max()` tie rule is reproducible only for a unique maximum (FFT length differs from SciPy's fast length); symmetric pairs decide on rounding noise."
becomes "... (the kit's fast length is the next 7-smooth length, SciPy's also allows 11) ...". Add: "Mixed radix: radices 11 and 13, a Float32 mixed-radix plan (Float32 500² runs Bluestein at 1.53 ms, 0.8x NumPy), a generic odd-radix pass."

## Benchmarks (docs/benchmarks.md, already edited by this branch)

Median (min-max) of fifteen timed calls (three alternating rounds of: kit before = parent commit sources, kit after, Python), ms, M5 Pro, load average 4 to 8 from other sessions.
Control: the untouched rows (gaussian, uniform, shift, profile_line, label, difference_of_gaussians) came out within 3% of "before" in the same runs.

| function | dtype | size | before ms: median (min–max) | after ms: median (min–max) | Python ms: median (min–max) | before / Python | after / Python |
|---|---|---|---|---|---|---|---|
| butterworth | float32 | 128 | 0.147 (0.145–0.152) | 0.0408 (0.0401–0.041) | 0.0765 (0.0731–0.105) | 1.92 | 0.53 |
| butterworth | float32 | 512 | 2.61 (2.53–2.64) | 1.02 (0.969–1.06) | 0.964 (0.907–1.17) | 2.71 | 1.06 |
| butterworth | float64 | 128 | 0.271 (0.269–0.285) | 0.0705 (0.07–0.0752) | 0.0956 (0.0927–0.104) | 2.83 | 0.74 |
| butterworth | float64 | 512 | 6.28 (6.11–6.54) | 1.45 (1.35–1.5) | 1.61 (1.44–2.14) | 3.91 | 0.91 |
| fft2 | float32 | 500 | 1.86 (1.83–1.95) | 1.53 (1.48–1.73) | 1.92 (1.73–2.08) | 0.97 | 0.80 |
| fft2 | float64 | 360 | 3.03 (2.92–3.11) | 0.596 (0.584–0.611) | 0.712 (0.66–0.929) | 4.26 | 0.84 |
| fft2 | float64 | 480 | 2.03 (1.97–2.5) | 1.53 (1.48–1.58) | 1.37 (1.3–1.71) | 1.49 | 1.12 |
| fft2 | float64 | 500 | 4.64 (4.39–4.77) | 1.46 (1.44–1.5) | 1.49 (1.29–1.9) | 3.11 | 0.98 |
| fft2 | float64 | 504 | 4.51 (4.41–4.71) | 1.26 (1.23–1.28) | 1.49 (1.39–2.02) | 3.03 | 0.85 |
| irfft2 | float32 | 128 | 0.34 (0.321–0.367) | 0.0187 (0.0185–0.0192) | 0.0274 (0.0255–0.0311) | 12.42 | 0.68 |
| irfft2 | float32 | 512 | 4.44 (4.31–4.82) | 0.442 (0.435–0.473) | 0.405 (0.389–0.429) | 10.96 | 1.09 |
| irfft2 | float64 | 128 | 0.3 (0.289–0.31) | 0.0334 (0.0331–0.0346) | 0.0377 (0.037–0.0393) | 7.95 | 0.89 |
| irfft2 | float64 | 512 | 4.34 (3.99–4.62) | 0.634 (0.562–0.758) | 0.716 (0.697–0.944) | 6.05 | 0.89 |
| masked phase_cross_correlation | float32 | 64 | 1.67 (1.64–1.73) | 0.491 (0.469–0.533) | 0.611 (0.574–0.765) | 2.73 | 0.80 |
| masked phase_cross_correlation | float32 | 256 | 28.3 (28–29.1) | 9.74 (9.59–10.3) | 11.3 (11.1–11.8) | 2.50 | 0.86 |
| masked phase_cross_correlation | float64 | 64 | 3.15 (3.09–3.25) | 0.696 (0.658–0.722) | 0.94 (0.925–1.09) | 3.36 | 0.74 |
| masked phase_cross_correlation | float64 | 256 | 62 (61.5–63.4) | 13.1 (12.8–13.5) | 17.9 (17–20.6) | 3.47 | 0.73 |
| phase_cross_correlation | float32 | 128 | 0.132 (0.13–0.137) | 0.105 (0.102–0.108) | 0.175 (0.164–0.195) | 0.76 | 0.60 |
| phase_cross_correlation | float32 | 512 | 2.65 (2.55–2.94) | 2.15 (2.06–2.23) | 3.35 (3.22–3.68) | 0.79 | 0.64 |
| phase_cross_correlation | float64 | 128 | 0.284 (0.282–0.298) | 0.159 (0.157–0.161) | 0.271 (0.264–0.282) | 1.05 | 0.59 |
| phase_cross_correlation | float64 | 512 | 7.66 (7.27–8.24) | 3.14 (3.02–3.2) | 5.22 (5.04–5.98) | 1.47 | 0.60 |
| phase_cross_correlation upsample=10 | float32 | 128 | 0.3 (0.297–0.314) | 0.281 (0.276–0.288) | 0.245 (0.24–0.262) | 1.22 | 1.15 |
| phase_cross_correlation upsample=10 | float32 | 512 | 5.24 (5.16–5.58) | 4.93 (4.82–5.01) | 3.89 (3.73–4.31) | 1.35 | 1.27 |
| rfft2 | float32 | 128 | 0.0285 (0.0253–0.0377) | 0.0155 (0.0147–0.0163) | 0.0379 (0.0369–0.0422) | 0.75 | 0.41 |
| rfft2 | float32 | 512 | 0.57 (0.558–0.598) | 0.43 (0.426–0.458) | 0.908 (0.82–1.17) | 0.63 | 0.47 |
| rfft2 | float64 | 128 | 0.0716 (0.066–0.075) | 0.03 (0.0289–0.0307) | 0.0313 (0.0293–0.0324) | 2.29 | 0.96 |
| rfft2 | float64 | 512 | 2.07 (1.96–2.23) | 0.721 (0.684–0.759) | 0.647 (0.589–0.766) | 3.20 | 1.12 |

DONE WHEN: `rfft2` Float64 512x512 = 1.12x NumPy by median (0.721 against 0.647 ms; by best times 0.684 against 0.589 = 1.16x): met. `fft2` Float64 500x500 = 0.98x: met.
Not met: `fft2` Float64 480x480 stays 1.12x (the vDSP path is kept by rule), `phase_cross_correlation upsample=10` Float32 1.15x / 1.27x (see open-items above).

## Agreement with NumPy 2.5.4 (tools/fft-real/pins.py, seed 20261023)

Scaled difference |got - want| / max(1, |want|), worst over all cases: rfft2 5.1e-14, irfft2 8.9e-14, rfft 1.9e-14, irfft 1.1e-14, fft lines (43 lengths) 2.7e-14, fft2/ifft2 planes 4.3e-14 (Float64);
Float32: rfft2 1.4e-5, rfft 6.8e-6, irfft 5.4e-6, fft lines 1.9e-5 (element-wise with a floor of 1; the tests hold Float32 to 1e-5 of the array maximum and Float64 to atol 1e-6 + rtol 1e-7, as test_pocketfft.py).
No tolerance was loosened.

## Deviations from the brief

- Commit granularity: rfft2 and its three callers are one commit (`dfa4db1`), not two; butterworth's half-spectrum mask, the transposed column pass and the fast-length masked
  correlation are three further commits, each green.
- Beyond the brief, each revertable alone: `0271eff` (butterworth: the mask build, not the transform, was most of what remained, so the transform alone did not bring it under 1.2x),
  `9a4c839` (column gathers), `6ca783c` (masked correlation at a fast length as scikit-image does: a rounding-level numerical change, and a change of the 0.6.x documented choice
  "the kit uses the exact length"; revert it if the exact length is wanted).
- `PhaseCorrelation` inverts with one complex-to-real transform, so the correlation's imaginary part is exactly 0 (see the CHANGELOG line on `phaseDifference`); the pins (including the pi case) pass.
- The pins script asserts NumPy 2.5.4 only (the Python in References is 3.12.14; `tools/fft1d/pins.py` asserts 3.14.7). The fixture `fft-real-pins.json` is 4.1 MB.
- 1D `rfft`/`irfft`: power-of-two lengths only, because the 1D plans are; the static forms return nil where NumPy raises. No `axis=`.
- Float32: `rfft2` gains less (0.57 -> 0.43 ms) because the vDSP 2D complex FFT is already fast; the mixed-radix plan is Float64 only (Float32 500² is Bluestein at 1.53 ms, 0.8x NumPy).
- `tools/benchmarks/bench.py` and `Benchmarks.swift` changed (`IMAGO_BENCH_ONLY`, irfft2, fft2 360/480/504).
- Temporary stage-timing and agreement tests were used and deleted; nothing of them is committed.
