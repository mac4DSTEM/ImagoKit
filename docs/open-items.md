# Open items

A plain list. The first entries are functions slower than their Python counterpart at the typical size (512², `docs/benchmarks.md`),
each with its cause as far as it is known (diagnose before changing a number, `../CLAUDE.md`).

## Slower than Python at the typical size

1. **No real-input 2D transform.** `rfft2` (Float64 2.7x), `butterworth` (2.3x Float32, 2.8x Float64) and masked registration
   (2.4x to 3.5x) all do complex transforms of real data, which is twice the work of a real-to-complex transform; NumPy/SciPy
   use r2c. Cause: the ratios sit at about 2x to 3x, and the only structural difference is the r2c path (the call counts are
   equal); not yet confirmed by a prototype. A Float32 `rfft2` is already faster than NumPy (0.39x) because the vDSP complex
   path is faster there. What would refute it: an r2c prototype that leaves `butterworth` still above 1.2x.
   Fix: `FFTSurface.rfft2/irfft2` on a packed-pair real transform; then `butterworth`, masked registration and `PhaseCorrelation`
   use it.
2. **Float64 `fft2` at 500² is 2.4x slower** (4.6 ms against 1.9 ms), Float32 at 500² is faster (0.61x). 500 = 2² · 5³ is not
   2ⁿ x {1,3,5,15}, so the Float64 path takes Bluestein (`CLAUDE.md`), while the Float32 path has exact radix-3/5 plans.
   Fix: a mixed-radix Float64 plan for lengths with small prime factors, Bluestein only for the rest.
3. **`label` is 2.1x slower** (4.4 ms against 2.1 ms at 512²). Not diagnosed: `Segmentation.grow` calls two closures per pixel
   (`isForeground`, `same`) and keeps Int arrays; profile before changing it.
4. **`phase_cross_correlation` Float64 at 512² is 1.25x** and `upsample_factor=10` Float32 is 1.26x: follow items 1 and 2.

## Work items from the audit

- Metal Float32 paths (D16): FFT, then the separable filters; measure the break-even at 2048² to 8192² in the nightly review.
- Study rows still missing (`docs/parity.md`): `ndimage.zoom` and `rotate`, `restoration.denoise_*`, `profile_line`
  multichannel, `peak_local_max` with `footprint`/`labels`.
- Spline shifts: `constant` and `wrap` modes at orders 2 to 5 (their coordinate rule is not ported; refused by name).
- `window`: the 14 `get_window` families beyond the 11 ported (triang, parzen, bohman, barthann, cosine, exponential, taylor,
  lanczos, kaiser_bessel_derived, general_cosine, general_gaussian, general_hamming, dpss, chebwin).
- Masked registration: scikit-image's `xcorr == xcorr.max()` tie rule is reproducible only for a unique maximum (FFT length
  differs from SciPy's fast length); symmetric pairs decide on rounding noise.
- `butterworth` with `channel_axis` and `difference_of_gaussians` with `channel_axis` (multichannel images).
