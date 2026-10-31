# Open items

A plain list. The first entries are functions slower than their Python counterpart at the typical size (512², `docs/benchmarks.md`),
each with its cause as far as it is known (diagnose before changing a number).

## Slower than Python at the typical size

1. **`label` is 2.1x slower** (4.4 ms against 2.1 ms at 512²). Not diagnosed: `Segmentation.grow` calls two closures per pixel
   (`isForeground`, `same`) and keeps Int arrays; profile before changing it.
2. **`phase_cross_correlation` with `upsample_factor=10` Float32 is 1.15x (128²) and 1.27x (512²)**; the integer-pixel case is now 0.6x. The FFTs
   are no longer the cost: at 512² the upsampled case takes 4.9 ms against 2.2 ms for the whole integer case, and the difference is the Hermitian
   completion of P plus the matrix-multiply DFT (`upsampledDFT`, scalar loops); not profiled stage by stage. What would refute it: a profile showing
   another stage. Fix: the two small complex matrix products on Accelerate (`cblas`/`vDSP_mmul`).
3. **`fft2` Float64 on the vDSP lengths (2^n x {3,5,15}) is up to 1.12x**, e.g. 480² 1.53 ms against 1.37 (NumPy). A batch of 480 lines takes the same time on
   `vDSP_DFT_zop` and on the mixed-radix plan (0.49 ms), so the remainder is the transposes around the vDSP calls; the mixed-radix plan needs none for the columns.
   The kit keeps the vDSP path by design.

## Other open work

- **1.0 item, naming:** the new surface takes `rows`/`columns` (decided 2026-10-22: `DistanceTransform`, `Segmentation.watershed`/`markers`, `RollingBall`, `Denoise`, `GeometricPhaseAnalysis`, `ImageResampling`); `LabelImage`, `Micrograph` and the older morphology and segmentation API still say `width`/`height`. Unify at 1.0 (a break). Four error enums carry a `.shape` case (`MorphologyError`, `RestorationError`, `DenoiseError`, `GeometricPhaseAnalysisError`); `RestorationError` and `DenoiseError` have no `LocalizedError` text yet.

- 0.8.0 gaps, none blocking: mixed radix has radices 2, 3, 5, 7 only (11 and 13, a Float32 plan and a generic odd-radix pass are open: Float32 500² runs Bluestein at 1.53 ms, 0.8x NumPy); 1D `rfft`/`irfft` take power-of-two lengths only; `hfft`/`ihfft` missing;
  `distance_transform_edt` is 2D; `zoom`/`rotate` are 2D; the denoisers are 2D single-channel, float only, and Bregman refuses an axis of length 1;
  GPA is Float64 CPU, two reflections, `validation: "none"` until one dataset with truth passes (the apps should show the amplitude maps).

- 0.7.0 (precipitation) gaps, none blocking: the thresholds, binary morphology, `ObjectStatistics`, `regionProperties` and `arealDensity` are CPU
  (Double or Float32 as the reference is) and not yet benchmarked against the Python (rows for `docs/benchmarks.md` are open);
  `BinaryMorphology` refuses even structures and an all-false structure (SciPy shifts the origin and answers) and runs a per-pixel loop
  (fine at the 512² of a micrograph, slow for a 4096² mask: a bit-packed or separable path is the fix if an app asks); `Thresholding.yen`
  can differ from scikit-image on an image of more than 2^24 pixels whose last histogram bin is nearly empty (Float32 `P1` reaches 1 and
  NumPy's `argmax` returns the first NaN); `regionProperties` computes every property per object (use `objectMoments` alone for just the moments);
  (`watershed`, `rolling_ball` and the Euclidean distance transform arrived in 0.8.0.)

- Metal Float32 paths: FFT, then the separable filters; measure the break-even at 2048² to 8192² in the nightly review.
- Study rows still missing (`docs/parity.md`): `profile_line` multichannel, `peak_local_max` with `footprint`/`labels`.
  (`ndimage.zoom` and `rotate` and the TV and non-local-means denoisers arrived in 0.8.0.)
- Spline shifts: `constant` and `wrap` modes at orders 2 to 5 (their coordinate rule is not ported; refused by name).
- Masked registration: scikit-image's `xcorr == xcorr.max()` tie rule is reproducible only for a unique maximum (FFT length
  differs from SciPy's fast length); symmetric pairs decide on rounding noise (the kit's fast length is the next 7-smooth length, SciPy's also allows 11).
- `butterworth` with `channel_axis` and `difference_of_gaussians` with `channel_axis` (multichannel images).
