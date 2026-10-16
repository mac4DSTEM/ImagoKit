# Parity with the references' documented API

The checklist that makes "complete" a measured claim (suite ROADMAP, phase 4): each row is a function of a reference's
documented API that a micrograph user reaches for, with its state here. Filled as slices land; updated at every tag.
States: **done** (ported, pinned, tested), **partial** (a subset of the parameters, named), **registered** (in a
registration, no code yet), **absent**.

Totals at 0.2.0 (32 rows): 17 done, 8 partial, 7 absent, 0 registered.

## NumPy 2.5.4, `numpy.fft`

| Function | State | Where | Notes |
|---|---|---|---|
| `fft(x)`, `ifft(x)` on complex 1D input | partial | `ImagoFFT.FFT1D`, `FFT1DDouble` (0.1.1) | radix-2 positive lengths only; split-complex in-place arrays; both directions unnormalised (`ifft(norm="forward")`); divide inverse output by n for default NumPy scaling. Pins at 1, 2, 4, 8, 64, 256, 1024; complex-bin error ≤ 1e-5 (Float) / 1e-12 (Double) × max(1, magnitude of oracle). No `axis=`, `n=` resizing, or padding |
| `fft2(x)` on a real 2D array, any shape | done | `ImagoCore` `FFT2D`, `FFT2DDouble` (slice 1a) | exact N-point DFT for every length (no padding); Float32 and Float64 paths; `s=`, `axes=` and `norm=` other than `"backward"` absent |
| `ifft2(X)` | done | `FFT2D`, `FFT2DDouble` (slice 1a) | `norm="backward"` (1/(nx·ny)) only |
| `fftshift`, `ifftshift` (2D) | partial | `PowerSpectrum`, `MaskedInverseFFT` (slice 1) | inside the products, not as a public function yet |
| `fftfreq(n, d)` | done | `PowerSpectrum.frequencies`, `FFT2D.fftfreq` (slice 1a) | `fftshift(fftfreq)` order, bit for bit |
| `rfft2`, `irfft2`, `fftn`, `hfft` | absent | | |
| `numpy.hanning(n)` | done | `SpotMeasurement.hanning` (slice 1a) | the window of the spot estimator |

## NumPy 2.5.4, `numpy.histogram`

| Function | State | Where | Notes |
|---|---|---|---|
| `histogram(a, bins=int, range=None or (lo, hi))` on a float32 array | done | `IntensityHistogram.histogram` (0.2.0) | non-finite pixels dropped first (`a[np.isfinite(a)]`); counts and float32 edges bit for bit on 40 pinned cases, NumPy's `ValueError`s as `HistogramError`; an overflowing `last - first` is refused (NumPy continues with non-finite edges); the `mean` field is not NumPy's. float64 input, array or string `bins`, `weights`, `density` absent. `compute` is the variant that keeps the statistics where NumPy raises |

## SciPy 1.18.1, `scipy.ndimage` and `scipy.fft`

| Function | State | Where | Notes |
|---|---|---|---|
| `ndimage.median_filter(size=3, mode="nearest")` | partial | `HRTEMFilter` step A1 (slice 1b) | 3 × 3, nearest, inside the filter only |
| `ndimage.map_coordinates(order=1, mode="nearest")` | partial | `HRTEMFilter.radialProfile` (slice 1b) | order 1, nearest, on circles about the centre |
| `ndimage.shift(order=0 or 1)` | partial | `ImageShift.shift` (slice 2) | all seven modes (`constant`, `nearest`, `wrap`, `reflect`, `mirror`, `grid-constant`, `grid-wrap`) and `cval`, 2D, one real channel; `order` 2 to 5 (the spline prefilter) absent, slice 2b; `output=` and `prefilter=` absent |
| `ndimage.fourier_shift` | partial | `ImageShift.fourierShift`, `fourierShiftWithResidual` (slice 2) | `real(ifft2(fourier_shift(fft2(x), s)))` of a real 2D image, with the largest imaginary part reported; `n=` and `axis=` (real-FFT input) absent |
| `ndimage.gaussian_filter`, `uniform_filter`, `zoom`, `rotate` | absent | | |

## scikit-image 0.26.0

| Function | State | Where | Notes |
|---|---|---|---|
| `measure.profile_line` | partial | `LineProfile` (0.2.0) | orders 0 and 1, modes `constant` `nearest` `reflect` `mirror` `wrap`, `linewidth`, `cval`, `reduce_func` (mean, sum, min, max, median, closure) and `reduce_func=None`; `.float32` bit for bit with scikit-image on a float32 image (368 pinned profiles), `.float64` within 8.9e-16 of the image maximum; orders 2 to 5, multichannel absent; width below 1 and a degenerate segment throw |
| `registration.phase_cross_correlation` (with `upsample_factor`) | partial | `PhaseCorrelation.register`, `DriftSeries` (slice 2) | real-space images of one size, integer `upsample_factor` (1 to 1024), `normalization` `"phase"` or `None`; the shift, the error and the phase difference; no masks (`reference_mask`, `moving_mask`, `overlap_ratio`), no `disambiguate`, no `space="fourier"` |
| a drift series and an aligned sum (not in scikit-image) | done (slice 2) | `DriftSeries`, `AlignedSum` | `.first`, `.mean`, `.previous` references; Fourier or bilinear summing with the overlap count |
| `filters.butterworth`, `filters.difference_of_gaussians`, `filters.window` | absent | | bandpass, a later slice |
| `feature.peak_local_max` | absent | | atom columns, a later slice |
| `restoration.denoise_*` | absent | | denoising, a later slice |

## Not from a Python package

| Quantity | State | Truth |
|---|---|---|
| distance and angle between marks (anisotropic scales, Kahan cross product) | done (0.2.0) | `ImageMeasurement`: planted angles, closed forms and 100 NumPy float64 triples |
| display normalisation, log contrast, thumbnails, colour mapping | done (0.2.0) | `DisplayNormalization`, `Thumbnail`, `DisplayMapping`: Float32 bit for bit with a NumPy reading of the definition; the log to the last Float32 bit |
| scale-bar numbers (1-2-5 length, unit ladder, text, drawn length) | done (0.2.0) | `ScaleBarMath`: exact on 250 random and 108 usual calibrations, plus the ladder grid across mantissa boundaries |
| square-region, bin and decade geometry, shown units and aspect | done (0.2.0) | `FFTRegion`, `SpectrumBin`, `RadialChartScale`, `FFTUnits`, `SpectrumShape`: exact against a transcribed definition on random cases |
| log-power spectrum display and its default window | done (0.2.0) | `SpectrumDisplay`: values to the last Float32 bit, window within a 2048th of the range of the (⌈n/2⌉)-th smallest value |
| axis as offset, scale, unit (RosettaSciIO convention) | done (0.2.0) | `value(at: 0) == offset`; round trip with `origin` |
| Å per pixel, Å⁻¹ per pixel (Å⁻¹, nm⁻¹, mrad), canonical unit spellings | done (0.2.0) | closed forms (1 nm = 10 Å, mrad = 1/(1000 λ) Å⁻¹); every spelling pinned to the transcribed consumer table |
| radial profile against physical \|q\| | done (slice 1a) | a planted ring at known \|q\| |
| spot spacing d = 1/\|q\| and the angle between two spots | done (slice 1a) | planted lattices, closed forms |
| masked inverse FFT (Bragg filter) | done (slice 1b) | NumPy's `ifft2(ifftshift(fftshift(fft2(x)) · M))` and a planted two-lattice image |
| HRTEM ABSF and Wiener filters (Kilaas 1998, Mitchell's script) | done (slice 1b) | a float64 NumPy/SciPy reading of the same steps; never compared with DigitalMicrograph |
| GPA strain (Hÿtch, Snoeck and Kilaas 1998) | absent | a later slice; planted strain fields |
| atom columns (Atomap, GPL-3.0) | absent | a later slice, its own GPL module |
