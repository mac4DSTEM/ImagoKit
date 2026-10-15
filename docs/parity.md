# Parity with the references' documented API

The checklist that makes "complete" a measured claim (suite ROADMAP, phase 4): each row is a function of a reference's
documented API that a micrograph user reaches for, with its state here. Filled as slices land; updated at every tag.
States: **done** (ported, pinned, tested), **partial** (a subset of the parameters, named), **registered** (in a
registration, no code yet), **absent**.

## NumPy 2.5.4, `numpy.fft`

| Function | State | Where | Notes |
|---|---|---|---|
| `fft2(x)` on a real 2D array, any shape | done | `ImagoCore` `FFT2D`, `FFT2DDouble` (slice 1a) | exact N-point DFT for every length (no padding); Float32 and Float64 paths; `s=`, `axes=` and `norm=` other than `"backward"` absent |
| `ifft2(X)` | done | `FFT2D`, `FFT2DDouble` (slice 1a) | `norm="backward"` (1/(nx·ny)) only |
| `fftshift`, `ifftshift` (2D) | partial | `PowerSpectrum`, `MaskedInverseFFT` (slice 1) | inside the products, not as a public function yet |
| `fftfreq(n, d)` | done | `PowerSpectrum.frequencies`, `FFT2D.fftfreq` (slice 1a) | `fftshift(fftfreq)` order, bit for bit |
| `rfft2`, `irfft2`, `fftn`, `hfft` | absent | | |
| `numpy.hanning(n)` | done | `SpotMeasurement.hanning` (slice 1a) | the window of the spot estimator |

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
| `registration.phase_cross_correlation` (with `upsample_factor`) | partial | `PhaseCorrelation.register`, `DriftSeries` (slice 2) | real-space images of one size, integer `upsample_factor` (1 to 1024), `normalization` `"phase"` or `None`; the shift, the error and the phase difference; no masks (`reference_mask`, `moving_mask`, `overlap_ratio`), no `disambiguate`, no `space="fourier"` |
| a drift series and an aligned sum (not in scikit-image) | done (slice 2) | `DriftSeries`, `AlignedSum` | `.first`, `.mean`, `.previous` references; Fourier or bilinear summing with the overlap count |
| `filters.butterworth`, `filters.difference_of_gaussians`, `filters.window` | absent | | bandpass, a later slice |
| `feature.peak_local_max` | absent | | atom columns, a later slice |
| `restoration.denoise_*` | absent | | denoising, a later slice |

## Not from a Python package

| Quantity | State | Truth |
|---|---|---|
| radial profile against physical \|q\| | done (slice 1a) | a planted ring at known \|q\| |
| spot spacing d = 1/\|q\| and the angle between two spots | done (slice 1a) | planted lattices, closed forms |
| masked inverse FFT (Bragg filter) | done (slice 1b) | NumPy's `ifft2(ifftshift(fftshift(fft2(x)) · M))` and a planted two-lattice image |
| HRTEM ABSF and Wiener filters (Kilaas 1998, Mitchell's script) | done (slice 1b) | a float64 NumPy/SciPy reading of the same steps; never compared with DigitalMicrograph |
| GPA strain (Hÿtch, Snoeck and Kilaas 1998) | absent | a later slice; planted strain fields |
| atom columns (Atomap, GPL-3.0) | absent | a later slice, its own GPL module |
