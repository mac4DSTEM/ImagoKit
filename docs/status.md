# Status

ImagoKit 0.8.0 (2026-10-22), against NumPy 2.5.4, SciPy 1.18.1 and scikit-image 0.26.0. Live: the FFT surface (`fft2`, `rfft2`, `fftshift`, axes and norm), power spectra and
spot measurement, Gaussian, uniform, difference-of-Gaussians, Butterworth and window filters, Fourier and spline-order shifts, phase correlation (plain and masked), HRTEM filter
steps, `peak_local_max`, profiles, colormaps, segmentation helpers (thresholds, binary morphology, labelled statistics, `regionprops` completion, areal density, distance transform, watershed, rolling ball), zoom and rotate, TV and non-local-means denoising, real-input 2D and 1D transforms and a mixed-radix Float64 plan, GPA strain (planted truth, validation none), CSV export. Parity: `docs/parity.md` (93 rows: 36 present, 12 partial, 45 missing); timings against Python:
`docs/benchmarks.md`; what is open (the upsampled registration, `label`, Metal paths): `docs/open-items.md`.

Test split (2026-10-21): the comparisons with the pinned NumPy, SciPy and scikit-image values and the benchmark harness are the `ImagoKitParityTests` target (fixtures stay in `Tests/Imago*Tests/Fixtures`, found through `ImagoKitTestSupport`); `swift test -c release --parallel --skip ImagoKitParityTests` is the per-session gate (84 tests at 0.7.0), the plain `swift test -c release --parallel` the full suite (222 tests: 84 fast + 138 parity). Mac, release, warm build: fast target 1.7 s, full suite 4.7 s (6.8 s before, one target set, no skip).
