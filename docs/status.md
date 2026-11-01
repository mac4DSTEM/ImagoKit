# Status

ImagoKit 0.8.0 (2026-10-22), against NumPy 2.5.4, SciPy 1.18.1 and scikit-image 0.26.0. Live: the FFT surface (`fft2`, `rfft2`, `fftshift`, axes and norm), power spectra and
spot measurement, Gaussian, uniform, difference-of-Gaussians, Butterworth and window filters, Fourier and spline-order shifts, phase correlation (plain and masked), HRTEM filter
steps, `peak_local_max`, profiles, colormaps, segmentation helpers (thresholds, binary morphology, labelled statistics, `regionprops` completion, areal density, distance transform, watershed, rolling ball), zoom and rotate, TV and non-local-means denoising, real-input 2D and 1D transforms and a mixed-radix Float64 plan, GPA strain (planted truth, validation none), CSV export. Parity: `docs/parity.md` (93 rows: 36 present, 12 partial, 45 missing); timings against Python:
`docs/benchmarks.md`; what is open: the GitHub issues (the upsampled registration, `label`, Metal paths); known limits: `docs/parity.md`.

Test split: the comparisons with the pinned NumPy, SciPy and scikit-image values and the benchmark harness are the `ImagoKitParityTests` target (fixtures stay in `Tests/Imago*Tests/Fixtures`, found through `ImagoKitTestSupport`); `swift test -c release --parallel --skip ImagoKitParityTests` runs the fast unit tests (what every pull request runs), the plain `swift test -c release --parallel` the full suite (before a release and nightly). On an Apple Silicon Mac, release, warm build, the fast target takes about 2 s and the full suite about 5 s.
