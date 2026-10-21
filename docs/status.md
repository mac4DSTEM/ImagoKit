# Status

ImagoKit 0.6.1 (2026-10-21), against NumPy 2.5.4, SciPy 1.18.1 and scikit-image 0.26.0. Live: the FFT surface (`fft2`, `rfft2`, `fftshift`, axes and norm), power spectra and
spot measurement, Gaussian, uniform, difference-of-Gaussians, Butterworth and window filters, Fourier and spline-order shifts, phase correlation (plain and masked), HRTEM filter
steps, `peak_local_max`, profiles, colormaps, segmentation helpers, CSV export. Parity: `docs/parity.md` (77 rows: 17 present, 13 partial, 47 missing); timings against Python:
`docs/benchmarks.md`; what is open (real-input FFT, Float64 mixed radix, zoom and rotate, Metal paths): `docs/open-items.md`.

Test split (2026-10-21): the comparisons with the pinned NumPy, SciPy and scikit-image values and the benchmark harness are the `ImagoKitParityTests` target (fixtures stay in `Tests/Imago*Tests/Fixtures`, found through `ImagoKitTestSupport`); `swift test -c release --parallel --skip ImagoKitParityTests` is the per-session gate (77 tests), the plain `swift test -c release --parallel` the full suite (180 tests, unchanged: 77 fast + 103 parity). Mac, release, warm build: fast target 1.7 s, full suite 4.7 s (6.8 s before, one target set, no skip).
