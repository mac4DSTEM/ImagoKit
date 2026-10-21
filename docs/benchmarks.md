# Benchmarks against NumPy, SciPy and scikit-image

Run 2026-10-21 on the audit session (ImagoKit 0.6.0 sources). **Header.** Apple M5 Pro, macOS (Darwin 27.0.0), Swift 6.4
(swiftlang-6.4.0.34.1), `swift test -c release`; Python 3.12.14 from `refs/skimage` with NumPy 2.5.4, SciPy 1.18.1,
scikit-image 0.26.0 (single-threaded `np.fft`/`ndimage`). **Method.** The same function, dtype and input size on both sides
(uniform random pixels 30 to 70); one warm-up call, then five timed calls; the table shows the median and (min–max) in
milliseconds. FFTs reuse their plan (`FFT2D`, `FFT2DDouble`; NumPy caches its own); `rfft2`, `butterworth`, the shifts,
registrations and filters are the public one-call API. **Caveat.** The five other kit audits ran on the machine at the same
time (load average 11 to 25 during these runs, 6 to 8 before): absolute times are noisy, and both sides were measured in the
same minutes so the ratios are comparable but not exact. The sizes are the small (128²) and typical (512²) ones plus one
non-power-of-two (500²); masked registration, which costs eleven transforms of (2n−1)², is at 64² and 256².
**Large sizes (2048², 4096², 8192²) are nightly:** `IMAGO_BENCH_LARGE=1` and `--large` add them; they were not run here.

Reproduce (under the suite's memory cap, one benchmark at a time machine-wide):

```sh
IMAGO_BENCH=1 IMAGO_BENCH_OUT=/tmp/swift.json ../tools/memcap.sh -- swift test -c release --filter ImagoBenchmarks
../tools/memcap.sh -- ../refs/skimage/bin/python -I -B tools/benchmarks/bench.py /tmp/python.json
python3 tools/benchmarks/report.py /tmp/swift.json /tmp/python.json
```

The harness is `Tests/ImagoBenchmarks/Benchmarks.swift` (its one test returns at once unless `IMAGO_BENCH=1`);
`tools/benchmarks/bench.py` is the Python side. "kit / Python" below 1 means the kit is faster.

## Kit CPU against Python

| function | dtype | size | kit CPU ms: median (min–max) | Python ms: median (min–max) | kit / Python |
|---|---|---|---|---|---|
| butterworth | float32 | 128² | 0.152 (0.151–0.158) | 0.0788 (0.0755–0.102) | 1.92 |
| butterworth | float32 | 512² | 2.61 (2.57–2.64) | 1.12 (1.06–1.23) | 2.34 |
| butterworth | float64 | 128² | 0.283 (0.28–0.311) | 0.0993 (0.0956–0.121) | 2.85 |
| butterworth | float64 | 512² | 6.44 (6.34–6.66) | 2.27 (2.02–2.64) | 2.83 |
| difference_of_gaussians | float32 | 128² | 0.286 (0.279–0.288) | 0.262 (0.242–0.274) | 1.09 |
| difference_of_gaussians | float32 | 512² | 4.46 (4.45–4.53) | 5.29 (5.16–5.49) | 0.84 |
| fft2 | float32 | 128² | 0.0191 (0.0175–0.0612) | 0.0867 (0.0855–0.0941) | 0.22 |
| fft2 | float32 | 500² | 1.89 (1.84–1.97) | 3.11 (2.68–3.37) | 0.61 |
| fft2 | float32 | 512² | 0.537 (0.528–0.537) | 3.61 (3.42–3.77) | 0.15 |
| fft2 | float64 | 128² | 0.0695 (0.0673–0.093) | 0.0759 (0.0716–0.0761) | 0.92 |
| fft2 | float64 | 500² | 4.63 (4.55–4.74) | 1.94 (1.78–1.99) | 2.38 |
| fft2 | float64 | 512² | 2.14 (2.06–2.23) | 2.66 (2.45–2.74) | 0.81 |
| gaussian_filter sigma=3 | float32 | 128² | 0.139 (0.136–0.157) | 0.127 (0.124–0.136) | 1.09 |
| gaussian_filter sigma=3 | float32 | 512² | 2.22 (2.18–2.23) | 2.52 (2.51–2.61) | 0.88 |
| gaussian_filter sigma=3 | float64 | 128² | 0.125 (0.124–0.129) | 0.145 (0.133–0.191) | 0.86 |
| gaussian_filter sigma=3 | float64 | 512² | 1.96 (1.94–2.01) | 3.09 (2.96–3.18) | 0.63 |
| label connectivity=2 | bool | 128² | 0.154 (0.126–0.231) | 0.117 (0.0848–0.167) | 1.32 |
| label connectivity=2 | bool | 512² | 4.37 (4.2–4.64) | 2.06 (2.04–2.1) | 2.13 |
| masked phase_cross_correlation | float32 | 64² | 1.66 (1.65–1.74) | 0.704 (0.693–0.789) | 2.36 |
| masked phase_cross_correlation | float32 | 256² | 29.6 (29.1–30.7) | 12.4 (12.2–13) | 2.38 |
| masked phase_cross_correlation | float64 | 64² | 3.39 (3.2–4.65) | 1.03 (1.02–1.28) | 3.27 |
| masked phase_cross_correlation | float64 | 256² | 66.5 (64.3–67.5) | 19.1 (18–19.7) | 3.48 |
| phase_cross_correlation | float32 | 128² | 0.136 (0.135–0.15) | 0.177 (0.173–0.207) | 0.77 |
| phase_cross_correlation | float32 | 512² | 2.62 (2.56–2.63) | 3.45 (3.19–3.63) | 0.76 |
| phase_cross_correlation | float64 | 128² | 0.294 (0.293–0.334) | 0.273 (0.266–0.278) | 1.08 |
| phase_cross_correlation | float64 | 512² | 7.57 (7.51–8.34) | 6.07 (5.52–6.59) | 1.25 |
| phase_cross_correlation upsample=10 | float32 | 128² | 0.317 (0.306–0.318) | 0.241 (0.237–0.251) | 1.32 |
| phase_cross_correlation upsample=10 | float32 | 512² | 5.25 (5.19–5.29) | 4.18 (3.83–4.61) | 1.26 |
| profile_line order=1 width=5 | float32 | 128² | 0.0114 (0.0112–0.0148) | 0.601 (0.594–0.634) | 0.02 |
| profile_line order=1 width=5 | float32 | 512² | 0.0449 (0.0444–0.0474) | 2.43 (2.42–2.71) | 0.02 |
| profile_line order=3 width=5 | float32 | 128² | 0.516 (0.502–0.527) | 0.801 (0.794–0.928) | 0.64 |
| profile_line order=3 width=5 | float32 | 512² | 5.33 (5.28–5.44) | 7.44 (7.21–7.6) | 0.72 |
| rfft2 | float32 | 128² | 0.0329 (0.0311–0.035) | 0.0405 (0.0401–0.0433) | 0.81 |
| rfft2 | float32 | 512² | 0.586 (0.509–0.598) | 1.52 (1.29–1.86) | 0.39 |
| rfft2 | float64 | 128² | 0.0799 (0.0755–0.0865) | 0.0313 (0.031–0.0322) | 2.55 |
| rfft2 | float64 | 512² | 2.12 (2.07–2.22) | 0.791 (0.724–0.888) | 2.68 |
| shift order=1 | float32 | 128² | 0.0466 (0.0464–0.0478) | 0.158 (0.154–0.159) | 0.29 |
| shift order=1 | float32 | 512² | 0.651 (0.587–0.895) | 2.51 (2.21–2.52) | 0.26 |
| shift order=3 | float32 | 128² | 0.491 (0.479–0.496) | 0.641 (0.605–0.655) | 0.77 |
| shift order=3 | float32 | 512² | 7.69 (7.67–7.79) | 10.9 (10.6–11.1) | 0.71 |
| uniform_filter size=5 | float32 | 128² | 0.0745 (0.0723–0.0846) | 0.0368 (0.0367–0.0381) | 2.02 |
| uniform_filter size=5 | float32 | 512² | 1.17 (1.17–1.18) | 1.27 (1.26–1.29) | 0.93 |
| uniform_filter size=5 | float64 | 128² | 0.052 (0.0517–0.0572) | 0.0466 (0.0455–0.0492) | 1.11 |
| uniform_filter size=5 | float64 | 512² | 0.778 (0.773–0.811) | 1.8 (1.77–2.05) | 0.43 |

Not in the table because there is no counterpart on the same inputs: `PeakLocalMax.find` (counterpart `peak_local_max`
needs `maximum_filter`; not timed this session).

## Backend decision per function (D16)

Today every function above is CPU (Accelerate, Float64 reference plus Float32 where it exists). **No function has a Metal path,
so no Metal break-even has been measured**; the column says what the CPU table implies and what the nightly run must measure.
A guess is not recorded as a measurement.

| function | backend today | decision |
|---|---|---|
| `fft2` Float32/Float64 | CPU | CPU only at 128² to 512² (0.02 to 2 ms: a Metal dispatch cannot win). Work item: a Metal Float32 FFT prototype, break-even to be measured nightly at 2048², 4096², 8192² |
| `rfft2`, `butterworth`, masked registration | CPU | CPU; first the real-input transform (open-items 1), then re-measure |
| `gaussian_filter`, `uniform_filter`, `difference_of_gaussians` | CPU | CPU only at 128² to 512² (1 to 4 ms, memory bound). Metal candidate above 2048² for large sigma; to be measured nightly |
| `shift` order 1 | CPU | CPU only: 0.65 ms at 512² |
| `shift` order 3 (prefilter plus taps) | CPU | CPU; the prefilter is sequential along lines, the tap sum is the Metal-friendly part: candidate above 2048², not measured |
| `phase_cross_correlation` | CPU | CPU; FFT-bound, follows `fft2` |
| `profile_line` | CPU | CPU: small arrays stay on the CPU (D16) |
| `label` | CPU | CPU: union-find is not a Metal workload |
| `peak_local_max` | CPU | CPU; the window maximum is a Metal candidate only above 4096² |
