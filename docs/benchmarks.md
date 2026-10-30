# Benchmarks against NumPy, SciPy and scikit-image

Run 2026-10-21 (ImagoKit 0.6.0 sources). **Header.** Apple M5 Pro, macOS (Darwin 27.0.0), Swift 6.4
(swiftlang-6.4.0.34.1), `swift test -c release`; Python 3.12.14 with NumPy 2.5.4, SciPy 1.18.1,
scikit-image 0.26.0 (single-threaded `np.fft`/`ndimage`). **Method.** The same function, dtype and input size on both sides
(uniform random pixels 30 to 70); one warm-up call, then five timed calls; the table shows the median and (min–max) in
milliseconds. FFTs reuse their plan (`FFT2D`, `FFT2DDouble`; NumPy caches its own); `rfft2`, `butterworth`, the shifts,
registrations and filters are the public one-call API. **Caveat.** Other builds ran on the machine at the same
time (load average 11 to 25 during these runs, 6 to 8 before): absolute times are noisy, and both sides were measured in the
same minutes so the ratios are comparable but not exact. The sizes are the small (128²) and typical (512²) ones plus one
non-power-of-two (500²); masked registration, which costs eleven transforms of (2n−1)², is at 64² and 256².
**Large sizes (2048², 4096², 8192²) are not measured yet:** `IMAGOKIT_BENCH_LARGE=1` and `--large` add them.

Re-run from a fresh clone of the kit, on a quiet Mac, one benchmark at a time:

```sh
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
IMAGOKIT_BENCH=1 IMAGOKIT_BENCH_OUT=/tmp/swift.json swift test -c release --filter ImagoKitParityTests.benchmarks
.venv/bin/python -I -B tools/benchmarks/bench.py /tmp/python.json
python3 tools/benchmarks/report.py /tmp/swift.json /tmp/python.json
```

The harness is `Tests/ImagoKitParityTests/Benchmarks/Benchmarks.swift` (its one test returns at once unless `IMAGOKIT_BENCH=1`);
`tools/benchmarks/bench.py` is the Python side. "kit / Python" below 1 means the kit is faster.

## Kit CPU against Python

| function | dtype | size | kit CPU ms: median (min–max) | Python ms: median (min–max) | kit / Python |
|---|---|---|---|---|---|
| butterworth | float32 | 128² | 0.0408 (0.0401–0.041) | 0.0765 (0.0731–0.105) | 0.53 |
| butterworth | float32 | 512² | 1.02 (0.969–1.06) | 0.964 (0.907–1.17) | 1.06 |
| butterworth | float64 | 128² | 0.0705 (0.07–0.0752) | 0.0956 (0.0927–0.104) | 0.74 |
| butterworth | float64 | 512² | 1.45 (1.35–1.5) | 1.61 (1.44–2.14) | 0.91 |
| difference_of_gaussians | float32 | 128² | 0.286 (0.279–0.288) | 0.262 (0.242–0.274) | 1.09 |
| difference_of_gaussians | float32 | 512² | 4.46 (4.45–4.53) | 5.29 (5.16–5.49) | 0.84 |
| fft2 | float32 | 128² | 0.0191 (0.0175–0.0612) | 0.0867 (0.0855–0.0941) | 0.22 |
| fft2 | float32 | 500² | 1.53 (1.48–1.73) | 1.92 (1.73–2.08) | 0.80 |
| fft2 | float32 | 512² | 0.537 (0.528–0.537) | 3.61 (3.42–3.77) | 0.15 |
| fft2 | float64 | 128² | 0.0695 (0.0673–0.093) | 0.0759 (0.0716–0.0761) | 0.92 |
| fft2 | float64 | 360² | 0.596 (0.584–0.611) | 0.712 (0.66–0.929) | 0.84 |
| fft2 | float64 | 480² | 1.53 (1.48–1.58) | 1.37 (1.3–1.71) | 1.12 |
| fft2 | float64 | 500² | 1.46 (1.44–1.5) | 1.49 (1.29–1.9) | 0.98 |
| fft2 | float64 | 504² | 1.26 (1.23–1.28) | 1.49 (1.39–2.02) | 0.85 |
| fft2 | float64 | 512² | 2.14 (2.06–2.23) | 2.66 (2.45–2.74) | 0.81 |
| gaussian_filter sigma=3 | float32 | 128² | 0.139 (0.136–0.157) | 0.127 (0.124–0.136) | 1.09 |
| gaussian_filter sigma=3 | float32 | 512² | 2.22 (2.18–2.23) | 2.52 (2.51–2.61) | 0.88 |
| gaussian_filter sigma=3 | float64 | 128² | 0.125 (0.124–0.129) | 0.145 (0.133–0.191) | 0.86 |
| gaussian_filter sigma=3 | float64 | 512² | 1.96 (1.94–2.01) | 3.09 (2.96–3.18) | 0.63 |
| irfft2 | float32 | 128² | 0.0187 (0.0185–0.0192) | 0.0274 (0.0255–0.0311) | 0.68 |
| irfft2 | float32 | 512² | 0.442 (0.435–0.473) | 0.405 (0.389–0.429) | 1.09 |
| irfft2 | float64 | 128² | 0.0334 (0.0331–0.0346) | 0.0377 (0.037–0.0393) | 0.89 |
| irfft2 | float64 | 512² | 0.634 (0.562–0.758) | 0.716 (0.697–0.944) | 0.89 |
| label connectivity=2 | bool | 128² | 0.154 (0.126–0.231) | 0.117 (0.0848–0.167) | 1.32 |
| label connectivity=2 | bool | 512² | 4.37 (4.2–4.64) | 2.06 (2.04–2.1) | 2.13 |
| masked phase_cross_correlation | float32 | 64² | 0.491 (0.469–0.533) | 0.611 (0.574–0.765) | 0.80 |
| masked phase_cross_correlation | float32 | 256² | 9.74 (9.59–10.3) | 11.3 (11.1–11.8) | 0.86 |
| masked phase_cross_correlation | float64 | 64² | 0.696 (0.658–0.722) | 0.94 (0.925–1.09) | 0.74 |
| masked phase_cross_correlation | float64 | 256² | 13.1 (12.8–13.5) | 17.9 (17–20.6) | 0.73 |
| phase_cross_correlation | float32 | 128² | 0.105 (0.102–0.108) | 0.175 (0.164–0.195) | 0.60 |
| phase_cross_correlation | float32 | 512² | 2.15 (2.06–2.23) | 3.35 (3.22–3.68) | 0.64 |
| phase_cross_correlation | float64 | 128² | 0.159 (0.157–0.161) | 0.271 (0.264–0.282) | 0.59 |
| phase_cross_correlation | float64 | 512² | 3.14 (3.02–3.2) | 5.22 (5.04–5.98) | 0.60 |
| phase_cross_correlation upsample=10 | float32 | 128² | 0.281 (0.276–0.288) | 0.245 (0.24–0.262) | 1.15 |
| phase_cross_correlation upsample=10 | float32 | 512² | 4.93 (4.82–5.01) | 3.89 (3.73–4.31) | 1.27 |
| profile_line order=1 width=5 | float32 | 128² | 0.0114 (0.0112–0.0148) | 0.601 (0.594–0.634) | 0.02 |
| profile_line order=1 width=5 | float32 | 512² | 0.0449 (0.0444–0.0474) | 2.43 (2.42–2.71) | 0.02 |
| profile_line order=3 width=5 | float32 | 128² | 0.516 (0.502–0.527) | 0.801 (0.794–0.928) | 0.64 |
| profile_line order=3 width=5 | float32 | 512² | 5.33 (5.28–5.44) | 7.44 (7.21–7.6) | 0.72 |
| rfft2 | float32 | 128² | 0.0155 (0.0147–0.0163) | 0.0379 (0.0369–0.0422) | 0.41 |
| rfft2 | float32 | 512² | 0.43 (0.426–0.458) | 0.908 (0.82–1.17) | 0.47 |
| rfft2 | float64 | 128² | 0.03 (0.0289–0.0307) | 0.0313 (0.0293–0.0324) | 0.96 |
| rfft2 | float64 | 512² | 0.721 (0.684–0.759) | 0.647 (0.589–0.766) | 1.12 |
| shift order=1 | float32 | 128² | 0.0466 (0.0464–0.0478) | 0.158 (0.154–0.159) | 0.29 |
| shift order=1 | float32 | 512² | 0.651 (0.587–0.895) | 2.51 (2.21–2.52) | 0.26 |
| shift order=3 | float32 | 128² | 0.491 (0.479–0.496) | 0.641 (0.605–0.655) | 0.77 |
| shift order=3 | float32 | 512² | 7.69 (7.67–7.79) | 10.9 (10.6–11.1) | 0.71 |
| uniform_filter size=5 | float32 | 128² | 0.0745 (0.0723–0.0846) | 0.0368 (0.0367–0.0381) | 2.02 |
| uniform_filter size=5 | float32 | 512² | 1.17 (1.17–1.18) | 1.27 (1.26–1.29) | 0.93 |
| uniform_filter size=5 | float64 | 128² | 0.052 (0.0517–0.0572) | 0.0466 (0.0455–0.0492) | 1.11 |
| uniform_filter size=5 | float64 | 512² | 0.778 (0.773–0.811) | 1.8 (1.77–2.05) | 0.43 |

## FFT work of 2026-10-22 (real-input transform, mixed radix): before and after

The rows of the main table above for `rfft2`, `irfft2`, `butterworth`, masked and plain `phase_cross_correlation` and `fft2`
at 360², 480², 500² and 504² were re-measured after the changes of 2026-10-22; the other rows are the 2026-10-21 run's. **Method.** Three
rounds, each running the kit before the changes (a copy of the sources of the parent commit), the kit after, and the Python
harness (`tools/benchmarks/bench.py`), in turn, one benchmark at a time; the cells show the median and (min–max) of the fifteen
timed calls of the three rounds, in milliseconds. The machine was shared with other builds (load average 4 to 8): the rows
of the table that the changes did not touch (`gaussian_filter`, `uniform_filter`, `shift`, `profile_line`, `label`, `difference_of_gaussians`)
came out within 3% of their "before" values in the same runs, which is the noise to read the table by.

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

What changed: `rfft2`/`irfft2` pack two real rows into one complex transform (`RealFFT2D.swift`) instead of truncating a complex
transform; `butterworth`, masked and plain `phase_cross_correlation` use them (and `butterworth` builds its mask on the half
spectrum only, with a product for order 2); the Float64 `fft2` takes the mixed-radix plan (`MixedRadixPlanD.swift`) for lengths of
prime factors 2, 3, 5, 7 that vDSP does not serve; the masked correlation transforms at the next fast length as scikit-image does.

**Still above 1.2 times Python**, with the cause as far as it is known:
- `fft2` Float64 480²: 1.12, and the axis is 480 = 15 · 32, a length `vDSP_DFT_zop` serves (the kit keeps that path by design). A batch of
  480 lines of 480 takes 0.49 ms on `vDSP_DFT_zop` and 0.49 ms on the mixed-radix plan, so the 1.5 ms are two such batches
  plus the transposes around the vDSP calls; the mixed-radix plan, which needs no transpose for the columns, is the lever if that
  rule is ever relaxed.
- `phase_cross_correlation upsample=10` Float32: 1.15 (128²) and 1.27 (512²). The FFTs are no longer the cost: the upsampled case takes
  4.9 ms against 2.2 ms for the whole integer-pixel case at 512², and that difference is the Hermitian completion of P plus
  the matrix-multiply DFT (`upsampledDFT`, scalar loops). Not changed yet.
- `irfft2` Float32 512²: 1.09 and `butterworth` Float32 512²: 1.06 are inside the noise of the shared machine.

Not in the table because there is no counterpart on the same inputs: `PeakLocalMax.find` (counterpart `peak_local_max`
needs `maximum_filter`; not measured yet).

## Backend decision per function

Today every function above is CPU (Accelerate, Float64 reference plus Float32 where it exists). **No function has a Metal path,
so no Metal break-even has been measured**; the column says what the CPU table implies and what a large-size run must measure.
A guess is not recorded as a measurement.

| function | backend today | decision |
|---|---|---|
| `fft2` Float32/Float64 | CPU | CPU only at 128² to 512² (0.02 to 2 ms: a Metal dispatch cannot win). Float64 lengths of prime factors 2, 3, 5, 7 run on the mixed-radix plan. Work item: a Metal Float32 FFT prototype, break-even not measured yet (2048², 4096², 8192²) |
| `rfft2`, `irfft2`, `butterworth`, masked registration | CPU | CPU; real-input transform since 2026-10-22 (0.7 ms Float64 at 512², 1.1 times NumPy); a Metal path would have to beat 0.4 to 0.7 ms at 512², which a dispatch cannot, so the break-even is again a question for 2048² and up, not measured yet |
| `gaussian_filter`, `uniform_filter`, `difference_of_gaussians` | CPU | CPU only at 128² to 512² (1 to 4 ms, memory bound). Metal candidate above 2048² for large sigma; not measured yet |
| `shift` order 1 | CPU | CPU only: 0.65 ms at 512² |
| `shift` order 3 (prefilter plus taps) | CPU | CPU; the prefilter is sequential along lines, the tap sum is the Metal-friendly part: candidate above 2048², not measured |
| `phase_cross_correlation` | CPU | CPU; FFT-bound, follows `fft2` |
| `profile_line` | CPU | CPU: small arrays stay on the CPU |
| `label` | CPU | CPU: union-find is not a Metal workload |
| `peak_local_max` | CPU | CPU; the window maximum is a Metal candidate only above 4096² |
