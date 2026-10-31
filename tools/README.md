# tools

Scripts that generate the test fixtures, check the public API from outside the package, and time the kit against the
Python packages. Run every command from the repository root.

## One Python environment

```sh
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
```

`tools/requirements.txt` includes `tools/oracle/references-lock.txt`: NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0 and
scikit-image's dependencies, exact versions. Every generator below runs as `.venv/bin/python -I -B tools/<folder>/<script>.py`.

## Fixture generators

Each writes JSON (and sometimes binary arrays) under `Tests/*/Fixtures/`; `Tests/Fixtures/README.md` lists every fixture and
the script that writes it. A regenerated fixture that changes is its own commit, with the reason.

| Folder | Script | What it pins |
|---|---|---|
| `alignment` | `pins.py` | spline-order shifts and masked phase correlation |
| `bandpass-channels` | `pins.py` | `butterworth` and `difference_of_gaussians` with `channel_axis` |
| `colormap` | `pins.py` | the four display colour tables and the sampler |
| `csvexport` | `pins.py` | the radial-profile, spots and tidy-results CSV text (see "Not yet from the Python reference") |
| `denoise` | `pins.py` | TV (Chambolle, Bregman) and non-local-means denoising |
| `display` | `pins.py` | display normalisation, thumbnails, scale bars, display mapping, spectrum geometry |
| `fft-real` | `pins.py` | `rfft2`/`irfft2`, 1D `rfft`/`irfft`, mixed-radix lines and planes |
| `fft1d` | `pins.py` | the complex 1D FFT |
| `fftsurface` | `pins.py` | `fft2`/`ifft2` with axes and norm, `fftshift` |
| `filters` | `pins.py` | Gaussian, uniform, difference of Gaussians, Butterworth, windows |
| `gpa` | `planted.py` | planted lattices with known strain (see `tools/gpa/README.md`) |
| `measure` | `pins.py`, `spline_pins.py` | histograms, line profiles, measurement |
| `peaks` | `pins.py`, `inf_pins.py` | `peak_local_max` |
| `precipitation` | `pins.py` | thresholds, binary morphology, `regionprops` |
| `radialchart` | `pins.py` | radial-chart samples and marks |
| `registration-drift` | `pins.py`, `full-parity.sh` | phase correlation, shifts, drift series and aligned sums; `full-parity.sh` runs the parity tests on the full case set |
| `resampling` | `pins.py` | `ndimage.zoom` and `rotate` |
| `rolling-ball-nansafe` | `pins.py` | `rolling_ball` with `nansafe=True` |
| `segmentation` | `pins.py` | `measure.label` and the object moments (see "Not yet from the Python reference") |
| `watershed-edt-rollingball` | `pins.py` | `distance_transform_edt`, `watershed`, `rolling_ball` |
| `window-families` | `pins.py` | the 14 further `get_window` families and their 2D `window` |

The FFT, spot, angle and HRTEM-filter pins (`fft-spot-angle-pins.json`, `hrtem-filter-pins.json` and their inputs) are frozen: their
generator could not run from a clone and was retired (`Tests/Fixtures/README.md`).

### Not yet from the Python reference

Three generators can also take an application's own output as a cross-check, which is circular for the kit:
`csvexport` (`CSVGEN_BINARY`, required: the CSV text has no Python package behind it), `segmentation` (`APP_SEG_BINARY`,
optional: the principal-axis extents and edge flag) and `colormap` (`APP_LUT_JSON`, optional). Without the optional
argument `segmentation` and `colormap` pin the NumPy/scikit-image reading only.

## Package checks

```sh
tools/consumer-check/run.sh     # a separate package that compiles and runs against the public API only
tools/readme-consumer/run.sh    # README.md's Swift program, checked line by line, compiled warning-free and run
tools/build-docs.sh             # the DocC site
```

## Benchmarks

`tools/benchmarks/README.md`.
