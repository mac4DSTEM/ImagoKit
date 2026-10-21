# ImagoKit

Image numerics: FFT 1D/2D (Double and Float), profiles, filters, alignment, measurement, segmentation, colormaps. Pure Swift, no UI, standalone; the suite's rules are in `../CLAUDE.md`. Start with `README.md`,
`docs/parity.md` and the task.

## Reference
NumPy 2.5.4, SciPy and scikit-image at the versions recorded in `tools/slice-*/pins.py` (fetched by the apps' `tools/lib/fetch-*.sh` into `References/`); planted lattices with a known answer beside them. Fixtures are generated from it (`tools/*/pins.py`) and committed under `Tests/*/Fixtures/`; a difference inside
the reference's own tolerance is agreement, outside it a bug. Readers, I/O, indices and metadata exact.

## Build and test
```sh
swift build
swift test -c release --parallel --skip ImagoKitParityTests   # per-session gate: the fast unit target
swift test -c release --parallel   # full suite with ImagoKitParityTests: before a tag and in the nightly (Debug is many times slower)
tools/consumer-check/run.sh    # a separate package that compiles against the public API
```
Never `swift build --build-tests -c release` (fails on this toolchain). Real-data tests skip when the data is absent.

## Modules
`ImagoCore` (images, FFT2D, profiles, measurement, colormaps), `ImagoFFT` (FFT1D), `ImagoFilters` (HRTEM and image filters), `ImagoAlignment` (phase correlation, drift, stacks).

## Rules of the kit
- Public API only; nothing names an app or an app type; error messages never say "mac4DSTEM" or "macTEM".
- Rows and columns, not x and y. Every accelerated path keeps the CPU float64 path as its reference (D16).
- A new or changed number gets a test against the reference at its upstream tolerance, and a row in `docs/parity.md`.
- The Double FFT runs on `vDSP_DFT_zop` for lengths 2^n x {1,3,5,15}, Bluestein for the rest; keep both paths tested.

## Closeout
`swift test -c release` green; `CHANGELOG.md` entry with a Migration paragraph where a consumer must change something;
version bumped in the entry; one commit on `main` whose last paragraph is the hand-off. Paul tags with
`../macEM-suite/tools/ship.sh` (tag = CHANGELOG top version).
