# ImagoKit status

## Now (2026-10-15 16:31 CEST): slice 1 done, 0.0.2 candidate; slice 2 registered

- **Public** at github.com/mac4DSTEM/ImagoKit (the owner, 16:10); BSD-3-Clause accepted (coordinator, 16:10). The owner's
  push stopped at `465c473`; the commits after it are local until the owner pushes.
- **Slice 1** (tier 1, `docs/registration-slice-1.md`, amendments 1–9):
  - `ImagoCore`: `Micrograph`, `AxisCalibration`, `PixelCrop`, `LengthUnit`, `Precision`, `FFT2D` (vDSP Float32),
    `FFT2DDouble` (the reference path).
  - `ImagoFFT`: `PowerSpectrum`, `SpectrumValues`, `RadialProfile`, `SpotMeasurement` (with the angle between spots and
    `maskSpot`), `LatticePair`, `MaskedInverseFFT`.
  - `ImagoFilters`: `HRTEMFilter` (ABSF and Wiener, Kilaas 1998 as Mitchell's script describes; never compared with
    DigitalMicrograph).
- **Checks at `4ba1a74`** (the gate's stamp):
  - Gate `tools/test.sh`: 39 test functions, 0 warnings, both precisions on every committed case.
  - Full set `tools/slice-1/full-parity.sh`: green (fft 19, spot 41, angle 30, hrtem 41 cases).
  - Apps parity `tools/slice-1/apps-parity/run.sh`: the Float32 path equals macTEM `3d17311` bit for bit, 19,773,475 of
    19,773,475 values (warm pass).
  - Consumer check: 45 checks. README consumer: the README's program compiled and run. iOS: `xcodebuild` for generic iOS
    succeeds with no warnings (not run on a device or simulator).
- **Gate B**: 1a upheld with findings (amendment 5), 1b upheld with findings (amendment 9); the should-fix items are fixed.
  **Mutants**: 1a 35 (33 caught, 2 equivalent) + 5 follow-ups; 1b 27 (26 caught after the follow-ups, 1 equivalent, 1 held by
  apps parity) + 8 follow-ups (`docs/slice-1-mutants-2026-10-15.md`).
- **Pins** `tools/slice-1/pins.py` (memcap, peak 534 MB): macTEM's fft-, spot- and hrtem-filter-parity harnesses at
  `3d17311` plus 30 angle plants; committed 9.30 MB of a 10 MB budget; the rest in `.build/pins-full`.
- **Slice 2** registered (`docs/registration-slice-2.md`): phase correlation (scikit-image 0.26.0), shifts (SciPy 1.18.1),
  drift series and aligned sums. No code yet.
- **0.1.0 bar**: slices 1 (done); `docs/parity.md` filled for slice 1; the consumer check (separate package); CI and `.spi.yml`;
  DocC per module; a stand-alone README with its consumer; the platform floor macOS 15 / iOS 18. Missing: CI green on the
  runner (the fix is in `35e0dcf`, not yet pushed), the app importing it (macTEM's switch, `docs/consumers-0.0.1.md` and
  `-0.0.2.md`), the API items of amendments 5 and 9.
- **Consumers**: `docs/consumers-0.0.1.md`, `docs/consumers-0.0.2.md`: at its switch macTEM deletes `FFT2D.swift` (its one
  `copies.manifest` line), `PowerSpectrum`, `RadialProfile`, `SpotMeasure`, `MaskedInverseFFT`, `HRTEMFilter` and
  `FFTMask.spot`, their tests and harnesses. mac4DSTEM: no change.

## Next

1. The owner pushes; CI on the runner (Xcode 26.6) must go green — `35e0dcf` fixed the first run's type-check timeout; any
   further runner-only failure is tier 3.
2. Slice 2's pins (`tools/slice-2/pins.py`) and code, then 0.1.0 with the API pass (amendments 5 item 7 and 9 item 6) and
   macTEM's switch in macTEM's session.
3. Gate D on the vDSP first-call finding (amendment 8).
4. The real-data check when a raw HRTEM image exists (amendment 4).

## Waiting on the owner

- Push `main` (8 commits after `465c473` at closeout).
- Tag **0.0.2 at the closeout head** once its CI run is green, and skip 0.0.1: `5a40748` fails on the runner's Xcode 26.6
  (the type-check timeout fixed in `35e0dcf`). `git tag -a 0.0.2 <head> -m "ImagoKit 0.0.2: slice 1" && git push origin 0.0.2`.
- Submit the URL to the Swift Package Index.
- A raw HRTEM image (Velox `.emd` or DM4, counts) in `data/tem/` for the real-data check (the HRTEM TIFFs are 8-bit renders).

## Open

- Gate D: a first call of a vDSP 1D FFT or DFT size can differ by one Float32 ulp (amendment 8).
- Float32 bands are claims on the planted cases (amendment 5 item 3, amendment 9 item 7).
- API pass for 0.1.0: `FFT2D.transform` traps on a size mismatch; `angle(qa:qb:)` range; subnormal scales; non-finite pixels in
  spots; `Spot` not validated; n² overflow traps in the filters; which helpers stay public.

## Unverified

- CI has not passed yet. iOS 18 built, never run. macOS 15 declared, built and tested on macOS 27 only.
