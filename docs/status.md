# ImagoKit status

## 0.1.1 preparation (2026-10-16)

- Entire package relicensed to GPL-3.0 by the owner, copyright Paul Lobpreis, 2026; the earlier BSD decision below is historical.

## Now (2026-10-15 17:36 CEST): 0.1.0 candidate at `a4f488b`

- The suite's process changed (owner, 2026-10-15 evening): focused commits with tests that fail before and pass after, one
  review lane, `CHANGELOG.md` with a Migration paragraph per release, CI the gate. Slice 2 had its one review (approve with
  changes): the `reflect`-mode defect it found is fixed test-first (`a44e674`); its other notes are in the CHANGELOG's known
  limits. Gate, both full parities, consumer check (84) and README consumer green at `a4f488b`. **0.1.0 candidate =
  `a4f488b`** once the owner's push has a green CI run.
- App switch patches (in the coordinator's notes folder): mac4DSTEM (`imagokit-mac4dstem-switch.patch`, reviewed: approve with
  changes, folded into its note); macTEM (`imagokit-mactem-switch.patch`, reviewed: changes requested, being revised).

## Earlier (2026-10-15 17:16 CEST): slice 2 at its green checkpoint

- **Released:** 0.0.2 tagged at `2cc5f07` (slice 1), public; CI green on the runner (run 38753074813: Xcode 26.6, Swift 6.3.3,
  39 tests, consumer check 45, README consumer).
- **Slice 2** (`docs/registration-slice-2.md`, amendments 1 and 2), local commits `ba70a20` (pins), `900fb9c` (code and tests),
  `f2764f2` (consumer check 84, README program, parity rows), implemented by a Sonnet lane: `ImagoAlignment` with
  `PhaseCorrelation`, `ImageShift` (orders 0 and 1, seven modes, Fourier shift), `DriftSeries`, `AlignedSum`, on Float32 and
  Float64. Gate green and stamped at `f2764f2` (81 test functions, 0 warnings); `tools/slice-2/full-parity.sh` green on 330
  cases; Float64 shifts equal scikit-image's on all 330. **Not yet done for slice 2: the mutants and the Gate B refuter**
  (paused by the owner's priority, 16:44).
- **For the refuter, from amendments 1 and 2 (deviations from the registration):** the plant's continuum was replaced (the
  registered one made phase correlation fail at σ 5); the `.mean` drift band is 0.20 px, not 0.05 (scikit-image itself errs
  0.185 px there) — a band set after measuring, to be argued; the near-tie rule became a per-case measured-noise rule (117
  of 330 Float32 cases flagged and held to their tie box); error and phase bands scale with each case's measured noise; the
  public Float64 call returns a `Micrograph` (Float32), so the 1e-12 band holds on the engine and 6.5e-8 on the call; the
  sources existed uncommitted while the pins were finalised (no source changed after the first test run). The 96² drift
  stack sits at 0.93–0.94 of its bands.
- **Slice 1** as at 0.0.2: see below and `docs/registration-slice-1.md` (amendments 1–9).

### Slice 1 (0.0.2)

- **Public** at github.com/mac4DSTEM/ImagoKit (the owner, 16:10); BSD-3-Clause accepted (coordinator, 16:10).
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

1. Slice 2's mutants (registration-slice-2 section 5) on a copy of `f2764f2` and one Gate B refuter (the deviations above).
2. 0.1.0: the API pass (slice-1 amendments 5 item 7 and 9 item 6), CI pinned to Xcode 26.6, `docs/consumers-0.1.0.md`;
   macTEM's switch (a patch prepared for macTEM's session in the coordinator's notes folder, `imagokit-mactem-switch.patch`).
3. Gate D on the vDSP first-call finding (amendment 8).
4. The real-data check when a raw HRTEM image exists (amendment 4).

## Waiting on the owner

- Push `main` (slice 2's local commits; not a tag candidate until its refuter).
- Submit the URL to the Swift Package Index.
- A raw HRTEM image (Velox `.emd` or DM4, counts) in `data/tem/` for the real-data check (the HRTEM TIFFs are 8-bit renders).

## Open

- Gate D: a first call of a vDSP 1D FFT or DFT size can differ by one Float32 ulp (amendment 8).
- Float32 bands are claims on the planted cases (amendment 5 item 3, amendment 9 item 7).
- API pass for 0.1.0: `FFT2D.transform` traps on a size mismatch; `angle(qa:qb:)` range; subnormal scales; non-finite pixels in
  spots; `Spot` not validated; n² overflow traps in the filters; which helpers stay public.

## Unverified

- CI has not passed yet. iOS 18 built, never run. macOS 15 declared, built and tested on macOS 27 only.
