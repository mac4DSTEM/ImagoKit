# ImagoKit 0.0.1: what a consumer changes

0.0.1 is slice 1a (`docs/registration-slice-1.md`): `ImagoCore` (`Micrograph`, `AxisCalibration`, `PixelCrop`, `LengthUnit`,
`Precision`, `FFT2D`, `FFT2DDouble`, `FFT2DTransform`) and `ImagoFFT` (`PowerSpectrum`, `SpectrumValues`, `RadialProfile`,
`SpotMeasurement`, `LatticePair`). `ImagoFilters` (HRTEM filters) and the masked inverse FFT come in slice 1b; `ImagoAlignment`
in slice 2. Platforms macOS 15 and iOS 18; Foundation and Accelerate only. Build the kit optimised (Release, or `-O` for the
dependency): the manifest carries no flags.

**The suite's rule (owner, 2026-10-15 13:10):** an app re-pins once per kit at its next minor tag (0.1.0), not at every
0.0.x, unless a number is wrong today or a room is blocked. Nothing here is wrong in the apps today, so this file is the
plan for that switch, kept current until 0.1.0; the switch itself is a tier-1 lane in the app's own session, with the app's
pinned tests as the second proof.

## The guarantee

The Float32 path (`precision: .float32`, the default) is the apps' code in the same order of operations: the apps-parity
check (`tools/slice-1/apps-parity`, registration section 6) compares it with macTEM `3d17311`'s TEMCore on every planted
case; bit for bit is the quick check (D12: if it fails, verify within the written tolerance against the Python reference).
A consumer that switches gets the same bits.

## macTEM (at 3d17311)

Deleted at the switch, replaced by the import:

| macTEM file | Replaced by | Note |
|---|---|---|
| `Sources/TEMCore/Compute/FFT2D.swift` (a verbatim copy; its `copies.manifest` line goes) | `ImagoCore.FFT2D` | public; same `transform(re:im:forward:scaleInverse:)` and `fftfreq` |
| `Sources/TEMCore/Compute/PowerSpectrum.swift` (with `PixelCrop` and `CalibratedImage.cropped`) | `ImagoFFT.PowerSpectrum`, `ImagoCore.PixelCrop`, `Micrograph.cropped` | `magnitude` and `power` are `SpectrumValues` (`.float32([Float])`; `doubles`, a `Double` subscript); `Failure` is `PowerSpectrumError` |
| `Sources/TEMCore/Compute/RadialProfile.swift` | `ImagoFFT.RadialProfile` | `compute` throws `RadialProfileError` instead of returning nil; `RadialChart.points` is `RadialProfile.points` |
| `Sources/TEMCore/Compute/SpotMeasure.swift` | `ImagoFFT.SpotMeasurement` | `clickCol:` is `clickColumn:`; `Failure` is `SpotMeasurementError` (same cases, `mixedUnits` carries both units); `Options` is gone (it existed for the harness's controls); new: `dNanometres`, `qPerNanometre`, `width`, `height`, `precision`, the angle and `LatticePair` |
| `macTEMTests/PowerSpectrumTests.swift`, `SpotMeasureTests.swift`, the profile half of `RadialProfileAndMaskTests.swift` | the kit's tests | keep the app tests that test the room's use of them |
| `tools/fft-parity` (spectrum and profile cases), `tools/spot-parity` | the kit's pins and `full-parity.sh` | their `sources.manifest` groups `fft` and `spot` shrink; the harnesses can stay as the app's consumer check until 1b moves the masked inverse |

Kept in macTEM: `CalibratedImage` (the reader's image with its file metadata) with a one-line conversion to `Micrograph`
(`Micrograph(width:height:pixels:x:y:)`, the axes as `AxisCalibration(scale:unit:origin:)`); `FFTRoomMath` (the room's
rules: the region, a click's bin, the chart, the drawn aspect, the units shown — UI rules, no reference); `SpectrumDisplay`,
`FFTExport` and the room. `FFTMask.spot` and `MaskedInverseFFT`, `HRTEMFilter` wait for slice 1b.

Behaviour differences, none numeric (registration amendment 5, item 6): `RadialProfile.compute` throws instead of
returning nil; refusal payloads carry the units; `mixedUnits` compares unit strings as written. One string differs: `SpotMeasurement.reciprocalUnits` of an image whose axis unit is already reciprocal
("1/nm", a diffraction pattern) is "nm", where macTEM's `SpotMeasure` wrote "1/1/nm". No planted case exercises it.

## mac4DSTEM (at 7a0f727f)

`mac4DSTEM/Core/Compute/FFT2D.swift` is the origin of `ImagoCore.FFT2D` (same arithmetic). Twelve analysis files call it
(DPC, disk detection, Friedel origin, origin calibration, the parallax family, probe kernels, ptychography preparation and
single-slice ptychography, rotation calibration). Most of those move to CuboKit, which carries its own Double FFT
(`CuboPhase`): which FFT the suite keeps for 4D-STEM is a coordinator question (the suite ROADMAP's "shared numerics layer"
candidate), not this kit's. Until then mac4DSTEM needs no change; `ImagoCore.FFT2D` is a drop-in when it chooses.
