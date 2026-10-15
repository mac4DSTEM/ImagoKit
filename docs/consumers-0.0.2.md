# ImagoKit 0.0.2: what a consumer changes

0.0.2 is slice 1b on top of 0.0.1 (`docs/consumers-0.0.1.md`), additive: `ImagoFFT.MaskedInverseFFT` (with
`MaskedInverseFFT.Spot`, `MaskedInverseFFTError`, `SpotMeasurement.maskSpot(radius:edge:)`) and the new module and product
`ImagoFilters` (`HRTEMFilter`, `HRTEMFilter.Parameters`, `HRTEMFilter.Result`, `HRTEMFilterError`). Nothing in 0.0.1's API
changes. `ImagoFiltersScope`, 0.0.1's placeholder, is gone (it only named the registration). Platforms, dependencies and
build advice as 0.0.1.

**The suite's rule:** an app re-pins at the kit's next minor tag (0.1.0). This file is the plan for that switch, kept current
until then.

## The guarantee

The Float32 path (`precision: .float32`, the default) is macTEM `3d17311`'s arithmetic: `tools/slice-1/apps-parity` compares
`MaskedInverseFFT` and `HRTEMFilter` (both outputs and every stage: the initial background, the median, pass 1's c, the
background, the spectrum, both masks, the flags with their first pass, the passes, the flags per pass, the final d) with
macTEM's on every planted case of `.build/pins-full`, bit for bit (results in `docs/status.md`). The Float64 path is new: the
same steps in Double, within 1e-10 of a float64 NumPy/SciPy reading of the filter (measured about 1e-15).

## macTEM (at 3d17311)

Deleted at the switch, replaced by the import:

| macTEM file | Replaced by | Note |
|---|---|---|
| `Sources/TEMCore/Compute/MaskedInverseFFT.swift` | `ImagoFFT.MaskedInverseFFT` | `compute(_:spots:includeFriedelPartners:precision:)` takes a `Micrograph`; `image` is `SpectrumValues` (`.float32([Float])`); `Failure` is `MaskedInverseFFTError`; `Spot` gains `weight(atDistance:)` |
| `Sources/TEMCore/Compute/HRTEMFilter.swift` | `ImagoFilters.HRTEMFilter` | `Parameters` has the same fields and ranges; `filter(_:parameters:)` on a `Micrograph` returns the `Result` (macTEM's returned two `CalibratedImage`s and the result: use `HRTEMFilter.outputs(_:like:)` for the two images); `absf` and `wiener` are `SpectrumValues`; `applyMask` takes a `precision` and returns `SpectrumValues`; `Failure` is `HRTEMFilterError` (same cases and texts); `Result` also carries `parameters` and `precision` |
| `FFTMask.spot` in `FFTRoomMath.swift` | `SpotMeasurement.maskSpot(radius:edge:)` | the same centroid-minus-centre rule; `FFTMask`'s defaults and ranges (UI) stay in the app |
| `macTEMTests/HRTEMFilterTests.swift`, the mask half of `RadialProfileAndMaskTests.swift` | the kit's tests | keep the app tests of the room's use |
| `tools/hrtem-filter-parity`, the `ifft` part of `tools/fft-parity` | the kit's pins, `full-parity.sh` and apps-parity | their `sources.manifest` groups `hrtem` and `fft` go with them |

Kept in macTEM: the FFT room, its state owner (`FFTAnalysis`, whose `FFTCompute` closures call the kit), `FFTRoomMath`'s
region, bin, chart, aspect and unit rules, `SpectrumDisplay`, `FFTExport`. The product badge "not compared with Digital
Micrograph" stays with the filters.

With 0.0.1's list, macTEM's image-analysis originals all leave: `FFT2D.swift` (its one `copies.manifest` line),
`PowerSpectrum.swift`, `RadialProfile.swift`, `SpotMeasure.swift`, `MaskedInverseFFT.swift`, `HRTEMFilter.swift`.

## mac4DSTEM

No change.
