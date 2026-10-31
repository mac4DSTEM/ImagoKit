<p align="center"><img src="docs/banner.png" width="100%" alt="ImagoKit: image analysis for electron micrographs, in Swift"></p>

# ImagoKit

[![CI](https://github.com/mac4DSTEM/ImagoKit/actions/workflows/ci.yml/badge.svg)](https://github.com/mac4DSTEM/ImagoKit/actions/workflows/ci.yml)
[![Docs](https://github.com/mac4DSTEM/ImagoKit/actions/workflows/docs.yml/badge.svg)](https://mac4dstem.github.io/ImagoKit/documentation/)
[![Licence: GPL-3.0-only](https://img.shields.io/badge/licence-GPL--3.0--only-blue.svg)](LICENSE)

Image analysis for electron micrographs, in Swift, for macOS and iPadOS: the Fourier transform of an image with its
calibration, lattice spacings and angles measured from FFT spots, Bragg (masked inverse FFT) filtering, the HRTEM
background-subtraction and Wiener filters, drift correction and stack alignment, thresholds, morphology, segmentation and
object statistics, denoising, zoom and rotate, and GPA strain. Each number is defined once, pinned against the Python
packages the field already uses, and checked against planted lattices whose answer is known.

**Status: pre-release (0.x), current version 0.8.0.** Four modules: `ImagoCore`, `ImagoFFT`, `ImagoFilters` and
`ImagoAlignment`; most functions have a Float32 and a Float64 path. The supported functions, set against the reference
packages' documented API, are listed in [`docs/parity.md`](docs/parity.md); the API reference is the
[DocC site](https://mac4dstem.github.io/ImagoKit/documentation/). The definitions and validation notes behind each number
are in [`docs/`](docs/).

**Platform.** Apple platforms only today (macOS 15 and iOS 18 or newer): the FFT engine uses Accelerate's vDSP, and the
package is built and tested on macOS only. Linux is not supported yet.

## What it computes

| Module | Scope | Reference | Since |
|---|---|---|---|
| `ImagoCore` | the calibrated image (pixels, per-axis scale, unit and origin, crops); length units; the exact N-point 2D DFT for every image size, Float32 on Accelerate's vDSP and a Float64 reference path | `numpy.fft.fft2` / `ifft2` | 0.0.1 |
| `ImagoFFT` | the centred magnitude and power spectrum with `fftshift(fftfreq(n, d))` axes per axis; the radial profile against physical \|q\|; a spot's position (snap, sub-pixel centroid of a Hann-windowed spectrum), its spacing d = 1/\|q\| and the angle between two spots; the masked inverse FFT with raised-cosine spots and Friedel partners | NumPy, and planted lattices | 0.0.1 |
| `ImagoFilters` | the HRTEM average background subtraction filter (ABSF) and Wiener filter (Kilaas 1998, as D. R. G. Mitchell's DigitalMicrograph script describes them) | a float64 NumPy/SciPy reading of the same steps (`scipy.ndimage.median_filter`, `map_coordinates`) | 0.0.2 |
| `ImagoAlignment` | phase cross-correlation with sub-pixel refinement, sub-pixel shifts, drift series and aligned stack sums | `skimage.registration.phase_cross_correlation`, `scipy.ndimage.shift`, `fourier_shift` | 0.1.0 |
| `ImagoCore` | `FFTSurface`: `fft2`/`ifft2` with `axes` and `norm`, `rfft2`/`irfft2`, `fftshift`/`ifftshift` | `numpy.fft` | 0.6.0 |
| `ImagoFilters` | `SeparableFilters` (Gaussian, uniform), `BandpassFilters` (Butterworth, difference of Gaussians), `WindowFunction`, `PeakLocalMax` | `scipy.ndimage`, `skimage.filters`, `scipy.signal.get_window`, `skimage.feature.peak_local_max` | 0.6.0 |
| `ImagoAlignment` | spline-order shifts (orders 0 to 5) and masked phase correlation | `scipy.ndimage.shift`, `phase_cross_correlation` with masks | 0.6.0 |
| `ImagoFilters` | `Thresholding`: Otsu, Li, Yen, triangle, mean, ISODATA, local, hysteresis | `skimage.filters` | 0.7.0 |
| `ImagoCore` | `BinaryMorphology`, labelled `ObjectStatistics` (sums, extrema, centre of mass, areal density, `N_V`), `Segmentation.regionProperties` (perimeter, Feret, solidity, Euler number, intensity) | `scipy.ndimage`, `skimage.measure.regionprops` | 0.7.0 |
| `ImagoCore` | `DistanceTransform`, `Segmentation.watershed` and `markers`, `RollingBall` | `scipy.ndimage.distance_transform_edt`, `skimage.segmentation.watershed`, `skimage.restoration.rolling_ball` | 0.8.0 |
| `ImagoFilters` | `Denoise` (total variation by Chambolle and Bregman, non-local means); `GeometricPhaseAnalysis` (GPA strain; planted truth only, not validated on real data) | `skimage.restoration`; planted lattices | 0.8.0 |
| `ImagoAlignment` | `ImageResampling`: `zoom` and `rotate` (2D, orders 0 to 5) | `scipy.ndimage.zoom`, `rotate` | 0.8.0 |
| `ImagoFFT` / `ImagoCore` | 1D `rfft`/`irfft`; a real-input 2D path and a mixed-radix Float64 plan (speed; same numbers) | `numpy.fft` | 0.8.0 |

`docs/benchmarks.md` has the timings against the Python packages. Planned, not yet available: atom-column finding (in a
module of its own); see [`ROADMAP.md`](ROADMAP.md).

## Complex 1D FFT

`ImagoFFT` also provides `FFT1D` (Float) and `FFT1DDouble` (Double). Both accept a positive power-of-two
length, transform caller-owned split-complex arrays in place, and are Sendable. A plan may be shared across
concurrent calls with separate buffers.

```swift
import ImagoFFT

var real = [Float](repeating: 0, count: 64)
var imaginary = real
real[0] = 1
let fft = FFT1D(n: real.count)!
fft.transform(re: &real, im: &imaginary, forward: true)
fft.transform(re: &real, im: &imaginary, forward: false)
// Both directions are unnormalised: divide these inverse results by Float(fft.n).
```

`FFT1D.nextPow2(_:)` returns the smallest power of two at least its argument (one for arguments ≤ 1);
values above the largest positive power of two representable by `Int` violate its precondition.
Length one is the identity. Initialisation returns `nil` for unsupported lengths or allocation failure;
transform buffers must each have exactly `n` elements. No windowing, padding or centring is implicit.
NumPy 2.5.4 pins cover lengths 1, 2, 4, 8, 64, 256 and 1024. The tests state the complex-bin tolerances:
1e-5 for Float and 1e-12 for Double, with an absolute floor for near-zero bins.

## Conventions (the short version)

- Pixels are row-major, x (column) fastest, row 0 first. A pixel's physical position is `(index − origin) · scale`.
- The zero frequency of a centred spectrum sits at index `n/2` (integer division) on each axis, odd and even `n` alike, as
  `numpy.fft.fftshift` puts it. The axis value at index `i` is `(i − n/2) / (n · d)`, `d` that axis's own scale, so a
  non-square or anisotropic image has different spacings along x and y.
- The 2D forward transform is unnormalised and its inverse divides by `nx · ny`, as NumPy's. Nothing is windowed, padded or
  mean-removed unless a function says so on its signature.
- Every Float32 result states its registered distance from NumPy's float64 result; speed never changes a number silently.

## Install

```text
// Package.swift
dependencies: [
    .package(url: "https://github.com/mac4DSTEM/ImagoKit.git", exact: "0.8.0"),
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "ImagoFFT", package: "ImagoKit"),
    ]),
]
```

Swift 6.2 tools or newer; macOS 15 or iOS 18. Foundation and Accelerate only; no other dependency. Build your app in Release
or with `-O` for the kit: the manifest carries no optimisation flags (a versioned dependency cannot).

## The first ten minutes

A lattice image with two spots: its spectrum, its radial profile, both spacings and the angle between them, a Bragg filter
on one spot, the HRTEM filters, the registration of a drifted copy, the particles of a field (denoise, threshold,
watershed), its rolling-ball background and a binary opening, and the GPA strain of the lattice. This program is compiled and run as it stands by `tools/readme-consumer` (every line
below is checked against that consumer's source in CI).

```swift
import Foundation
import ImagoAlignment
import ImagoCore
import ImagoFFT
import ImagoFilters

// A 256 × 256 image, 0.02 nm pixels: two lattice fringes on a constant background.
let n = 256
var pixels = [Float](repeating: 0, count: n * n)
for row in 0..<n {
    for col in 0..<n {
        let x = Double(col) / Double(n), y = Double(row) / Double(n)
        pixels[row * n + col] = Float(100 + 10 * cos(2 * Double.pi * (40 * x + 12 * y)) + 6 * cos(2 * Double.pi * (-15 * x + 35 * y)))
    }
}
let image = try Micrograph(width: n, height: n, pixels: pixels,
                           x: AxisCalibration(scale: 0.02, unit: "nm"), y: AxisCalibration(scale: 0.02, unit: "nm"))

// The centred spectrum (NumPy's fftshift(fft2(x))) and its radial profile against physical |q|.
let spectrum = try PowerSpectrum.compute(image)
let profile = try RadialProfile.compute(spectrum)
print("spectrum", spectrum.width, "×", spectrum.height, "in", spectrum.unitsX, "; profile bins:", profile.points.count)

// Click near each spot (row, column of the centred spectrum): the estimator snaps and refines to sub-pixel.
let a = try SpotMeasurement.measure(image, clickRow: n / 2 + 12, clickColumn: n / 2 + 40)
let b = try SpotMeasurement.measure(image, clickRow: n / 2 + 35, clickColumn: n / 2 - 15)
let pair = try LatticePair.measure(a, b)
print(String(format: "d₁ = %.4f nm, d₂ = %.4f nm, angle = %.2f°", a.dNanometres!, b.dNanometres!, pair.angle))

// Keep only the first lattice (a Bragg filter), then the HRTEM background-subtraction and Wiener filters.
let bragg = try MaskedInverseFFT.compute(image, spots: [a.maskSpot(radius: 3, edge: 2)])
let filtered = try HRTEMFilter.filter(image)
print("Bragg-filtered image:", bragg.image.count, "pixels; HRTEM filters ran", filtered.passes, "passes")

// Drift: a field of three blobs, displaced by a known sub-pixel amount, is registered back (phase correlation refined to
// 1/20 pixel) and the pair is summed after alignment.
var blobs = [Float](repeating: 5, count: n * n)
for (r0, c0) in [(60.0, 70.0), (120.0, 180.0), (190.0, 90.0)] {
    for row in 0..<n {
        for col in 0..<n {
            let d2 = (Double(row) - r0) * (Double(row) - r0) + (Double(col) - c0) * (Double(col) - c0)
            blobs[row * n + col] += Float(100 * exp(-d2 / 18))
        }
    }
}
let field = try Micrograph(width: n, height: n, pixels: blobs)
let drifted = try ImageShift.fourierShift(field, by: PixelShift(row: -2.5, column: 1.25))
let drift = try PhaseCorrelation.register(reference: field, moving: drifted, upsampleFactor: 20, normalization: .unnormalized)
let summed = try AlignedSum.sum([field, drifted], shifts: [.zero, drift.shift])
print("registered drift:", drift.shift.row, drift.shift.column, "px; aligned sum of", summed.frameCount, "frames")

// Particles in the same field: total-variation denoising, Otsu's threshold, one marker per blob from peak_local_max, and a
// marker-controlled watershed inside the thresholded mask.
let smooth = try Denoise.totalVariationChambolle(blobs, rows: n, columns: n, weight: 2)
let level = Float(try Thresholding.otsu(.float32(smooth)))
let foreground = Thresholding.mask(smooth, threshold: level)
let peaks = try PeakLocalMax.find(smooth, rows: n, columns: n, minDistance: 10, thresholdAbs: Double(level))
let seeds = try Segmentation.markers(peakRows: peaks.map(\.row), peakColumns: peaks.map(\.column), rows: n, columns: n)
let particles = try Segmentation.watershed(image: smooth.map { -Double($0) }, markers: seeds, rows: n, columns: n,
                                           mask: foreground)
print("particles:", Set(particles.filter { $0 > 0 }).count, "above Otsu's level", level)

// The field's background by a rolling ball of radius 20 px, and the mask cleaned by a binary opening.
let background = try RollingBall.background(image: blobs, rows: n, columns: n, radius: 20)
let cleaned = try BinaryMorphology.opening(foreground, width: n, height: n)
print("background at a corner:", background[0], "; mask pixels before and after the opening:",
      foreground.filter { $0 }.count, cleaned.filter { $0 }.count)

// GPA strain of the lattice image from its two strongest reflections (planted truth only; not validated on real data).
let lattice = pixels.map(Double.init)
let reflections = try GeometricPhaseAnalysis.findReflections(lattice, rows: n, columns: n)
let strain = try GeometricPhaseAnalysis.analyze(lattice, rows: n, columns: n, g1: reflections.g1, g2: reflections.g2)
print("GPA reflections:", reflections.g1, reflections.g2, "; strain at the centre:", strain.strainRR[n / 2 * n + n / 2])
```

## Checking it yourself

```bash
swift test -c release              # the test suite
tools/consumer-check/run.sh        # a separate package that uses only the public API
tools/readme-consumer/run.sh       # this README's program, compiled and run
```

The fixtures the tests read are generated by the `pins.py` scripts under `tools/` against the pinned NumPy, SciPy and
scikit-image (the versions are in `tools/oracle/references-lock.txt`; install them into a virtual environment with
`pip install -r tools/oracle/references-lock.txt`). Numbers are held to the Python
reference within its own tolerance; `docs/parity.md` lists each function, its state and its tolerance. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) before changing a number.

## Known limits

- iOS 18 builds but has never been run on a device or simulator; macOS 15 is declared but built and tested on macOS 27 only.
- A first call of a vDSP 1D FFT or DFT size can differ by one Float32 ulp.
- Float32 tolerances are claims on the planted and pinned cases, not proofs for every input.
- Spline-order shifts refuse the `constant` and `wrap` modes; the masked registration returns the shift only, and a symmetric pair with exactly tied maxima is not reproducible (`docs/open-items.md`).
- No number is validated on a microscope; the HRTEM filters have never been compared with DigitalMicrograph.
- Not yet checked on a raw (counts) HRTEM image; the available TIFFs are 8-bit renders.

## Please cite

If ImagoKit contributes to published work, please cite it using [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this
repository" button reads it), and cite the reference packages whose definitions it follows: NumPy, SciPy and
scikit-image.

## Licence

GPL-3.0-only ([`LICENSE`](LICENSE)). Copyright (c) 2026 Paul Lobpreis.
[`NOTICE`](NOTICE) records the provenance and reference packages.

## Not claimed

No result is validated on a microscope. The HRTEM filters have never been compared with DigitalMicrograph. Agreement with
NumPy is agreement; the planted lattices and closed forms are the truth the tests hold.
