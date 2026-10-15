# ImagoKit

Image analysis for electron micrographs, in Swift, for macOS and iPadOS: the Fourier transform of an image with its
calibration, lattice spacings and angles measured from FFT spots, Bragg (masked inverse FFT) filtering, the HRTEM
background-subtraction and Wiener filters, and, next, drift correction and stack alignment. Each number is defined once,
pinned against the Python packages the field already uses, and checked against planted lattices whose answer is known.

**Status: pre-release.** Slice 1 is in: `ImagoCore`, `ImagoFFT` (spectrum, radial profile, spot spacings and angles, the
masked inverse FFT) and `ImagoFilters` (the HRTEM filters), each on a Float32 and a Float64 path
([`docs/registration-slice-1.md`](docs/registration-slice-1.md)). Slice 2 (`ImagoAlignment`: phase correlation, drift and
stack alignment) is registered ([`docs/registration-slice-2.md`](docs/registration-slice-2.md)) and not yet written.

## What it computes

| Module | Scope | Reference | Slice |
|---|---|---|---|
| `ImagoCore` | the calibrated image (pixels, per-axis scale, unit and origin, crops); length units; the exact N-point 2D DFT for every image size, Float32 on Accelerate's vDSP and a Float64 reference path | `numpy.fft.fft2` / `ifft2` | 1 |
| `ImagoFFT` | the centred magnitude and power spectrum with `fftshift(fftfreq(n, d))` axes per axis; the radial profile against physical \|q\|; a spot's position (snap, sub-pixel centroid of a Hann-windowed spectrum), its spacing d = 1/\|q\| and the angle between two spots; the masked inverse FFT with raised-cosine spots and Friedel partners | NumPy, and planted lattices | 1 |
| `ImagoFilters` | the HRTEM average background subtraction filter (ABSF) and Wiener filter (Kilaas 1998, as D. R. G. Mitchell's DigitalMicrograph script describes them) | a float64 NumPy/SciPy reading of the same steps (`scipy.ndimage.median_filter`, `map_coordinates`) | 1 |
| `ImagoAlignment` | phase cross-correlation with sub-pixel refinement, sub-pixel shifts, drift series and aligned stack sums | `skimage.registration.phase_cross_correlation`, `scipy.ndimage.shift`, `fourier_shift` | 2 |

Later slices, registered when they start: GPA strain (Hÿtch, Snoeck and Kilaas 1998), atom-column finding (Atomap, GPL-3.0,
in a module of its own), bandpass and denoising filters (scikit-image, SciPy).

## Conventions (the short version)

- Pixels are row-major, x (column) fastest, row 0 first. A pixel's physical position is `(index − origin) · scale`.
- The zero frequency of a centred spectrum sits at index `n/2` (integer division) on each axis, odd and even `n` alike, as
  `numpy.fft.fftshift` puts it. The axis value at index `i` is `(i − n/2) / (n · d)`, `d` that axis's own scale, so a
  non-square or anisotropic image has different spacings along x and y.
- The forward transform is unnormalised and the inverse divides by `nx · ny`, as NumPy's. Nothing is windowed, padded or
  mean-removed unless a function says so on its signature.
- Every Float32 result states its registered distance from NumPy's float64 result; speed never changes a number silently.

## Install

```text
// Package.swift
dependencies: [
    .package(url: "https://github.com/mac4DSTEM/ImagoKit.git", exact: "0.0.1"),
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
on one spot, and the HRTEM filters. This program is compiled and run as it stands by `tools/readme-consumer` (every line
below is checked against that consumer's source).

```swift
import Foundation
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
```

## Checking it yourself

```bash
tools/test.sh                      # the gate: swift test -c release
tools/consumer-check/run.sh        # a separate package that uses only the public API
tools/readme-consumer/run.sh       # this README's program, compiled and run
tools/hooks/test-no-sleep-poll.sh  # the maintainers' shell guard
```

The fixtures the tests read are generated by `tools/slice-*/pins.py` against the pinned NumPy, SciPy and scikit-image
(`tools/oracle/references-lock.txt`; `tools/oracle/provision.sh` builds the environment).

## Licence

BSD-3-Clause ([`LICENSE`](LICENSE)), the licence of NumPy, SciPy and scikit-image. [`NOTICE`](NOTICE) says where the code
comes from and why it is permissive. A future port of GPL-3.0 code (for example Atomap) will be a separate module carrying
GPL-3.0, named as such here and in `Package.swift`; the other modules stay BSD-3-Clause.

## Not claimed

No result is validated on a microscope. The HRTEM filters have never been compared with DigitalMicrograph. Agreement with
NumPy is agreement; the planted lattices and closed forms are the truth the tests hold.
