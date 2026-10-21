// swift-tools-version: 6.2
// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Paul Lobpreis
// ImagoKit — image analysis for electron micrographs on Apple platforms: the calibrated image and the exact-length 2D
// FFT (ImagoCore); the centred power spectrum, its radial profile, FFT-spot spacings and angles, and the masked inverse
// FFT (ImagoFFT); the HRTEM background-subtraction and Wiener filters (ImagoFilters); drift and stack alignment
// (ImagoAlignment). NumPy, SciPy and scikit-image are the references; docs/registration-slice-1.md and
// docs/registration-slice-2.md hold the definitions and validation notes.
//
// Swift 6 language mode, nonisolated by default (no `.defaultIsolation`), the three upcoming features enabled below, no
// unsafe flags (a versioned dependency cannot carry them): a consumer builds the kit optimised itself, and tools/test.sh
// runs this package's tests with `-c release`. One product per module. What one module needs from another is `package`,
// never `public`; what an app reads is `public` and is guarded by tools/consumer-check. Foundation and Accelerate only:
// every module builds for iOS as well as macOS.
import PackageDescription

let settings: [SwiftSetting] = [
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("MemberImportVisibility"),
]

let package = Package(
    name: "ImagoKit",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "ImagoCore", targets: ["ImagoCore"]),
        .library(name: "ImagoFFT", targets: ["ImagoFFT"]),
        .library(name: "ImagoFilters", targets: ["ImagoFilters"]),
        .library(name: "ImagoAlignment", targets: ["ImagoAlignment"]),
    ],
    targets: [
        // The calibrated image, its axes and crops, length units, and the
        // exact N-point 2D DFT for every size (Float32 on vDSP, the Float64 reference path).
        .target(name: "ImagoCore", swiftSettings: settings),
        // The centred spectrum and its axes, the radial profile, spot measurement, lattice spacings and
        // the angle between spots, the masked inverse FFT.
        .target(name: "ImagoFFT", dependencies: ["ImagoCore"], swiftSettings: settings),
        // The HRTEM filters (average background subtraction and Wiener). Later: bandpass, denoising.
        .target(name: "ImagoFilters", dependencies: ["ImagoCore", "ImagoFFT"], swiftSettings: settings),
        // Phase cross-correlation with sub-pixel refinement, shifts of order 0 and
        // 1 and the Fourier shift, drift series and aligned stack sums, on both precisions.
        .target(name: "ImagoAlignment", dependencies: ["ImagoCore"], swiftSettings: settings),
        .testTarget(name: "ImagoCoreTests", dependencies: ["ImagoCore"], exclude: ["Fixtures"], swiftSettings: settings),
        .testTarget(name: "ImagoFFTTests", dependencies: ["ImagoCore", "ImagoFFT"], exclude: ["Fixtures"],
                    swiftSettings: settings),
        .testTarget(name: "ImagoFiltersTests", dependencies: ["ImagoCore", "ImagoFFT", "ImagoFilters"], exclude: ["Fixtures"],
                    swiftSettings: settings),
        // The benchmark harness (tools/benchmarks/README: IMAGO_BENCH=1); without that variable its one test returns at once.
        .testTarget(name: "ImagoBenchmarks", dependencies: ["ImagoCore", "ImagoFFT", "ImagoFilters", "ImagoAlignment"],
                    swiftSettings: settings),
        .testTarget(name: "ImagoAlignmentTests", dependencies: ["ImagoCore", "ImagoAlignment"], exclude: ["Fixtures"],
                    swiftSettings: settings),
    ],
    swiftLanguageModes: [.v6]
)
