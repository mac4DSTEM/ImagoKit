// swift-tools-version: 6.2
// ImagoKit — image analysis for electron micrographs on Apple platforms: the calibrated image and the exact-length 2D
// FFT (ImagoCore); the centred power spectrum, its radial profile, FFT-spot spacings and angles, and the masked inverse
// FFT (ImagoFFT); the HRTEM background-subtraction and Wiener filters (ImagoFilters); drift and stack alignment
// (ImagoAlignment, slice 2). NumPy, SciPy and scikit-image are the references. docs/registration-slice-1.md is the first
// registration; until it lands the modules are placeholders that state their scope.
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
        // Slice 1 (docs/registration-slice-1.md 1.1): the calibrated image, its axes and crops, length units, and the
        // exact N-point 2D DFT for every size (Float32 on vDSP, the Float64 reference path).
        .target(name: "ImagoCore", swiftSettings: settings),
        // Slice 1 (1.2): the centred spectrum and its axes, the radial profile, spot measurement, lattice spacings and
        // the angle between spots, the masked inverse FFT.
        .target(name: "ImagoFFT", dependencies: ["ImagoCore"], swiftSettings: settings),
        // Slice 1 (1.3): the HRTEM filters (average background subtraction and Wiener). Later: bandpass, denoising.
        .target(name: "ImagoFilters", dependencies: ["ImagoCore", "ImagoFFT"], swiftSettings: settings),
        // Slice 2 (registered in scope only, docs/registration-slice-1.md section 10): phase cross-correlation, drift
        // and stack alignment, shifted sums. Empty until registered.
        .target(name: "ImagoAlignment", dependencies: ["ImagoCore"], swiftSettings: settings),
        .testTarget(name: "ImagoCoreTests", dependencies: ["ImagoCore", "ImagoFilters", "ImagoAlignment"],
                    swiftSettings: settings),
        .testTarget(name: "ImagoFFTTests", dependencies: ["ImagoCore", "ImagoFFT"], exclude: ["Fixtures"],
                    swiftSettings: settings),
    ],
    swiftLanguageModes: [.v6]
)
