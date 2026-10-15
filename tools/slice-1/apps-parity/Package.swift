// swift-tools-version: 6.2
// apps-parity (ImagoKit slice 1, registration section 6): macTEM 3d17311's Float32 originals (target MacTEMRef, extracted
// unchanged by `git show` into Sources/MacTEMRef by run.sh, never committed here) against ImagoKit's Float32 path, bit for
// bit, on every planted case of .build/pins-full. Maintainer tooling: it needs macTEM checked out beside this repository.
import Foundation
import PackageDescription

// The kit is the repository three levels up; a path dependency's identity is its directory's name.
let kit = URL(fileURLWithPath: Context.packageDirectory).deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().lastPathComponent

let package = Package(
    name: "apps-parity",
    platforms: [.macOS(.v15)],
    dependencies: [.package(path: "../../..")],
    targets: [
        .target(name: "MacTEMRef", swiftSettings: [
            .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
            .enableUpcomingFeature("InferIsolatedConformances"),
            .enableUpcomingFeature("MemberImportVisibility"),
        ]),
        .executableTarget(name: "apps-parity", dependencies: [
            "MacTEMRef",
            .product(name: "ImagoCore", package: kit),
            .product(name: "ImagoFFT", package: kit),
        ]),
    ],
    swiftLanguageModes: [.v6]
)
