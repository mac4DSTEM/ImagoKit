// swift-tools-version: 6.2
// ImagoKit consumer check: a SEPARATE package that depends on ImagoKit by path and uses, from outside the package, the
// public API an app reads. `package` access does not cross a package boundary, so this compiles only while each symbol
// it names is `public`.
import Foundation
import PackageDescription

// A path dependency's identity is its directory's name (lower-cased): "imagokit" in the repository, but not in a scratch
// copy; `.product` needs that name, so it is taken from the directory two levels above this manifest.
let kit = URL(fileURLWithPath: Context.packageDirectory).deletingLastPathComponent().deletingLastPathComponent()
    .lastPathComponent

let package = Package(
    name: "ConsumerCheck",
    platforms: [.macOS(.v15)],
    dependencies: [.package(path: "../..")],
    targets: [
        .executableTarget(name: "consumer-check", dependencies: [
            .product(name: "ImagoCore", package: kit),
            .product(name: "ImagoFFT", package: kit),
            .product(name: "ImagoFilters", package: kit),
            .product(name: "ImagoAlignment", package: kit),
        ]),
    ],
    swiftLanguageModes: [.v6]
)
