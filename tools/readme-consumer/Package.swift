// swift-tools-version: 6.2
// ImagoKit README consumer: a SEPARATE package that depends on ImagoKit by path and holds README.md's Swift program
// verbatim (tools/readme-consumer/run.sh checks every README line is in main.swift, builds it in Debug and runs it).
import Foundation
import PackageDescription

let kit = URL(fileURLWithPath: Context.packageDirectory).deletingLastPathComponent().deletingLastPathComponent()
    .lastPathComponent

let package = Package(
    name: "ReadmeConsumer",
    platforms: [.macOS(.v15)],
    dependencies: [.package(path: "../..")],
    targets: [
        .executableTarget(name: "readme-consumer", dependencies: [
            .product(name: "ImagoCore", package: kit),
            .product(name: "ImagoFFT", package: kit),
            .product(name: "ImagoFilters", package: kit),
            .product(name: "ImagoAlignment", package: kit),
        ]),
    ],
    swiftLanguageModes: [.v6]
)
