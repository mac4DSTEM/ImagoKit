// SPDX-License-Identifier: GPL-3.0-only
//
// rolling_ball on a Float16 image against the pins of tools/rolling-ball-float16/pins.py (scikit-image 0.26.0): scikit-image
// computes a float16 image in float32 and casts the background back to float16, so the kit does too. Held exact (one rounding to
// Float16 of the same float32 minimum), NaN where the pin is NaN.
#if arch(arm64)
import Foundation
import ImagoCore
import ImagoKitTestSupport
import Testing

struct RollingBallFloat16Pins: Decodable, Sendable {
    /// A pinned float: a number, or the strings "inf", "-inf" and "nan".
    struct Value: Decodable, Sendable {
        let value: Double
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) {
                value = s == "nan" ? .nan : (s == "-inf" ? -.infinity : .infinity)
            } else {
                value = try c.decode(Double.self)
            }
        }
    }
    struct Kernel: Decodable, Sendable {
        let height, width: Int
        let values: [Value]
    }
    struct Case: Decodable, Sendable, CustomStringConvertible {
        let name: String
        let width, height: Int
        let image: [Value]
        let nansafe: Bool
        let radius: Double?
        let kernel: Kernel?
        let background: [Value]
        var description: String { name }
    }
    struct Meta: Decodable, Sendable { let skimage, numpy, scipy: String }
    let meta: Meta
    let cases: [Case]

    static let all: RollingBallFloat16Pins = {
        let url = FixtureDirs.fixtures(of: "ImagoCoreTests").appendingPathComponent("rolling-ball-float16-pins.json")
        return try! JSONDecoder().decode(RollingBallFloat16Pins.self, from: Data(contentsOf: url))
    }()
}

@Suite("rolling ball Float16 (scikit-image 0.26.0)") struct RollingBallFloat16Tests {
    static let pins = RollingBallFloat16Pins.all

    @Test func versions() {
        #expect(Self.pins.meta.skimage == "0.26.0" && Self.pins.meta.numpy == "2.5.4" && Self.pins.meta.scipy == "1.18.1")
        #expect(Self.pins.cases.contains { $0.kernel != nil } && Self.pins.cases.contains { $0.radius != nil })
        #expect(Self.pins.cases.contains { $0.nansafe })
    }

    /// A Float16 image is computed in Float and rounded back to Float16 once: a kernel value of 70000, +infinity in Float16 but
    /// finite in Float, stays the finite offset it is in scikit-image.
    @Test(arguments: pins.cases) func computedInFloatAndCastBack(_ c: RollingBallFloat16Pins.Case) throws {
        let image = c.image.map { Float16($0.value) }
        let got: [Float16]
        if let k = c.kernel {
            let kernel = try RollingBall.Kernel(rows: k.height, columns: k.width, values: k.values.map(\.value))
            got = try RollingBall.background(image: image, rows: c.height, columns: c.width, kernel: kernel, nanSafe: c.nansafe)
        } else {
            got = try RollingBall.background(image: image, rows: c.height, columns: c.width, radius: try #require(c.radius),
                                             nanSafe: c.nansafe)
        }
        let expected = c.background.map { Float16($0.value) }
        #expect(got.count == expected.count)
        for (i, (g, e)) in zip(got, expected).enumerated() {
            #expect(g.isNaN ? e.isNaN : g == e, "\(c.name) pixel \(i): \(g) vs \(e)")
        }
    }

    /// A Float16 image is refused for the same reasons, with the same errors, as a Float one.
    @Test func refusalsAreTheFloatOnes() throws {
        let image: [Float16] = [1, 2, 3, 4]
        let kernel = try RollingBall.ballKernel(radius: 1)
        #expect(throws: RestorationError.shape) { try RollingBall.background(image: image, rows: 3, columns: 2, radius: 1) }
        #expect(throws: RestorationError.shape) { try RollingBall.background(image: image, rows: 3, columns: 2, kernel: kernel) }
        #expect(throws: RestorationError.nonFinite) {
            try RollingBall.background(image: [1, .nan, 3, 4] as [Float16], rows: 2, columns: 2, radius: 1)
        }
        #expect(throws: RestorationError.nonFinite) {
            try RollingBall.background(image: [1, .nan, 3, 4] as [Float16], rows: 2, columns: 2, kernel: kernel)
        }
        for radius in [-1, Double.nan] {
            #expect(throws: RestorationError.self) { try RollingBall.background(image: image, rows: 2, columns: 2, radius: radius) }
        }
    }

    /// The radius path beyond `maximumRadius` (its own offsets, no kernel) is computed in Float too.
    @Test func aHugeRadiusIsComputedInFloat() throws {
        let image: [Float16] = [1, 2, 3, 4, 5, 6]
        let half = try RollingBall.background(image: image, rows: 2, columns: 3, radius: 1e5)
        let single = try RollingBall.background(image: image.map { Float($0) }, rows: 2, columns: 3, radius: 1e5)
        #expect(half == single.map { Float16($0) })
    }
}
#endif
