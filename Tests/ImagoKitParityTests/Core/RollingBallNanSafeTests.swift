// rolling_ball with nansafe=True against the pins of tools/rolling-ball-nansafe/pins.py (scikit-image 0.26.0).
// scikit-image ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 Float64, 1e-5 Float32); NaN must be NaN.
import Foundation
import ImagoCore
import ImagoKitTestSupport
import Testing

struct RollingBallNanSafePins: Decodable, Sendable {
    /// A pinned float: a number, null for NaN, or the strings "inf" and "-inf".
    struct Value: Decodable, Sendable {
        let value: Double
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if c.decodeNil() { value = .nan } else if let s = try? c.decode(String.self) {
                value = s == "-inf" ? -.infinity : .infinity
            } else { value = try c.decode(Double.self) }
        }
    }
    struct Kernel: Decodable, Sendable {
        let height, width: Int
        let values: [Value]
    }
    struct Case: Decodable, Sendable, CustomStringConvertible {
        let name: String
        let width, height: Int
        let dtype: String
        let image: [Value]
        let radius: Double?
        let kernel: Kernel?
        let background: [Value]
        let nanCount: Int
        var description: String { name }
    }
    struct Meta: Decodable, Sendable { let skimage, numpy, scipy: String }
    let meta: Meta
    let cases: [Case]

    static let all: RollingBallNanSafePins = {
        let url = FixtureDirs.fixtures(of: "ImagoCoreTests").appendingPathComponent("rolling-ball-nansafe-pins.json")
        return try! JSONDecoder().decode(RollingBallNanSafePins.self, from: Data(contentsOf: url))
    }()
}

@Suite("rolling ball nansafe (scikit-image 0.26.0)") struct RollingBallNanSafeTests {
    static let pins = RollingBallNanSafePins.all

    @Test func versions() {
        #expect(Self.pins.meta.skimage == "0.26.0" && Self.pins.meta.numpy == "2.5.4" && Self.pins.meta.scipy == "1.18.1")
        #expect(Self.pins.cases.contains { $0.nanCount > 0 } && Self.pins.cases.contains { $0.nanCount == 0 })
    }

    @Test(arguments: pins.cases) func nanSafe(_ c: RollingBallNanSafePins.Case) throws {
        let kernel = try c.kernel.map { k in try RollingBall.Kernel(rows: k.height, columns: k.width, values: k.values.map(\.value)) }
        let expected = c.background.map(\.value)
        let image = c.image.map(\.value)
        let got: [Double]
        let rtol: Double
        if c.dtype == "float32" {
            let f = image.map { Float($0) }
            let out = kernel == nil ? try RollingBall.background(image: f, rows: c.height, columns: c.width, radius: c.radius ?? 0, nanSafe: true)
                : try RollingBall.background(image: f, rows: c.height, columns: c.width, kernel: kernel!, nanSafe: true)
            got = out.map { Double($0) }
            rtol = 1e-5
        } else {
            got = kernel == nil ? try RollingBall.background(image: image, rows: c.height, columns: c.width, radius: c.radius ?? 0, nanSafe: true)
                : try RollingBall.background(image: image, rows: c.height, columns: c.width, kernel: kernel!, nanSafe: true)
            rtol = 1e-7
        }
        #expect(got.count == expected.count)
        #expect(got.filter(\.isNaN).count == c.nanCount, "\(c.name): NaN count")
        for (i, (g, e)) in zip(got, expected).enumerated() {
            if e.isNaN || e.isInfinite { #expect(g.isNaN == e.isNaN && (e.isNaN || g == e), "\(c.name) pixel \(i): \(g) vs \(e)"); continue }
            #expect(abs(g - e) <= rtol * abs(e), "\(c.name) pixel \(i): \(g) vs \(e)")
        }
    }

    @Test func withoutNanSafeANaNIsStillRefused() {
        #expect(throws: RestorationError.nonFinite) { try RollingBall.background(image: [1.0, .nan], rows: 1, columns: 2, radius: 1) }
        #expect(throws: RestorationError.nonFinite) {
            try RollingBall.background(image: [1.0, .nan], rows: 1, columns: 2, kernel: RollingBall.ballKernel(radius: 1))
        }
    }

    /// Beyond `maximumRadius` the ball covers a small image from every pixel, so one NaN makes every pixel NaN.
    @Test func aBallBeyondTheKernelLimitSeesEveryNaN() throws {
        var image = [Double](repeating: 2, count: 12)
        let clean = try RollingBall.background(image: image, rows: 3, columns: 4, radius: 10000, nanSafe: true)
        #expect(clean.allSatisfy { abs($0 - 2) < 1e-9 })
        image[5] = .nan
        #expect(try RollingBall.background(image: image, rows: 3, columns: 4, radius: 10000, nanSafe: true).allSatisfy(\.isNaN))
    }
}
