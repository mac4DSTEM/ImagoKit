// SPDX-License-Identifier: GPL-3.0-only
//
// rolling_ball(image, kernel=) without nansafe, with a kernel value that is finite in Double but overflows Float, against the pins
// of tools/rolling-ball-kernel-overflow/pins.py (scikit-image 0.26.0). assert_allclose dtype defaults (rtol 1e-7 Float64, 1e-5 Float32).
import Foundation
import ImagoCore
import ImagoKitTestSupport
import Testing

struct RollingBallKernelOverflowPins: Decodable, Sendable {
    /// A pinned float: a number, or the strings "inf" and "-inf".
    struct Value: Decodable, Sendable {
        let value: Double
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) { value = s == "-inf" ? -.infinity : .infinity } else { value = try c.decode(Double.self) }
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
        let kernel: Kernel
        let background: [Value]
        var description: String { name }
    }
    struct Meta: Decodable, Sendable { let skimage, numpy, scipy: String }
    let meta: Meta
    let cases: [Case]

    static let all: RollingBallKernelOverflowPins = {
        let url = FixtureDirs.fixtures(of: "ImagoCoreTests").appendingPathComponent("rolling-ball-kernel-overflow-pins.json")
        return try! JSONDecoder().decode(RollingBallKernelOverflowPins.self, from: Data(contentsOf: url))
    }()
}

@Suite("rolling ball kernel overflow (scikit-image 0.26.0)") struct RollingBallKernelOverflowTests {
    static let pins = RollingBallKernelOverflowPins.all

    @Test func versions() {
        #expect(Self.pins.meta.skimage == "0.26.0" && Self.pins.meta.numpy == "2.5.4" && Self.pins.meta.scipy == "1.18.1")
        #expect(Self.pins.cases.contains { $0.dtype == "float32" } && Self.pins.cases.contains { $0.dtype == "float64" })
    }

    /// The kernel is converted to the image type before the infinity test, as scikit-image does: on a Float image 1e300 is a cell
    /// outside the kernel, not an offset of -infinity.
    @Test(arguments: pins.cases) func kernelConvertedBeforeTheInfinityTest(_ c: RollingBallKernelOverflowPins.Case) throws {
        let kernel = try RollingBall.Kernel(rows: c.kernel.height, columns: c.kernel.width, values: c.kernel.values.map(\.value))
        let expected = c.background.map(\.value)
        let image = c.image.map(\.value)
        let got: [Double]
        let rtol: Double
        if c.dtype == "float32" {
            got = try RollingBall.background(image: image.map { Float($0) }, rows: c.height, columns: c.width, kernel: kernel).map { Double($0) }
            rtol = 1e-5
        } else {
            got = try RollingBall.background(image: image, rows: c.height, columns: c.width, kernel: kernel)
            rtol = 1e-7
        }
        #expect(got.count == expected.count)
        for (i, (g, e)) in zip(got, expected).enumerated() {
            #expect(g.isFinite && abs(g - e) <= rtol * abs(e), "\(c.name) pixel \(i): \(g) vs \(e)")
        }
    }
}
