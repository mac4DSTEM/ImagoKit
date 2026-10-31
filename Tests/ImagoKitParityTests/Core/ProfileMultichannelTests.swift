// skimage.measure.profile_line on a (rows, columns, channels) image against the pinned runs of
// tools/profile-multichannel/pins.py (NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0).
// Tolerance: scikit-image ships no tests in its wheel, so assert_allclose dtype defaults per element (rtol 1e-5 float32,
// 1e-7 float64, atol 0).
import Foundation
import ImagoKitTestSupport
import ImagoCore
import Testing

struct ProfileMultichannelPins: Decodable, Sendable {
    struct Case: Decodable, Sendable, CustomStringConvertible {
        let src, dst: [Double]; let order: Int; let mode: String; let cval: Double; let lw: Int; let reduce: String?
        let shape: [Int]; let f32, f64: [Double]
        var description: String { "\(src)->\(dst) o\(order) \(mode) w\(lw) \(reduce ?? "none")" }
    }
    let height, width, channels: Int; let image: [Double]; let cases: [Case]
    static let all: ProfileMultichannelPins = {
        let url = FixtureDirs.fixtures(of: "ImagoCoreTests").appendingPathComponent("profile-multichannel-pins.json")
        return try! JSONDecoder().decode(ProfileMultichannelPins.self, from: Data(contentsOf: url))
    }()
    /// One Micrograph per channel of the channel-last image.
    var planes: [Micrograph] {
        (0..<channels).map { k in
            try! Micrograph(width: width, height: height,
                            pixels: (0..<(height * width)).map { Float(image[$0 * channels + k]) })
        }
    }
}

private func reduction(_ name: String) -> ProfileReduction {
    switch name {
    case "sum": .sum
    case "min": .min
    case "max": .max
    case "median": .median
    default: .mean
    }
}

private func check(_ got: [Double], _ want: [Double], rtol: Double, _ label: String) {
    #expect(got.count == want.count, "\(label) count \(got.count) vs \(want.count)")
    var worst = -Double.infinity
    for (g, w) in zip(got, want) { worst = max(worst, abs(g - w) - rtol * abs(w)) }
    #expect(worst <= 0, "\(label): exceeds rtol \(rtol) by \(worst)")
}

@Test(arguments: ProfileMultichannelPins.all.cases)
func multichannelProfileMatchesSciKitImage(_ c: ProfileMultichannelPins.Case) throws {
    let planes = ProfileMultichannelPins.all.planes
    let from = PixelPoint(x: c.src[1], y: c.src[0]), to = PixelPoint(x: c.dst[1], y: c.dst[0])
    let order = ProfileOrder(rawValue: c.order)!, mode = ProfileMode(rawValue: c.mode)!
    for (arithmetic, want, rtol) in [(ProfileArithmetic.float32, c.f32, 1e-5), (.float64, c.f64, 1e-7)] {
        // scikit-image's layout: (n, channels) reduced, (n, w, channels) unreduced, channel-last.
        let got: [Double]
        if let name = c.reduce {
            let profiles = try LineProfile.sample(channels: planes, from: from, to: to, linewidth: c.lw, order: order, mode: mode,
                                                  cval: c.cval, reduce: reduction(name), arithmetic: arithmetic)
            #expect(profiles.count == planes.count)
            got = (0..<profiles[0].samples.count).flatMap { i in profiles.map { $0.samples[i] } }
        } else {
            let rows = try LineProfile.unreduced(channels: planes, from: from, to: to, linewidth: c.lw, order: order, mode: mode,
                                                 cval: c.cval, arithmetic: arithmetic)
            #expect(rows.count == planes.count)
            got = (0..<rows[0].count).flatMap { i in (0..<c.lw).flatMap { k in rows.map { $0[i][k] } } }
        }
        check(got, want, rtol: rtol, "\(arithmetic) \(c)")
    }
}

@Test func multichannelProfileRefusesMismatchedOrMissingChannels() throws {
    let a = try Micrograph(width: 4, height: 3, pixels: [Float](repeating: 1, count: 12))
    let b = try Micrograph(width: 3, height: 4, pixels: [Float](repeating: 1, count: 12))
    let o = PixelPoint(x: 0, y: 0), e = PixelPoint(x: 2, y: 2)
    #expect(throws: LineProfileError.channelShape) { try LineProfile.sample(channels: [a, b], from: o, to: e) }
    #expect(throws: LineProfileError.channelShape) { try LineProfile.sample(channels: [], from: o, to: e) }
    #expect(throws: LineProfileError.channelShape) { try LineProfile.unreduced(channels: [], from: o, to: e) }
}
