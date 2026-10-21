// LineProfile at spline orders 2 to 5 against tools/measure/spline_pins.py (NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0):
// 1080 `profile_line` cases over six image shapes, the five modes, widths 1, 3, 4 and 7 and the five reductions.
import Foundation
import ImagoCore
import Testing

private struct SplinePins: Decodable {
    struct Versions: Decodable { let numpy, scipy, skimage, python: String }
    struct Image: Decodable { let height, width: Int; let bits: [UInt32] }
    struct Profile: Decodable {
        let image: Int
        let src, dst: [Double]
        let lw, order: Int
        let mode, reduce: String
        let cval: Double
        let f32, f64: [Double]
    }
    let versions: Versions
    let images: [Image]
    let profile: [Profile]

    static func load() throws -> SplinePins {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/spline-profile-pins.json")
        return try JSONDecoder().decode(SplinePins.self, from: Data(contentsOf: url))
    }
}

@Suite("LineProfile splines") struct SplineProfileTests {
    private func reduction(_ name: String) -> ProfileReduction {
        switch name {
        case "sum": .sum
        case "min": .min
        case "max": .max
        case "median": .median
        default: .mean
        }
    }

    private func run(_ p: SplinePins.Profile, _ images: [Micrograph], _ a: ProfileArithmetic) throws -> [Double] {
        try LineProfile.sample(images[p.image], from: PixelPoint(x: p.src[1], y: p.src[0]), to: PixelPoint(x: p.dst[1], y: p.dst[0]),
                               linewidth: p.lw, order: ProfileOrder(rawValue: p.order)!, mode: ProfileMode(rawValue: p.mode)!,
                               cval: p.cval, reduce: reduction(p.reduce), arithmetic: a).samples
    }

    private func load() throws -> (SplinePins, [Micrograph]) {
        let pins = try SplinePins.load()
        let images = try pins.images.map {
            try Micrograph(width: $0.width, height: $0.height, pixels: $0.bits.map { Float(bitPattern: $0) })
        }
        return (pins, images)
    }

    @Test func versions() throws {
        let v = try SplinePins.load().versions
        #expect(v.numpy == "2.5.4" && v.scipy == "1.18.1" && v.skimage == "0.26.0")
    }

    /// Float64 path against scikit-image on the same pixels widened to float64: written tolerance 1e-13 of a fixed scale of 130 (the
    /// images' magnitudes stay below it); measured 3.9e-15.
    @Test func float64MatchesSkimage() throws {
        let (pins, images) = try load()
        var worst = 0.0
        for p in pins.profile {
            let got = try run(p, images, .float64)
            #expect(got.count == p.f64.count)
            for (g, w) in zip(got, p.f64) { #expect(g.isFinite); worst = max(worst, abs(g - w) / 130) }
        }
        #expect(worst <= 1e-13)
    }

    /// Float32 path against scikit-image on a float32 image (its output and mean are float32): 2e-6 of the same scale; measured 4.7e-7 (float32 rounding of the samples).
    @Test func float32MatchesSkimage() throws {
        let (pins, images) = try load()
        var worst = 0.0
        for p in pins.profile {
            let got = try run(p, images, .float32)
            #expect(got.count == p.f32.count)
            for (g, w) in zip(got, p.f32) { #expect(g.isFinite); worst = max(worst, abs(g - w) / 130) }
        }
        #expect(worst <= 2e-6)
    }
}
