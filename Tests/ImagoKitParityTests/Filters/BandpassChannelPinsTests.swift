// skimage.filters.butterworth and difference_of_gaussians with channel_axis=-1 against the pinned runs of
// tools/bandpass-channels/pins.py (NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0). Images are channel-last.
// Tolerance as the single-channel filter pins: scikit-image's wheel ships no tests, so assert_allclose dtype defaults; Float64
// rtol 1e-7 + atol 1e-6 (the SciPy Gaussian tests use decimal 6; butterworth keeps the same gate as its single-channel pins),
// Float32 1e-5 of the largest magnitude.
import Foundation
import ImagoKitTestSupport
import ImagoFilters
import Testing

struct BandpassChannelPins: Decodable, Sendable {
    struct Butter: Decodable, Sendable, CustomStringConvertible {
        let h, w, channels: Int; let cutoff: Double; let high: Bool; let order: Double; let squared: Bool; let npad: Int
        let x, f32, f64: [Double]
        var description: String { "\(h)x\(w)x\(channels) c\(cutoff) hp\(high) o\(order) sq\(squared) pad\(npad)" }
    }
    struct DoG: Decodable, Sendable, CustomStringConvertible {
        let h, w, channels: Int; let low: [Double]; let high: [Double]?
        let x, f32, f64: [Double]
        var description: String { "\(h)x\(w)x\(channels) \(low) \(String(describing: high))" }
    }
    let butter: [Butter]; let dog: [DoG]
    static let all: BandpassChannelPins = {
        let url = FixtureDirs.fixtures(of: "ImagoFiltersTests").appendingPathComponent("bandpass-channel-pins.json")
        return try! JSONDecoder().decode(BandpassChannelPins.self, from: Data(contentsOf: url))
    }()
}

private func close64(_ got: [Double], _ want: [Double], _ label: String) {
    #expect(got.count == want.count, "\(label) count")
    var worst = -Double.infinity
    for (g, w) in zip(got, want) { worst = max(worst, abs(g - w) - (1e-6 + 1e-7 * abs(w))) }
    #expect(worst <= 0, "\(label): exceeds atol 1e-6 + rtol 1e-7 by \(worst)")
}

private func close32(_ got: [Float], _ want: [Double], _ label: String) {
    #expect(got.count == want.count, "\(label) count")
    let scale = max(1, want.map(abs).max() ?? 1)
    var worst = 0.0
    for (g, w) in zip(got, want) { worst = max(worst, abs(Double(g) - w)) }
    #expect(worst <= 1e-5 * scale, "\(label): \(worst) vs \(1e-5 * scale)")
}

@Test(arguments: BandpassChannelPins.all.butter)
func butterworthWithChannelsMatchesSciKitImage(_ c: BandpassChannelPins.Butter) throws {
    close64(try BandpassFilters.butterworth(c.x, rows: c.h, columns: c.w, channels: c.channels, cutoffFrequencyRatio: c.cutoff,
                                            highPass: c.high, order: c.order, squaredButterworth: c.squared, npad: c.npad),
            c.f64, "butter64 \(c)")
    close32(try BandpassFilters.butterworth(c.x.map(Float.init), rows: c.h, columns: c.w, channels: c.channels,
                                            cutoffFrequencyRatio: c.cutoff, highPass: c.high, order: c.order,
                                            squaredButterworth: c.squared, npad: c.npad),
            c.f32, "butter32 \(c)")
}

@Test(arguments: BandpassChannelPins.all.dog)
func differenceOfGaussiansWithChannelsMatchesSciKitImage(_ c: BandpassChannelPins.DoG) throws {
    let low = (row: c.low[0], column: c.low[1])
    let high = c.high.map { (row: $0[0], column: $0[1]) }
    close64(try BandpassFilters.differenceOfGaussians(c.x, rows: c.h, columns: c.w, channels: c.channels, lowSigma: low, highSigma: high),
            c.f64, "dog64 \(c)")
    close32(try BandpassFilters.differenceOfGaussians(c.x.map(Float.init), rows: c.h, columns: c.w, channels: c.channels,
                                                     lowSigma: low, highSigma: high),
            c.f32, "dog32 \(c)")
}

@Test func bandpassRefusesAChannelCountThatDoesNotFit() {
    let a = [Double](repeating: 1, count: 4 * 4 * 3)
    #expect(throws: FilterError.shape) { try BandpassFilters.butterworth(a, rows: 4, columns: 4, channels: 2) }
    #expect(throws: FilterError.shape) { try BandpassFilters.butterworth(a, rows: 4, columns: 4, channels: 0) }
    #expect(throws: FilterError.shape) { try BandpassFilters.butterworth(a, rows: 4, columns: 4, channels: -1) }
    #expect(throws: FilterError.shape) { try BandpassFilters.butterworth(a, rows: Int.max / 2, columns: 2, channels: 2) }
    #expect(throws: FilterError.shape) {
        try BandpassFilters.differenceOfGaussians(a, rows: 4, columns: 4, channels: 0, lowSigma: (1, 1))
    }
    #expect(throws: FilterError.shape) {
        try BandpassFilters.differenceOfGaussians(a, rows: 4, columns: 4, channels: 2, lowSigma: (1, 1))
    }
}
