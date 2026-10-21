// scipy.ndimage.gaussian_filter / uniform_filter and skimage.filters.difference_of_gaussians / butterworth / window against
// the pinned runs of tools/filters/pins.py (NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0).
// Tolerance: the libraries' own tests use assert_allclose / assert_array_almost_equal at the dtype default; Float64 is held to
// rtol 1e-7 + atol 1e-6 (scipy/ndimage/tests/test_filters.py uses decimal 6), Float32 to 1e-5 of the largest magnitude.
// scikit-image's wheel ships no tests (checked in refs/skimage), so its functions use the same dtype defaults.
import Foundation
import ImagoCore
import ImagoFilters
import Testing

struct FilterPins: Decodable, Sendable {
    struct Gauss: Decodable, Sendable, CustomStringConvertible {
        let h, w: Int; let sigma: [Double]; let mode: String; let truncate, cval: Double
        let x, f32, f64: [Double]
        var description: String { "\(h)x\(w) σ\(sigma) \(mode) t\(truncate)" }
    }
    struct Uniform: Decodable, Sendable, CustomStringConvertible {
        let h, w: Int; let size: [Int]; let mode: String; let cval: Double
        let x, f32, f64: [Double]
        var description: String { "\(h)x\(w) size\(size) \(mode)" }
    }
    struct DoG: Decodable, Sendable, CustomStringConvertible {
        let h, w: Int; let low: [Double]; let high: [Double]?
        let x, f32, f64: [Double]
        var description: String { "\(h)x\(w) \(low) \(String(describing: high))" }
    }
    struct Butter: Decodable, Sendable, CustomStringConvertible {
        let h, w: Int; let cutoff: Double; let high: Bool; let order: Double; let squared: Bool; let npad: Int
        let x, f32, f64: [Double]
        var description: String { "\(h)x\(w) c\(cutoff) hp\(high) o\(order) sq\(squared) pad\(npad)" }
    }
    struct Win: Decodable, Sendable, CustomStringConvertible {
        let type: String; let param: Double?; let h, w: Int; let window1d, window2d: [Double]
        var description: String { "\(type)\(param.map { "(\($0))" } ?? "") \(h)x\(w)" }
        var windowType: WindowType {
            switch type {
            case "boxcar": .boxcar
            case "hann": .hann
            case "hamming": .hamming
            case "blackman": .blackman
            case "blackmanharris": .blackmanharris
            case "nuttall": .nuttall
            case "flattop": .flattop
            case "bartlett": .bartlett
            case "tukey": .tukey(alpha: param!)
            case "gaussian": .gaussian(std: param!)
            default: .kaiser(beta: param!)
            }
        }
    }
    let gauss: [Gauss]; let uniform: [Uniform]; let dog: [DoG]; let butter: [Butter]; let windows: [Win]
    static let all: FilterPins = {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/filter-pins.json")
        return try! JSONDecoder().decode(FilterPins.self, from: Data(contentsOf: url))
    }()
}

private func mode(_ s: String) -> BoundaryMode { BoundaryMode(rawValue: s)! }

private func match64(_ got: [Double], _ want: [Double], _ label: String) {
    #expect(got.count == want.count, "\(label) count")
    var worst = -Double.infinity
    for (g, w) in zip(got, want) { worst = max(worst, abs(g - w) - (1e-6 + 1e-7 * abs(w))) }
    #expect(worst <= 0, "\(label): exceeds atol 1e-6 + rtol 1e-7 by \(worst)")
}

private func match32(_ got: [Float], _ want: [Double], _ label: String) {
    #expect(got.count == want.count, "\(label) count")
    let scale = max(1, want.map(abs).max() ?? 1)
    var worst = 0.0
    for (g, w) in zip(got, want) { worst = max(worst, abs(Double(g) - w)) }
    #expect(worst <= 1e-5 * scale, "\(label): \(worst) vs \(1e-5 * scale)")
}

@Test(arguments: FilterPins.all.gauss)
func gaussianFilterMatchesSciPy(_ c: FilterPins.Gauss) throws {
    let sigma = (row: c.sigma[0], column: c.sigma[1])
    let g64 = try SeparableFilters.gaussianFilter(c.x, rows: c.h, columns: c.w, sigma: sigma, truncate: c.truncate, mode: mode(c.mode), cval: c.cval)
    match64(g64, c.f64, "gaussian64 \(c)")
    let g32 = try SeparableFilters.gaussianFilter(c.x.map(Float.init), rows: c.h, columns: c.w, sigma: sigma, truncate: c.truncate,
                                                  mode: mode(c.mode), cval: c.cval)
    match32(g32, c.f32, "gaussian32 \(c)")
}

@Test(arguments: FilterPins.all.uniform)
func uniformFilterMatchesSciPy(_ c: FilterPins.Uniform) throws {
    let size = (row: c.size[0], column: c.size[1])
    match64(try SeparableFilters.uniformFilter(c.x, rows: c.h, columns: c.w, size: size, mode: mode(c.mode), cval: c.cval), c.f64, "uniform64 \(c)")
    match32(try SeparableFilters.uniformFilter(c.x.map(Float.init), rows: c.h, columns: c.w, size: size, mode: mode(c.mode), cval: c.cval),
            c.f32, "uniform32 \(c)")
}

@Test(arguments: FilterPins.all.dog)
func differenceOfGaussiansMatchesSciKitImage(_ c: FilterPins.DoG) throws {
    let low = (row: c.low[0], column: c.low[1])
    let high = c.high.map { (row: $0[0], column: $0[1]) }
    match64(try BandpassFilters.differenceOfGaussians(c.x, rows: c.h, columns: c.w, lowSigma: low, highSigma: high), c.f64, "dog64 \(c)")
    match32(try BandpassFilters.differenceOfGaussians(c.x.map(Float.init), rows: c.h, columns: c.w, lowSigma: low, highSigma: high),
            c.f32, "dog32 \(c)")
}

@Test func differenceOfGaussiansRefusesAHighSigmaBelowTheLow() {
    #expect(throws: FilterError.sigmaOrder) {
        try BandpassFilters.differenceOfGaussians([Double](repeating: 1, count: 16), rows: 4, columns: 4,
                                                  lowSigma: (2, 2), highSigma: (1, 3))
    }
}

@Test(arguments: FilterPins.all.butter)
func butterworthMatchesSciKitImage(_ c: FilterPins.Butter) throws {
    match64(try BandpassFilters.butterworth(c.x, rows: c.h, columns: c.w, cutoffFrequencyRatio: c.cutoff, highPass: c.high,
                                            order: c.order, squaredButterworth: c.squared, npad: c.npad), c.f64, "butter64 \(c)")
    match32(try BandpassFilters.butterworth(c.x.map(Float.init), rows: c.h, columns: c.w, cutoffFrequencyRatio: c.cutoff, highPass: c.high,
                                            order: c.order, squaredButterworth: c.squared, npad: c.npad), c.f32, "butter32 \(c)")
}

@Test func butterworthRefusesOutOfRangeParameters() {
    let a = [Double](repeating: 1, count: 16)
    for cutoff in [-0.1, 0, 0.6] {
        #expect(throws: FilterError.cutoff) { try BandpassFilters.butterworth(a, rows: 4, columns: 4, cutoffFrequencyRatio: cutoff) }
    }
    #expect(throws: FilterError.cutoff) { try BandpassFilters.butterworth(a, rows: 4, columns: 4, npad: -1) }
}

@Test(arguments: FilterPins.all.windows)
func windowsMatchSciPyAndSciKitImage(_ c: FilterPins.Win) throws {
    let m = max(c.h, c.w)
    match64(try Window.window1D(c.windowType, count: m), c.window1d, "get_window \(c)")
    match64(try Window.window2D(c.windowType, rows: c.h, columns: c.w), c.window2d, "window \(c)")
}
