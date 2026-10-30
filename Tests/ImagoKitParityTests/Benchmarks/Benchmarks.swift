// Per-function timings for docs/benchmarks.md. Off unless IMAGOKIT_BENCH=1 (a normal `swift test -c release` returns at once).
//   IMAGOKIT_BENCH=1 IMAGOKIT_BENCH_OUT=/path/swift.json swift test -c release --filter ImagoKitParityTests.benchmarks
//   IMAGOKIT_BENCH_LARGE=1   adds the large sizes (the nightly review's)
// Each case: one warm-up call, then five timed calls; the JSON holds all five times in milliseconds. The Python side is
// tools/benchmarks/bench.py on the same sizes and dtypes; tools/benchmarks/report.py joins the two files.
import Foundation
import ImagoKitTestSupport
import ImagoAlignment
import ImagoCore
import ImagoFFT
import ImagoFilters
import Testing

private struct Gen {
    var state: UInt64 = 0x9E3779B97F4A7C15
    mutating func next() -> Double {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Double(state >> 11) / Double(1 << 53)
    }
    mutating func array(_ n: Int) -> [Double] { (0..<n).map { _ in next() * 40 + 30 } }
}

private struct Result: Codable { let name, dtype: String; let size: Int; let ms: [Double] }

private func time(_ body: () throws -> Void) rethrows -> [Double] {
    try body()
    var out: [Double] = []
    for _ in 0..<5 {
        let t0 = ContinuousClock.now
        try body()
        let d = ContinuousClock.now - t0
        out.append(Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
    }
    return out
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["IMAGOKIT_BENCH"] == "1"))
func benchmarks() throws {
    let env = ProcessInfo.processInfo.environment
    var sizes = [128, 512]
    if env["IMAGOKIT_BENCH_LARGE"] == "1" { sizes += [2048, 4096, 8192] }
    var results: [Result] = []
    // IMAGOKIT_BENCH_ONLY=rfft2,fft2 runs only the cases whose name starts with one of the comma-separated prefixes.
    let only = env["IMAGOKIT_BENCH_ONLY"]?.split(separator: ",").map(String.init)
    func record(_ name: String, _ dtype: String, _ n: Int, _ body: () throws -> Void) rethrows {
        if let only, !only.contains(where: { name.hasPrefix($0) }) { return }
        let t = try time(body)
        results.append(Result(name: name, dtype: dtype, size: n, ms: t))
        print(String(format: "BENCH %@ %@ %d  best %.3f ms", name, dtype, n, t.min()!))
    }
    var gen = Gen()
    for n in sizes {
        let d = gen.array(n * n), f = d.map(Float.init)
        // fft2 forward, plan built once (np.fft.fft2 caches its plan too).
        if let p = FFT2D(nx: n, ny: n) {
            record("fft2", "float32", n) { var re = f, im = [Float](repeating: 0, count: n * n); p.transform(re: &re, im: &im, forward: true, scaleInverse: false) }
        }
        if let p = FFT2DDouble(nx: n, ny: n) {
            record("fft2", "float64", n) { var re = d, im = [Double](repeating: 0, count: n * n); p.transform(re: &re, im: &im, forward: true, scaleInverse: false) }
        }
        try record("rfft2", "float32", n) { _ = try FFTSurface.rfft2(real: f, rows: n, columns: n) }
        try record("rfft2", "float64", n) { _ = try FFTSurface.rfft2(real: d, rows: n, columns: n) }
        do {
            let spec = try FFTSurface.rfft2(real: d, rows: n, columns: n), spec32 = try FFTSurface.rfft2(real: f, rows: n, columns: n)
            try record("irfft2", "float32", n) { _ = try FFTSurface.irfft2(spec32, columns: n) }
            try record("irfft2", "float64", n) { _ = try FFTSurface.irfft2(spec, columns: n) }
        }
        // A non-power-of-two size beside it (the exact-length path).
        if n == 512 {
            // Lengths of small prime factors (mixed radix) beside 500.
            for m in [360, 480, 504] {
                let dm = gen.array(m * m)
                if let p = FFT2DDouble(nx: m, ny: m) {
                    record("fft2", "float64", m) { var re = dm, im = [Double](repeating: 0, count: m * m); p.transform(re: &re, im: &im, forward: true, scaleInverse: false) }
                }
            }
            let m = 500, dm = gen.array(m * m), fm = dm.map(Float.init)
            if let p = FFT2D(nx: m, ny: m) {
                record("fft2", "float32", m) { var re = fm, im = [Float](repeating: 0, count: m * m); p.transform(re: &re, im: &im, forward: true, scaleInverse: false) }
            }
            if let p = FFT2DDouble(nx: m, ny: m) {
                record("fft2", "float64", m) { var re = dm, im = [Double](repeating: 0, count: m * m); p.transform(re: &re, im: &im, forward: true, scaleInverse: false) }
            }
        }
        try record("gaussian_filter sigma=3", "float32", n) { _ = try SeparableFilters.gaussianFilter(f, rows: n, columns: n, sigma: (3, 3)) }
        try record("gaussian_filter sigma=3", "float64", n) { _ = try SeparableFilters.gaussianFilter(d, rows: n, columns: n, sigma: (3, 3)) }
        try record("uniform_filter size=5", "float32", n) { _ = try SeparableFilters.uniformFilter(f, rows: n, columns: n, size: (5, 5)) }
        try record("uniform_filter size=5", "float64", n) { _ = try SeparableFilters.uniformFilter(d, rows: n, columns: n, size: (5, 5)) }
        try record("difference_of_gaussians", "float32", n) { _ = try BandpassFilters.differenceOfGaussians(f, rows: n, columns: n, lowSigma: (2, 2)) }
        try record("butterworth", "float32", n) { _ = try BandpassFilters.butterworth(f, rows: n, columns: n, cutoffFrequencyRatio: 0.1) }
        try record("butterworth", "float64", n) { _ = try BandpassFilters.butterworth(d, rows: n, columns: n, cutoffFrequencyRatio: 0.1) }
        let image = try Micrograph(width: n, height: n, pixels: f)
        let other = try Micrograph(width: n, height: n, pixels: gen.array(n * n).map(Float.init))
        for order in [ShiftOrder.linear, .cubic] {
            try record("shift order=\(order.rawValue)", "float32", n) { _ = try ImageShift.shift(image, by: PixelShift(row: 2.3, column: -1.7), order: order, mode: .reflect) }
        }
        if n <= 2048 {
            try record("phase_cross_correlation", "float32", n) { _ = try PhaseCorrelation.register(reference: image, moving: other) }
            try record("phase_cross_correlation upsample=10", "float32", n) { _ = try PhaseCorrelation.register(reference: image, moving: other, upsampleFactor: 10) }
            try record("phase_cross_correlation", "float64", n) { _ = try PhaseCorrelation.register(reference: image, moving: other, precision: .float64) }
        }
        let p0 = PixelPoint(x: 5, y: 5), p1 = PixelPoint(x: Double(n - 6), y: Double(n - 6))
        try record("profile_line order=1 width=5", "float32", n) { _ = try LineProfile.sample(image, from: p0, to: p1, linewidth: 5) }
        try record("profile_line order=3 width=5", "float32", n) { _ = try LineProfile.sample(image, from: p0, to: p1, linewidth: 5, order: .cubic) }
        let mask = f.map { $0 > 50 }
        try record("label connectivity=2", "bool", n) { _ = try Segmentation.label(mask, width: n, height: n, connectivity: .eight) }
    }
    // Masked registration is O(11 FFTs of (2n)²): its sizes are 64 and 256.
    for n in [64, 256] {
        let f = gen.array(n * n).map(Float.init)
        let a = try Micrograph(width: n, height: n, pixels: f), b = try Micrograph(width: n, height: n, pixels: gen.array(n * n).map(Float.init))
        let m = [Bool](repeating: true, count: n * n)
        try record("masked phase_cross_correlation", "float32", n) { _ = try PhaseCorrelation.registerMasked(reference: a, moving: b, referenceMask: m) }
        try record("masked phase_cross_correlation", "float64", n) { _ = try PhaseCorrelation.registerMasked(reference: a, moving: b, referenceMask: m, precision: .float64) }
    }
    if let out = env["IMAGOKIT_BENCH_OUT"] {
        try JSONEncoder().encode(results).write(to: URL(fileURLWithPath: out))
    }
}
