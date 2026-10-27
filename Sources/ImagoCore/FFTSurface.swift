//
//  FFTSurface.swift (ImagoCore)
//  NumPy's `numpy.fft` surface over the kit's exact-length 2D transforms: `fft2` / `ifft2` with `axes=` and `norm=`,
//  `rfft2` / `irfft2`, `fftshift` / `ifftshift`. Arrays are row-major `[rows * columns]`, split complex. Every length is the
//  exact DFT (no padding), Float32 on `FFT2D`, Float64 on `FFT2DDouble`, the Float64 path being the reference.
//

import Foundation

/// NumPy's `norm=`: which direction carries the 1/N (or 1/√N).
public nonisolated enum FFTNorm: Sendable, Equatable, CaseIterable {
    /// Forward unscaled, inverse 1/N (NumPy's default).
    case backward
    /// Forward 1/N, inverse unscaled.
    case forward
    /// Both 1/√N.
    case ortho
}

/// NumPy's `axes=` for a 2D array: axis 0 runs down the rows (the transform mixes values of one column), axis 1 along them.
public nonisolated enum FFTAxes: Sendable, Equatable, CaseIterable {
    case both, axis0, axis1
}

public nonisolated enum FFTSurfaceError: Error, Equatable, Sendable {
    /// Counts do not match `rows * columns`, or a length is below 1.
    case shape
    /// An Accelerate setup could not be allocated.
    case setup
}

/// A complex row-major image in split form.
public nonisolated struct ComplexImage<Scalar: BinaryFloatingPoint & Sendable>: Sendable, Equatable {
    public var rows: Int
    public var columns: Int
    public var re: [Scalar]
    public var im: [Scalar]

    public init(rows: Int, columns: Int, re: [Scalar], im: [Scalar]) {
        self.rows = rows; self.columns = columns; self.re = re; self.im = im
    }

    /// A real image with a zero imaginary part.
    public init(rows: Int, columns: Int, real: [Scalar]) {
        self.init(rows: rows, columns: columns, re: real, im: [Scalar](repeating: 0, count: real.count))
    }
}

/// The last few plans `FFTSurface` built, so a loop of same-size calls (a stack of frames, the six transforms of a masked
/// registration) pays for the setup once, as NumPy's cached plans do. A plan is immutable and documented shareable across
/// threads; the cache holds at most `capacity` of them and drops the oldest.
nonisolated private final class PlanCache: @unchecked Sendable {
    static let shared = PlanCache()
    private let lock = NSLock()
    private let capacity = 6
    private var double: [(nx: Int, ny: Int, plan: FFT2DDouble)] = []
    private var single: [(nx: Int, ny: Int, plan: FFT2D)] = []

    func plan(nx: Int, ny: Int) -> FFT2DDouble? {
        lock.lock(); defer { lock.unlock() }
        if let hit = double.first(where: { $0.nx == nx && $0.ny == ny }) { return hit.plan }
        guard let plan = FFT2DDouble(nx: nx, ny: ny) else { return nil }
        double.append((nx, ny, plan))
        if double.count > capacity { double.removeFirst() }
        return plan
    }

    func plan(nx: Int, ny: Int) -> FFT2D? {
        lock.lock(); defer { lock.unlock() }
        if let hit = single.first(where: { $0.nx == nx && $0.ny == ny }) { return hit.plan }
        guard let plan = FFT2D(nx: nx, ny: ny) else { return nil }
        single.append((nx, ny, plan))
        if single.count > capacity { single.removeFirst() }
        return plan
    }
}

public nonisolated enum FFTSurface {
    // MARK: fftshift

    /// `numpy.fft.fftshift` of a row-major `rows × columns` array over `axes` (zero frequency to index n/2).
    public static func fftshift<T>(_ a: [T], rows: Int, columns: Int, axes: FFTAxes = .both) throws -> [T] {
        try roll(a, rows: rows, columns: columns, axes: axes, inverse: false)
    }

    /// `numpy.fft.ifftshift`: the inverse of `fftshift`, which differs from it for odd lengths.
    public static func ifftshift<T>(_ a: [T], rows: Int, columns: Int, axes: FFTAxes = .both) throws -> [T] {
        try roll(a, rows: rows, columns: columns, axes: axes, inverse: true)
    }

    private static func roll<T>(_ a: [T], rows: Int, columns: Int, axes: FFTAxes, inverse: Bool) throws -> [T] {
        guard rows > 0, columns > 0, a.count == rows * columns else { throw FFTSurfaceError.shape }
        // fftshift rolls by n/2 (output[i] = input[(i − n/2) mod n]); ifftshift by −(n/2).
        let sr = axes == .axis1 ? 0 : (inverse ? -(rows / 2) : rows / 2)
        let sc = axes == .axis0 ? 0 : (inverse ? -(columns / 2) : columns / 2)
        var out = a
        for r in 0..<rows {
            let rr = ((r + sr) % rows + rows) % rows
            for c in 0..<columns {
                out[rr * columns + ((c + sc) % columns + columns) % columns] = a[r * columns + c]
            }
        }
        return out
    }

    // MARK: fft2, ifft2

    /// `numpy.fft.fft2(x, axes=, norm=)` in Float64.
    public static func fft2(_ x: ComplexImage<Double>, axes: FFTAxes = .both, norm: FFTNorm = .backward) throws -> ComplexImage<Double> {
        try run(x, forward: true, axes: axes, norm: norm) { PlanCache.shared.plan(nx: $0, ny: $1) as FFT2DDouble? }
    }

    /// `numpy.fft.ifft2(x, axes=, norm=)` in Float64.
    public static func ifft2(_ x: ComplexImage<Double>, axes: FFTAxes = .both, norm: FFTNorm = .backward) throws -> ComplexImage<Double> {
        try run(x, forward: false, axes: axes, norm: norm) { PlanCache.shared.plan(nx: $0, ny: $1) as FFT2DDouble? }
    }

    /// `numpy.fft.fft2(x, axes=, norm=)` in Float32.
    public static func fft2(_ x: ComplexImage<Float>, axes: FFTAxes = .both, norm: FFTNorm = .backward) throws -> ComplexImage<Float> {
        try run(x, forward: true, axes: axes, norm: norm) { PlanCache.shared.plan(nx: $0, ny: $1) as FFT2D? }
    }

    /// `numpy.fft.ifft2(x, axes=, norm=)` in Float32.
    public static func ifft2(_ x: ComplexImage<Float>, axes: FFTAxes = .both, norm: FFTNorm = .backward) throws -> ComplexImage<Float> {
        try run(x, forward: false, axes: axes, norm: norm) { PlanCache.shared.plan(nx: $0, ny: $1) as FFT2D? }
    }

    // MARK: rfft2, irfft2

    /// `numpy.fft.rfft2(x, norm=)` of a real `rows × columns` image: `rows × (columns / 2 + 1)`, Float64. Two real rows share
    /// one complex transform (`RealFFT2D.swift`), so it costs about half of `fft2`.
    public static func rfft2(real: [Double], rows: Int, columns: Int, norm: FFTNorm = .backward) throws -> ComplexImage<Double> {
        guard rows > 0, columns > 0, real.count == rows * columns else { throw FFTSurfaceError.shape }
        guard let plan = PlanCache.shared.plan(nx: columns, ny: rows) as FFT2DDouble? else { throw FFTSurfaceError.setup }
        var (re, im) = plan.realForward(real)
        scale(&re, &im, by: scaleFactor(norm, forward: true, count: rows * columns))
        return ComplexImage(rows: rows, columns: columns / 2 + 1, re: re, im: im)
    }

    /// `numpy.fft.rfft2(x, norm=)`, Float32.
    public static func rfft2(real: [Float], rows: Int, columns: Int, norm: FFTNorm = .backward) throws -> ComplexImage<Float> {
        guard rows > 0, columns > 0, real.count == rows * columns else { throw FFTSurfaceError.shape }
        guard let plan = PlanCache.shared.plan(nx: columns, ny: rows) as FFT2D? else { throw FFTSurfaceError.setup }
        var (re, im) = plan.realForward(real)
        scale(&re, &im, by: scaleFactor(norm, forward: true, count: rows * columns))
        return ComplexImage(rows: rows, columns: columns / 2 + 1, re: re, im: im)
    }

    /// `numpy.fft.irfft2(X, s=(rows, columns), norm=)`: the real image of a half spectrum; `columns` defaults to
    /// `2 · (X.columns − 1)`. Like NumPy the imaginary parts of the zero-frequency and Nyquist columns do not count.
    public static func irfft2(_ x: ComplexImage<Double>, columns: Int? = nil, norm: FFTNorm = .backward) throws -> [Double] {
        let n = columns ?? 2 * (x.columns - 1)
        guard x.rows > 0, x.columns > 0, n > 0, x.re.count == x.rows * x.columns, x.im.count == x.re.count else { throw FFTSurfaceError.shape }
        guard let plan = PlanCache.shared.plan(nx: n, ny: x.rows) as FFT2DDouble? else { throw FFTSurfaceError.setup }
        var out = plan.realInverse(re: x.re, im: x.im, inputColumns: x.columns)
        var none: [Double] = []
        scale(&out, &none, by: scaleFactor(norm, forward: false, count: x.rows * n))
        return out
    }

    /// `numpy.fft.irfft2`, Float32.
    public static func irfft2(_ x: ComplexImage<Float>, columns: Int? = nil, norm: FFTNorm = .backward) throws -> [Float] {
        let n = columns ?? 2 * (x.columns - 1)
        guard x.rows > 0, x.columns > 0, n > 0, x.re.count == x.rows * x.columns, x.im.count == x.re.count else { throw FFTSurfaceError.shape }
        guard let plan = PlanCache.shared.plan(nx: n, ny: x.rows) as FFT2D? else { throw FFTSurfaceError.setup }
        var out = plan.realInverse(re: x.re, im: x.im, inputColumns: x.columns)
        var none: [Float] = []
        scale(&out, &none, by: scaleFactor(norm, forward: false, count: x.rows * n))
        return out
    }

    /// The factor `norm` puts on a transform of `count` points in the given direction (nil: none).
    private static func scaleFactor(_ norm: FFTNorm, forward: Bool, count: Int) -> Double? {
        switch (norm, forward) {
        case (.backward, true), (.forward, false): return nil
        case (.backward, false), (.forward, true): return 1 / Double(count)
        case (.ortho, _): return 1 / Double(count).squareRoot()
        }
    }

    private static func scale<S: BinaryFloatingPoint>(_ re: inout [S], _ im: inout [S], by factor: Double?) {
        guard let factor else { return }
        let s = S(factor)
        for i in 0..<re.count { re[i] *= s }
        for i in 0..<im.count { im[i] *= s }
    }

    // MARK: engine

    private static func run<F: FFT2DTransform>(_ x: ComplexImage<F.Scalar>, forward: Bool, axes: FFTAxes, norm: FFTNorm,
                                               make: (Int, Int) -> F?) throws -> ComplexImage<F.Scalar> {
        let h = x.rows, w = x.columns
        guard h > 0, w > 0, x.re.count == h * w, x.im.count == h * w else { throw FFTSurfaceError.shape }
        var re = x.re, im = x.im
        switch axes {
        case .both:
            guard let fft = make(w, h) else { throw FFTSurfaceError.setup }
            fft.transform(re: &re, im: &im, forward: forward, scaleInverse: false)
        case .axis1 where w == 1: break
        case .axis0 where h == 1: break
        case .axis1:
            guard let fft = make(w, 1) else { throw FFTSurfaceError.setup }
            var rr = [F.Scalar](repeating: 0, count: w), ri = rr
            for r in 0..<h {
                for c in 0..<w { rr[c] = re[r * w + c]; ri[c] = im[r * w + c] }
                fft.transform(re: &rr, im: &ri, forward: forward, scaleInverse: false)
                for c in 0..<w { re[r * w + c] = rr[c]; im[r * w + c] = ri[c] }
            }
        case .axis0:
            guard let fft = make(h, 1) else { throw FFTSurfaceError.setup }
            var cr = [F.Scalar](repeating: 0, count: h), ci = cr
            for c in 0..<w {
                for r in 0..<h { cr[r] = re[r * w + c]; ci[r] = im[r * w + c] }
                fft.transform(re: &cr, im: &ci, forward: forward, scaleInverse: false)
                for r in 0..<h { re[r * w + c] = cr[r]; im[r * w + c] = ci[r] }
            }
        }
        let n = Double(axes == .both ? h * w : (axes == .axis0 ? h : w))
        let scale: Double?
        switch (norm, forward) {
        case (.backward, true), (.forward, false): scale = nil
        case (.backward, false), (.forward, true): scale = 1 / n
        case (.ortho, _): scale = 1 / n.squareRoot()
        }
        if let scale {
            let s = F.Scalar(scale)
            for i in 0..<re.count { re[i] *= s; im[i] *= s }
        }
        return ComplexImage(rows: h, columns: w, re: re, im: im)
    }
}

/// `Float` and `Double`, so numeric code can be written once over the precision and still reach `FFTSurface`.
package protocol FFTSurfaceScalar: BinaryFloatingPoint, Sendable {
    static func fft2(_ x: ComplexImage<Self>, axes: FFTAxes, norm: FFTNorm) throws -> ComplexImage<Self>
    static func ifft2(_ x: ComplexImage<Self>, axes: FFTAxes, norm: FFTNorm) throws -> ComplexImage<Self>
    static func rfft2(real: [Self], rows: Int, columns: Int, norm: FFTNorm) throws -> ComplexImage<Self>
    static func irfft2(_ x: ComplexImage<Self>, columns: Int?, norm: FFTNorm) throws -> [Self]
}

extension Float: FFTSurfaceScalar {
    package static func rfft2(real: [Float], rows: Int, columns: Int, norm: FFTNorm) throws -> ComplexImage<Float> {
        try FFTSurface.rfft2(real: real, rows: rows, columns: columns, norm: norm)
    }
    package static func irfft2(_ x: ComplexImage<Float>, columns: Int?, norm: FFTNorm) throws -> [Float] {
        try FFTSurface.irfft2(x, columns: columns, norm: norm)
    }
    package static func fft2(_ x: ComplexImage<Float>, axes: FFTAxes, norm: FFTNorm) throws -> ComplexImage<Float> {
        try FFTSurface.fft2(x, axes: axes, norm: norm)
    }
    package static func ifft2(_ x: ComplexImage<Float>, axes: FFTAxes, norm: FFTNorm) throws -> ComplexImage<Float> {
        try FFTSurface.ifft2(x, axes: axes, norm: norm)
    }
}

extension Double: FFTSurfaceScalar {
    package static func rfft2(real: [Double], rows: Int, columns: Int, norm: FFTNorm) throws -> ComplexImage<Double> {
        try FFTSurface.rfft2(real: real, rows: rows, columns: columns, norm: norm)
    }
    package static func irfft2(_ x: ComplexImage<Double>, columns: Int?, norm: FFTNorm) throws -> [Double] {
        try FFTSurface.irfft2(x, columns: columns, norm: norm)
    }
    package static func fft2(_ x: ComplexImage<Double>, axes: FFTAxes, norm: FFTNorm) throws -> ComplexImage<Double> {
        try FFTSurface.fft2(x, axes: axes, norm: norm)
    }
    package static func ifft2(_ x: ComplexImage<Double>, axes: FFTAxes, norm: FFTNorm) throws -> ComplexImage<Double> {
        try FFTSurface.ifft2(x, axes: axes, norm: norm)
    }
}
