//
//  FFTSurface.swift (ImagoCore)
//  NumPy's `numpy.fft` surface over the kit's exact-length 2D transforms: `fft2` / `ifft2` with `axes=` and `norm=`,
//  `rfft2` / `irfft2`, `fftshift` / `ifftshift`. Arrays are row-major `[rows * columns]`, split complex. Every length is the
//  exact DFT (no padding), Float32 on `FFT2D`, Float64 on `FFT2DDouble`, the Float64 path being the reference (D16).
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
        try run(x, forward: true, axes: axes, norm: norm) { FFT2DDouble(nx: $0, ny: $1) }
    }

    /// `numpy.fft.ifft2(x, axes=, norm=)` in Float64.
    public static func ifft2(_ x: ComplexImage<Double>, axes: FFTAxes = .both, norm: FFTNorm = .backward) throws -> ComplexImage<Double> {
        try run(x, forward: false, axes: axes, norm: norm) { FFT2DDouble(nx: $0, ny: $1) }
    }

    /// `numpy.fft.fft2(x, axes=, norm=)` in Float32.
    public static func fft2(_ x: ComplexImage<Float>, axes: FFTAxes = .both, norm: FFTNorm = .backward) throws -> ComplexImage<Float> {
        try run(x, forward: true, axes: axes, norm: norm) { FFT2D(nx: $0, ny: $1) }
    }

    /// `numpy.fft.ifft2(x, axes=, norm=)` in Float32.
    public static func ifft2(_ x: ComplexImage<Float>, axes: FFTAxes = .both, norm: FFTNorm = .backward) throws -> ComplexImage<Float> {
        try run(x, forward: false, axes: axes, norm: norm) { FFT2D(nx: $0, ny: $1) }
    }

    // MARK: rfft2, irfft2

    /// `numpy.fft.rfft2(x, norm=)` of a real `rows × columns` image: `rows × (columns / 2 + 1)`, Float64.
    public static func rfft2(real: [Double], rows: Int, columns: Int, norm: FFTNorm = .backward) throws -> ComplexImage<Double> {
        guard rows > 0, columns > 0, real.count == rows * columns else { throw FFTSurfaceError.shape }
        return halfColumns(try fft2(ComplexImage(rows: rows, columns: columns, real: real), norm: norm))
    }

    /// `numpy.fft.rfft2(x, norm=)`, Float32.
    public static func rfft2(real: [Float], rows: Int, columns: Int, norm: FFTNorm = .backward) throws -> ComplexImage<Float> {
        guard rows > 0, columns > 0, real.count == rows * columns else { throw FFTSurfaceError.shape }
        return halfColumns(try fft2(ComplexImage(rows: rows, columns: columns, real: real), norm: norm))
    }

    /// `numpy.fft.irfft2(X, s=(rows, columns), norm=)`: the real image of a half spectrum; `columns` defaults to
    /// `2 · (X.columns − 1)`. Like NumPy the imaginary parts of the zero-frequency and Nyquist columns do not count.
    public static func irfft2(_ x: ComplexImage<Double>, columns: Int? = nil, norm: FFTNorm = .backward) throws -> [Double] {
        try inverseReal(x, columns: columns, norm: norm, ifft: { try ifft2($0, axes: $1, norm: norm) })
    }

    /// `numpy.fft.irfft2`, Float32.
    public static func irfft2(_ x: ComplexImage<Float>, columns: Int? = nil, norm: FFTNorm = .backward) throws -> [Float] {
        try inverseReal(x, columns: columns, norm: norm, ifft: { try ifft2($0, axes: $1, norm: norm) })
    }

    private static func halfColumns<S>(_ x: ComplexImage<S>) -> ComplexImage<S> {
        let m = x.columns / 2 + 1
        var re = [S](), im = [S]()
        re.reserveCapacity(x.rows * m); im.reserveCapacity(x.rows * m)
        for r in 0..<x.rows {
            re.append(contentsOf: x.re[r * x.columns ..< r * x.columns + m])
            im.append(contentsOf: x.im[r * x.columns ..< r * x.columns + m])
        }
        return ComplexImage(rows: x.rows, columns: m, re: re, im: im)
    }

    private static func inverseReal<S>(_ x: ComplexImage<S>, columns: Int?, norm: FFTNorm,
                                       ifft: (ComplexImage<S>, FFTAxes) throws -> ComplexImage<S>) throws -> [S] {
        let m = x.columns, h = x.rows
        let n = columns ?? 2 * (m - 1)
        guard h > 0, m > 0, n > 0, x.re.count == h * m, x.im.count == h * m else { throw FFTSurfaceError.shape }
        // The complex inverse down the rows first (axis 0), then the Hermitian completion and the inverse along each row.
        let down = try ifft(x, .axis0)
        var re = [S](repeating: 0, count: h * n), im = [S](repeating: 0, count: h * n)
        let usable = min(m, n / 2 + 1)
        for r in 0..<h {
            for c in 0..<usable {
                let vr = down.re[r * m + c]
                let vi = (c == 0 || (n % 2 == 0 && c == n / 2)) ? S(0) : down.im[r * m + c]
                re[r * n + c] = vr; im[r * n + c] = vi
                if c > 0, n - c != c { re[r * n + n - c] = vr; im[r * n + n - c] = -vi }
            }
        }
        let rowsDone = try ifft(ComplexImage(rows: h, columns: n, re: re, im: im), .axis1)
        return rowsDone.re
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
public protocol FFTSurfaceScalar: BinaryFloatingPoint, Sendable {
    static func fft2(_ x: ComplexImage<Self>, axes: FFTAxes, norm: FFTNorm) throws -> ComplexImage<Self>
    static func ifft2(_ x: ComplexImage<Self>, axes: FFTAxes, norm: FFTNorm) throws -> ComplexImage<Self>
}

extension Float: FFTSurfaceScalar {
    public static func fft2(_ x: ComplexImage<Float>, axes: FFTAxes, norm: FFTNorm) throws -> ComplexImage<Float> {
        try FFTSurface.fft2(x, axes: axes, norm: norm)
    }
    public static func ifft2(_ x: ComplexImage<Float>, axes: FFTAxes, norm: FFTNorm) throws -> ComplexImage<Float> {
        try FFTSurface.ifft2(x, axes: axes, norm: norm)
    }
}

extension Double: FFTSurfaceScalar {
    public static func fft2(_ x: ComplexImage<Double>, axes: FFTAxes, norm: FFTNorm) throws -> ComplexImage<Double> {
        try FFTSurface.fft2(x, axes: axes, norm: norm)
    }
    public static func ifft2(_ x: ComplexImage<Double>, axes: FFTAxes, norm: FFTNorm) throws -> ComplexImage<Double> {
        try FFTSurface.ifft2(x, axes: axes, norm: norm)
    }
}
