//
//  Window.swift (ImagoFilters)
//  scikit-image 0.26.0's `filters.window(window_type, shape)` for a 2D shape: a 1D `scipy.signal.get_window(type, max(shape),
//  fftbins=False)` (the symmetric window) sampled by distance from the centre with linear interpolation (`warp`, order 1,
//  `mode='constant'`, `cval=0`), zero beyond the 1D window's ends. The coordinates are stored as float32, as scikit-image does.
//

import Foundation

/// The window families `get_window` offers that need at most one parameter.
public nonisolated enum WindowType: Sendable, Equatable {
    case boxcar, hann, hamming, blackman, blackmanharris, nuttall, flattop, bartlett
    /// `alpha` is the taper fraction (SciPy's default 0.5).
    case tukey(alpha: Double)
    case gaussian(std: Double)
    case kaiser(beta: Double)
}

public nonisolated enum WindowFunction {
    /// `scipy.signal.get_window(type, count, fftbins=False)`: the symmetric 1D window.
    public static func window1D(_ type: WindowType, count m: Int) throws -> [Double] {
        guard m >= 0 else { throw FilterError.parameter("count") }
        if m <= 1 { return [Double](repeating: 1, count: m) }
        func cosineSum(_ a: [Double]) -> [Double] {
            (0..<m).map { i in
                let fac = -Double.pi + 2 * Double.pi * Double(i) / Double(m - 1)
                return a.enumerated().reduce(0.0) { $0 + $1.element * cos(Double($1.offset) * fac) }
            }
        }
        switch type {
        case .boxcar: return [Double](repeating: 1, count: m)
        case .hann: return cosineSum([0.5, 0.5])
        case .hamming: return cosineSum([0.54, 0.46])
        case .blackman: return cosineSum([0.42, 0.50, 0.08])
        case .blackmanharris: return cosineSum([0.35875, 0.48829, 0.14128, 0.01168])
        case .nuttall: return cosineSum([0.3635819, 0.4891775, 0.1365995, 0.0106411])
        case .flattop: return cosineSum([0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368])
        case .bartlett:
            return (0..<m).map { i in
                let n = Double(i), half = Double(m - 1) / 2
                return n <= half ? 2 * n / Double(m - 1) : 2 - 2 * n / Double(m - 1)
            }
        case .tukey(let alpha):
            if alpha <= 0 { return [Double](repeating: 1, count: m) }
            if alpha >= 1 { return cosineSum([0.5, 0.5]) }
            let width = Int(floor(alpha * Double(m - 1) / 2))
            return (0..<m).map { i in
                let n = Double(i)
                if i <= width { return 0.5 * (1 + cos(Double.pi * (-1 + 2 * n / alpha / Double(m - 1)))) }
                if i < m - width - 1 { return 1 }
                return 0.5 * (1 + cos(Double.pi * (-2 / alpha + 1 + 2 * n / alpha / Double(m - 1))))
            }
        case .gaussian(let std):
            guard std > 0 else { throw FilterError.parameter("std") }
            return (0..<m).map { i in let n = Double(i) - Double(m - 1) / 2; return exp(-n * n / (2 * std * std)) }
        case .kaiser(let beta):
            let alpha = Double(m - 1) / 2
            return (0..<m).map { i in
                let r = (Double(i) - alpha) / alpha
                return besselI0(beta * max(0, 1 - r * r).squareRoot()) / besselI0(beta)
            }
        }
    }

    /// `skimage.filters.window(type, (rows, columns))`.
    public static func window2D(_ type: WindowType, rows: Int, columns: Int) throws -> [Double] {
        guard rows >= 0, columns >= 0 else { throw FilterError.parameter("shape") }
        let m = max(rows, columns)
        let w = try window1D(type, count: m)
        let center = Double(m) / 2 - 0.5
        // scikit-image builds the distances in Double (a NumPy float64 scalar widens the float32 `arange`) and stores the
        // coordinate in a float32 array, which is what `map_coordinates` reads.
        func g(_ i: Int, _ n: Int) -> Double { Double(i) * (Double(m) / Double(n)) - center }
        var out = [Double](repeating: 0, count: rows * columns)
        for r in 0..<rows {
            let gr = g(r, rows)
            for c in 0..<columns {
                let gc = g(c, columns)
                let dist = (gr * gr + gc * gc).squareRoot()
                let x = Double(Float(dist + center))
                // SciPy's constant mode with order 1 still blends the last sample with cval over the next pixel.
                guard x >= 0, x < Double(m) else { continue }
                let i0 = Int(x.rounded(.down)), f = x - Double(i0)
                let v0 = w[i0], v1 = i0 + 1 < m ? w[i0 + 1] : 0
                out[r * columns + c] = v0 * (1 - f) + v1 * f
            }
        }
        // `warp(clip=True)`: the output is clipped to the 1D window's range, widened to hold the fill value 0 when some
        // pixel took it.
        var lo = w.min() ?? 0, hi = w.max() ?? 0
        if !(lo <= 0 && 0 <= hi), let oMin = out.min(), let oMax = out.max(), oMin <= 0, 0 <= oMax { lo = min(lo, 0); hi = max(hi, 0) }
        return out.map { min(max($0, lo), hi) }
    }

    /// The window applied to an image: `pixels · window2D(type, rows, columns)` elementwise, as `filters.window` is used (the image
    /// is multiplied by it before the FFT). Each factor is rounded to the image's type first (a Float32 image multiplies by Float(w)).
    public static func apply<S: BinaryFloatingPoint & Sendable>(_ type: WindowType, to pixels: [S], rows: Int, columns: Int) throws -> [S] {
        guard rows > 0, columns > 0, rows <= Int.max / columns, pixels.count == rows * columns else { throw FilterError.shape }
        let w = try window2D(type, rows: rows, columns: columns)
        var out = pixels
        for i in out.indices { out[i] *= S(w[i]) }
        return out
    }

    /// The product of an existing 2D window (`window2D`) and an image of the same shape, elementwise.
    public static func apply<S: BinaryFloatingPoint & Sendable>(window: [Double], to pixels: [S]) throws -> [S] {
        guard window.count == pixels.count, !pixels.isEmpty else { throw FilterError.shape }
        var out = pixels
        for i in out.indices { out[i] *= S(window[i]) }
        return out
    }

    /// Modified Bessel function of the first kind, order 0 (power series; `numpy.i0`).
    static func besselI0(_ x: Double) -> Double {
        var term = 1.0, sum = 1.0
        let q = x * x / 4
        var k = 1.0
        while term > 1e-17 * sum {
            term *= q / (k * k)
            sum += term
            k += 1
            if k > 500 { break }
        }
        return sum
    }
}
