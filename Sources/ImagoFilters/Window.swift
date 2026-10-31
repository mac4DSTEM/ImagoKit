// SPDX-License-Identifier: GPL-3.0-only
//
//  Window.swift (ImagoFilters)
//  scikit-image 0.26.0's `filters.window(window_type, shape)` for a 2D shape: a 1D `scipy.signal.get_window(type, max(shape),
//  fftbins=False)` (the symmetric window) sampled by distance from the centre with linear interpolation (`warp`, order 1,
//  `mode='constant'`, `cval=0`), zero beyond the 1D window's ends. The coordinates are stored as float32, as scikit-image does.
//

import Foundation

/// The window families of `scipy.signal.get_window`, each with SciPy's parameters under SciPy's names.
public nonisolated enum WindowType: Sendable, Equatable {
    case boxcar, hann, hamming, blackman, blackmanharris, nuttall, flattop, bartlett
    /// `alpha` is the taper fraction (SciPy's default 0.5).
    case tukey(alpha: Double)
    case gaussian(std: Double)
    case kaiser(beta: Double)
    case triang, parzen, bohman, barthann, cosine, lanczos
    /// `exp(-|n - centre| / tau)` about the window's centre (the symmetric window has no other centre).
    case exponential(tau: Double = 1)
    /// `nbar` nearly constant-level sidelobes next to the main lobe, `sll` dB down; `norm` scales the centre to 1.
    case taylor(nbar: Int = 4, sll: Double = 30, norm: Bool = true)
    /// Even lengths only (`kaiser_bessel_derived`); an odd length is refused, as SciPy does.
    case kaiserBesselDerived(beta: Double)
    /// `sum_k a[k] cos(k z)`, z from -pi to pi.
    case generalCosine(a: [Double])
    /// `exp(-0.5 |n / sig|^(2p))`.
    case generalGaussian(p: Double, sig: Double)
    /// `alpha - (1 - alpha) cos(2 pi n / (M - 1))`.
    case generalHamming(alpha: Double)
    /// The first discrete prolate spheroidal sequence with `NW` the standardised half bandwidth (0 < NW < M/2), scaled
    /// as SciPy's `norm='approximate'` (the default of `get_window`). `nw` is SciPy's `NW`; a NaN is refused (SciPy carries
    /// on with it), and lengths 0 and 1 answer ones before `nw` is checked, as SciPy does.
    case dpss(nw: Double)
    /// Dolph-Chebyshev with `at` dB sidelobe attenuation.
    case chebwin(at: Double)
}

public nonisolated enum WindowFunction {
    /// `scipy.signal.get_window(type, count, fftbins=False)`: the symmetric 1D window.
    public static func window1D(_ type: WindowType, count m: Int) throws -> [Double] {
        guard m >= 0 else { throw FilterError.parameter("count") }
        if case .kaiserBesselDerived(let beta) = type { return try kaiserBesselDerived(m, beta: beta) }
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
            return kaiser(m, beta: beta)
        case .triang:
            let half = (0..<((m + 1) / 2)).map { i -> Double in
                let n = Double(i + 1)
                return m % 2 == 0 ? (2 * n - 1) / Double(m) : 2 * n / (Double(m) + 1)
            }
            let mirror: [Double] = m % 2 == 0 ? Array(half.reversed()) : Array(half.dropLast().reversed())
            return half + mirror
        case .parzen:
            let mm = Double(m)
            return (0..<m).map { i in
                let a = abs(-(mm - 1) / 2 + Double(i)), r = a / (mm / 2)
                return a <= (mm - 1) / 4 ? 1 - 6 * (r * r) + 6 * pow(r, 3) : 2 * pow(1 - r, 3)
            }
        case .bohman:
            let step = 2 / Double(m - 1)
            return (0..<m).map { i in
                if i == 0 || i == m - 1 { return 0 }
                let fac = abs(Double(i) * step - 1)
                return (1 - fac) * cos(Double.pi * fac) + 1 / Double.pi * sin(Double.pi * fac)
            }
        case .barthann:
            return (0..<m).map { i in
                let fac = abs(Double(i) / (Double(m) - 1) - 0.5)
                return 0.62 - 0.48 * fac + 0.38 * cos(2 * Double.pi * fac)
            }
        case .cosine:
            return (0..<m).map { i in sin(Double.pi / Double(m) * (Double(i) + 0.5)) }
        case .exponential(let tau):
            let center = Double(m - 1) / 2
            return (0..<m).map { i in exp(-abs(Double(i) - center) / tau) }
        case .taylor(let nbar, let sll, let norm):
            return try taylor(m, nbar: nbar, sll: sll, norm: norm)
        case .lanczos:
            func sinc(_ x: Double) -> Double { x == 0 ? 1 : sin(Double.pi * x) / (Double.pi * x) }
            let right = (((m + 1) / 2)..<m).map { i in sinc(2 * Double(i) / Double(m - 1) - 1) }
            let middle: [Double] = m % 2 == 0 ? [] : [1]
            return Array(right.reversed()) + middle + right
        case .kaiserBesselDerived:
            return []  // answered above
        case .generalCosine(let a):
            return cosineSum(a)
        case .generalGaussian(let p, let sig):
            return (0..<m).map { i in exp(-0.5 * pow(abs((Double(i) - (Double(m) - 1) / 2) / sig), 2 * p)) }
        case .generalHamming(let alpha):
            return cosineSum([alpha, 1 - alpha])
        case .dpss(let nw):
            return try dpss(m, nw: nw)
        case .chebwin(let at):
            return chebwin(m, at: at)
        }
    }

    /// `scipy.signal.windows.kaiser(m, beta)`, symmetric.
    static func kaiser(_ m: Int, beta: Double) -> [Double] {
        if m <= 1 { return [Double](repeating: 1, count: max(m, 0)) }
        let alpha = Double(m - 1) / 2
        return (0..<m).map { i in
            let r = (Double(i) - alpha) / alpha
            return besselI0(beta * max(0, 1 - r * r).squareRoot()) / besselI0(beta)
        }
    }

    /// `kaiser_bessel_derived(m, beta)`: the square root of the running sum of a Kaiser window of m/2 + 1 points, mirrored.
    static func kaiserBesselDerived(_ m: Int, beta: Double) throws -> [Double] {
        if m < 1 { return [] }
        guard m % 2 == 0 else { throw FilterError.parameter("count") }
        let k = kaiser(m / 2 + 1, beta: beta)
        var csum = [Double](repeating: 0, count: k.count)
        var s = 0.0
        for i in k.indices { s += k[i]; csum[i] = s }
        let half = csum.dropLast().map { ($0 / s).squareRoot() }
        return half + Array(half.reversed())
    }

    /// `taylor(m, nbar, sll, norm)`, symmetric.
    static func taylor(_ m: Int, nbar: Int, sll: Double, norm: Bool) throws -> [Double] {
        guard nbar >= 1 else { throw FilterError.parameter("nbar") }
        let b = pow(10, sll / 20)
        let a = acosh(b) / Double.pi
        let s2 = Double(nbar * nbar) / (a * a + (Double(nbar) - 0.5) * (Double(nbar) - 0.5))
        let ma = (1..<nbar).map(Double.init)
        let m2 = ma.map { $0 * $0 }
        var fm = [Double](repeating: 0, count: ma.count)
        for mi in ma.indices {
            let sign: Double = mi % 2 == 0 ? 1 : -1
            var numer = 1.0
            for j in ma.indices { numer *= 1 - m2[mi] / s2 / (a * a + (ma[j] - 0.5) * (ma[j] - 0.5)) }
            var below = 1.0, above = 1.0
            for j in 0..<mi { below *= 1 - m2[mi] / m2[j] }
            for j in (mi + 1)..<ma.count { above *= 1 - m2[mi] / m2[j] }
            fm[mi] = sign * numer / (2 * below * above)
        }
        let mm = Double(m)
        func w(_ n: Double) -> Double {
            var s = 0.0
            for j in ma.indices { s += fm[j] * cos(2 * Double.pi * ma[j] * (n - mm / 2 + 0.5) / mm) }
            return 1 + 2 * s
        }
        var out = (0..<m).map { w(Double($0)) }
        if norm {
            let scale = 1 / w((mm - 1) / 2)
            for i in out.indices { out[i] *= scale }
        }
        return out
    }

    /// `chebwin(m, at)`, symmetric: the real part of the DFT of the Chebyshev polynomial samples, normalised to a peak of 1.
    /// The DFT is summed directly with the twiddles taken from one table of cos(pi t / m), t reduced mod 2m.
    static func chebwin(_ m: Int, at: Double) -> [Double] {
        let order = Double(m - 1), mm = Double(m)
        let beta = cosh(1 / order * acosh(pow(10, abs(at) / 20)))
        let p = (0..<m).map { k -> Double in
            let x = beta * cos(Double.pi * Double(k) / mm)
            if x > 1 { return cosh(order * acosh(x)) }
            if x < -1 { return Double(2 * (m % 2) - 1) * cosh(order * acosh(-x)) }
            return cos(order * acos(x))
        }
        let table = (0..<(2 * m)).map { cos(Double.pi * Double($0) / mm) }
        // Odd m: Re F[k] = sum_j p_j cos(2 pi j k / m). Even m: Re of the DFT of p_j exp(i pi j / m) is
        // sum_j p_j cos(pi j (1 - 2k) / m).
        func term(_ k: Int) -> Double {
            // The table index j * step mod 2m, stepped instead of multiplied.
            var step = (m % 2 == 1 ? 2 * k : 1 - 2 * k) % (2 * m)
            if step < 0 { step += 2 * m }
            var s = 0.0, t = 0
            for j in 0..<m {
                s += p[j] * table[t]
                t += step
                if t >= 2 * m { t -= 2 * m }
            }
            return s
        }
        var w: [Double]
        if m % 2 == 1 {
            let n = (m + 1) / 2
            let head = (0..<n).map(term)
            w = Array(head.dropFirst().reversed()) + head
        } else {
            let n = m / 2 + 1
            let head = (1..<n).map(term)
            w = Array(head.reversed()) + head
        }
        let peak = w.max() ?? 1
        for i in w.indices { w[i] /= peak }
        return w
    }

    /// `dpss(m, NW)` with SciPy's `get_window` defaults (one window, `norm='approximate'`, symmetric): the eigenvector of the
    /// largest eigenvalue of SciPy's tridiagonal matrix, by bisection and inverse iteration (LAPACK's stebz and stein, which
    /// SciPy's `eigh_tridiagonal` calls), signed to a positive sum, scaled to a peak of 1 and, for even m, by m²/(m² + NW).
    static func dpss(_ m: Int, nw: Double) throws -> [Double] {
        guard nw < Double(m) / 2 else { throw FilterError.parameter("nw") }
        guard nw > 0 else { throw FilterError.parameter("nw") }
        let mm = Double(m), wBand = nw / mm
        let d = (0..<m).map { i -> Double in let h = (mm - 1 - 2 * Double(i)) / 2; return h * h * cos(2 * Double.pi * wBand) }
        let e = (1..<m).map { i in Double(i) * (mm - Double(i)) / 2 }
        var v = largestEigenvector(diagonal: d, offDiagonal: e)
        if v.reduce(0, +) < 0 { for i in v.indices { v[i] = -v[i] } }
        let peak = v.max() ?? 1
        for i in v.indices { v[i] /= peak }
        if m % 2 == 0 {
            let correction = Double(m * m) / (Double(m * m) + nw)
            for i in v.indices { v[i] *= correction }
        }
        return v
    }

    /// The unit eigenvector of the largest eigenvalue of the symmetric tridiagonal matrix (diagonal d, off-diagonal e).
    static func largestEigenvector(diagonal d: [Double], offDiagonal e: [Double]) -> [Double] {
        let n = d.count
        if n == 1 { return [1] }
        // Gershgorin bounds, then bisection on the Sturm count for the largest eigenvalue.
        var lo = Double.infinity, hi = -Double.infinity, norm = 0.0
        for i in 0..<n {
            let r = (i > 0 ? abs(e[i - 1]) : 0) + (i < n - 1 ? abs(e[i]) : 0)
            lo = min(lo, d[i] - r); hi = max(hi, d[i] + r); norm = max(norm, abs(d[i]) + r)
        }
        let pivmin = Double.leastNormalMagnitude * max(1, e.map { $0 * $0 }.max() ?? 1)
        func below(_ x: Double) -> Int {  // the number of eigenvalues smaller than x
            var count = 0, q = d[0] - x
            if abs(q) < pivmin { q = -pivmin }
            if q < 0 { count += 1 }
            for i in 1..<n {
                q = d[i] - x - e[i - 1] * e[i - 1] / q
                if abs(q) < pivmin { q = -pivmin }
                if q < 0 { count += 1 }
            }
            return count
        }
        // Invariant: fewer than n eigenvalues below lo, all n below hi.
        lo -= 1e-12 * norm + pivmin; hi += 1e-12 * norm + pivmin
        for _ in 0..<200 {
            let mid = lo + (hi - lo) / 2
            if mid <= lo || mid >= hi { break }
            if below(mid) == n { hi = mid } else { lo = mid }
        }
        let lambda = lo + (hi - lo) / 2
        // Inverse iteration on T - lambda I, factored once by Gaussian elimination with partial pivoting (LAPACK gttrf).
        var dl = e, dd = d.map { $0 - lambda }, du = e, du2 = [Double](repeating: 0, count: max(n - 2, 0))
        var swapped = [Bool](repeating: false, count: n - 1)
        for i in 0..<(n - 1) {
            if abs(dd[i]) >= abs(dl[i]) {
                if dd[i] != 0 { let f = dl[i] / dd[i]; dl[i] = f; dd[i + 1] -= f * du[i] }
            } else {
                let f = dd[i] / dl[i]
                dd[i] = dl[i]; dl[i] = f
                let t = du[i]; du[i] = dd[i + 1]; dd[i + 1] = t - f * dd[i + 1]
                if i < n - 2 { du2[i] = du[i + 1]; du[i + 1] = -f * du[i + 1] }
                swapped[i] = true
            }
        }
        let tiny = Double.ulpOfOne * norm
        for i in 0..<n where abs(dd[i]) < tiny { dd[i] = dd[i] < 0 ? -tiny : tiny }
        var x = [Double](repeating: 1 / Double(n).squareRoot(), count: n)
        for _ in 0..<5 {
            for i in 0..<(n - 1) {
                if swapped[i] { let t = x[i]; x[i] = x[i + 1]; x[i + 1] = t - dl[i] * x[i] } else { x[i + 1] -= dl[i] * x[i] }
            }
            x[n - 1] /= dd[n - 1]
            x[n - 2] = (x[n - 2] - du[n - 2] * x[n - 1]) / dd[n - 2]
            if n > 2 { for i in stride(from: n - 3, through: 0, by: -1) { x[i] = (x[i] - du[i] * x[i + 1] - du2[i] * x[i + 2]) / dd[i] } }
            let scale = 1 / x.reduce(0) { $0 + $1 * $1 }.squareRoot()
            for i in x.indices { x[i] *= scale }
        }
        return x
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
