//
//  Bandpass.swift (ImagoFilters)
//  scikit-image 0.26.0's `filters.difference_of_gaussians` and `filters.butterworth` for one real 2D image.
//
//  difference_of_gaussians: `gaussian(low) − gaussian(high)` with `mode='nearest'`, `truncate=4.0`, high = 1.6 · low when not
//  given (both subtractions in the image's precision, as NumPy does for float32). butterworth: the Fourier-domain mask
//  `1/(1 + q^order)` (high pass `q^order/(1 + q^order)`, optional square root), q² = Σ (f_axis / cutoff)², f the ifftshifted
//  centred frequency index over (n · cutoff); the image is edge-padded by `npad` (0 by default in 0.26.0), filtered through
//  `irfft2(mask · rfft2(image))` and cropped, on the real-input transforms as scikit-image does (`irfftn`).
//

import Foundation
import ImagoCore

public nonisolated enum BandpassFilters {
    /// `skimage.filters.difference_of_gaussians(image, low_sigma, high_sigma, mode='nearest', cval=0, truncate=4.0)`.
    /// `highSigma` defaults to 1.6 × `lowSigma` per axis; a high sigma below the low sigma on an axis is refused.
    @_specialize(where S == Float)
    @_specialize(where S == Double)
    public static func differenceOfGaussians<S: BinaryFloatingPoint & Sendable>(
        _ a: [S], rows: Int, columns: Int, lowSigma: (row: Double, column: Double),
        highSigma: (row: Double, column: Double)? = nil, mode: BoundaryMode = .nearest, cval: Double = 0, truncate: Double = 4
    ) throws -> [S] {
        let high = highSigma ?? (lowSigma.row * 1.6, lowSigma.column * 1.6)
        guard high.row >= lowSigma.row, high.column >= lowSigma.column else { throw FilterError.sigmaOrder }
        let one = try SeparableFilters.gaussianFilter(a, rows: rows, columns: columns, sigma: lowSigma, truncate: truncate, mode: mode, cval: cval)
        let two = try SeparableFilters.gaussianFilter(a, rows: rows, columns: columns, sigma: high, truncate: truncate, mode: mode, cval: cval)
        return zip(one, two).map { $0 - $1 }
    }

    /// `skimage.filters.butterworth(image, cutoff_frequency_ratio, high_pass, order, squared_butterworth=, npad=)` of a real image.
    /// `cutoffFrequencyRatio` is in (0, 0.5]; 0, which scikit-image accepts and answers NaN to, is refused.
    public static func butterworth(_ a: [Double], rows: Int, columns: Int, cutoffFrequencyRatio: Double = 0.005,
                                   highPass: Bool = true, order: Double = 2, squaredButterworth: Bool = true,
                                   npad: Int = 0) throws -> [Double] {
        try butterworthCore(a, rows: rows, columns: columns, cutoff: cutoffFrequencyRatio, highPass: highPass, order: order,
                            squared: squaredButterworth, npad: npad)
    }

    /// The Float32 path (scikit-image on a float32 image: mask and transforms in single precision).
    public static func butterworth(_ a: [Float], rows: Int, columns: Int, cutoffFrequencyRatio: Double = 0.005,
                                   highPass: Bool = true, order: Double = 2, squaredButterworth: Bool = true,
                                   npad: Int = 0) throws -> [Float] {
        try butterworthCore(a, rows: rows, columns: columns, cutoff: cutoffFrequencyRatio, highPass: highPass, order: order,
                            squared: squaredButterworth, npad: npad)
    }

    /// The mask, `rows × columns`, in Double: `1/(1+q²^order)` (× q²^order for a high pass), square-rooted when not squared.
    static func butterworthMask(rows: Int, columns: Int, cutoffFrequencyRatio: Double, highPass: Bool, order: Double,
                                       squaredButterworth: Bool) -> [Double] {
        let q2 = q2Grid(rows, columns, cutoffFrequencyRatio)
        return q2.map { maskValue($0, order, highPass, squaredButterworth) }
    }

    private static func maskValue(_ q2: Double, _ order: Double, _ high: Bool, _ squared: Bool) -> Double {
        let p = pow(q2, order)
        var v = 1 / (1 + p)
        if high { v *= p }
        return squared ? v : v.squareRoot()
    }

    /// q² at every ifftshifted bin: (Σ_axis (index / (n · factor))²) with scikit-image's index range −(n−1)//2 … (n−1)//2 (floored).
    private static func axisTerms(_ n: Int, _ factor: Double) -> [Double] {
        let start = -(n / 2)                       // Python's −(n − 1) // 2
        var v = (0..<n).map { i -> Double in let x = Double(start + i) / (Double(n) * factor); return x * x }
        // ifftshift: roll by −(n / 2)
        let s = n / 2
        v = (0..<n).map { v[($0 + s) % n] }
        return v
    }

    private static func q2Grid(_ h: Int, _ w: Int, _ factor: Double) -> [Double] {
        let ar = axisTerms(h, factor), ac = axisTerms(w, factor)
        var out = [Double](repeating: 0, count: h * w)
        for r in 0..<h { for c in 0..<w { out[r * w + c] = ar[r] + ac[c] } }
        return out
    }

    private static func butterworthCore<S: FFTSurfaceScalar>(
        _ a: [S], rows: Int, columns: Int, cutoff: Double, highPass: Bool, order: Double, squared: Bool, npad: Int
    ) throws -> [S] {
        guard rows > 0, columns > 0, rows <= Int.max / columns, a.count == rows * columns else { throw FilterError.shape }
        guard npad >= 0, npad < (1 << 20), cutoff.isFinite, cutoff > 0, cutoff <= 0.5 else { throw FilterError.cutoff }
        var image = a, h = rows, w = columns
        if npad > 0 {
            h = rows + 2 * npad; w = columns + 2 * npad
            image = [S](repeating: 0, count: h * w)
            for r in 0..<h {
                let sr = min(max(r - npad, 0), rows - 1)
                for c in 0..<w { image[r * w + c] = a[sr * columns + min(max(c - npad, 0), columns - 1)] }
            }
        }
        let filtered: [S]
        do {
            // scikit-image filters with `irfftn(mask * rfftn(image))`: the mask is even in both axes, so the half spectrum
            // (`h × (w/2 + 1)`) carries everything and only its mask values are computed.
            var spectrum = try S.rfft2(real: image, rows: h, columns: w, norm: .backward)
            let m = spectrum.columns
            let rowTerms = axisTerms(h, cutoff), columnTerms = axisTerms(w, cutoff)
            spectrum.re.withUnsafeMutableBufferPointer { re in spectrum.im.withUnsafeMutableBufferPointer { im in
                for r in 0..<h {
                    for c in 0..<m {
                        // The mask in the image's precision as scikit-image builds it: q² cast, powered and combined in
                        // that precision. Order 2, the default, is a product (NumPy squares a power of exactly 2).
                        let q2 = rowTerms[r] + columnTerms[c]
                        let k: S
                        if S.self == Float.self {
                            let f = Float(q2)
                            let p = order == 2 ? f * f : pow(f, Float(order))
                            var v = 1 / (1 + p)
                            if highPass { v *= p }
                            k = S(squared ? v : v.squareRoot())
                        } else {
                            let p = order == 2 ? q2 * q2 : pow(q2, order)
                            var v = 1 / (1 + p)
                            if highPass { v *= p }
                            k = S(squared ? v : v.squareRoot())
                        }
                        re[r * m + c] *= k
                        im[r * m + c] *= k
                    }
                }
            } }
            filtered = try S.irfft2(spectrum, columns: w, norm: .backward)
        } catch is FFTSurfaceError { throw FilterError.setup }
        guard npad > 0 else { return filtered }
        var out = [S](repeating: 0, count: rows * columns)
        for r in 0..<rows { for c in 0..<columns { out[r * columns + c] = filtered[(r + npad) * w + c + npad] } }
        return out
    }
}
