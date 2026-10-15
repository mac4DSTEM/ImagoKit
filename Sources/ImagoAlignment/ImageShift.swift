//
//  ImageShift.swift (ImagoAlignment)
//  A shifted copy of an image: SciPy 1.18.1's `ndimage.shift` of order 0 and 1 in its seven boundary modes, and the exact
//  shift of a periodic image, `ifft2(fourier_shift(fft2(x), s)).real` (registration-slice-2.md 3.2 and 3.3).
//
//  THE COORDINATE: output pixel (i, j) takes the input at (i − s_row, j − s_col), mapped into the image by the mode (a
//  continuous coordinate, as SciPy maps it), then rounded half up (order 0) or interpolated between floor and floor + 1
//  (order 1); a neighbour beyond the edge is mapped by the same mode (the integer index for the periodic and symmetric
//  modes, `cval` for "grid-constant"). "constant" gives `cval` for any coordinate outside [0, n − 1]. The mapping is
//  SciPy's, read from its behaviour at the pins (a transcription of its C `map_coordinate` for the coordinate), and the pins
//  decide ties. Non-finite pixels propagate as SciPy's do (a zero weight times a NaN is NaN).
//

import Foundation
import ImagoCore

public nonisolated enum ImageShift {
    /// The largest shift magnitude accepted, in pixels.
    public static let maximumShift: Double = 1e9

    /// `image` translated by `shift` (content moves toward larger row and column indices for a positive shift), same size
    /// and calibration. The result has Float32 pixels whichever `precision` computed it.
    public static func shift(_ image: Micrograph, by shift: PixelShift, order: ShiftOrder = .linear,
                             mode: ShiftMode = .constant, cval: Double = 0,
                             precision: Precision = .float32) throws -> Micrograph {
        try checkShift(shift, index: 0)
        let pixels: [Float]
        switch precision {
        case .float32:
            pixels = shifted(image.pixels, image.width, image.height, shift, order, mode, Float(cval))
        case .float64:
            let values: [Double] = shifted(image.pixels.map(Double.init), image.width, image.height, shift, order, mode, cval)
            pixels = values.map { Float($0) }
        }
        return try rebuilt(image, pixels)
    }

    /// The exact shift of a periodic image: every Fourier coefficient is multiplied by `exp(−2πi (s_row f_row + s_col f_col))`,
    /// `f = fftfreq(n)`, and the real part is kept. Exact for an image that is periodic and has no content at the Nyquist
    /// frequency; at an even size's Nyquist row or column a non-integer shift leaves an imaginary part (the definition's),
    /// which `fourierShiftWithResidual` reports.
    public static func fourierShift(_ image: Micrograph, by shift: PixelShift,
                                    precision: Precision = .float32) throws -> Micrograph {
        try fourierShiftWithResidual(image, by: shift, precision: precision).image
    }

    /// `fourierShift` and the largest |imaginary part| the inverse transform left (rounding for an integer shift or an odd
    /// size; the Nyquist content's for an even size and a non-integer shift).
    public static func fourierShiftWithResidual(_ image: Micrograph, by shift: PixelShift,
                                                precision: Precision = .float32) throws -> (image: Micrograph, largestImaginaryPart: Double) {
        try checkShift(shift, index: 0)
        guard image.pixels.allSatisfy(\.isFinite) else { throw AlignmentError.nonFinitePixel(index: 0) }
        let w = image.width, h = image.height
        let pixels: [Float]
        let residual: Double
        switch precision {
        case .float32:
            guard let fft = FFT2D(nx: w, ny: h) else { throw AlignmentError.fftSetup(width: w, height: h) }
            let (values, imag) = fourierShifted(image.pixels, w, h, shift, fft)
            pixels = values
            residual = imag
        case .float64:
            guard let fft = FFT2DDouble(nx: w, ny: h) else { throw AlignmentError.fftSetup(width: w, height: h) }
            let (values, imag) = fourierShifted(image.pixels.map(Double.init), w, h, shift, fft)
            pixels = values.map { Float($0) }
            residual = imag
        }
        return (try rebuilt(image, pixels), residual)
    }

    static func checkShift(_ shift: PixelShift, index: Int) throws {
        guard shift.row.isFinite, shift.column.isFinite, abs(shift.row) <= maximumShift,
              abs(shift.column) <= maximumShift else { throw AlignmentError.invalidShift(index: index) }
    }

    private static func rebuilt(_ image: Micrograph, _ pixels: [Float]) throws -> Micrograph {
        try Micrograph(width: image.width, height: image.height, pixels: pixels, x: image.x, y: image.y)
    }

    // MARK: the Fourier shift

    /// The per-axis factors `exp(−2πi s k / n)`, k the signed frequency index of `fftfreq`, as (re, im) in Double.
    private static func phases(_ n: Int, _ s: Double) -> ([Double], [Double]) {
        var re = [Double](repeating: 0, count: n)
        var im = [Double](repeating: 0, count: n)
        for k in 0..<n {
            let signed = k <= (n - 1) / 2 ? k : k - n
            let angle = -(2.0 * Double.pi) * s * Double(signed) / Double(n)
            re[k] = cos(angle)
            im[k] = sin(angle)
        }
        return (re, im)
    }

    package static func fourierShifted<T: FFT2DTransform>(_ pixels: [T.Scalar], _ w: Int, _ h: Int, _ shift: PixelShift,
                                                           _ fft: T) -> ([T.Scalar], Double) {
        var re = pixels
        var im = [T.Scalar](repeating: 0, count: pixels.count)
        fft.transform(re: &re, im: &im, forward: true, scaleInverse: false)
        let (rowRe, rowIm) = phases(h, shift.row)
        let (colRe, colIm) = phases(w, shift.column)
        for r in 0..<h {
            for c in 0..<w {
                let fr = rowRe[r] * colRe[c] - rowIm[r] * colIm[c]
                let fi = rowRe[r] * colIm[c] + rowIm[r] * colRe[c]
                let i = r * w + c
                let xr = Double(re[i]), xi = Double(im[i])
                re[i] = T.Scalar(xr * fr - xi * fi)
                im[i] = T.Scalar(xr * fi + xi * fr)
            }
        }
        fft.transform(re: &re, im: &im, forward: false, scaleInverse: true)
        var residual = 0.0
        for v in im { residual = max(residual, abs(Double(v))) }
        return (re, residual)
    }

    // MARK: ndimage.shift, orders 0 and 1

    /// What one output index of one axis reads: nothing (the constant), or one or two input indices with weights. An index
    /// of −1 stands for `cval` (grid-constant, beyond the edge).
    package nonisolated struct AxisSample: Sendable, Equatable {
        package var constant: Bool
        package var i0: Int
        package var i1: Int
        package var w0: Double
        package var w1: Double
    }

    /// SciPy's `map_coordinate`: the continuous coordinate `x` mapped into [0, n − 1] by `mode` (−1 for "constant" outside).
    package static func mapped(_ x: Double, _ n: Int, _ mode: ShiftMode) -> Double {
        var x = x
        if x < 0 {
            switch mode {
            case .mirror:
                if n <= 1 {
                    x = 0
                } else {
                    let s2 = Double(2 * n - 2)
                    x = s2 * (-x / s2).rounded(.towardZero) + x
                    x = x <= Double(1 - n) ? x + s2 : -x
                }
            case .reflect:
                if n <= 1 {
                    x = 0
                } else {
                    let s2 = Double(2 * n)
                    if x < -s2 { x = s2 * (-x / s2).rounded(.towardZero) + x }
                    x = x < Double(-n) ? x + s2 : -x - 1
                }
            case .wrap:
                if n <= 1 {
                    x = 0
                } else {
                    let sz = Double(n - 1)
                    x += sz * ((-x / sz).rounded(.towardZero) + 1)
                }
            case .gridWrap:
                if n <= 1 {
                    x = 0
                } else {
                    let sz = Double(n)
                    x += sz * (((-1 - x) / sz).rounded(.towardZero) + 1)
                }
            case .nearest:
                x = 0
            case .constant:
                x = -1
            case .gridConstant:
                break
            }
        } else if x > Double(n - 1) {
            switch mode {
            case .mirror:
                if n <= 1 {
                    x = 0
                } else {
                    let s2 = Double(2 * n - 2)
                    x -= s2 * (x / s2).rounded(.towardZero)
                    if x >= Double(n) { x = s2 - x }
                }
            case .reflect:
                if n <= 1 {
                    x = 0
                } else {
                    let s2 = Double(2 * n)
                    x -= s2 * (x / s2).rounded(.towardZero)
                    if x >= Double(n) { x = s2 - x - 1 }
                }
            case .wrap:
                if n <= 1 {
                    x = 0
                } else {
                    let sz = Double(n - 1)
                    x -= sz * (x / sz).rounded(.towardZero)
                }
            case .gridWrap:
                if n <= 1 {
                    x = 0
                } else {
                    let sz = Double(n)
                    x -= sz * (x / sz).rounded(.towardZero)
                }
            case .nearest:
                x = Double(n - 1)
            case .constant:
                x = -1
            case .gridConstant:
                break
            }
        }
        return x
    }

    /// A neighbour's integer index mapped by the mode; −1 when the neighbour is `cval` ("grid-constant" beyond the edge).
    private static func neighbour(_ index: Int, _ n: Int, _ mode: ShiftMode) -> Int {
        switch mode {
        case .gridConstant:
            return index >= 0 && index < n ? index : -1
        case .constant:
            return min(max(index, 0), n - 1)
        default:
            if index >= 0 && index < n { return index }
            return Int(mapped(Double(index), n, mode))
        }
    }

    package static func axisSamples(_ n: Int, _ shift: Double, _ order: ShiftOrder, _ mode: ShiftMode) -> [AxisSample] {
        var out = [AxisSample]()
        out.reserveCapacity(n)
        for k in 0..<n {
            let cc = mapped(Double(k) - shift, n, mode)
            guard cc > -1 || mode == .gridConstant else {
                out.append(AxisSample(constant: true, i0: 0, i1: 0, w0: 0, w1: 0))
                continue
            }
            switch order {
            case .nearest:
                let start = Int((cc + 0.5).rounded(.down))
                out.append(AxisSample(constant: false, i0: neighbour(start, n, mode), i1: 0, w0: 1, w1: 0))
            case .linear:
                let floor = cc.rounded(.down)
                let start = Int(floor)
                let f = cc - floor
                out.append(AxisSample(constant: false, i0: neighbour(start, n, mode), i1: neighbour(start + 1, n, mode),
                                      w0: 1 - f, w1: f))
            }
        }
        return out
    }

    package static func shifted<S: BinaryFloatingPoint>(_ pixels: [S], _ w: Int, _ h: Int, _ shift: PixelShift,
                                                        _ order: ShiftOrder, _ mode: ShiftMode, _ cval: S) -> [S] {
        let rows = axisSamples(h, shift.row, order, mode)
        let columns = axisSamples(w, shift.column, order, mode)
        var out = [S](repeating: cval, count: w * h)
        for r in 0..<h {
            let a = rows[r]
            if a.constant { continue }
            for c in 0..<w {
                let b = columns[c]
                if b.constant { continue }
                out[r * w + c] = combine(pixels, w, a, b, order, cval)
            }
        }
        return out
    }

    /// The value at one output pixel from its row and column samples.
    @inline(__always)
    private static func combine<S: BinaryFloatingPoint>(_ pixels: [S], _ w: Int, _ a: AxisSample, _ b: AxisSample,
                                                        _ order: ShiftOrder, _ cval: S) -> S {
        func value(_ i: Int, _ j: Int) -> S { i < 0 || j < 0 ? cval : pixels[i * w + j] }
        if order == .nearest { return value(a.i0, b.i0) }
        var t: S = 0
        t += S(a.w0 * b.w0) * value(a.i0, b.i0)
        t += S(a.w0 * b.w1) * value(a.i0, b.i1)
        t += S(a.w1 * b.w0) * value(a.i1, b.i0)
        t += S(a.w1 * b.w1) * value(a.i1, b.i1)
        return t
    }
}
