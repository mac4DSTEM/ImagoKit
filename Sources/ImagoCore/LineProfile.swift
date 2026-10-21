//
//  LineProfile.swift (ImagoCore)
//  `skimage.measure.profile_line(image, src, dst, linewidth, order, mode, cval, reduce_func=...)` of a `Micrograph`, for
//  orders 0 to 5, scikit-image 0.26.0 with SciPy 1.18.1's `ndimage.map_coordinates`.
//
//  A PORT of skimage/measure/profile.py (`_line_profile_coordinates`, then `profile_line`): n = ⌈hypot(Δr, Δc) + 1⌉
//  samples, `linspace` from the start to the end INCLUDING the end (one sample, at the start, for a zero-length line); at
//  each, `linewidth` points along the perpendicular, one pixel apart and centred on the line (offsets (cos θ, −sin θ)·
//  (w − 1)/2 with θ = atan2(Δr, Δc), `linspace`d over w points); each read by `map_coordinates`; the w values reduced.
//  scikit-image maps its mode names to SciPy's grid modes: "constant" is SciPy's "grid-constant" (cval beyond the edge,
//  interpolated toward from the edge pixels) and "wrap" is "grid-wrap" (period n); "reflect" (d c b a | a b c d | d c b a),
//  "mirror" and "nearest" are SciPy's own.
//
//  ARITHMETIC. scikit-image runs on the image's dtype. `.float32` is that run on the Micrograph's float32 pixels:
//  SciPy forms each sample in double and writes it as float32; the reduction is NumPy's float32 reduction along the
//  profile's width (first value, then NumPy's pairwise sum of the rest; the median of two is their float32 mean).
//  `.float64` keeps every sample in Double and reduces in Double: scikit-image on the same pixels widened to float64, to
//  rounding of the sum. The tests state both gaps against the pinned NumPy output.
//
//  SPLINES (orders 2 to 5). As SciPy: the image is prefiltered once into B-spline coefficients (Float64, the poles and gain
//  of ni_splines.c, the causal and anticausal recursions of `spline_filter1d`, mirror initialisation for "mirror" and
//  "constant", reflect for "reflect" and "nearest", wrap for "wrap"), then each sample is the sum of (order + 1)² taps
//  weighted by the B-spline. "constant" and "nearest" first pad the image by 12 pixels of `cval`, or of the edge pixel
//  (`_prepad_for_spline_filter`); a tap beyond the padded array is `cval` ("constant") or the edge coefficient
//  ("nearest", whose coordinate is NOT clamped, so the weights stay those of the true position); "reflect", "mirror"
//  and "wrap" fold the coordinate and the tap indices. The whole image is filtered on every call: at order 2 to 5 a
//  profile costs one pass over the image (and 8 bytes per pixel), as in SciPy.
//
//  Absent: multichannel images, modes other than scikit-image's five.
//
//  DEVIATION from scikit-image: a width below 1 (an empty axis, a NaN mean), a segment with a non-finite end, an end
//  beyond 1e15 px or more than 2²⁴ samples throw (scikit-image would produce NaN, raise, lose every digit of the fold, or
//  exhaust memory). A NaN pixel poisons every sample that has it among its four neighbours, even at weight 0, as in SciPy.
//

import Foundation

/// scikit-image's `mode` for `profile_line`.
public nonisolated enum ProfileMode: String, Sendable, Equatable, Hashable, CaseIterable {
    /// `cval` beyond the edge (SciPy's grid-constant).
    case constant
    /// The nearest edge pixel.
    case nearest
    /// Half-sample symmetric: d c b a | a b c d | d c b a. scikit-image's default.
    case reflect
    /// Whole-sample symmetric: d c b | a b c d | c b a.
    case mirror
    /// The image repeats with period n (SciPy's grid-wrap).
    case wrap
}

/// The interpolation order of `profile_line`: scikit-image's default is 1 for float images.
public nonisolated enum ProfileOrder: Int, Sendable, Equatable, Hashable, CaseIterable {
    case nearest = 0
    case linear = 1
    case quadratic = 2
    case cubic = 3
    case quartic = 4
    case quintic = 5
}

/// The `reduce_func` applied across the width at each sample.
public nonisolated enum ProfileReduction: Sendable {
    case mean, sum, min, max, median
    /// Any function of the w perpendicular values (in scikit-image's order, the reverse of the perpendicular's), as
    /// `reduce_func` accepts one that has no `axis` argument.
    case custom(@Sendable ([Double]) -> Double)
}

/// The arithmetic of the profile (see the file header).
public nonisolated enum ProfileArithmetic: String, Sendable, Equatable, Hashable, CaseIterable {
    case float32
    case float64
}

public nonisolated enum LineProfileError: Error, Equatable, Sendable {
    case invalidWidth(Int)
    /// An end that is not finite or lies beyond 1e15 px, or a line of more than `LineProfile.maxSamples` samples.
    case invalidSegment
}

public nonisolated struct LineProfile: Sendable {
    /// The reduced intensity at each of the n points.
    public let samples: [Double]
    /// Where each sample sits along the line, in `unit`; the first is 0 and, with n > 1, the last is the line's length.
    public let positions: [Double]
    /// The axes' shared unit, or "px" when the positions are pixels.
    public let unit: String
    /// Whether `positions` are physical (both axes calibrated in one unit).
    public let isPhysical: Bool
    /// The width averaged over, in pixels.
    public let width: Int

    public init(samples: [Double], positions: [Double], unit: String, isPhysical: Bool, width: Int) {
        self.samples = samples; self.positions = positions; self.unit = unit; self.isPhysical = isPhysical
        self.width = width
    }

    /// More samples than this is not a line anyone drew on an image.
    public static let maxSamples = 1 << 24

    /// The profile along `from` → `to` (rows and columns as `PixelPoint.y` and `.x`), with `reduce` across `linewidth`.
    public static func sample(_ image: Micrograph, from: PixelPoint, to: PixelPoint, linewidth: Int = 1,
                              order: ProfileOrder = .linear, mode: ProfileMode = .reflect, cval: Double = 0,
                              reduce: ProfileReduction = .mean,
                              arithmetic: ProfileArithmetic = .float32) throws -> LineProfile {
        let rows = try perpendicularValues(image, from: from, to: to, linewidth: linewidth, order: order, mode: mode,
                                           cval: cval, arithmetic: arithmetic)
        let n = rows.count / linewidth
        var samples = [Double](repeating: 0, count: n)
        var scratch = [Double](repeating: 0, count: linewidth)
        for i in 0..<n {
            for k in 0..<linewidth { scratch[k] = rows[i * linewidth + k] }
            samples[i] = reduced(scratch, reduce, arithmetic)
        }
        let length = ImageMeasurement.distance(from, to, x: image.x, y: image.y)
        let total = length.physical ?? length.pixels
        let posStep = n > 1 ? total / Double(n - 1) : 0
        var positions = [Double](repeating: 0, count: n)
        for i in 0..<n { positions[i] = n > 1 && i == n - 1 ? total : Double(i) * posStep }
        return LineProfile(samples: samples, positions: positions, unit: length.unit ?? "px",
                           isPhysical: length.unit != nil, width: linewidth)
    }

    /// `profile_line(..., reduce_func=None)`: n rows of `linewidth` values, each row in scikit-image's order (the
    /// perpendicular reversed, as `np.flip(pixels, axis=1)` leaves it).
    public static func unreduced(_ image: Micrograph, from: PixelPoint, to: PixelPoint, linewidth: Int = 1,
                                 order: ProfileOrder = .linear, mode: ProfileMode = .reflect, cval: Double = 0,
                                 arithmetic: ProfileArithmetic = .float32) throws -> [[Double]] {
        let flat = try perpendicularValues(image, from: from, to: to, linewidth: linewidth, order: order, mode: mode,
                                           cval: cval, arithmetic: arithmetic)
        return stride(from: 0, to: flat.count, by: linewidth).map { Array(flat[$0..<($0 + linewidth)]) }
    }

    // MARK: coordinates and sampling

    /// All n × w values, row-major, each row already reversed (`np.flip(axis=1)`).
    private static func perpendicularValues(_ image: Micrograph, from: PixelPoint, to: PixelPoint, linewidth w: Int,
                                            order: ProfileOrder, mode: ProfileMode, cval: Double,
                                            arithmetic: ProfileArithmetic) throws -> [Double] {
        guard w >= 1 else { throw LineProfileError.invalidWidth(w) }
        let dRow = to.y - from.y, dCol = to.x - from.x
        let hyp = hypot(dRow, dCol)
        // A coordinate beyond 1e15 px is no mark on an image, and folding it into the reflected range loses every digit.
        guard hyp.isFinite, hyp + 1 <= Double(maxSamples),
              max(abs(from.x), abs(from.y), abs(to.x), abs(to.y)) <= 1e15 else { throw LineProfileError.invalidSegment }
        let n = Int((hyp + 1).rounded(.up))
        let theta = atan2(dRow, dCol)
        let colHalf = Double(w - 1) * sin(-theta) / 2          // skimage's col_width
        let rowHalf = Double(w - 1) * cos(theta) / 2           // skimage's row_width
        let rowStep = n > 1 ? dRow / Double(n - 1) : 0         // numpy.linspace(src, dst, n)
        let colStep = n > 1 ? dCol / Double(n - 1) : 0
        let boundary = Boundary(mode)
        let field = order.rawValue >= 2
            ? SplineField(image, order: order.rawValue, boundary: boundary, cval: cval) : nil
        var out = [Double](repeating: 0, count: n * w)
        image.pixels.withUnsafeBufferPointer { pixels in
            for i in 0..<n {
                let last = n > 1 && i == n - 1
                let lineRow = last ? to.y : Double(i) * rowStep + from.y
                let lineCol = last ? to.x : Double(i) * colStep + from.x
                let rowStart = lineRow - rowHalf, rowEnd = lineRow + rowHalf
                let colStart = lineCol - colHalf, colEnd = lineCol + colHalf
                let perpRowStep = w > 1 ? (rowEnd - rowStart) / Double(w - 1) : 0      // numpy.linspace(·, ·, w)
                let perpColStep = w > 1 ? (colEnd - colStart) / Double(w - 1) : 0
                for k in 0..<w {
                    let pRow = w > 1 && k == w - 1 ? rowEnd : Double(k) * perpRowStep + rowStart
                    let pCol = w > 1 && k == w - 1 ? colEnd : Double(k) * perpColStep + colStart
                    var v = field.map { $0.value(row: pRow, col: pCol, cval: cval) }
                        ?? mapCoordinates(pixels, width: image.width, height: image.height, row: pRow, col: pCol,
                                          order: order, boundary: boundary, cval: cval)
                    if arithmetic == .float32 { v = Double(Float(v)) }
                    out[i * w + (w - 1 - k)] = v                // np.flip(pixels, axis=1)
                }
            }
        }
        return out
    }

    package nonisolated enum Boundary { case gridConstant, nearest, reflect, mirror, gridWrap
        init(_ mode: ProfileMode) {
            switch mode {
            case .constant: self = .gridConstant
            case .nearest: self = .nearest
            case .reflect: self = .reflect
            case .mirror: self = .mirror
            case .wrap: self = .gridWrap
            }
        }
    }

    /// One axis's neighbours and weights; an index of −1 stands for `cval`.
    private struct Axis { var i0: Int, i1: Int, w0: Double, w1: Double }

    /// SciPy's `map_coordinate` (ni_interpolation.c) for the five modes scikit-image offers.
    private static func mapped(_ x: Double, _ n: Int, _ boundary: Boundary) -> Double {
        var x = x
        if x < 0 {
            switch boundary {
            case .mirror:
                if n <= 1 { x = 0 } else {
                    let s2 = Double(2 * n - 2)
                    x = s2 * (-x / s2).rounded(.towardZero) + x
                    x = x <= Double(1 - n) ? x + s2 : -x
                }
            case .reflect:
                if n <= 1 { x = 0 } else {
                    let s2 = Double(2 * n)
                    if x < -s2 { x = s2 * (-x / s2).rounded(.towardZero) + x }
                    x = x < Double(-n) ? x + s2 : (x == 0 ? 0 : -x - 1)
                }
            case .gridWrap:
                if n <= 1 { x = 0 } else {
                    let sz = Double(n)
                    x += sz * (((-1 - x) / sz).rounded(.towardZero) + 1)
                }
            case .nearest: x = 0
            case .gridConstant: break
            }
        } else if x > Double(n - 1) {
            switch boundary {
            case .mirror:
                if n <= 1 { x = 0 } else {
                    let s2 = Double(2 * n - 2)
                    x -= s2 * (x / s2).rounded(.towardZero)
                    if x >= Double(n) { x = s2 - x }
                }
            case .reflect:
                if n <= 1 { x = 0 } else {
                    let s2 = Double(2 * n)
                    x -= s2 * (x / s2).rounded(.towardZero)
                    if x >= Double(n) { x = s2 - x - 1 }
                }
            case .gridWrap:
                if n <= 1 { x = 0 } else {
                    let sz = Double(n)
                    x -= sz * (x / sz).rounded(.towardZero)
                }
            case .nearest: x = Double(n - 1)
            case .gridConstant: break
            }
        }
        return x
    }

    private static func neighbour(_ index: Int, _ n: Int, _ boundary: Boundary) -> Int {
        if boundary == .gridConstant { return index >= 0 && index < n ? index : -1 }
        if index >= 0 && index < n { return index }
        return Int(mapped(Double(index), n, boundary))
    }

    private static func axis(_ x: Double, _ n: Int, _ order: ProfileOrder, _ boundary: Boundary) -> Axis {
        let cc = mapped(x, n, boundary)
        switch order {
        case .nearest:
            return Axis(i0: neighbour(Int((cc + 0.5).rounded(.down)), n, boundary), i1: 0, w0: 1, w1: 0)
        case .linear:
            let floor = cc.rounded(.down)
            let f = cc - floor
            return Axis(i0: neighbour(Int(floor), n, boundary), i1: neighbour(Int(floor) + 1, n, boundary), w0: 1 - f, w1: f)
        default:
            preconditionFailure("orders 2 to 5 read a SplineField")
        }
    }

    @inline(__always)
    private static func mapCoordinates(_ pixels: UnsafeBufferPointer<Float>, width: Int, height: Int, row: Double,
                                       col: Double, order: ProfileOrder, boundary: Boundary, cval: Double) -> Double {
        let a = axis(row, height, order, boundary), b = axis(col, width, order, boundary)
        func value(_ i: Int, _ j: Int) -> Double { i < 0 || j < 0 ? cval : Double(pixels[i * width + j]) }
        if order == .nearest { return value(a.i0, b.i0) }
        var t = 0.0
        t += value(a.i0, b.i0) * a.w0 * b.w0
        t += value(a.i0, b.i1) * a.w0 * b.w1
        t += value(a.i1, b.i0) * a.w1 * b.w0
        t += value(a.i1, b.i1) * a.w1 * b.w1
        return t
    }


    // MARK: splines (orders 2 to 5)

    /// SciPy's prefiltered B-spline coefficients of an image, padded as `_prepad_for_spline_filter` does.
    package nonisolated struct SplineField {
        let coefficients: [Double]
        package let rows: Int, cols: Int
        let pad: Int
        let order: Int
        let boundary: Boundary

        package init(_ image: Micrograph, order: Int, boundary: Boundary, cval: Double) {
            self.order = order
            self.boundary = boundary
            pad = boundary == .nearest || boundary == .gridConstant ? 12 : 0
            rows = image.height + 2 * pad
            cols = image.width + 2 * pad
            var c = [Double](repeating: boundary == .gridConstant ? cval : 0, count: rows * cols)
            for r in 0..<rows {
                let sr = min(max(r - pad, 0), image.height - 1)
                for k in 0..<cols {
                    let sc = min(max(k - pad, 0), image.width - 1)
                    let inside = r >= pad && r < rows - pad && k >= pad && k < cols - pad
                    if inside || boundary == .nearest { c[r * cols + k] = Double(image.pixels[sr * image.width + sc]) }
                }
            }
            let kind: SplineBoundary = switch boundary {
            case .mirror, .gridConstant: .mirror
            case .reflect, .nearest: .reflect
            case .gridWrap: .wrap
            }
            var line = [Double](repeating: 0, count: max(rows, cols))
            for r in 0..<rows {                                   // along the columns (axis 1), then the rows (axis 0)
                for k in 0..<cols { line[k] = c[r * cols + k] }
                SplineFilter.filter(&line, count: cols, order: order, kind)
                for k in 0..<cols { c[r * cols + k] = line[k] }
            }
            for k in 0..<cols {
                for r in 0..<rows { line[r] = c[r * cols + k] }
                SplineFilter.filter(&line, count: rows, order: order, kind)
                for r in 0..<rows { c[r * cols + k] = line[r] }
            }
            coefficients = c
        }

        /// One axis: the (order + 1) tap indices (mapped by the boundary; −1 is a `cval` tap) and the B-spline weights.
        package func taps(_ x: Double, _ n: Int) -> (indices: [Int], weights: [Double]) {
            let shifted = x + Double(pad)
            // SciPy folds the coordinate for mirror, reflect and wrap; nearest and constant keep it (the taps carry the edge).
            let cc = boundary == .nearest || boundary == .gridConstant ? shifted : LineProfile.mapped(shifted, n, boundary)
            let first = (order & 1 == 1 ? cc.rounded(.down) : (cc + 0.5).rounded(.down)) - Double(order / 2)
            let start = Int(first)
            let f = order & 1 == 1 ? cc - cc.rounded(.down) : cc - (cc + 0.5).rounded(.down)
            var weights = [Double](repeating: 0, count: order + 1)
            for k in 0...order { weights[k] = SplineFilter.basis(order, Double(k - order / 2) - f) }
            var indices = [Int](repeating: 0, count: order + 1)
            for k in 0...order {
                let i = start + k
                if i >= 0 && i < n { indices[k] = i; continue }
                switch boundary {
                case .gridConstant: indices[k] = -1
                case .nearest: indices[k] = i < 0 ? 0 : n - 1
                default: indices[k] = Int(LineProfile.mapped(Double(i), n, boundary))
                }
            }
            return (indices, weights)
        }

        /// The tap sum for one output pixel from its precomputed row and column taps (`taps(_, rows)` / `taps(_, cols)`).
        package func value(rowTaps a: (indices: [Int], weights: [Double]), columnTaps b: (indices: [Int], weights: [Double]),
                           cval: Double) -> Double {
            var t = 0.0
            for (i, wi) in zip(a.indices, a.weights) {
                for (j, wj) in zip(b.indices, b.weights) {
                    t += (i < 0 || j < 0 ? cval : coefficients[i * cols + j]) * wi * wj
                }
            }
            return t
        }

        package func value(row: Double, col: Double, cval: Double) -> Double {
            let a = taps(row, rows), b = taps(col, cols)
            var t = 0.0
            for (i, wi) in zip(a.indices, a.weights) {
                for (j, wj) in zip(b.indices, b.weights) {
                    t += (i < 0 || j < 0 ? cval : coefficients[i * cols + j]) * wi * wj
                }
            }
            return t
        }
    }

    fileprivate nonisolated enum SplineBoundary { case mirror, reflect, wrap }

    // MARK: reductions along the width

    private static func reduced(_ values: [Double], _ reduce: ProfileReduction, _ arithmetic: ProfileArithmetic) -> Double {
        if case .custom(let f) = reduce { return f(values) }
        switch arithmetic {
        case .float64: return apply(values)
        case .float32: return Double(apply(values.map { Float($0) }))
        }
        func apply<S: BinaryFloatingPoint>(_ v: [S]) -> S {
            switch reduce {
            case .sum: return numpySum(v)
            case .mean: return numpySum(v) / S(v.count)
            case .min: return v.contains { $0.isNaN } ? .nan : v.min()!
            case .max: return v.contains { $0.isNaN } ? .nan : v.max()!
            case .median:
                if v.contains(where: { $0.isNaN }) { return .nan }
                let s = v.sorted(), m = s.count / 2
                return s.count % 2 == 1 ? s[m] : (s[m - 1] + s[m]) / 2
            case .custom: preconditionFailure()
            }
        }
    }

    /// NumPy's `np.add.reduce` over one row of the flipped view: `pairwise_sum` of the whole row, in the flipped order
    /// (measured bit for bit against NumPy at widths 3 to 25; the first-value-then-rest form does not match).
    private static func numpySum<S: BinaryFloatingPoint>(_ flipped: [S]) -> S {
        pairwise(flipped, 0, flipped.count)
    }

    private static func pairwise<S: BinaryFloatingPoint>(_ a: [S], _ start: Int, _ n: Int) -> S {
        if n < 8 {
            var res: S = -0.0
            for i in 0..<n { res += a[start + i] }
            return res
        } else if n <= 128 {
            var r = (0..<8).map { a[start + $0] }
            var i = 8
            while i < n - (n % 8) {
                for j in 0..<8 { r[j] += a[start + i + j] }
                i += 8
            }
            var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
            while i < n { res += a[start + i]; i += 1 }
            return res
        } else {
            var n2 = n / 2
            n2 -= n2 % 8
            return pairwise(a, start, n2) + pairwise(a, start + n2, n - n2)
        }
    }
}

/// SciPy's `ni_splines.c`: the B-spline basis and the 1-D prefilter (`spline_filter1d`), orders 2 to 5.
private nonisolated enum SplineFilter {
    /// The poles to the digits SciPy carries (its closed forms lose about 12 digits to cancellation at orders 4 and 5).
    static func poles(_ order: Int) -> [Double] {
        switch order {
        case 2: return [Double(8).squareRoot() - 3]
        case 3: return [Double(3).squareRoot() - 2]
        case 4: return [-0.361341225900220177092, -0.0137254292973391537]
        default: return [-0.430575347099973791852, -0.043096288203264653822]
        }
    }

    /// The centred cardinal B-spline of degree `d` at `t` (support (−(d+1)/2, (d+1)/2), half-open at the ends).
    static func basis(_ d: Int, _ t: Double) -> Double {
        if d == 0 { return t >= -0.5 && t < 0.5 ? 1 : 0 }
        let h = Double(d + 1) / 2
        return ((t + h) * basis(d - 1, t + 0.5) + (h - t) * basis(d - 1, t - 0.5)) / Double(d)
    }

    /// `spline_filter1d` of `c[0..<n]` in place, for the boundary kind SciPy's mode selects.
    static func filter(_ c: inout [Double], count n: Int, order: Int, _ kind: LineProfile.SplineBoundary) {
        if n < 2 { return }
        let zs = poles(order)
        var gain = 1.0
        for z in zs { gain *= (1 - z) * (1 - 1 / z) }
        for i in 0..<n { c[i] *= gain }
        for z in zs {
            let maxIter = Int((log(1e-15) / log(abs(z))).rounded(.up))
            let truncated = maxIter < n
            var c0: Double
            switch kind {
            case .mirror:
                if truncated {
                    var zi = z
                    c0 = c[0]
                    for i in 1..<maxIter { c0 += zi * c[i]; zi *= z }
                } else {
                    var zi = z
                    let zn1 = pow(z, Double(n - 1))
                    c0 = c[0] + zn1 * c[n - 1]
                    for i in 1..<(n - 1) { c0 += (zi + zn1 * zn1 / zi) * c[i]; zi *= z }
                    c0 /= 1 - zn1 * zn1
                }
            case .reflect:
                if truncated {
                    var zi = z
                    var s = c[0]
                    for i in 1..<maxIter { s += zi * c[i]; zi *= z }
                    c0 = c[0] + z * s
                } else {
                    var zi = z
                    let zn = pow(z, Double(n))
                    var s = c[0] + zn * c[n - 1]
                    for i in 1..<n { s += zi * (c[i] + zn * c[n - 1 - i]); zi *= z }
                    s /= 1 - zn * zn
                    c0 = c[0] + z * s
                }
            case .wrap:
                var zi = z
                let zn = pow(z, Double(n))
                c0 = c[0]
                for i in 1..<n { c0 += zi * c[n - i]; zi *= z }
                c0 /= 1 - zn
            }
            c[0] = c0
            for i in 1..<n { c[i] += z * c[i - 1] }
            switch kind {
            case .mirror: c[n - 1] = (z / (z * z - 1)) * (z * c[n - 2] + c[n - 1])
            case .reflect: c[n - 1] = c[n - 1] * z / (z - 1)
            case .wrap:
                var zi = z
                let zn = pow(z, Double(n))
                var s = c[n - 1]
                for i in 0..<(n - 1) { s += zi * c[i]; zi *= z }
                c[n - 1] = z * s / (zn - 1)
            }
            var i = n - 2
            while i >= 0 { c[i] = z * (c[i + 1] - c[i]); i -= 1 }
        }
    }
}
