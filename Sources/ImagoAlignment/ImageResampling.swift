// SPDX-License-Identifier: GPL-3.0-only
//
//  ImageResampling.swift (ImagoAlignment)
//  SciPy 1.18.1's `ndimage.zoom` and `ndimage.rotate` on a 2D image, on the kit's spline machinery (the B-spline prefilter and
//  taps of `ndimage.shift`): orders 0 to 5, modes `constant` (SciPy's default), `grid-constant`, `nearest`, `reflect`, `mirror` and
//  `grid-wrap` (`wrap` for orders 0 and 1 only: its spline rule is ported for shifts but not checked here, `AlignmentError.splineMode`).
//
//  ZOOM. The output has `round(n * zoom)` pixels per axis (Python's round: half to even). Output index k reads the input at
//  k * (n - 1) / (m - 1) (the corners of the first and last pixel coincide; `gridMode` false, SciPy's default) or at
//  (k + 0.5) * n / m - 0.5 (`gridMode` true: the pixel grids coincide at their edges). A zoom of 1 returns the image.
//
//  ROTATE. `angle` degrees in the plane of (rows, columns), SciPy's rotation matrix [[c, s], [-s, c]] with the exact
//  cosine and sine of an angle in degrees (`cosdg`, `sindg`: a quarter turn has exactly 0 and 1), `reshape` true enlarging the
//  output to hold the whole rotated image (`int(ptp(bounds) + 0.5)` per axis) and false keeping the shape; the output pixel o
//  reads the input at M o + offset with offset = in_centre - M out_centre, the arithmetic done in SciPy's order and with its fused
//  multiply-adds (SciPy's C build contracts `cc += m * i`; an order-0 sample on an exact half pixel, as a 30 degree turn makes,
//  rounds differently without it: measured on the pins).
//
//  "constant" mode: a source coordinate outside [0, n - 1] on either axis gives `cval` for the whole pixel; one inside reads the
//  spline with its taps mirrored (so a zoom, whose coordinates are always inside, is the same in "constant" and "mirror").
//  Calibration: a zoom rescales each axis's scale by n / m and its origin by m / n; a rotation drops the calibration (the
//  rotated frame has no row or column axis). The result is Float32 whatever the arithmetic, as `ImageShift`'s.
//

import Foundation
import ImagoCore

public nonisolated enum ImageResampling {
    /// `ndimage.zoom(image, (zoom.rows, zoom.columns), order:, mode:, cval:, grid_mode:)`.
    public static func zoom(_ image: Micrograph, zoom: (rows: Double, columns: Double), order: ShiftOrder = .cubic,
                            mode: ShiftMode = .constant, cval: Double = 0, gridMode: Bool = false) throws -> Micrograph {
        guard zoom.rows.isFinite, zoom.columns.isFinite, zoom.rows > 0, zoom.columns > 0, cval.isFinite else {
            throw AlignmentError.invalidResampling("zoom factors must be finite and positive")
        }
        let h = image.height, w = image.width
        let fh = (Double(h) * zoom.rows).rounded(.toNearestOrEven), fw = (Double(w) * zoom.columns).rounded(.toNearestOrEven)
        guard fh >= 1, fw >= 1, fh <= Double(maximumSide), fw <= Double(maximumSide),
              fh * fw <= Double(maximumPixels) else {
            throw AlignmentError.invalidResampling("the zoomed image would be \(fh) x \(fw) pixels")
        }
        let outH = Int(fh), outW = Int(fw)
        if zoom.rows == 1, zoom.columns == 1 { return image }
        let rowCoordinates = coordinates(inCount: h, outCount: outH, gridMode: gridMode)
        let columnCoordinates = coordinates(inCount: w, outCount: outW, gridMode: gridMode)
        var out = [Float](repeating: 0, count: outH * outW)
        if order.rawValue >= 2 {
            let field = try splineField(image, order, mode, cval)
            let rowTaps = rowCoordinates.map { axisTaps(field, $0, field.rows, mode, h) }
            let columnTaps = columnCoordinates.map { axisTaps(field, $0, field.cols, mode, w) }
            for r in 0..<outH {
                for c in 0..<outW {
                    guard let a = rowTaps[r], let b = columnTaps[c] else { out[r * outW + c] = Float(cval); continue }
                    out[r * outW + c] = Float(field.value(rowTaps: a, columnTaps: b, cval: cval))
                }
            }
        } else {
            let rows = rowCoordinates.map { ImageShift.axisSample($0, h, order, mode) }
            let columns = columnCoordinates.map { ImageShift.axisSample($0, w, order, mode) }
            let pixels = image.pixels.map(Double.init)
            for r in 0..<outH {
                for c in 0..<outW {
                    let a = rows[r], b = columns[c]
                    out[r * outW + c] = a.constant || b.constant ? Float(cval)
                        : Float(ImageShift.combine(pixels, w, a, b, order, cval))
                }
            }
        }
        let rowScale = Double(h) / Double(outH), columnScale = Double(w) / Double(outW)
        func rescaled(_ axis: AxisCalibration?, by factor: Double) -> AxisCalibration? {
            axis.map { AxisCalibration(scale: $0.scale * factor, unit: $0.unit, origin: $0.origin / factor) }
        }
        return try Micrograph(width: outW, height: outH, pixels: out, x: rescaled(image.x, by: columnScale), y: rescaled(image.y, by: rowScale))
    }

    /// `ndimage.rotate(image, angle, reshape:, order:, mode:, cval:)` in the (rows, columns) plane.
    public static func rotate(_ image: Micrograph, degrees angle: Double, reshape: Bool = true, order: ShiftOrder = .cubic,
                              mode: ShiftMode = .constant, cval: Double = 0) throws -> Micrograph {
        guard angle.isFinite, abs(angle) <= 1e9, cval.isFinite else { throw AlignmentError.invalidResampling("the angle must be finite") }
        let h = image.height, w = image.width
        let c = cosdg(angle), s = sindg(angle)
        // rot_matrix = [[c, s], [-s, c]] applied to (row, column) coordinates.
        var outH = h, outW = w
        if reshape {
            // out_bounds = rot_matrix @ [[0, 0, iy, iy], [0, ix, 0, ix]]; the shape is int(ptp + 0.5) per axis.
            let iy = Double(h), ix = Double(w)
            let rows = [c * 0 + s * 0, c * 0 + s * ix, c * iy + s * 0, c * iy + s * ix]
            let columns = [-s * 0 + c * 0, -s * 0 + c * ix, -s * iy + c * 0, -s * iy + c * ix]
            outH = Int((rows.max()! - rows.min()! + 0.5).rounded(.towardZero))
            outW = Int((columns.max()! - columns.min()! + 0.5).rounded(.towardZero))
        }
        guard outH >= 1, outW >= 1, outH <= maximumSide, outW <= maximumSide, outH * outW <= maximumPixels else {
            throw AlignmentError.invalidResampling("the rotated image would be \(outH) x \(outW) pixels")
        }
        let outCentreRow = c * (Double(outH - 1) / 2) + s * (Double(outW - 1) / 2)
        let outCentreColumn = -s * (Double(outH - 1) / 2) + c * (Double(outW - 1) / 2)
        let offsetRow = Double(h - 1) / 2 - outCentreRow, offsetColumn = Double(w - 1) / 2 - outCentreColumn
        var out = [Float](repeating: 0, count: outH * outW)
        let field = order.rawValue >= 2 ? try splineField(image, order, mode, cval) : nil
        let pixels = field == nil ? image.pixels.map(Double.init) : []
        for r in 0..<outH {
            for k in 0..<outW {
                // The coordinate as SciPy's geometric transform builds it: offset, then each matrix term added in turn.
                var cr = offsetRow; cr = fma(c, Double(r), cr); cr = fma(s, Double(k), cr)
                var cc = offsetColumn; cc = fma(-s, Double(r), cc); cc = fma(c, Double(k), cc)
                if let field {
                    guard let a = axisTaps(field, cr, field.rows, mode, h), let b = axisTaps(field, cc, field.cols, mode, w) else {
                        out[r * outW + k] = Float(cval); continue
                    }
                    out[r * outW + k] = Float(field.value(rowTaps: a, columnTaps: b, cval: cval))
                } else {
                    let a = ImageShift.axisSample(cr, h, order, mode), b = ImageShift.axisSample(cc, w, order, mode)
                    out[r * outW + k] = a.constant || b.constant ? Float(cval)
                        : Float(ImageShift.combine(pixels, w, a, b, order, cval))
                }
            }
        }
        return try Micrograph(width: outW, height: outH, pixels: out)
    }

    /// The largest output side either call produces.
    public static let maximumSide = 1 << 15
    /// The largest output area either call produces (64 mega-pixels, 256 MB of Float32).
    public static let maximumPixels = 1 << 26

    // MARK: pieces

    /// The input coordinate of every output index (SciPy's `zoom_shift`: `cc = k * zoom`, or `(k + 0.5) * zoom - 0.5` on the grid).
    private static func coordinates(inCount n: Int, outCount m: Int, gridMode: Bool) -> [Double] {
        let nominator = gridMode ? Double(n) : Double(n - 1)
        let divisor = gridMode ? Double(m) : Double(m - 1)
        let zoom = divisor != 0 ? nominator / divisor : 1
        return (0..<m).map { k in gridMode ? (Double(k) + 0.5) * zoom - 0.5 : Double(k) * zoom }
    }

    private static func splineField(_ image: Micrograph, _ order: ShiftOrder, _ mode: ShiftMode, _ cval: Double) throws -> LineProfile.SplineField {
        let boundary: LineProfile.Boundary
        switch mode {
        case .constant, .mirror: boundary = .mirror
        case .gridConstant: boundary = .gridConstant
        case .nearest: boundary = .nearest
        case .reflect: boundary = .reflect
        case .gridWrap: boundary = .gridWrap
        case .wrap: throw AlignmentError.splineMode(mode)
        }
        return LineProfile.SplineField(image, order: order.rawValue, boundary: boundary, cval: cval)
    }

    /// One axis's taps and weights at coordinate `x`, or nil when "constant" mode puts `x` outside [0, n - 1].
    private static func axisTaps(_ field: LineProfile.SplineField, _ x: Double, _ fieldCount: Int, _ mode: ShiftMode,
                                 _ n: Int) -> (indices: [Int], weights: [Double])? {
        if mode == .constant, x < 0 || x > Double(n - 1) { return nil }
        return field.taps(x, fieldCount)
    }

    // MARK: cosdg and sindg (Cephes, as scipy.special)

    private static let sincof: [Double] = [1.58962301576546568060e-10, -2.50507477628578072866e-8, 2.75573136213857245213e-6,
                                           -1.98412698295895385996e-4, 8.33333333332211858878e-3, -1.66666666666666307295e-1]
    private static let coscof: [Double] = [-1.13585365213876817300e-11, 2.08757008419747316778e-9, -2.75573141792967388112e-7,
                                           2.48015872888517045348e-5, -1.38888888888730564116e-3, 4.16666666666665929218e-2]
    private static let pi180 = 1.74532925199432957692e-2

    private static func polevl(_ x: Double, _ coefficients: [Double]) -> Double {
        var ans = coefficients[0]
        for c in coefficients.dropFirst() { ans = ans * x + c }
        return ans
    }

    /// Cephes `sindg` / `cosdg`: the angle is reduced in degrees to within 45 of a multiple of 90 before the conversion, so
    /// the quarter turns are exact.
    static func sindg(_ degrees: Double) -> Double { trigDegrees(degrees, cosine: false) }
    static func cosdg(_ degrees: Double) -> Double { trigDegrees(degrees, cosine: true) }

    private static func trigDegrees(_ angle: Double, cosine: Bool) -> Double {
        var x = angle
        var sign = 1.0
        if x < 0 { x = -x; if !cosine { sign = -1 } }
        var y = (x / 45.0).rounded(.down)
        let z16 = (y / 16.0).rounded(.down)
        var j = Int(y - z16 * 16.0)
        if j & 1 == 1 { j += 1; y += 1 }
        j &= 7
        if j > 3 { j -= 4; sign = -sign }
        if cosine, j > 1 { sign = -sign }
        let z = (x - y * 45.0) * pi180
        let zz = z * z
        let sineSeries = cosine ? (j == 1 || j == 2) : !(j == 1 || j == 2)
        let value = sineSeries ? z + z * (zz * polevl(zz, sincof)) : 1.0 - zz * 0.5 + zz * zz * polevl(zz, coscof)
        return sign * value
    }
}
