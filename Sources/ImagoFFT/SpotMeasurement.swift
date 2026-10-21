//
//  SpotMeasurement.swift (ImagoFFT)
//  The spacing of one spot of an image's spectrum, from where the user clicked, and the angle between two spots
//  (registration-slice-1.md 1.2).
//
//  THE ESTIMATOR (every step registered): the image (or crop) minus its mean, times a Hann window (NumPy's
//  hanning(n) = 0.5 − 0.5·cos(2πk/(n − 1)), x and y), transformed; the power of the centred spectrum; the clicked bin snapped
//  to the largest power within ±2 bins; the power-weighted centroid of the 5 × 5 window around it, no background; the
//  centroid's signed offsets from the centre times each axis's own spacing give |q|, and d = 1/|q|. The window belongs to
//  the measurement: a `PowerSpectrum` stays unwindowed.
//
//  ACCURACY (planted lattices): |d − d₀|/d₀ ≤ 0.2 % noise-free; ≤ 0.5 % with noise σ = 1 per pixel from 256 px and 0.1
//  Nyquist up. Smaller or noisier images are not promised that (1–5 % measured at 64 px). A spot within 3 bins (index
//  distance) of the centre is refused (the DC's skirt), and one within 2 bins of the spectrum's edge (the window would leave
//  it); another peak above 10 % of the spot's power within 3 bins is flagged (it biases d by up to 0.3 %).
//

import Foundation
import ImagoCore

/// Why a spot cannot be measured, or two spots compared.
public nonisolated enum SpotMeasurementError: Error, Equatable, Sendable {
    /// The snapped bin is within 3 bins of the zero frequency.
    case nearDC
    /// The snapped bin is within the centroid window's half-width of the spectrum's edge.
    case nearEdge
    case crop(PixelCrop)
    case mixedUnits(String, String)
    case fftSetup(width: Int, height: Int)
    case outside(row: Int, col: Int)
    /// One of two spots compared has |q| = 0.
    case zeroFrequency
    /// The image (or crop) holds a NaN or an infinity: its spectrum is not defined.
    case nonFinite
    /// The window has no power once its mean is removed (a constant image or crop), or none in the 5 × 5 window about the
    /// snapped bin: there is no spot to centre on, and the centroid would be 0/0.
    case noPower
}

public nonisolated struct SpotMeasurement: Sendable, Equatable {
    public let clickedRow: Int
    public let clickedCol: Int
    public let snappedRow: Int
    public let snappedCol: Int
    /// The centroid in centred-spectrum index units (row, column), fractional.
    public let centroidRow: Double
    public let centroidCol: Double
    /// The image's size the spectrum was taken of (the crop's, when there is one).
    public let width: Int
    public let height: Int
    /// |q| and its components, in `reciprocalUnits`; d = 1/|q| in `units`.
    public let qx: Double
    public let qy: Double
    public let q: Double
    public let d: Double
    public let units: String
    public let reciprocalUnits: String
    /// Another peak above 10 % of the spot's power within 3 bins: d may be biased (up to 0.3 % measured).
    public let neighbour: Bool
    public let precision: Precision

    /// A measurement from its values (a stored one, or a planted one in a test); `measure` is the estimator.
    public init(clickedRow: Int, clickedCol: Int, snappedRow: Int, snappedCol: Int, centroidRow: Double, centroidCol: Double,
                width: Int, height: Int, qx: Double, qy: Double, q: Double, d: Double, units: String, reciprocalUnits: String,
                neighbour: Bool, precision: Precision) {
        self.clickedRow = clickedRow; self.clickedCol = clickedCol; self.snappedRow = snappedRow; self.snappedCol = snappedCol
        self.centroidRow = centroidRow; self.centroidCol = centroidCol; self.width = width; self.height = height
        self.qx = qx; self.qy = qy; self.q = q; self.d = d; self.units = units; self.reciprocalUnits = reciprocalUnits
        self.neighbour = neighbour; self.precision = precision
    }

    /// d in nanometres when `units` is a length unit; nil otherwise.
    public var dNanometres: Double? { LengthUnit.nanometres(per: units).map { d * $0 } }
    /// |q| in reciprocal nanometres when `units` is a length unit; nil otherwise.
    public var qPerNanometre: Double? { LengthUnit.nanometres(per: units).map { q / $0 } }

    public static func measure(_ image: Micrograph, crop: PixelCrop? = nil, clickRow: Int, clickColumn: Int,
                               precision: Precision = .float32) throws -> SpotMeasurement {
        let source: Micrograph
        if let crop {
            guard let c = image.cropped(crop) else { throw SpotMeasurementError.crop(crop) }
            source = c
        } else {
            source = image
        }
        let w = source.width, h = source.height
        guard (0..<h).contains(clickRow), (0..<w).contains(clickColumn) else {
            throw SpotMeasurementError.outside(row: clickRow, col: clickColumn)
        }
        let (dx, ux) = LengthUnit.realSampling(source.x), (dy, uy) = LengthUnit.realSampling(source.y)
        guard ux == uy else { throw SpotMeasurementError.mixedUnits(ux, uy) }

        // A NaN or infinity makes every bin undefined; a constant window has no power after the mean is removed.
        guard source.pixels.allSatisfy({ $0.isFinite }) else { throw SpotMeasurementError.nonFinite }
        if let lo = source.pixels.min(), let hi = source.pixels.max(), lo == hi { throw SpotMeasurementError.noPower }

        // Mean removed, Hann applied, in Double; the Float32 path's FFT takes Float.
        let mean = source.pixels.reduce(0.0) { $0 + Double($1) } / Double(w * h)
        let hx = hanning(w), hy = hanning(h)
        var power = [Double](repeating: 0, count: w * h)
        switch precision {
        case .float32:
            var re = [Float](repeating: 0, count: w * h)
            for r in 0..<h { for c in 0..<w { re[r * w + c] = Float((Double(source.pixels[r * w + c]) - mean) * hy[r] * hx[c]) } }
            var im = [Float](repeating: 0, count: w * h)
            guard let fft = FFT2D(nx: w, ny: h) else { throw SpotMeasurementError.fftSetup(width: w, height: h) }
            fft.transform(re: &re, im: &im, forward: true)
            centredPower(re, im, w: w, h: h, into: &power)
        case .float64:
            var re = [Double](repeating: 0, count: w * h)
            for r in 0..<h { for c in 0..<w { re[r * w + c] = (Double(source.pixels[r * w + c]) - mean) * hy[r] * hx[c] } }
            var im = [Double](repeating: 0, count: w * h)
            guard let fft = FFT2DDouble(nx: w, ny: h) else { throw SpotMeasurementError.fftSetup(width: w, height: h) }
            fft.transform(re: &re, im: &im, forward: true)
            centredPower(re, im, w: w, h: h, into: &power)
        }
        func p(_ r: Int, _ c: Int) -> Double { (0..<h).contains(r) && (0..<w).contains(c) ? power[r * w + c] : 0 }

        // Snap: the largest power within ±2 bins, every bin within 1e-6 (relative) of it counting as largest (the float32
        // transform's noise is ~1e-7 of the maximum), and the first of them in row-major order.
        var sr = clickRow, sc = clickColumn
        let window = ((clickRow - 2)...(clickRow + 2)).flatMap { r in ((clickColumn - 2)...(clickColumn + 2)).map { (r, $0) } }
            .filter { (0..<h).contains($0.0) && (0..<w).contains($0.1) }
        let best = window.map { p($0.0, $0.1) }.max() ?? 0
        if let first = window.first(where: { p($0.0, $0.1) >= best * (1 - 1e-6) }) { (sr, sc) = first }
        let (cr0, cc0) = (h / 2, w / 2)
        guard (Double((sr - cr0) * (sr - cr0) + (sc - cc0) * (sc - cc0))).squareRoot() > 3 else {
            throw SpotMeasurementError.nearDC
        }
        let half = 2
        guard sr >= half, sc >= half, sr < h - half, sc < w - half else { throw SpotMeasurementError.nearEdge }

        // The centroid over the 5 × 5 window, no background.
        var sum = 0.0, sumR = 0.0, sumC = 0.0
        for r in (sr - half)...(sr + half) { for c in (sc - half)...(sc + half) {
            let v = p(r, c)
            sum += v; sumR += v * Double(r); sumC += v * Double(c)
        } }
        guard sum > 0, sum.isFinite else { throw SpotMeasurementError.noPower }
        let centroidRow = sumR / sum, centroidCol = sumC / sum
        let qx = (centroidCol - Double(cc0)) * (1.0 / (Double(w) * dx))
        let qy = (centroidRow - Double(cr0)) * (1.0 / (Double(h) * dy))
        let q = (qx * qx + qy * qy).squareRoot()

        // Another local maximum above 10 % of the spot's power within 3 bins, outside the spot itself.
        var neighbour = false
        let spot = p(sr, sc)
        for r in (sr - 3)...(sr + 3) { for c in (sc - 3)...(sc + 3) {
            guard (r, c) != (sr, sc), (0..<h).contains(r), (0..<w).contains(c),
                  Double((r - sr) * (r - sr) + (c - sc) * (c - sc)).squareRoot() <= 3 else { continue }
            let v = p(r, c)
            // A local maximum by more than 1e-6 relative: a float-noise tie is not a second peak.
            let isMax = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                .allSatisfy { p(r + $0.0, c + $0.1) * (1 + 1e-6) < v }
            if isMax, v > 0.1 * spot { neighbour = true }
        } }

        let reciprocal = ux == "px" ? "1/px" : (ux.hasPrefix("1/") && ux.count > 2 ? String(ux.dropFirst(2)) : "1/" + ux)
        return SpotMeasurement(clickedRow: clickRow, clickedCol: clickColumn, snappedRow: sr, snappedCol: sc,
                               centroidRow: centroidRow, centroidCol: centroidCol, width: w, height: h,
                               qx: qx, qy: qy, q: q, d: 1 / q, units: ux, reciprocalUnits: reciprocal,
                               neighbour: neighbour, precision: precision)
    }

    /// NumPy's `hanning(n)`: 0.5 − 0.5·cos(2πk/(n − 1)); `[1]` for n = 1.
    public static func hanning(_ n: Int) -> [Double] {
        n == 1 ? [1] : (0..<n).map { 0.5 - 0.5 * cos(2 * .pi * Double($0) / Double(n - 1)) }
    }

    /// The angle between two spots' physical q vectors, in degrees, 0 … 180: atan2(|a × b|, a · b) (not acos, which loses
    /// small angles). Refused when the spots' reciprocal units differ or either |q| is zero.
    public static func angle(_ a: SpotMeasurement, _ b: SpotMeasurement) throws -> Double {
        guard a.reciprocalUnits == b.reciprocalUnits else {
            throw SpotMeasurementError.mixedUnits(a.reciprocalUnits, b.reciprocalUnits)
        }
        return try angle(qa: (a.qx, a.qy), qb: (b.qx, b.qy))
    }

    /// The angle between two q vectors (x, y) in one unit, in degrees, 0 … 180: atan2(|a × b|, a · b). Refused when either
    /// vector is zero.
    public static func angle(qa: (Double, Double), qb: (Double, Double)) throws -> Double {
        guard qa != (0, 0), qb != (0, 0) else { throw SpotMeasurementError.zeroFrequency }
        let cross = qa.0 * qb.1 - qa.1 * qb.0
        let dot = qa.0 * qb.0 + qa.1 * qb.1
        return atan2(abs(cross), dot) * 180 / .pi
    }

    /// Centred power (fftshift: output (r, c) ← input ((r − h/2) mod h, (c − w/2) mod w)), in Double.
    private static func centredPower<T: BinaryFloatingPoint>(_ re: [T], _ im: [T], w: Int, h: Int, into power: inout [Double]) {
        for r in 0..<h {
            let sr = (r + h - h / 2) % h
            for c in 0..<w {
                let i = sr * w + (c + w - w / 2) % w
                power[r * w + c] = Double(re[i]) * Double(re[i]) + Double(im[i]) * Double(im[i])
            }
        }
    }
}

/// Two spots of one lattice as an indexing step reads them: both spacings, the angle between them and the ratio d_a/d_b.
public nonisolated struct LatticePair: Sendable, Equatable {
    public let a: SpotMeasurement
    public let b: SpotMeasurement
    /// Degrees, 0 … 180 (`SpotMeasurement.angle`).
    public let angle: Double
    /// d_a / d_b.
    public let ratio: Double

    public static func measure(_ a: SpotMeasurement, _ b: SpotMeasurement) throws -> LatticePair {
        LatticePair(a: a, b: b, angle: try SpotMeasurement.angle(a, b), ratio: a.d / b.d)
    }
}
