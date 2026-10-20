//
//  RadialProfile.swift (ImagoFFT)
//  The power of a centred spectrum against the PHYSICAL spatial frequency |q| (registration-slice-1.md 1.2).
//
//  DEFINITION: for the signed indices (i, j) of a bin from the spectrum's centre, |q| = √((i·Δq_x)² + (j·Δq_y)²) with each
//  axis's own spacing, so a ring on a non-square or anisotropic grid lands in one bin along every direction. Bin
//  k = ⌊|q| / w⌋, w = max(Δq_x, Δq_y) unless given. Per bin: the mean power and the pixel count, accumulated in Double. The DC
//  term is in bin 0 and counted there — stated, not removed. An empty bin has mean 0 and count 0. No profile when the axes
//  carry different units (|q| would mix them) or the width is not a positive finite number.
//
//  ACCURACY on the Float32 path: every bin within 1e-5 of the largest non-DC bin; bins far below that carry the float32
//  FFT's noise (a ring fainter than about 1e-6 of the DC bin is not reliable there). The Float64 path holds 1e-11.
//

import Foundation
import ImagoCore

/// Why a profile cannot be computed.
public nonisolated enum RadialProfileError: Error, Equatable, Sendable {
    case mixedUnits(String, String)
    case unusableWidth(Double)
}

public nonisolated struct RadialProfile: Sendable, Equatable {
    /// The bin width, in `units`.
    public let width: Double
    /// The spectrum's unit, shared by both axes ("1/nm", or "1/px" uncalibrated).
    public let units: String
    /// Mean power per bin; bin k spans [k·width, (k+1)·width).
    public let mean: [Double]
    public let count: [Int]

    /// A drawn bin: k ≥ 1 with at least one pixel, at its centre (k + ½)·width.
    public struct Point: Sendable, Equatable {
        public let bin: Int
        public let q: Double
        public let mean: Double
        public let count: Int
    }

    /// A profile from its bins (a stored or computed one); nil unless `mean` and `count` have the same length, no count is negative, and the width is a
    /// positive finite number.
    public init?(width: Double, units: String, mean: [Double], count: [Int]) {
        guard mean.count == count.count, !count.contains(where: { $0 < 0 }), width.isFinite, width > 0 else { return nil }
        self.init(unchecked: width, units: units, mean: mean, count: count)
    }

    private init(unchecked width: Double, units: String, mean: [Double], count: [Int]) {
        self.width = width; self.units = units; self.mean = mean; self.count = count
    }

    public static func compute(_ spectrum: PowerSpectrum, width: Double? = nil) throws -> RadialProfile {
        guard spectrum.unitsX == spectrum.unitsY else { throw RadialProfileError.mixedUnits(spectrum.unitsX, spectrum.unitsY) }
        let dqx = spectrum.spacingX, dqy = spectrum.spacingY
        let wb = width ?? max(dqx, dqy)
        // A width so small that the profile would have more bins than the spectrum has pixels is refused too.
        let qMax = (pow(Double(spectrum.width / 2 + 1) * dqx, 2) + pow(Double(spectrum.height / 2 + 1) * dqy, 2)).squareRoot()
        guard wb.isFinite, wb > 0, qMax / wb < Double(spectrum.width * spectrum.height) else {
            throw RadialProfileError.unusableWidth(wb)
        }
        let w = spectrum.width, h = spectrum.height
        var sums: [Double] = []
        var counts: [Int] = []
        for r in 0..<h {
            let qy = Double(r - h / 2) * dqy
            for c in 0..<w {
                let qx = Double(c - w / 2) * dqx
                let k = Int((qx * qx + qy * qy).squareRoot() / wb)      // ⌊·⌋: |q| ≥ 0
                if k >= sums.count {
                    sums.append(contentsOf: repeatElement(0, count: k - sums.count + 1))
                    counts.append(contentsOf: repeatElement(0, count: k - counts.count + 1))
                }
                sums[k] += spectrum.power[r * w + c]
                counts[k] += 1
            }
        }
        let mean = zip(sums, counts).map { $1 > 0 ? $0 / Double($1) : 0 }
        return RadialProfile(unchecked: wb, units: spectrum.unitsX, mean: mean, count: counts)
    }

    /// The bins drawn: k ≥ 1 (bin 0 holds the DC) with at least one pixel, each at its centre.
    public var points: [Point] {
        guard mean.count > 1 else { return [] }
        return (1..<mean.count).compactMap { k in
            count[k] > 0 ? Point(bin: k, q: (Double(k) + 0.5) * width, mean: mean[k], count: count[k]) : nil
        }
    }
}
