// SPDX-License-Identifier: GPL-3.0-only
//
//  DisplayMapping.swift (ImagoCore)
//  One normalised value to an 8-bit colour: the window, the gamma and a 256-entry colour table, in Double.
//

import Foundation

/// The display mapping of a normalised value, the arithmetic of a fragment shader written out in Double:
///
///     n < 0 (or NaN)  -> the masked grey (82, 82, 87)
///     v = clamp(Double(n), 0, 1); v = clamp((v − lo) / max(hi − lo, 1e-6), 0, 1); v = pow(v, 1 / max(gamma, 0.05))
///     p = clamp(v · 256 − 0.5, 0, 255); i0 = ⌊p⌋; i1 = min(i0 + 1, 255); t = p − i0
///     channel = UInt8(clamp((a + (b − a) · t).rounded(), 0, 255)), a and b the table bytes at i0 and i1
///
/// A GPU does this in Float32, so it differs from this by a measured amount; this is the exact reference for the table
/// step. The table is 1024 bytes, 256 RGBA texels.
public nonisolated struct DisplayMapping: Sendable {
    /// What a negative normalised value (a masked pixel) shows: (0.32, 0.32, 0.34) · 255, rounded.
    public static let maskedGrey: (UInt8, UInt8, UInt8) = (82, 82, 87)

    private let low: Double
    private let span: Double
    private let exponent: Double
    private let lut: [UInt8]

    /// nil unless `lut` holds 1024 bytes.
    public init?(low: Float, high: Float, gamma: Double, lut: [UInt8]) {
        guard lut.count == 1024 else { return nil }
        self.low = Double(low)
        let width = Double(high) - Double(low)
        span = width > 1e-6 ? width : 1e-6                      // max(hi − lo, 1e-6); a NaN width gives 1e-6
        exponent = 1 / (gamma > 0.05 ? gamma : 0.05)             // 1 / max(gamma, 0.05); a NaN gamma gives 0.05
        self.lut = lut
    }

    @inline(__always) private static func clamp(_ x: Double, _ lower: Double, _ upper: Double) -> Double {
        x > lower ? (x < upper ? x : upper) : lower                // a NaN gives `lower`: nothing below can trap
    }

    @inline(__always) public func rgb(_ n: Float) -> (UInt8, UInt8, UInt8) {
        if !(n >= 0) { return Self.maskedGrey }                    // the masked sentinel (n < 0), or NaN
        var v = Self.clamp(Double(n), 0, 1)
        v = Self.clamp((v - low) / span, 0, 1)
        v = pow(v, exponent)
        let p = Self.clamp(v * 256 - 0.5, 0, 255)
        let i0 = Int(p.rounded(.down))
        let i1 = min(i0 + 1, 255)
        let t = p - Double(i0)
        @inline(__always) func channel(_ c: Int) -> UInt8 {
            let a = Double(lut[i0 * 4 + c]), b = Double(lut[i1 * 4 + c])
            return UInt8(Self.clamp((a + (b - a) * t).rounded(), 0, 255))
        }
        return (channel(0), channel(1), channel(2))
    }

    /// `rgb` for every value, as packed RGB bytes (3 per value).
    public func rgbBytes(_ values: [Float]) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: values.count * 3)
        for (i, n) in values.enumerated() {
            let c = rgb(n)
            out[3 * i] = c.0; out[3 * i + 1] = c.1; out[3 * i + 2] = c.2
        }
        return out
    }
}
