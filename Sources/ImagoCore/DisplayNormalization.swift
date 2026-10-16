//
//  DisplayNormalization.swift (ImagoCore)
//  The arithmetic that takes raw pixels to the [0, 1] values a display samples, in Float32: the finite range, the linear and
//  the zero-centred normalisation, and the log contrast of a diffraction pattern. Non-finite pixels never reach a valid
//  value: they map to `invalidValue`, so "no data" stays distinguishable from any real value.
//

import Foundation

public nonisolated enum DisplayNormalization {
    /// The value a non-finite (masked) pixel takes after normalisation. Valid pixels fall in [0, 1], so a renderer can show
    /// every negative value as an explicit masked colour.
    public static let invalidValue: Float = -1

    /// The minimum and maximum over the FINITE pixels only: one NaN or infinite pixel (a dead pixel, a division artefact)
    /// must not blank the display scaling. (0, 1) when no pixel is finite.
    public static func finiteMinMax(_ pixels: [Float]) -> (Float, Float) {
        var lo = Float.greatestFiniteMagnitude
        var hi = -Float.greatestFiniteMagnitude
        for v in pixels where v.isFinite {
            if v < lo { lo = v }
            if v > hi { hi = v }
        }
        guard lo <= hi else { return (0, 1) }
        return (lo, hi)
    }

    /// Linearly normalised to [0, 1] in Float32: (p − lo)/(hi − lo) over the finite range. With `symmetric`, the scale is
    /// centred on zero: 0.5 + 0.5·(p/m) with m the largest finite magnitude. A constant image maps to 0 (0.5 symmetric).
    /// Non-finite pixels map to `invalidValue`. A finite range whose span overflows Float32 maps every pixel to 0.
    public static func normalized(_ pixels: [Float], symmetric: Bool = false) -> [Float] {
        let invalid = invalidValue
        let (lo, hi) = finiteMinMax(pixels)
        if symmetric {
            let magnitude = max(abs(lo), abs(hi))
            guard magnitude > 0 else { return pixels.map { $0.isFinite ? 0.5 : invalid } }
            return pixels.map { $0.isFinite ? 0.5 + 0.5 * ($0 / magnitude) : invalid }
        }
        let span = hi - lo
        guard span > 0 else { return pixels.map { $0.isFinite ? 0.0 : invalid } }
        return pixels.map { $0.isFinite ? ($0 - lo) / span : invalid }
    }

    /// The log contrast of a pattern that spans many orders of magnitude: log₁₀(1 + max(p, 0)) computed in Double and
    /// rounded to Float32; a non-finite pixel becomes 0.
    public static func logContrast(_ pixels: [Float]) -> [Float] {
        pixels.map { $0.isFinite ? Float(log10(1.0 + Double(max($0, 0)))) : 0 }
    }

    /// `normalized` of `logContrast`.
    public static func normalizedLog(_ pixels: [Float], symmetric: Bool = false) -> [Float] {
        normalized(logContrast(pixels), symmetric: symmetric)
    }
}
