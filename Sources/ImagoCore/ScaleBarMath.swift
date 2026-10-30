// SPDX-License-Identifier: GPL-3.0-only
//
//  ScaleBarMath.swift (ImagoCore)
//  The numbers of a scale bar drawn over an image of `width` × `height` pixels: which unit a calibrated axis reads in, the
//  1-2-5 length, the unit ladder, the number's text, and the bar's length in pixels. The plate (text metrics, fit) and the
//  drawing are the caller's; so is the spelling a unit is shown with (`unitLabel`).
//
//  The bar measures along the SCREEN HORIZONTAL, so it takes x's sampling (y's when x has none): on anisotropic pixels it
//  measures x. The overlays it sits with are drawn as on a pane where one screen point is `u` export pixels,
//  u = max(1, round(min(W, H) / 512)).
//

import Foundation

/// What a scale bar says and how long it is drawn.
public nonisolated struct ScaleBarPlan: Sendable, Equatable {
    /// Shown units per image pixel along x.
    public var perPixel: Double
    /// `perPixel`'s unit: "nm", "µm", "px", "1/nm", "Å", …
    public var unit: String
    /// The label's number after `ScaleBarMath.ladder`.
    public var value: Double
    /// The label's unit after `ScaleBarMath.ladder`.
    public var labelUnit: String
    /// "<format(value)> <labelUnit>".
    public var label: String
    /// nice / perPixel, in image pixels.
    public var lengthPixels: Double
    /// `Int(lengthPixels.rounded())` (half away from zero): the run of pixels to fill.
    public var drawnPixels: Int
    /// u: image pixels per screen point.
    public var overlayScale: Int

    public init(perPixel: Double, unit: String, value: Double, labelUnit: String, label: String,
                lengthPixels: Double, drawnPixels: Int, overlayScale: Int) {
        self.perPixel = perPixel
        self.unit = unit
        self.value = value
        self.labelUnit = labelUnit
        self.label = label
        self.lengthPixels = lengthPixels
        self.drawnPixels = drawnPixels
        self.overlayScale = overlayScale
    }
}

public nonisolated enum ScaleBarMath {
    /// u: image pixels per screen point, max(1, Int((min(W, H) / 512).rounded())): a 512² image as at 1 pt per pixel, 4096² at u = 8.
    public static func overlayScale(width: Int, height: Int) -> Int {
        max(1, Int((Double(min(width, height)) / 512).rounded()))
    }

    /// The sampling along the screen horizontal: x's (y's when x has none), or pixels when neither is calibrated; a metre
    /// axis is brought to nm, µm or mm first (`shown`). `unitLabel` spells the unit for display; the default shows it as
    /// written.
    public static func sampling(x: AxisCalibration?, y: AxisCalibration?, width: Int, height: Int,
                                unitLabel: (String) -> String = { $0 }) -> (perPixel: Double, unit: String) {
        let shownX = x.map { shown($0, reference: $0.scale * Double(width)) }
        let shownY = y.map { shown($0, reference: $0.scale * Double(height)) }
        let along = shownX?.scale ?? shownY?.scale
        guard let unit = shownX?.unit, !unit.isEmpty, let along, along.isFinite, along > 0 else { return (1, "px") }
        return (along, unitLabel(unit))
    }

    /// A metre axis rescaled into the unit a length of `reference` metres reads in (nm below 1 µm, µm below 1 mm, mm above);
    /// any other axis unchanged (the origin is in pixels and stays).
    public static func shown(_ axis: AxisCalibration, reference: Double) -> AxisCalibration {
        let isMetres = ["m", "meter", "meters", "metre", "metres"].contains(LengthUnit.normalized(axis.unit))
        guard isMetres, axis.isCalibrated, reference.isFinite, reference > 0 else { return axis }
        let length = abs(reference)
        let (perMetre, label): (Double, String) = length < 1e-6 ? (1e9, "nm") : (length < 1e-3 ? (1e6, "\u{00B5}m") : (1e3, "mm"))
        return AxisCalibration(scale: axis.scale * perMetre, unit: label, origin: axis.origin)
    }

    /// Round to the nearest 1, 2 or 5 × 10ⁿ (mantissa boundaries 1.5, 3.5, 7.5). 1 for a value that is not positive and finite.
    public static func nice125(_ v: Double) -> Double {
        guard v > 0, v.isFinite else { return 1 }
        let exponent = floor(log10(v))
        let base = pow(10, exponent)
        let mantissa = v / base
        let nice: Double = mantissa < 1.5 ? 1 : (mantissa < 3.5 ? 2 : (mantissa < 7.5 ? 5 : 10))
        return nice * base
    }

    /// The length unit the bar is READ in: nm / µm / mm only, chosen so the number lies in [1, 1000), by exact ×1000 steps
    /// re-snapped to the 1-2-5 value. Å, pm, reciprocal units, mrad and px are left alone, so is anything unrecognised.
    public static func ladder(_ v: Double, unit: String) -> (value: Double, unit: String) {
        let rungs = ["nm", "\u{00B5}m", "mm"]
        let canonical: String
        switch unit {
        case "nm", "nanometre", "nanometres", "nanometer", "nanometers": canonical = "nm"
        case "\u{00B5}m", "\u{03BC}m", "um": canonical = "\u{00B5}m"
        case "mm": canonical = "mm"
        default: return (v, unit)
        }
        guard v > 0, v.isFinite, var index = rungs.firstIndex(of: canonical) else { return (v, unit) }
        var scaled = v
        while scaled >= 1000, index < rungs.count - 1 { scaled /= 1000; index += 1 }
        while scaled < 1, index > 0 { scaled *= 1000; index -= 1 }
        return (index == rungs.firstIndex(of: canonical) ? v : nice125(scaled), rungs[index])
    }

    /// The bar's number as C-format digits with a point (locale-free): "%.0f" from 10, "%g" from 1, "%.3g" below.
    public static func format(_ v: Double) -> String {
        if v >= 10 { return String(format: "%.0f", v) }
        if v >= 1 { return String(format: "%g", v) }
        return String(format: "%.3g", v)
    }

    /// The bar: aims at T = min(70·u, W/4) pixels, `nice` = nice125(perPixel · T), drawn nice / perPixel long (rounded to
    /// whole pixels for the fill; the plan keeps the unrounded length). Its label is `ladder` then `format`. nil when the
    /// bar would be under half a pixel or not shorter than the image. Whether a plate with this label fits is the caller's.
    public static func plan(x: AxisCalibration?, y: AxisCalibration?, width: Int, height: Int,
                            unitLabel: (String) -> String = { $0 }) -> ScaleBarPlan? {
        let s = sampling(x: x, y: y, width: width, height: height, unitLabel: unitLabel)
        let u = overlayScale(width: width, height: height)
        let target = min(70.0 * Double(u), Double(width) / 4)
        let nice = nice125(s.perPixel * target)
        let length = nice / s.perPixel
        // Wider than the image cannot fit; and it keeps `Int(…)` below away from inf and from a trap.
        guard length.isFinite, length >= 0.5, length < Double(width) else { return nil }
        let reading = ladder(nice, unit: s.unit)
        return ScaleBarPlan(perPixel: s.perPixel, unit: s.unit, value: reading.value, labelUnit: reading.unit,
                            label: "\(format(reading.value)) \(reading.unit)", lengthPixels: length,
                            drawnPixels: Int(length.rounded()), overlayScale: u)
    }
}
