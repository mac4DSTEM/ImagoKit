// SPDX-License-Identifier: GPL-3.0-only
//
//  SpectrumGeometry.swift (ImagoFFT)
//  The pure geometry around a spectrum on screen: which pixels a drawn region transforms, which bin a click names, the decades
//  a log axis labels, the spectrum's drawn aspect, and the units a d-spacing and a frequency are shown in. Closed forms; the
//  cores are called unchanged on what these rules give. Which gestures exist, defaults and ranges of controls are the
//  caller's.
//
//  COORDINATES: image coordinates are column x and row y with a pixel's centre at integers, so the image spans
//  −0.5 … W − 0.5 (`PixelPoint`); a pixel EDGE index e (0 … W) lies at x = e − 0.5. A spectrum is drawn the same way, bin
//  (r, c) at integer coordinates, its zero frequency at (h/2, w/2) (`PowerSpectrum`).
//

import Foundation
import ImagoCore

/// A region to transform: a square of even side, at least `minimumSide`, inside the image, as a `PixelCrop`. The whole image
/// (no region) may be any shape; a region never is.
public nonisolated enum FFTRegion {
    public static let minimumSide = 32

    /// A corner of a region, for a resize.
    public enum Corner: CaseIterable, Sendable {
        case topLeft, topRight, bottomLeft, bottomRight
    }

    /// S, the largest even number ≤ min(width, height); nil when it is under `minimumSide` (no region is possible).
    public static func maximumSide(width: Int, height: Int) -> Int? {
        let smaller = min(width, height)
        let even = smaller - (smaller & 1)          // & 1 is the parity of a negative number too
        return even >= minimumSide ? even : nil
    }

    /// The region drawn from the press `p0` to the pointer `p1`: its corner is the pixel edge nearest the press,
    /// (round(x₀ + ½), round(y₀ + ½)) half away from zero, held to 0…W and 0…H; s = clamp(2⌊max(|x₁ − x₀|, |y₁ − y₀|) / 2⌋,
    /// 32, S); the square extends from that corner toward `p1` on each axis (a zero extent counts as positive), then is
    /// shifted — never shrunk — into the image. nil when no region is possible, or when a point is not finite.
    public static func drawn(from p0: PixelPoint, to p1: PixelPoint, width: Int, height: Int) -> PixelCrop? {
        guard let largest = maximumSide(width: width, height: height),
              p0.x.isFinite, p0.y.isFinite, p1.x.isFinite, p1.y.isFinite else { return nil }
        let dx = p1.x - p0.x, dy = p1.y - p0.y
        let extent = max(abs(dx), abs(dy))                                  // may be +∞ from two huge finite points
        let side = Int(min(max(2 * (extent / 2).rounded(.down), Double(minimumSide)), Double(largest)))
        let edgeX = edge(p0.x, upTo: width), edgeY = edge(p0.y, upTo: height)
        let col = dx >= 0 ? edgeX : edgeX - side
        let row = dy >= 0 ? edgeY : edgeY - side
        return shifted(row: row, col: col, height: side, width: side, imageWidth: width, imageHeight: height)
    }

    /// `crop` moved by a drag of (dx, dy) image pixels, each rounded half away from zero, then shifted into the image.
    public static func moved(_ crop: PixelCrop, dx: Double, dy: Double, width: Int, height: Int) -> PixelCrop {
        shifted(row: crop.row + whole(dy), col: crop.col + whole(dx), height: crop.height, width: crop.width,
                imageWidth: width, imageHeight: height)
    }

    /// Where `corner` of `crop` is drawn, in image coordinates (a pixel edge: the top-left corner is (col − ½, row − ½)).
    public static func point(of corner: Corner, in crop: PixelCrop) -> PixelPoint {
        let left = Double(crop.col) - 0.5, top = Double(crop.row) - 0.5
        let right = left + Double(crop.width), bottom = top + Double(crop.height)
        switch corner {
        case .topLeft: return PixelPoint(x: left, y: top)
        case .topRight: return PixelPoint(x: right, y: top)
        case .bottomLeft: return PixelPoint(x: left, y: bottom)
        case .bottomRight: return PixelPoint(x: right, y: bottom)
        }
    }

    /// `crop` resized by dragging `corner` to `p`: drawn from the opposite corner (as if pressed there) to `p`.
    public static func resized(_ crop: PixelCrop, corner: Corner, to p: PixelPoint, width: Int, height: Int) -> PixelCrop? {
        let opposite: Corner
        switch corner {
        case .topLeft: opposite = .bottomRight
        case .topRight: opposite = .bottomLeft
        case .bottomLeft: opposite = .topRight
        case .bottomRight: opposite = .topLeft
        }
        return drawn(from: point(of: opposite, in: crop), to: p, width: width, height: height)
    }

    /// The region typed as Left, Top and Size (px): the size taken to the even number at or below it and clamped to
    /// 32…S, then left and top clamped to 0…W − s and 0…H − s. nil when no region is possible.
    public static func typed(left: Int, top: Int, size: Int, width: Int, height: Int) -> PixelCrop? {
        guard let largest = maximumSide(width: width, height: height) else { return nil }
        let side = min(max(size - (size & 1), minimumSide), largest)
        return shifted(row: top, col: left, height: side, width: side, imageWidth: width, imageHeight: height)
    }

    /// The centred square: s = S at row ⌊(H − s)/2⌋, column ⌊(W − s)/2⌋. nil when no region is possible.
    public static func centredSquare(width: Int, height: Int) -> PixelCrop? {
        guard let side = maximumSide(width: width, height: height) else { return nil }
        return PixelCrop(row: (height - side) / 2, col: (width - side) / 2, height: side, width: side)
    }

    /// Whether `crop` is a region on an image of this size: square, even, at least 32, inside.
    public static func isValid(_ crop: PixelCrop, width: Int, height: Int) -> Bool {
        crop.width == crop.height && crop.width >= minimumSide && crop.width % 2 == 0
            && crop.row >= 0 && crop.col >= 0
            && crop.height <= height - crop.row && crop.width <= width - crop.col    // no overflow for a huge row or col
    }

    /// Whether the HRTEM filters can take these pixels as they are: square and even, at least 32 — a region always, the
    /// whole image only when W = H is even.
    public static func filterable(width: Int, height: Int) -> Bool {
        width == height && width >= minimumSide && width % 2 == 0
    }

    /// The pixel edge index nearest image coordinate `x` (edge e lies at x = e − ½), ties away from zero, held to 0…`count`.
    private static func edge(_ x: Double, upTo count: Int) -> Int {
        Int(min(max((x + 0.5).rounded(), 0), Double(count)))
    }

    /// `d` rounded to whole pixels, ties away from zero; a drag beyond a billion pixels is a billion (it is shifted in anyway).
    private static func whole(_ d: Double) -> Int {
        guard d.isFinite else { return d.isNaN ? 0 : (d > 0 ? 1_000_000_000 : -1_000_000_000) }
        return Int(min(max(d.rounded(), -1e9), 1e9))
    }

    /// A `width` × `height` rectangle at (row, col) moved into the image, its size kept.
    private static func shifted(row: Int, col: Int, height: Int, width: Int, imageWidth: Int, imageHeight: Int) -> PixelCrop {
        PixelCrop(row: min(max(row, 0), max(imageHeight - height, 0)), col: min(max(col, 0), max(imageWidth - width, 0)),
                  height: height, width: width)
    }
}

/// A click on the spectrum.
public nonisolated enum SpectrumBin {
    /// The bin under image point `q` of a `width` × `height` spectrum: (round(y), round(x)) half away from zero, clamped
    /// to the spectrum. A point that is not a number is bin 0.
    public static func at(_ q: PixelPoint, width: Int, height: Int) -> (row: Int, col: Int) {
        (row: nearest(q.y, count: height), col: nearest(q.x, count: width))
    }

    /// Whole bins (kx, ky) from the centre, the bin (h/2 + ky, w/2 + kx); nil when outside the spectrum.
    public static func at(kx: Int, ky: Int, width: Int, height: Int) -> (row: Int, col: Int)? {
        // Checked on kx and ky themselves, so a huge value cannot overflow the sum.
        guard kx >= -(width / 2), kx < width - width / 2, ky >= -(height / 2), ky < height - height / 2 else { return nil }
        return (row: height / 2 + ky, col: width / 2 + kx)
    }

    private static func nearest(_ v: Double, count: Int) -> Int {
        guard !v.isNaN else { return 0 }
        return Int(min(max(v.rounded(), 0), Double(count - 1)))
    }
}

public nonisolated enum RadialChartScale {
    /// The decades a log y axis labels: powers of ten from ⌊log₁₀ lowest⌋, every `step`-th, so that two neighbouring labels
    /// are at least `minimumSpacing` points apart on a plot `plotHeight` points tall (a decade is plotHeight / span points).
    /// `lowest` and `highest` are the smallest and largest value drawn; a span under one decade is one decade. Empty when
    /// the data or the sizes are not usable (nothing positive and finite).
    public static func decadeMarks(lowest: Double, highest: Double, plotHeight: Double,
                                   minimumSpacing: Double) -> [Double] {
        guard lowest > 0, highest >= lowest, lowest.isFinite, highest.isFinite,
              plotHeight > 0, plotHeight.isFinite, minimumSpacing > 0, minimumSpacing.isFinite else { return [] }
        let first = Int(floor(log10(lowest)))
        let span = max(Int(ceil(log10(highest))) - first, 1)
        let step = max(Int(ceil(minimumSpacing * Double(span) / plotHeight)), 1)
        return stride(from: first, through: first + span, by: step).map { pow(10, Double($0)) }
    }
}

public nonisolated enum SpectrumShape {
    /// The spectrum's box aspect, width over height: (w·Δq_x) / (h·Δq_y), so a ring is drawn as a circle. The two spacings
    /// are compared as numbers only when they are in the same unit: equal units as they are, units that
    /// `FFTUnits.spectrumUnit` brings to one label (nm⁻¹, or nm for a diffraction image: "1/nm" beside "1/Å") after that
    /// conversion; any other pair — a ratio of numbers in different units is not a shape — is drawn square, as is a spectrum
    /// with no usable spacing (not a positive, finite number).
    public static func aspect(width: Int, height: Int, spacingX: Double, unitsX: String,
                              spacingY: Double, unitsY: String) -> Double {
        let sx: Double, sy: Double
        if LengthUnit.normalized(unitsX) == LengthUnit.normalized(unitsY) {
            (sx, sy) = (spacingX, spacingY)
        } else {
            let x = FFTUnits.spectrumUnit(unitsX), y = FFTUnits.spectrumUnit(unitsY)
            guard x.label == y.label else { return 1 }
            (sx, sy) = (spacingX * x.factor, spacingY * y.factor)
        }
        let ratio = (Double(width) * sx) / (Double(height) * sy)
        return ratio.isFinite && ratio > 0 ? ratio : 1
    }
}

/// The units a d-spacing and a frequency are shown in.
public nonisolated enum FFTUnits {
    /// "nm⁻¹": the label of a frequency shown in the reciprocal of the nanometre.
    public static let reciprocalNanometreLabel = "nm\u{207B}\u{00B9}"
    /// "px⁻¹": the label of a frequency of an uncalibrated image.
    public static let reciprocalPixelLabel = "px\u{207B}\u{00B9}"

    /// Whether `unit` says the image has no calibration: "px", or nothing.
    private static func isUncalibrated(_ unit: String) -> Bool {
        let normalized = LengthUnit.normalized(unit)
        return normalized.isEmpty || normalized == "px"
    }

    /// A length `d` in the image's `unit` as shown: nm when the unit is a length (`LengthUnit.nanometres(per:)`), else the
    /// unit as written; "px" for an uncalibrated image (`unit` "px" or empty).
    public static func length(_ d: Double, unit: String) -> (value: Double, label: String) {
        if let nm = LengthUnit.nanometres(per: unit) { return (d * nm, "nm") }
        if isUncalibrated(unit) { return (d, "px") }
        return (d, unit)
    }

    /// A spatial frequency `q`, in the reciprocal of the image's `unit`, as shown: nm⁻¹ when the unit is a length, else
    /// "1/<unit>"; "px⁻¹" uncalibrated. An image whose axis is already reciprocal ("1/nm", "1/Å": a diffraction pattern) has
    /// frequencies in the unit after the "1/" — as `PowerSpectrum` names them, never "1/1/nm" — and they are shown as
    /// `spectrumUnit` shows them (nm for "1/nm", a length in another unit converted to nm).
    public static func reciprocal(_ q: Double, realUnit unit: String) -> (value: Double, label: String) {
        if let nm = LengthUnit.nanometres(per: unit) { return (q / nm, reciprocalNanometreLabel) }
        if isUncalibrated(unit) { return (q, reciprocalPixelLabel) }
        if let own = reciprocalAxisRemainder(unit) {
            let shown = spectrumUnit(own)
            return (q * shown.factor, shown.label)
        }
        return (q, "1/" + unit)
    }

    /// The unit after the "1/" of an axis unit that is already a reciprocal ("1/nm" → "nm", as written), the way
    /// `PowerSpectrum` reads it; nil for any other unit.
    private static func reciprocalAxisRemainder(_ unit: String) -> String? {
        let trimmed = unit.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.hasPrefix("1/") && trimmed.count > 2 ? String(trimmed.dropFirst(2)) : nil
    }

    /// A spectrum's own unit (`PowerSpectrum.unitsX`: "1/nm", "1/m", "1/px", or "nm" for a diffraction image) as shown: the
    /// factor that takes a value in it to the label's unit, and the label. "1/<length>" goes to nm⁻¹ and a length to nm
    /// (the same factors as `reciprocal` and `length`); "1/px" is px⁻¹; any other unit stays as written, and no unit at all
    /// is read as uncalibrated (px⁻¹).
    public static func spectrumUnit(_ units: String) -> (factor: Double, label: String) {
        let normalized = LengthUnit.normalized(units)
        if normalized.isEmpty || normalized == "1/px" { return (1, reciprocalPixelLabel) }
        if normalized.hasPrefix("1/"), let nm = LengthUnit.nanometres(per: String(normalized.dropFirst(2))) {
            return (1 / nm, reciprocalNanometreLabel)
        }
        if let nm = LengthUnit.nanometres(per: units) { return (nm, "nm") }
        return (1, units)
    }
}
