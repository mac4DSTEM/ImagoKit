//
//  Micrograph.swift (ImagoCore)
//  The image the kit computes on: one 2D Float32 image and the calibration of each axis exactly as the caller states it. Nothing here converts, guesses or rounds a calibration: a unit is the caller's string.
//

import Foundation

/// One calibrated axis. The physical value at pixel index `i` is `(i − origin) · scale`.
public nonisolated struct AxisCalibration: Sendable, Equatable, Hashable {
    /// Physical units per pixel.
    public let scale: Double
    /// The unit as written ("nm", "µm", "1/nm"); not normalised.
    public let unit: String
    /// The pixel index of the physical zero, in pixels.
    public let origin: Double

    public init(scale: Double, unit: String, origin: Double = 0) {
        self.scale = scale
        self.unit = unit
        self.origin = origin
    }

    /// The physical value at pixel index `i`.
    public func value(at i: Double) -> Double { (i - origin) * scale }

    /// The axis written as an offset: the physical value at pixel index 0, `-origin · scale` (RosettaSciIO's `offset`).
    public var offset: Double { -origin * scale }

    /// An axis from an offset (the physical value at pixel index 0), a scale and a unit: `origin = -offset / scale`.
    /// A zero or non-finite scale has no origin; the origin is then 0 and `isCalibrated` is false.
    public init(offset: Double, scale: Double, unit: String) {
        self.init(scale: scale, unit: unit, origin: scale != 0 && scale.isFinite ? -offset / scale : 0)
    }

    /// A physical sampling: a unit, and a finite, positive scale. An axis that is not calibrated computes in pixels.
    public var isCalibrated: Bool { !unit.isEmpty && scale.isFinite && scale > 0 }
}

/// A region of an image in pixel coordinates, half-open: rows `row ..< row + height`, columns `col ..< col + width`.
public nonisolated struct PixelCrop: Sendable, Equatable, Hashable {
    public let row: Int
    public let col: Int
    public let height: Int
    public let width: Int

    public init(row: Int, col: Int, height: Int, width: Int) {
        self.row = row
        self.col = col
        self.height = height
        self.width = width
    }
}

/// The arithmetic a product is computed in. `.float32` runs Accelerate's single-precision FFT (the fast path); `.float64`
/// runs the same algorithm in Double (the reference path). The registration states each one's distance from NumPy.
public nonisolated enum Precision: String, Sendable, Equatable, Hashable, CaseIterable {
    case float32
    case float64
}

/// Why a `Micrograph` cannot be made.
public nonisolated enum MicrographError: Error, Equatable, Sendable {
    case emptySize(width: Int, height: Int)
    case pixelCount(expected: Int, actual: Int)
}

/// One 2D image: `width × height` Float32 pixels, row-major, x (column) fastest, row 0 first, and each axis's calibration.
public nonisolated struct Micrograph: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let pixels: [Float]
    /// Calibration along columns (x) and rows (y); nil when there is none.
    public let x: AxisCalibration?
    public let y: AxisCalibration?

    public init(width: Int, height: Int, pixels: [Float], x: AxisCalibration? = nil, y: AxisCalibration? = nil) throws {
        guard width > 0, height > 0 else { throw MicrographError.emptySize(width: width, height: height) }
        let (count, overflow) = width.multipliedReportingOverflow(by: height)
        guard !overflow, count == pixels.count else {
            throw MicrographError.pixelCount(expected: overflow ? Int.max : count, actual: pixels.count)
        }
        self.width = width
        self.height = height
        self.pixels = pixels
        self.x = x
        self.y = y
    }

    /// The pixels of `crop`, each axis's scale and unit kept, its origin moved by the offset so a physical position is the
    /// same in the crop as in the image. nil when the crop is empty or leaves the image.
    public func cropped(_ crop: PixelCrop) -> Micrograph? {
        guard crop.row >= 0, crop.col >= 0, crop.height > 0, crop.width > 0,
              crop.height <= height - crop.row, crop.width <= width - crop.col else { return nil }
        var out = [Float]()
        out.reserveCapacity(crop.width * crop.height)
        for r in crop.row..<(crop.row + crop.height) {
            out.append(contentsOf: pixels[(r * width + crop.col)..<(r * width + crop.col + crop.width)])
        }
        func moved(_ axis: AxisCalibration?, by offset: Int) -> AxisCalibration? {
            axis.map { AxisCalibration(scale: $0.scale, unit: $0.unit, origin: $0.origin - Double(offset)) }
        }
        return try? Micrograph(width: crop.width, height: crop.height, pixels: out,
                               x: moved(x, by: crop.col), y: moved(y, by: crop.row))
    }
}

/// Length units as image formats write them, and the names of their reciprocals.
public nonisolated enum LengthUnit {
    /// A unit folded for comparison: canonical Unicode composition, trimmed, lower-cased, spaces removed, "ångström" spelled
    /// "angstrom".
    public static func normalized(_ unit: String) -> String {
        unit.precomposedStringWithCanonicalMapping
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: " ", with: "")
            .replacingOccurrences(of: "ångström", with: "angstrom")
    }

    /// Nanometres in one `unit` for a length unit (m, mm, µm, nm, Å, pm in their spellings); nil for anything else. µ is
    /// accepted as the micro sign (U+00B5), the Greek mu (U+03BC) or "u".
    public static func nanometres(per unit: String) -> Double? {
        switch normalized(unit) {
        case "m", "meter", "meters", "metre", "metres": 1e9
        case "mm", "millimeter", "millimeters", "millimetre", "millimetres": 1e6
        case "\u{00B5}m", "\u{03BC}m", "um", "micron", "microns", "micrometer", "micrometers", "micrometre", "micrometres": 1e3
        case "nm", "nanometer", "nanometers", "nanometre", "nanometres": 1
        case "\u{00E5}", "a", "angstrom", "angstroms", "ang": 0.1
        case "pm", "picometer", "picometers", "picometre", "picometres": 1e-3
        default: nil
        }
    }

    /// The sampling of one axis for a Fourier transform: its scale and the unit of its frequencies. A calibrated axis gives
    /// its scale and "1/<unit>"; an axis already reciprocal ("1/nm", a diffraction pattern) gives the unit after the "1/"
    /// (never "1/1/nm"); an uncalibrated axis computes in pixels, scale 1 and "1/px".
    public static func frequencySampling(_ axis: AxisCalibration?) -> (scale: Double, unit: String) {
        guard let axis, axis.isCalibrated else { return (1, "1/px") }
        if axis.unit.hasPrefix("1/"), axis.unit.count > 2 { return (axis.scale, String(axis.unit.dropFirst(2))) }
        return (axis.scale, "1/" + axis.unit)
    }

    /// The sampling of one axis in real space: its scale and unit, or 1 and "px" when it is not calibrated.
    public static func realSampling(_ axis: AxisCalibration?) -> (scale: Double, unit: String) {
        guard let axis, axis.isCalibrated else { return (1, "px") }
        return (axis.scale, axis.unit)
    }
}

/// The two FFT classes under one name: `FFT2D` (Float32, vDSP) and `FFT2DDouble` (the Double reference path).
public protocol FFT2DTransform: AnyObject, Sendable {
    associatedtype Scalar: BinaryFloatingPoint & Sendable
    var nx: Int { get }
    var ny: Int { get }
    /// In-place complex 2D DFT over row-major `[ny * nx]` split-complex data; the inverse divides by `nx · ny` when
    /// `scaleInverse` is true.
    func transform(re: inout [Scalar], im: inout [Scalar], forward: Bool, scaleInverse: Bool)
}
