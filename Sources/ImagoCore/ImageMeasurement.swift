// SPDX-License-Identifier: GPL-3.0-only
//
//  ImageMeasurement.swift (ImagoCore)
//  Distance and angle between marks on a calibrated image, in pixels and, when the calibration allows it, in the axes' unit.
//
//  CONVENTION: a `PixelPoint` is (x = column, y = row) and the centre of pixel (row r, column c) is (c, r), at integers;
//  a mark between pixels has fractional coordinates. A physical length uses each axis's OWN scale,
//  √((Δc·s_x)² + (Δr·s_y)²), so anisotropic pixels measure correctly. A measurement is physical only when BOTH axes are
//  calibrated (`AxisCalibration.isCalibrated`) and state the SAME unit string; the unit is never converted or guessed here
//  (convert before passing the axes). Otherwise the physical value is nil and only pixels are reported.
//
//  ANGLE at the middle point B, between B→A and B→C: atan2(|cross|, dot) in degrees, 0 to 180. Not acos, which loses
//  every digit near 0° and 180°. It is taken in the physical plane (Δc·s_x, Δr·s_y) when the length is physical, else in
//  pixel space, and the result says which. No angle when an arm has zero length. The cross product is Kahan's difference
//  of products, so two long, almost parallel arms keep the small angle between them.
//

import Foundation

/// A position on an image in pixel coordinates: x = column, y = row; pixel (r, c) has its centre at (c, r).
public nonisolated struct PixelPoint: Sendable, Equatable, Hashable, Codable {
    public var x: Double
    public var y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

public nonisolated enum ImageMeasurement {
    /// A distance in pixels and, when both axes allow it, in the axes' unit.
    public nonisolated struct Distance: Sendable, Equatable {
        /// The length in pixels, √(Δc² + Δr²).
        public let pixels: Double
        /// The length in `unit`, √((Δc·s_x)² + (Δr·s_y)²); nil unless both axes are calibrated in one unit.
        public let physical: Double?
        /// The axes' shared unit exactly as written; non-nil exactly when `physical` is.
        public let unit: String?
    }

    /// The plane an angle was taken in.
    public nonisolated enum AnglePlane: Sendable, Equatable {
        /// (Δc·s_x, Δr·s_y), in the axes' shared unit.
        case physical
        /// (Δc, Δr), the pixel grid: the image is not calibrated, or its axes disagree on the unit.
        case pixel
    }

    public nonisolated struct Angle: Sendable, Equatable {
        /// Degrees, 0 to 180.
        public let degrees: Double
        public let plane: AnglePlane
    }

    /// Both axes' scales and their shared unit; nil when either axis is uncalibrated or the two units differ.
    public static func physicalScales(x: AxisCalibration?, y: AxisCalibration?) -> (sx: Double, sy: Double, unit: String)? {
        guard let x, let y, x.isCalibrated, y.isCalibrated, x.unit == y.unit else { return nil }
        return (x.scale, y.scale, x.unit)
    }

    public static func distance(_ a: PixelPoint, _ b: PixelPoint, x: AxisCalibration?, y: AxisCalibration?) -> Distance {
        let dc = b.x - a.x, dr = b.y - a.y
        let pixels = hypot(dc, dr)
        guard let s = physicalScales(x: x, y: y) else { return Distance(pixels: pixels, physical: nil, unit: nil) }
        return Distance(pixels: pixels, physical: hypot(dc * s.sx, dr * s.sy), unit: s.unit)
    }

    /// The angle at `vertex` between the arms to `a` and to `c`; nil when an arm has zero length.
    public static func angle(_ a: PixelPoint, vertex b: PixelPoint, _ c: PixelPoint,
                             x: AxisCalibration?, y: AxisCalibration?) -> Angle? {
        let scales = physicalScales(x: x, y: y)
        let sx = scales?.sx ?? 1, sy = scales?.sy ?? 1
        let ux = (a.x - b.x) * sx, uy = (a.y - b.y) * sy
        let vx = (c.x - b.x) * sx, vy = (c.y - b.y) * sy
        guard !(ux == 0 && uy == 0), !(vx == 0 && vy == 0) else { return nil }
        let p = uy * vx
        let cross = fma(ux, vy, -p) + fma(-uy, vx, p)
        let dot = ux * vx + uy * vy
        return Angle(degrees: atan2(abs(cross), dot) * 180 / .pi, plane: scales == nil ? .pixel : .physical)
    }
}
