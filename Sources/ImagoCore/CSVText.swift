//
//  CSVText.swift (ImagoCore)
//  The text primitives of the kit's CSV writers: a number, a header value, a cell and a calibration line.
//
//  NUMBERS are `%.17g` in the POSIX locale (`String(format:)` with no locale): 17 significant digits round-trip every
//  Double exactly, trailing zeros are dropped, and a comma-decimal locale still writes a point. NaN, +∞ and −∞ are spelled
//  `nan`, `inf` and `-inf` (never 0, never an empty cell), whatever sign bit a NaN has.
//
//  HEADER VALUES (a file name, a unit, a note) become one line: C0 and C1 controls, DEL, U+2028 and U+2029 turn into spaces,
//  on unicode scalars (`"\r\n"` is one Character, which a Character-wise replace misses), so a value can never end its own
//  line or start another.
//
//  CELLS are RFC 4180: a cell holding a comma, a quote, CR or LF is wrapped in quotes with its quotes doubled.
//
//  Cells are not guarded against spreadsheet formulas (a leading =, +, - or @); the files are data for scripts.
//
//  Pure text: exact, tested byte for byte.
//

import Foundation

public nonisolated enum CSVText {
    /// `%.17g` in the POSIX locale; `nan`, `inf`, `-inf` spelled out.
    public static func number(_ v: Double) -> String {
        if v.isNaN { return "nan" }
        if v.isInfinite { return v < 0 ? "-inf" : "inf" }
        return String(format: "%.17g", v)
    }

    /// One header line's value: C0 controls, DEL, C1 controls (including U+0085), U+2028 and U+2029 become a space.
    public static func headerValue(_ s: String) -> String {
        var scalars = String.UnicodeScalarView()
        for scalar in s.unicodeScalars {
            let v = scalar.value
            let isControl = v < 0x20 || (v >= 0x7F && v <= 0x9F) || v == 0x2028 || v == 0x2029
            scalars.append(isControl ? " " : scalar)
        }
        return String(scalars)
    }

    /// RFC 4180 quoting of one cell.
    public static func quoted(_ cell: String) -> String {
        let needsQuotes = cell.unicodeScalars.contains { $0 == "," || $0 == "\"" || $0 == "\r" || $0 == "\n" }
        guard needsQuotes else { return cell }
        return "\"" + cell.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }

    /// `# x scale: <scale> <unit> per px, origin <origin> px` (the scale and origin as Swift prints a Double), or
    /// `# x scale: none` when the axis is absent or not calibrated (`AxisCalibration.isCalibrated`).
    public static func scaleLine(_ name: String, _ axis: AxisCalibration?) -> String {
        guard let axis, axis.isCalibrated else { return "# \(name) scale: none" }
        return "# \(name) scale: \(axis.scale) \(headerValue(axis.unit)) per px, origin \(axis.origin) px"
    }
}
