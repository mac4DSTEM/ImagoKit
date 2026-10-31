// SPDX-License-Identifier: GPL-3.0-only
//
//  ProfileCSV.swift (ImagoCore)
//  A `LineProfile` as CSV text: a `# key: value` header that says what the numbers are and where they came from, the
//  column row `position (<unit>),value`, then one row per sample. Values in, text out; the caller writes the file.
//
//  NUMBERS are `CSVText.number` (`%.17g`, POSIX locale); a NaN sample (a NaN pixel among the neighbours) is `nan`, never 0
//  or an empty cell. POSITIONS are exactly what `LineProfile` holds: the distance from the start along the line, in the
//  axes' unit as the FILE states it (a Velox metre axis stays in metres), `px` when the image is not calibrated; `shown`
//  only adds a line saying what the app shows them in. HEADER values pass `CSVText.headerValue`, the column header
//  `CSVText.quoted`. The title, the validation sentence and the lineage sentence are the caller's (the kit authors no claim);
//  when a note of the source says "unvalidated" (any case) `TidyResultsCSV.unvalidatedClause` follows the validation.
//
//  UTF-8, LF line ends, tested byte for byte. Same lines, in the same order, as macTEM's former app copy.
//

import Foundation

public nonisolated enum ProfileCSV {
    /// The `# method:` line for the default profile: order 1, scipy's `reflect`, the mean of `width` samples.
    public static func method(width: Int) -> String {
        "bilinear interpolation (order 1), scipy's \"reflect\" boundary (half-sample symmetric), "
            + "mean of \(width) samples 1 px apart along the perpendicular on the pixel grid"
    }

    /// The default `# lineage:` sentence (marks carry no lineage id).
    public static let noLineage = "none — marks carry no lineage id; the mark's parameters above stand in"

    /// The whole text; nil when the profile has no samples.
    ///
    /// - Parameters:
    ///   - title: the first line without `# ` ("macTEM line profile").
    ///   - appName: names the app in the `# shown in <appName> as:` line ("macTEM").
    ///   - start, end: the line's ends in image pixels (x column, y row, pixel centres at integers).
    ///   - source: the file and its calibration as the file states them.
    ///   - validation: the validation sentence (the clause for an unvalidated source is added).
    ///   - method: the `# method:` sentence; `method(width:)` for order 1, `reflect`, mean.
    ///   - lineage: the `# lineage:` sentence.
    ///   - shown: the unit and factor the app shows positions in when it differs from the file's; nil when the same.
    public static func text(_ profile: LineProfile, title: String, appName: String, start: PixelPoint, end: PixelPoint,
                            source: TidyResultsCSV.Source, validation: String, method: String? = nil,
                            lineage: String = noLineage, shown: (unit: String, factor: Double)? = nil) -> String? {
        guard let length = profile.positions.last, !profile.samples.isEmpty else { return nil }
        let unit = CSVText.headerValue(profile.unit)
        let unvalidated = source.notes.contains { $0.lowercased().contains("unvalidated") }
        let notes = source.notes.isEmpty ? "none" : source.notes.map(CSVText.headerValue).joined(separator: "; ")

        var header: [String] = []
        header.append("# \(CSVText.headerValue(title))")
        header.append("# software: \(CSVText.headerValue(source.software))")
        header.append("# file: \(CSVText.headerValue(source.fileName))")
        header.append("# object: \(CSVText.headerValue(source.object))")
        header.append("# image: \(source.width) x \(source.height) px")
        header.append("# start: x=\(start.x), y=\(start.y) (image pixels: x column, y row, pixel centres at integers)")
        header.append("# end: x=\(end.x), y=\(end.y)")
        header.append("# width: \(profile.width) px")
        header.append("# samples: \(profile.samples.count)")
        header.append("# length: \(CSVText.number(length)) \(unit)")
        header.append(CSVText.scaleLine("x", source.x))
        header.append(CSVText.scaleLine("y", source.y))
        header.append("# position: distance from the start along the line, in \(unit) as the file states it "
                      + "(not an axis coordinate; origins unused)")
        if let shown {
            header.append("# shown in \(CSVText.headerValue(appName)) as: \(CSVText.headerValue(shown.unit)) (x \(shown.factor))")
        }
        header.append("# method: \(CSVText.headerValue(method ?? Self.method(width: profile.width)))")
        header.append("# notes: \(notes)")
        header.append("# validation: \(CSVText.headerValue(validation))"
                      + (unvalidated ? TidyResultsCSV.unvalidatedClause : ""))
        header.append("# lineage: \(CSVText.headerValue(lineage))")
        header.append("# numbers: %.17g (exact round trip); nan where a NaN pixel enters a sample")
        header.append(CSVText.quoted("position (\(unit))") + ",value")

        var out = header.joined(separator: "\n")
        out.append("\n")
        out.reserveCapacity(out.utf8.count + profile.samples.count * 40)
        for (position, value) in zip(profile.positions, profile.samples) {
            out.append(CSVText.number(position))
            out.append(",")
            out.append(CSVText.number(value))
            out.append("\n")
        }
        return out
    }
}
