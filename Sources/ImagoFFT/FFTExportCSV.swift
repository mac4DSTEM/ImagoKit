//
//  FFTExportCSV.swift (ImagoFFT)
//  The radial profile and the measured spots as CSV text: `# key: value` header lines that say what the numbers are, a
//  column row, and one line per bin or spot. Values in, text out.
//
//  EXACT: every number is written `%.17g` (`CSVText.number`), so a reader's `float()` is the in-memory Double bit for bit;
//  a "shown" column is that Double times the factor of `FFTUnits` (one multiplication, restated by a reader).
//
//  WHAT THE NUMBERS ARE: the transform and the radial profile are of the region's pixels as read, with no window and the
//  mean not removed (`PowerSpectrum`); only the spot estimator works on a Hann-windowed, mean-removed copy
//  (`SpotMeasurement`). Each header says which. The caller supplies what only it can claim: the file's first-line title,
//  the validation sentence and the lineage sentence.
//
//  Text is UTF-8 with LF line ends, tested byte for byte.
//

import Foundation
import ImagoCore

/// What every FFT export names: the file, the region transformed, the file's calibration and the spectrum's frequency steps.
public nonisolated struct FFTExportHeader: Sendable, Equatable {
    public var software: String
    public var fileName: String
    public var object: String
    public var imageWidth: Int
    public var imageHeight: Int
    /// The FILE's axes, never display-converted.
    public var x: AxisCalibration?
    public var y: AxisCalibration?
    public var notes: [String]
    /// The region transformed; nil is the whole image.
    public var crop: PixelCrop?
    public var spectrumWidth: Int
    public var spectrumHeight: Int
    /// `PowerSpectrum`'s frequency step per bin and its unit, per axis.
    public var spacingX: Double
    public var spacingY: Double
    public var unitsX: String
    public var unitsY: String

    public init(software: String, fileName: String, object: String, imageWidth: Int, imageHeight: Int,
                x: AxisCalibration?, y: AxisCalibration?, notes: [String], crop: PixelCrop?,
                spectrumWidth: Int, spectrumHeight: Int, spacingX: Double, spacingY: Double,
                unitsX: String, unitsY: String) {
        self.software = software; self.fileName = fileName; self.object = object
        self.imageWidth = imageWidth; self.imageHeight = imageHeight; self.x = x; self.y = y; self.notes = notes
        self.crop = crop; self.spectrumWidth = spectrumWidth; self.spectrumHeight = spectrumHeight
        self.spacingX = spacingX; self.spacingY = spacingY; self.unitsX = unitsX; self.unitsY = unitsY
    }
}

public nonisolated enum FFTExportText {
    public static let transformNote = "unnormalised forward FFT of the region's pixels as read: NO window, the mean NOT removed "
        + "(the DC bin is the region's sum and is not drawn); |F|^2 centred, the zero frequency at bin (height/2, width/2)"
    public static let spotNote = "spots: the centroid of |F|^2 (power weights) over the 5 x 5 bins around the largest power within "
        + "2 bins of the click, measured on a Hann-windowed, mean-removed copy of the region (the transform above is not); "
        + "d = 1/|q|; neighbour = another peak above 10 % of the spot's power within 3 bins (d may be off by up to 0.3 %); "
        + "In Mask = the spot passes the masked inverse"

    /// "the whole image, 2048 x 1024 px" or "region 256 x 256 px at (812, 404) (column, row of its top-left pixel)".
    public static func subject(crop: PixelCrop?, imageWidth: Int, imageHeight: Int) -> String {
        guard let crop else { return "the whole image, \(imageWidth) x \(imageHeight) px" }
        return "region \(crop.width) x \(crop.height) px at (\(crop.col), \(crop.row)) (column, row of its top-left pixel)"
    }

    /// The lines both CSVs open with, after their title.
    public static func commonLines(_ h: FFTExportHeader) -> [String] {
        let notes = h.notes.isEmpty ? "none" : h.notes.map(CSVText.headerValue).joined(separator: "; ")
        let unitX = FFTUnits.spectrumUnit(h.unitsX), unitY = FFTUnits.spectrumUnit(h.unitsY)
        return [
            "# software: \(CSVText.headerValue(h.software))",
            "# file: \(CSVText.headerValue(h.fileName))",
            "# object: \(CSVText.headerValue(h.object))",
            "# subject: \(subject(crop: h.crop, imageWidth: h.imageWidth, imageHeight: h.imageHeight))",
            "# image: \(h.imageWidth) x \(h.imageHeight) px",
            CSVText.scaleLine("x", h.x),
            CSVText.scaleLine("y", h.y),
            "# spectrum: \(h.spectrumWidth) x \(h.spectrumHeight) bins",
            "# dq x: \(CSVText.number(h.spacingX)) \(CSVText.headerValue(h.unitsX)) per bin "
                + "(\(CSVText.number(h.spacingX * unitX.factor)) \(unitX.label))",
            "# dq y: \(CSVText.number(h.spacingY)) \(CSVText.headerValue(h.unitsY)) per bin "
                + "(\(CSVText.number(h.spacingY * unitY.factor)) \(unitY.label))",
            "# notes: \(notes)",
        ]
    }
}

public nonisolated enum RadialProfileCSV {
    /// The whole file, or nil when no bin outside the DC holds a pixel (a chart draws none either). Rows are
    /// `RadialProfile.points`: the file holds the bins a chart draws, no more. `title` is the first line's words (`# <title>`).
    public static func text(_ profile: RadialProfile, header h: FFTExportHeader, title: String, validation: String,
                            lineage: String) -> String? {
        let points = profile.points
        guard !points.isEmpty else { return nil }
        let shown = FFTUnits.spectrumUnit(profile.units)
        let unit = CSVText.headerValue(profile.units)
        var lines = ["# \(CSVText.headerValue(title))"]
        lines += FFTExportText.commonLines(h)
        lines.append("# transform: \(FFTExportText.transformNote)")
        lines.append("# mean |F|^2: the mean over the pixels of each bin of |F|^2 of that transform (not log, not divided by anything)")
        lines.append("# bin width: \(CSVText.number(profile.width)) \(unit) (\(CSVText.number(profile.width * shown.factor)) \(shown.label))")
        lines.append("# bins: bin k spans [k*width, (k+1)*width) in |q|; the centre is (k + 1/2)*width; bin 0 (the DC bin) is not written")
        lines.append("# omitted: bin 0, the DC bin (the region's sum, which the room's chart does not draw), and bins with no pixel (the profile holds mean 0 there)")
        lines.append("# validation: \(CSVText.headerValue(validation))")
        lines.append("# lineage: \(CSVText.headerValue(lineage))")
        lines.append("# numbers: %.17g (exact round trip); the shown column is the unit column times \(CSVText.number(shown.factor))")
        lines.append(CSVText.quoted("bin") + "," + CSVText.quoted("|q| centre (\(shown.label))") + ","
                     + CSVText.quoted("|q| centre (\(unit))") + "," + CSVText.quoted("mean |F|^2") + ",count")
        var out = lines.joined(separator: "\n")
        out.append("\n")
        for point in points {
            let k = point.bin, centre = point.q
            out.append("\(k),\(CSVText.number(centre * shown.factor)),\(CSVText.number(centre)),"
                       + "\(CSVText.number(profile.mean[k])),\(profile.count[k])\n")
        }
        return out
    }
}

public nonisolated enum SpotsCSV {
    /// One measured spot as the list holds it.
    public struct Row: Sendable, Equatable {
        public var number: Int
        public var measured: SpotMeasurement
        public var inMask: Bool

        public init(number: Int, measured: SpotMeasurement, inMask: Bool) {
            self.number = number; self.measured = measured; self.inMask = inMask
        }
    }

    /// The whole file, or nil without a row. `omitted`: spots in the list that have no numbers (measuring, refused, no peak,
    /// or of another region). `notes`: further sentences written verbatim as `# note:` lines. The units of every column come from the
    /// first row (rows are expected to share them) and `header`. `title` is the first line's words.
    public static func text(rows: [Row], omitted: Int, maskRadius: Double, maskEdge: Double, notes: [String],
                            header h: FFTExportHeader, title: String, validation: String, lineage: String) -> String? {
        guard let first = rows.first else { return nil }
        let realUnit = CSVText.headerValue(first.measured.units)
        let reciprocalUnit = CSVText.headerValue(first.measured.reciprocalUnits)
        let shownLength = FFTUnits.length(first.measured.d, unit: first.measured.units).label
        let shownQ = FFTUnits.reciprocal(first.measured.q, realUnit: first.measured.units).label
        var lines = ["# \(CSVText.headerValue(title))"]
        lines += FFTExportText.commonLines(h)
        lines.append("# \(FFTExportText.spotNote)")
        lines.append("# transform: \(FFTExportText.transformNote)")
        lines.append("# listed: \(rows.count) measured spots; not listed: \(omitted) (measuring, refused, no peak, or of another region)")
        lines.append("# mask: radius \(CSVText.number(maskRadius)) bins, edge \(CSVText.number(maskEdge)) bins (the masked inverse)")
        lines.append("# qx, qy: from the centre of the spectrum, signed, in \(reciprocalUnit); Friedel partners are not listed")
        lines.append("# centroid, click: bins of the centred spectrum, column and row, the zero frequency at (width/2, height/2)")
        for note in notes { lines.append("# note: \(CSVText.headerValue(note))") }
        lines.append("# validation: \(CSVText.headerValue(validation))")
        lines.append("# lineage: \(CSVText.headerValue(lineage))")
        lines.append("# numbers: %.17g (exact round trip); the shown columns are the app's conversion (FFTUnits) of the unit columns")
        let columns = ["spot", "d (\(shownLength))", "|q| (\(shownQ))", "in mask", "neighbour",
                       "d (\(realUnit))", "|q| (\(reciprocalUnit))", "qx (\(reciprocalUnit))", "qy (\(reciprocalUnit))",
                       "centroid col", "centroid row", "click col", "click row"]
        lines.append(columns.map(CSVText.quoted).joined(separator: ","))
        var out = lines.joined(separator: "\n")
        out.append("\n")
        for row in rows {
            let m = row.measured
            let d = FFTUnits.length(m.d, unit: m.units).value
            let q = FFTUnits.reciprocal(m.q, realUnit: m.units).value
            let cells = ["\(row.number)", CSVText.number(d), CSVText.number(q),
                         row.inMask ? "1" : "0", m.neighbour ? "1" : "0",
                         CSVText.number(m.d), CSVText.number(m.q), CSVText.number(m.qx), CSVText.number(m.qy),
                         CSVText.number(m.centroidCol), CSVText.number(m.centroidRow),
                         "\(m.clickedCol)", "\(m.clickedRow)"]
            out.append(cells.joined(separator: ",") + "\n")
        }
        return out
    }
}
