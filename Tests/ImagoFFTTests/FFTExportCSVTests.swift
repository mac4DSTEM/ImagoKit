// RadialProfileCSV and SpotsCSV against tools/csvexport/pins.py: the application's own writers, run offline on seeded
// cases (NaN, infinities, subnormals, controls, quotes, commas, newlines, uncalibrated axes, crops, empty tables).
// Text is tested exact: the kit must reproduce each file byte for byte.
import Foundation
import ImagoCore
import ImagoFFT
import Testing

private func dbl(_ b: UInt64) -> Double { Double(bitPattern: b) }

private struct AxisJ: Decodable, Sendable {
    let originBits: UInt64, scaleBits: UInt64
    let unit: String
    var axis: AxisCalibration { AxisCalibration(scale: dbl(scaleBits), unit: unit, origin: dbl(originBits)) }
}

private struct HeaderJ: Decodable, Sendable {
    let software, fileName, object: String
    let imageWidth, imageHeight: Int
    let x, y: AxisJ?
    let notes: [String]
    let crop: [Int]?
    let spectrumWidth, spectrumHeight: Int
    let spacingXBits, spacingYBits: UInt64
    let unitsX, unitsY: String
    var header: FFTExportHeader {
        FFTExportHeader(software: software, fileName: fileName, object: object, imageWidth: imageWidth,
                        imageHeight: imageHeight, x: x?.axis, y: y?.axis, notes: notes,
                        crop: crop.map { PixelCrop(row: $0[0], col: $0[1], height: $0[2], width: $0[3]) },
                        spectrumWidth: spectrumWidth, spectrumHeight: spectrumHeight, spacingX: dbl(spacingXBits),
                        spacingY: dbl(spacingYBits), unitsX: unitsX, unitsY: unitsY)
    }
}

private struct SpotJ: Decodable, Sendable {
    let clickedRow, clickedCol, snappedRow, snappedCol: Int
    let centroidRowBits, centroidColBits: UInt64
    let width, height: Int
    let qxBits, qyBits, qBits, dBits: UInt64
    let units, reciprocalUnits: String
    let neighbour: Bool
    var measurement: SpotMeasurement {
        SpotMeasurement(clickedRow: clickedRow, clickedCol: clickedCol, snappedRow: snappedRow, snappedCol: snappedCol,
                        centroidRow: dbl(centroidRowBits), centroidCol: dbl(centroidColBits), width: width, height: height,
                        qx: dbl(qxBits), qy: dbl(qyBits), q: dbl(qBits), d: dbl(dBits), units: units,
                        reciprocalUnits: reciprocalUnits, neighbour: neighbour, precision: .float32)
    }
}

private struct ExportPins: Decodable, Sendable {
    struct Constants: Decodable, Sendable { let profileValidation, spotsValidation, lineage: String }
    struct Radial: Decodable, Sendable {
        struct Input: Decodable, Sendable {
            let header: HeaderJ
            let units: String
            let widthBits: UInt64
            let meanBits: [UInt64]
            let count: [Int]
        }
        let input: Input
        let expected: String?
    }
    struct Spots: Decodable, Sendable {
        struct Row: Decodable, Sendable { let number: Int; let inMask: Bool; let m: SpotJ }
        struct Input: Decodable, Sendable {
            let header: HeaderJ
            let rows: [Row]
            let omitted: Int
            let maskRadiusBits, maskEdgeBits: UInt64
            let notes: [String]
        }
        let input: Input
        let expected: String?
    }
    let constants: Constants
    let radial: [Radial]
    let spots: [Spots]

    static func load() throws -> ExportPins {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/csv-export-pins.json")
        return try JSONDecoder().decode(ExportPins.self, from: Data(contentsOf: url))
    }
}

@Suite("FFTExportCSV") struct FFTExportCSVTests {
    @Test func radialProfileTextIsTheApplicationsByteForByte() throws {
        let pins = try ExportPins.load()
        var written = 0
        for c in pins.radial {
            let i = c.input
            let profile = try #require(RadialProfile(width: dbl(i.widthBits), units: i.units,
                                                     mean: i.meanBits.map(dbl), count: i.count))
            let text = RadialProfileCSV.text(profile, header: i.header.header, title: "macTEM radial profile",
                                             validation: pins.constants.profileValidation, lineage: pins.constants.lineage)
            #expect(text == c.expected)
            if text != nil { written += 1 }
        }
        #expect(written == pins.radial.filter { $0.expected != nil }.count && written >= 8)
    }

    @Test func spotsTextIsTheApplicationsByteForByte() throws {
        let pins = try ExportPins.load()
        var written = 0
        for c in pins.spots {
            let i = c.input
            let rows = i.rows.map { SpotsCSV.Row(number: $0.number, measured: $0.m.measurement, inMask: $0.inMask) }
            let text = SpotsCSV.text(rows: rows, omitted: i.omitted, maskRadius: dbl(i.maskRadiusBits), maskEdge: dbl(i.maskEdgeBits),
                                     notes: i.notes, header: i.header.header, title: "macTEM spots",
                                     validation: pins.constants.spotsValidation, lineage: pins.constants.lineage)
            #expect(text == c.expected)
            if text != nil { written += 1 }
        }
        #expect(written == pins.spots.filter { $0.expected != nil }.count && written >= 8)
    }

    @Test func subjectAndNoFileWithoutBins() throws {
        #expect(FFTExportText.subject(crop: nil, imageWidth: 2048, imageHeight: 1024) == "the whole image, 2048 x 1024 px")
        #expect(FFTExportText.subject(crop: PixelCrop(row: 404, col: 812, height: 256, width: 256), imageWidth: 1, imageHeight: 1)
                == "region 256 x 256 px at (812, 404) (column, row of its top-left pixel)")
        let profile = try #require(RadialProfile(width: 1, units: "1/nm", mean: [5], count: [1]))     // only the DC bin
        let h = try ExportPins.load().radial[0].input.header.header
        #expect(RadialProfileCSV.text(profile, header: h, title: "t", validation: "v", lineage: "l") == nil)
        #expect(SpotsCSV.text(rows: [], omitted: 0, maskRadius: 3, maskEdge: 2, notes: [], header: h, title: "t",
                              validation: "v", lineage: "l") == nil)
    }
}
