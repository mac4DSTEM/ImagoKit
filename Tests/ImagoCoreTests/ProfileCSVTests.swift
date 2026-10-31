// ProfileCSV: the whole text of a planted profile byte for byte (the lines and order macTEM's app copy wrote), the shown
// line, the unvalidated clause, header sanitising, nan, and an empty profile that is not a file. Text is tested exact.
import Foundation
import ImagoCore
import Testing

@Suite("ProfileCSV") struct ProfileCSVTests {
    private let nmX = AxisCalibration(scale: 0.5, unit: "nm")
    private let nmY = AxisCalibration(scale: 0.8, unit: "nm")
    private let validation = "samples compared with scikit-image 0.26.0 profile_line(order=1, mode=\"reflect\", "
        + "reduce_func=mean) (1e-5 relative;1e-5 of the image's largest magnitude near zero); "
        + "calibration: as the file states it"

    /// v = 100 + 3 r + 2 c on 8 x 6 pixels, the line (1, 2) to (5, 2): five samples on row 2, 0.5 nm apart, 106 + 2 c.
    private func planted(notes: [String] = [], fileName: String = "planted.dm4", object: String = "ImageList.1",
                         pixel: (Int, Int) -> Float = { r, c in Float(100 + 3 * r + 2 * c) },
                         shown: (unit: String, factor: Double)? = nil) throws -> String? {
        let pixels = (0..<48).map { pixel($0 / 8, $0 % 8) }
        let image = try Micrograph(width: 8, height: 6, pixels: pixels, x: nmX, y: nmY)
        let profile = try LineProfile.sample(image, from: PixelPoint(x: 1, y: 2), to: PixelPoint(x: 5, y: 2))
        let source = TidyResultsCSV.Source(software: "macTEM 0.1.0 (1)", fileName: fileName, object: object, width: 8,
                                           height: 6, x: nmX, y: nmY, notes: notes)
        return ProfileCSV.text(profile, title: "macTEM line profile", appName: "macTEM", start: PixelPoint(x: 1, y: 2),
                               end: PixelPoint(x: 5, y: 2), source: source, validation: validation, shown: shown)
    }

    @Test func textIsExactForAPlantedProfile() throws {
        let expected = [
            "# macTEM line profile",
            "# software: macTEM 0.1.0 (1)",
            "# file: planted.dm4",
            "# object: ImageList.1",
            "# image: 8 x 6 px",
            "# start: x=1.0, y=2.0 (image pixels: x column, y row, pixel centres at integers)",
            "# end: x=5.0, y=2.0",
            "# width: 1 px",
            "# samples: 5",
            "# length: 2 nm",
            "# x scale: 0.5 nm per px, origin 0.0 px",
            "# y scale: 0.8 nm per px, origin 0.0 px",
            "# position: distance from the start along the line, in nm as the file states it "
                + "(not an axis coordinate; origins unused)",
            "# method: bilinear interpolation (order 1), scipy's \"reflect\" boundary (half-sample symmetric), "
                + "mean of 1 samples 1 px apart along the perpendicular on the pixel grid",
            "# notes: none",
            "# validation: " + validation,
            "# lineage: none \u{2014} marks carry no lineage id; the mark's parameters above stand in",
            "# numbers: %.17g (exact round trip); nan where a NaN pixel enters a sample",
            "position (nm),value",
            "0,108", "0.5,110", "1,112", "1.5,114", "2,116",
            "",
        ].joined(separator: "\n")
        #expect(try planted() == expected)
    }

    @Test func shownLineAndUnvalidatedClauseAndSanitising() throws {
        let text = try #require(try planted(notes: ["scale UNVALIDATED\r\nnext"], fileName: "a\u{2028}b.dm4",
                                            shown: (unit: "nm", factor: 1e9)))
        let lines = text.components(separatedBy: "\n")
        #expect(lines.contains("# shown in macTEM as: nm (x 1000000000.0)"))
        #expect(lines.contains("# notes: scale UNVALIDATED  next"))
        #expect(lines.contains("# validation: " + validation + ", unvalidated (reader note)"))
        #expect(lines.contains("# file: a b.dm4"))
        let keys = lines.prefix { $0.hasPrefix("#") }.compactMap { $0.dropFirst(2).components(separatedBy: ": ").first }
        #expect(keys.contains("shown in macTEM as"))
    }

    @Test func nanIsSpelledAndNeverZero() throws {
        let text = try #require(try planted(pixel: { r, c in r == 2 && c == 3 ? .nan : Float(c) }))
        let rows = text.components(separatedBy: "\n").drop { !$0.hasPrefix("position") }.dropFirst().filter { !$0.isEmpty }
        #expect(rows.contains { $0.hasSuffix(",nan") })
        #expect(!text.contains("-nan"))
    }

    @Test func emptyProfileIsNotAFile() throws {
        let image = try Micrograph(width: 2, height: 2, pixels: [0, 1, 2, 3])
        let profile = try LineProfile.sample(image, from: PixelPoint(x: 0, y: 0), to: PixelPoint(x: 1, y: 0))
        let empty = LineProfile(samples: [], positions: [], unit: "px", isPhysical: false, width: 1)
        _ = profile
        let source = TidyResultsCSV.Source(software: "s", fileName: "f", object: "o", width: 2, height: 2, x: nil, y: nil, notes: [])
        #expect(ProfileCSV.text(empty, title: "t", appName: "a", start: PixelPoint(x: 0, y: 0), end: PixelPoint(x: 1, y: 0),
                                source: source, validation: "v") == nil)
    }
}
