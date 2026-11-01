// CSVText primitives and TidyResultsCSV against tools/csvexport/pins.py: the application's own writers, run offline on seeded
// cases. Text is tested exact (byte for byte); the primitives also against planted values.
import Foundation
import ImagoCore
import Testing

private func dbl(_ b: UInt64) -> Double { Double(bitPattern: b) }

private struct AxisJ: Decodable, Sendable {
    let originBits: UInt64, scaleBits: UInt64
    let unit: String
    var axis: AxisCalibration { AxisCalibration(scale: dbl(scaleBits), unit: unit, origin: dbl(originBits)) }
}

private struct TidyPins: Decodable, Sendable {
    struct Source: Decodable, Sendable {
        let software, fileName, object: String
        let width, height: Int
        let x, y: AxisJ?
        let notes: [String]
    }
    struct Entry: Decodable, Sendable { let name: String; let valueBits: UInt64?; let unit, text: String }
    struct Row: Decodable, Sendable {
        let group, id, label, product: String
        let entries: [Entry]
        let refusal, lineage: String?
    }
    struct Input: Decodable, Sendable {
        let source: Source
        let rows: [Row]
        let session: String
        let validations: [[String]]
    }
    struct Case: Decodable, Sendable { let input: Input; let expected: String? }
    let title: String
    let validations: [[String]]
    let descriptionLines: [String]
    let lineageLine: String
    let cases: [Case]

    static func load() throws -> TidyPins {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/csv-tidy-pins.json")
        return try JSONDecoder().decode(TidyPins.self, from: Data(contentsOf: url))
    }
}

@Suite("CSVText") struct CSVTextTests {
    @Test func numbers() {
        #expect(CSVText.number(0.1) == "0.10000000000000001")
        #expect(CSVText.number(1) == "1" && CSVText.number(-0.0) == "-0" && CSVText.number(1e300) == "1.0000000000000001e+300")
        #expect(CSVText.number(.nan) == "nan" && CSVText.number(-Double.nan) == "nan")
        #expect(CSVText.number(.infinity) == "inf" && CSVText.number(-.infinity) == "-inf")
        #expect(CSVText.number(5e-324) == "4.9406564584124654e-324")
        for v in [1 / 3.0, 123456789.123456789, 2.2250738585072014e-308, Double.greatestFiniteMagnitude] {
            #expect(Double(CSVText.number(v)) == v)
        }
    }

    @Test func headerValuesAndCells() {
        #expect(CSVText.headerValue("a\r\nb\tc\u{0085}d\u{2028}e\u{2029}f\u{7F}g") == "a  b c d e f g")
        #expect(CSVText.headerValue("Å µm") == "Å µm")
        #expect(CSVText.quoted("plain") == "plain" && CSVText.quoted("a,b") == "\"a,b\"")
        #expect(CSVText.quoted("say \"hi\"") == "\"say \"\"hi\"\"\"" && CSVText.quoted("x\ny") == "\"x\ny\"")
    }

    @Test func scaleLines() {
        #expect(CSVText.scaleLine("x", nil) == "# x scale: none")
        #expect(CSVText.scaleLine("y", AxisCalibration(scale: 0.5, unit: "nm", origin: 2)) == "# y scale: 0.5 nm per px, origin 2.0 px")
        #expect(CSVText.scaleLine("x", AxisCalibration(scale: 0, unit: "nm")) == "# x scale: none")
        #expect(CSVText.scaleLine("x", AxisCalibration(scale: 1, unit: "")) == "# x scale: none")
    }

    @Test func tidyTextIsTheApplicationsByteForByte() throws {
        let pins = try TidyPins.load()
        var written = 0
        for c in pins.cases {
            let i = c.input
            let s = i.source
            let source = TidyResultsCSV.Source(software: s.software, fileName: s.fileName, object: s.object, width: s.width,
                                               height: s.height, x: s.x?.axis, y: s.y?.axis, notes: s.notes)
            let rows = i.rows.map { r in
                TidyResultsCSV.Row(group: r.group, id: r.id, label: r.label, product: r.product,
                                   entries: r.entries.map { TidyResultsCSV.Entry($0.name, $0.valueBits.map(dbl), unit: $0.unit, text: $0.text) },
                                   refusal: r.refusal, lineage: r.lineage)
            }
            let text = try TidyResultsCSV.text(rows, source: source, title: pins.title, session: i.session,
                                               descriptionLines: pins.descriptionLines,
                                               validations: pins.validations.map { (product: $0[0], sentence: $0[1]) },
                                               lineageLine: pins.lineageLine)
            #expect(text == c.expected)
            if text != nil { written += 1 }
        }
        #expect(written >= 12)
    }

    /// A source with reader notes: every validation sentence (header line and cell) is followed by the notes as the reader
    /// wrote them, and nothing is called unvalidated. Without notes the sentence stands alone.
    @Test func tidyQuotesTheReaderNoteAfterEachValidation() throws {
        let row = TidyResultsCSV.Row(group: "g", id: "i", label: "l", product: "known", entries: [TidyResultsCSV.Entry("d", 1)])
        func text(_ notes: [String]) throws -> String {
            let source = TidyResultsCSV.Source(software: "s", fileName: "f", object: "o", width: 1, height: 1, x: nil, y: nil,
                                               notes: notes)
            return try #require(try TidyResultsCSV.text([row], source: source, title: "t", session: "s", descriptionLines: [],
                                                        validations: [(product: "known", sentence: "closed form")],
                                                        lineageLine: "# lineage: none"))
        }
        let noted = try text(["scale from the tag", "UNVALIDATED"])
        let lines = noted.components(separatedBy: "\n")
        #expect(lines.contains("# validation (known): closed form; reader note: scale from the tag; UNVALIDATED"))
        #expect(lines.contains("g,i,l,d,1,,,closed form; reader note: scale from the tag; UNVALIDATED,"))
        #expect(!noted.contains("unvalidated"))     // the reader's own word is quoted, never a lower-case verdict
        let plain = try text(["origin from the header"])
        #expect(plain.contains("closed form; reader note: origin from the header"))
        #expect(!plain.lowercased().contains("unvalidated"))
        let bare = try text([])
        #expect(bare.contains("# validation (known): closed form\n") && bare.hasSuffix("g,i,l,d,1,,,closed form,\n"))
        // a cell keeps one line (the note passes headerValue) and a blank note adds nothing
        let multi = try text(["cr\r\nlf", "", " "])
        #expect(multi.hasSuffix("g,i,l,d,1,,,closed form; reader note: cr  lf,\n"))
        #expect(try text(["", "\n"]) == bare.replacingOccurrences(of: "# notes: none", with: "# notes: ;  "))
    }

    @Test func tidyRefusesAProductWithoutASentenceAndWritesNothingForNoRow() throws {
        let source = TidyResultsCSV.Source(software: "s", fileName: "f", object: "o", width: 1, height: 1, x: nil, y: nil, notes: [])
        let row = TidyResultsCSV.Row(group: "g", id: "i", label: "l", product: "mystery", entries: [])
        #expect(throws: TidyResultsCSV.TidyResultsError.noValidation(product: "mystery")) {
            try TidyResultsCSV.text([row], source: source, title: "t", session: "s", descriptionLines: [],
                                    validations: [(product: "known", sentence: "x")], lineageLine: "# lineage: none")
        }
        #expect(try TidyResultsCSV.text([], source: source, title: "t", session: "s", descriptionLines: [], validations: [],
                                        lineageLine: "") == nil)
        // a refusal is a quantity "outcome" with no value; a row with entries and no refusal writes only its entries
        let refused = TidyResultsCSV.Row(group: "g", id: "i", label: "l", product: "known", entries: [], refusal: "no, because")
        let text = try #require(try TidyResultsCSV.text([refused], source: source, title: "t", session: "s", descriptionLines: [],
                                                    validations: [(product: "known", sentence: "x")], lineageLine: "# lineage: none"))
        #expect(text.hasSuffix("\ng,i,l,outcome,,,\"no, because\",x,\n"))
    }
}
