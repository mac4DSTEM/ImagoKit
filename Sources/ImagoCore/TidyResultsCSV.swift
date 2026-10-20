//
//  TidyResultsCSV.swift (ImagoCore)
//  Every finished measurement of one file as ONE tidy CSV: a `# key: value` header, then the columns
//  `group,id,label,quantity,value,unit,text,validation,lineage` and one line per quantity of each row. One long format,
//  because rows differ in columns and a script wants one shape.
//
//  NUMBERS are `CSVText.number` (`%.17g`); a quantity with no number (a refusal's words, a text-only line) has an EMPTY value
//  cell and its words in `text`. Text cells and header values go through `CSVText.quoted` and `CSVText.headerValue`.
//
//  VALIDATION: the `validation` column holds the product's own sentence, and the header holds one `# validation (<product>):`
//  line for each product present, in the order `validations` lists them. When a note of the source says "unvalidated" (any
//  case), `unvalidatedClause` follows every sentence. The kit authors no claim: the caller supplies the sentences.
//
//  Pure text: exact, tested byte for byte. UTF-8, LF line ends.
//

import Foundation

public nonisolated enum TidyResultsCSV {
    /// Added to every validation sentence when the source's notes say it is unvalidated.
    public static let unvalidatedClause = ", unvalidated (reader note)"

    public static let columns = ["group", "id", "label", "quantity", "value", "unit", "text", "validation", "lineage"]

    /// The file and its calibration as the file states them.
    public struct Source: Sendable, Equatable {
        public var software: String
        public var fileName: String
        public var object: String
        public var width: Int
        public var height: Int
        public var x: AxisCalibration?
        public var y: AxisCalibration?
        public var notes: [String]

        public init(software: String, fileName: String, object: String, width: Int, height: Int,
                    x: AxisCalibration?, y: AxisCalibration?, notes: [String]) {
            self.software = software; self.fileName = fileName; self.object = object
            self.width = width; self.height = height; self.x = x; self.y = y; self.notes = notes
        }
    }

    /// One quantity of a row: a number, or words only (`value` nil).
    public struct Entry: Sendable, Equatable {
        public var name: String
        public var value: Double?
        public var unit: String
        public var text: String

        public init(_ name: String, _ value: Double?, unit: String = "", text: String = "") {
            self.name = name; self.value = value; self.unit = unit; self.text = text
        }
    }

    /// One measurement: its group, id, label, product (which names its validation sentence), quantities, an optional
    /// refusal (written as a quantity `outcome` with no value) and an optional lineage id.
    public struct Row: Sendable, Equatable {
        public var group: String
        public var id: String
        public var label: String
        public var product: String
        public var entries: [Entry]
        public var refusal: String?
        public var lineage: String?

        public init(group: String, id: String, label: String, product: String, entries: [Entry],
                    refusal: String? = nil, lineage: String? = nil) {
            self.group = group; self.id = id; self.label = label; self.product = product
            self.entries = entries; self.refusal = refusal; self.lineage = lineage
        }
    }

    public enum TidyResultsError: Error, Equatable, Sendable {
        /// A row's product has no entry in `validations`.
        case noValidation(product: String)
    }

    /// The whole file, or nil without a row.
    ///
    /// - `title`: the first header line's words, written `# <title>`.
    /// - `session`: the sentence that says where the file's marks are kept.
    /// - `descriptionLines` and `lineageLine` must each be one line (they are written as given).
    /// - `descriptionLines`: complete header lines (each starting `# `) that say what the points and units of this table mean,
    ///   written after `# notes:` in the order given.
    /// - `validations`: (product, sentence) in header order; every row's product must appear.
    /// - `lineageLine`: the complete `# lineage: ...` header line.
    public static func text(_ rows: [Row], source: Source, title: String, session: String, descriptionLines: [String],
                            validations: [(product: String, sentence: String)], lineageLine: String) throws -> String? {
        guard !rows.isEmpty else { return nil }
        var sentences: [String: String] = [:]
        for v in validations where sentences[v.product] == nil { sentences[v.product] = v.sentence }
        for row in rows where sentences[row.product] == nil { throw TidyResultsError.noValidation(product: row.product) }
        let notes = source.notes.isEmpty ? "none" : source.notes.map(CSVText.headerValue).joined(separator: "; ")
        var lines: [String] = []
        lines.append("# \(CSVText.headerValue(title))")
        lines.append("# software: \(CSVText.headerValue(source.software))")
        lines.append("# file: \(CSVText.headerValue(source.fileName))")
        lines.append("# object: \(CSVText.headerValue(source.object))")
        lines.append("# image: \(source.width) x \(source.height) px")
        lines.append(CSVText.scaleLine("x", source.x))
        lines.append(CSVText.scaleLine("y", source.y))
        lines.append("# session: \(CSVText.headerValue(session))")
        lines.append("# notes: \(notes)")
        lines.append(contentsOf: descriptionLines)
        let flagged = source.notes.contains { $0.lowercased().contains("unvalidated") } ? unvalidatedClause : ""
        let present = Set(rows.map(\.product))
        var written = Set<String>()
        for v in validations where present.contains(v.product) && written.insert(v.product).inserted {
            lines.append("# validation (\(v.product)): \(CSVText.headerValue(v.sentence + flagged))")
        }
        lines.append(lineageLine)
        lines.append("# numbers: %.17g (exact round trip); nan where the owner holds NaN; value empty where a line is words only")
        lines.append(columns.map(CSVText.quoted).joined(separator: ","))
        var out = lines.joined(separator: "\n")
        out.append("\n")
        for row in rows {
            let validation = (sentences[row.product] ?? "") + flagged
            func line(_ quantity: String, _ value: String, _ unit: String, _ text: String) {
                let cells = [row.group, row.id, row.label, quantity, value, unit, text, validation, row.lineage ?? ""]
                out.append(cells.map(CSVText.quoted).joined(separator: ","))
                out.append("\n")
            }
            for entry in row.entries {
                line(entry.name, entry.value.map(CSVText.number) ?? "", entry.unit, entry.text)
            }
            if let refusal = row.refusal { line("outcome", "", "", refusal) }
        }
        return out
    }
}
