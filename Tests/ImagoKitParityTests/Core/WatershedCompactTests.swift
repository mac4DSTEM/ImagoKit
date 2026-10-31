// watershed with compactness and watershed_line against the pins of tools/watershed-compact/pins.py (scikit-image 0.26.0).
// The wheel ships no tests; labels are exact.
import Foundation
import ImagoCore
import ImagoKitTestSupport
import Testing

struct WatershedCompactPins: Decodable, Sendable {
    struct Case: Decodable, Sendable, CustomStringConvertible {
        let name: String
        let width, height: Int
        let image: [Double]
        let markers: [Int]
        let mask: String?
        let connectivity: Int
        let compactness: Double
        let watershedLine: Bool
        let labels: [Int]
        var description: String { name }
    }
    struct Meta: Decodable, Sendable { let skimage, numpy, scipy: String }
    let meta: Meta
    let cases: [Case]

    static let all: WatershedCompactPins = {
        let url = FixtureDirs.fixtures(of: "ImagoCoreTests").appendingPathComponent("watershed-compact-pins.json")
        return try! JSONDecoder().decode(WatershedCompactPins.self, from: Data(contentsOf: url))
    }()
}

@Suite("watershed compactness and line (scikit-image 0.26.0)") struct WatershedCompactTests {
    static let pins = WatershedCompactPins.all

    @Test func versions() {
        #expect(Self.pins.meta.skimage == "0.26.0" && Self.pins.cases.count >= 50)
        #expect(Self.pins.cases.contains { $0.watershedLine && $0.labels.contains(0) })
    }

    @Test(arguments: pins.cases) func labels(_ c: WatershedCompactPins.Case) throws {
        let got = try Segmentation.watershed(
            image: c.image, markers: c.markers.map { Int32($0) }, rows: c.height, columns: c.width,
            mask: c.mask.map { $0.map { $0 == "1" } }, connectivity: c.connectivity == 1 ? .four : .eight,
            compactness: c.compactness, watershedLine: c.watershedLine)
        let expected = c.labels.map { Int32($0) }
        #expect(got == expected, "\(c.name): \(zip(got, expected).filter { $0 != $1 }.count) of \(got.count) labels differ")
    }

    @Test func refusals() {
        for bad in [-1.0, .nan, .infinity] {
            #expect(throws: MorphologyError.self) {
                try Segmentation.watershed(image: [0, 1], markers: [1, 0], rows: 1, columns: 2, compactness: bad)
            }
        }
    }
}
