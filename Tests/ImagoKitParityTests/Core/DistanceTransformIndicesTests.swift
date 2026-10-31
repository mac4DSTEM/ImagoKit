// distance_transform_edt(return_indices=True) against the pins of tools/edt-indices/pins.py (SciPy 1.18.1).
// Indices are integers and exact; the distances, computed from them as SciPy does, are held to 1e-12 (SciPy's tests: decimal 6).
import Foundation
import ImagoCore
import ImagoKitTestSupport
import Testing

struct DistanceTransformIndicesPins: Decodable, Sendable {
    struct Case: Decodable, Sendable, CustomStringConvertible {
        let name: String
        let width, height: Int
        let mask: String
        let sampling: [Double]?
        let distance: [Double]
        let nearestRows, nearestColumns: [Int]
        var description: String { name }
    }
    struct Meta: Decodable, Sendable { let skimage, numpy, scipy: String }
    let meta: Meta
    let cases: [Case]

    static let all: DistanceTransformIndicesPins = {
        let url = FixtureDirs.fixtures(of: "ImagoCoreTests").appendingPathComponent("edt-indices-pins.json")
        return try! JSONDecoder().decode(DistanceTransformIndicesPins.self, from: Data(contentsOf: url))
    }()
}

@Suite("distance transform indices (SciPy 1.18.1)") struct DistanceTransformIndicesTests {
    static let pins = DistanceTransformIndicesPins.all

    @Test func versions() {
        #expect(Self.pins.meta.scipy == "1.18.1" && Self.pins.meta.numpy == "2.5.4" && Self.pins.cases.count >= 30)
    }

    @Test(arguments: pins.cases) func indices(_ c: DistanceTransformIndicesPins.Case) throws {
        let mask = c.mask.map { $0 == "1" }
        let sampling = c.sampling.map { (rows: $0[0], columns: $0[1]) } ?? (1, 1)
        let got = try DistanceTransform.euclideanWithIndices(foreground: mask, rows: c.height, columns: c.width, sampling: sampling)
        #expect(got.nearestRows == c.nearestRows, "\(c.name): \(zip(got.nearestRows, c.nearestRows).filter { $0 != $1 }.count) rows differ")
        #expect(got.nearestColumns == c.nearestColumns,
                "\(c.name): \(zip(got.nearestColumns, c.nearestColumns).filter { $0 != $1 }.count) columns differ")
        #expect(got.distances.count == c.distance.count)
        for (i, (g, e)) in zip(got.distances, c.distance).enumerated() {
            #expect(abs(g - e) <= 1e-12 * max(1, e), "\(c.name) pixel \(i): \(g) vs \(e)")
        }
        // The distances agree with the envelope transform, and each one is the distance to the pixel the indices name.
        let plain = try DistanceTransform.euclidean(foreground: mask, rows: c.height, columns: c.width, sampling: sampling)
        for (i, (g, e)) in zip(got.distances, plain).enumerated() { #expect(abs(g - e) <= 1e-12 * max(1, e), "\(c.name) pixel \(i)") }
        for i in 0..<mask.count { #expect(!mask[got.nearestRows[i] * c.width + got.nearestColumns[i]], "\(c.name) pixel \(i): not background") }
    }

    @Test func refusals() {
        #expect(throws: MorphologyError.shape) { try DistanceTransform.euclideanWithIndices(foreground: [true], rows: 1, columns: 2) }
        #expect(throws: MorphologyError.noBackground) { try DistanceTransform.euclideanWithIndices(foreground: [true, true], rows: 1, columns: 2) }
        #expect(throws: MorphologyError.self) {
            try DistanceTransform.euclideanWithIndices(foreground: [true, false], rows: 1, columns: 2, sampling: (0, 1))
        }
    }
}
