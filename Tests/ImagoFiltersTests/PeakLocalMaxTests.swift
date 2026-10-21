// skimage.feature.peak_local_max against scikit-image 0.26.0 (tools/peaks/pins.py): coordinates exact, in order.
// scikit-image's wheel ships no tests (checked in refs/skimage); indices are integers, so the comparison is exact.
import Foundation
import ImagoFilters
import ImagoCore
import Testing

struct PeakPin: Decodable, Sendable, CustomStringConvertible {
    let h, w: Int; let dtype: String; let x: [Double]; let min_distance: Int
    let threshold_abs, threshold_rel: Double?; let border: [Int]?; let num_peaks: Int?; let p_norm: Double?
    let peaks: [[Int]]
    var description: String { "\(h)x\(w) \(dtype) d\(min_distance) abs\(String(describing: threshold_abs)) rel\(String(describing: threshold_rel)) border\(String(describing: border)) n\(String(describing: num_peaks)) p\(String(describing: p_norm))" }
    static let all: [PeakPin] = {
        struct File: Decodable { let cases: [PeakPin] }
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/peak-pins.json")
        return try! JSONDecoder().decode(File.self, from: Data(contentsOf: url)).cases
    }()
}

@Test(arguments: PeakPin.all)
func peakLocalMaxMatchesSciKitImage(_ c: PeakPin) throws {
    let border: PeakBorder = c.border.map { .width(row: $0[0], column: $0[1]) } ?? .minDistance
    let p = c.p_norm ?? .infinity
    let got: [PeakCoordinate]
    if c.dtype == "float32" {
        got = try PeakLocalMax.find(c.x.map(Float.init), rows: c.h, columns: c.w, minDistance: c.min_distance, thresholdAbs: c.threshold_abs,
                                    thresholdRel: c.threshold_rel, excludeBorder: border, numPeaks: c.num_peaks, pNorm: p)
    } else {
        got = try PeakLocalMax.find(c.x, rows: c.h, columns: c.w, minDistance: c.min_distance, thresholdAbs: c.threshold_abs,
                                    thresholdRel: c.threshold_rel, excludeBorder: border, numPeaks: c.num_peaks, pNorm: p)
    }
    #expect(got.map { [$0.row, $0.column] } == c.peaks, "\(c)")
}

@Test func hugeParametersAreRefusedOrHarmless() throws {
    let a = [Float](repeating: 0, count: 16)
    #expect(try PeakLocalMax.find(a, rows: 4, columns: 4, minDistance: Int.max / 2).isEmpty)
    #expect(throws: FilterError.self) { try SeparableFilters.gaussianFilter([Float](repeating: 1, count: 4), rows: 2, columns: 2, sigma: (1e300, 1)) }
    #expect(throws: FilterError.self) { try PeakLocalMax.find([Float.nan, 1, 2, 3], rows: 2, columns: 2) }
    #expect(PeakCoordinate(row: 1, column: 2).row == 1)
}
