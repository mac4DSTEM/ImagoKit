// skimage.feature.peak_local_max against scikit-image 0.26.0 (tools/peaks/pins.py): coordinates exact, in order.
// scikit-image's wheel ships no tests (checked in References/python/skimage); indices are integers, so the comparison is exact.
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

/// 0.6.1: infinite pixels are accepted and match scikit-image 0.26.0 (tools/peaks/inf_pins.py); a NaN pixel stays refused.
struct InfPin: Decodable, Sendable {
    let h, w: Int; let dtype: String; let x: [Pixel]; let min_distance: Int; let border: Bool
    let threshold_abs: Double?; let num_peaks: Int?; let peaks: [[Int]]
    enum Pixel: Decodable, Sendable {
        case number(Double), inf, negInf
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let v = try? c.decode(Double.self) { self = .number(v); return }
            self = try c.decode(String.self) == "inf" ? .inf : .negInf
        }
        var value: Double { switch self { case .number(let v): return v; case .inf: return .infinity; case .negInf: return -.infinity } }
    }
    static let all: [InfPin] = {
        struct File: Decodable { let cases: [InfPin] }
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/peak-inf-pins.json")
        return try! JSONDecoder().decode(File.self, from: Data(contentsOf: url)).cases
    }()
}

@Test func peakLocalMaxOnInfinitePixelsMatchesSciKitImage() throws {
    #expect(InfPin.all.count == 7)
    for c in InfPin.all {
        let border: PeakBorder = c.border ? .minDistance : .width(row: 0, column: 0)
        let got: [PeakCoordinate]
        if c.dtype == "float32" {
            got = try PeakLocalMax.find(c.x.map { Float($0.value) }, rows: c.h, columns: c.w, minDistance: c.min_distance,
                                        thresholdAbs: c.threshold_abs, excludeBorder: border, numPeaks: c.num_peaks)
        } else {
            got = try PeakLocalMax.find(c.x.map(\.value), rows: c.h, columns: c.w, minDistance: c.min_distance,
                                        thresholdAbs: c.threshold_abs, excludeBorder: border, numPeaks: c.num_peaks)
        }
        #expect(got.map { [$0.row, $0.column] } == c.peaks, "\(c.h)x\(c.w) \(c.dtype) d\(c.min_distance)")
    }
    #expect(throws: FilterError.self) { try PeakLocalMax.find([Double.nan, 1, 2, 3], rows: 2, columns: 2) }
}
