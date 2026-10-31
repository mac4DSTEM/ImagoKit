// skimage.feature.peak_local_max with footprint= and labels= against scikit-image 0.26.0
// (tools/peak-footprint-labels/pins.py): coordinates exact, in order (indices are integers; the wheel ships no tests).
import Foundation
import ImagoKitTestSupport
import ImagoFilters
import Testing

struct PeakFootprintLabelsPins: Decodable, Sendable {
    struct Image: Decodable, Sendable { let h, w: Int; let x: [Double] }
    struct Labels: Decodable, Sendable { let h, w: Int; let labels: [Int] }
    struct Footprint: Decodable, Sendable { let h, w: Int; let mask: [Bool] }
    struct Case: Decodable, Sendable, CustomStringConvertible {
        let image: Int; let dtype: String; let footprint: Int?; let labels: Int?; let min_distance: Int
        let threshold_abs, threshold_rel: Double?; let border: [Int]?; let num_peaks, num_peaks_per_label: Int?
        let p_norm: Double?; let peaks: [[Int]]
        var description: String {
            "img\(image) \(dtype) fp\(footprint.map(String.init) ?? "-") lab\(labels.map(String.init) ?? "-") d\(min_distance) "
                + "abs\(threshold_abs.map { "\($0)" } ?? "-") rel\(threshold_rel.map { "\($0)" } ?? "-") "
                + "border\(border.map { "\($0)" } ?? "-") n\(num_peaks.map(String.init) ?? "-") "
                + "nl\(num_peaks_per_label.map(String.init) ?? "-") p\(p_norm.map { "\($0)" } ?? "-")"
        }
    }
    let images: [Image]; let labels: [Labels]; let footprints: [Footprint]; let cases: [Case]
    static let all: PeakFootprintLabelsPins = {
        let url = FixtureDirs.fixtures(of: "ImagoFiltersTests").appendingPathComponent("peak-footprint-labels-pins.json")
        return try! JSONDecoder().decode(PeakFootprintLabelsPins.self, from: Data(contentsOf: url))
    }()
}

@Test(arguments: PeakFootprintLabelsPins.all.cases)
func peakLocalMaxWithFootprintAndLabelsMatchesSciKitImage(_ c: PeakFootprintLabelsPins.Case) throws {
    let pins = PeakFootprintLabelsPins.all
    let image = pins.images[c.image]
    let footprint = try c.footprint.map { i in
        try PeakFootprint(rows: pins.footprints[i].h, columns: pins.footprints[i].w, mask: pins.footprints[i].mask)
    }
    let labels = c.labels.map { pins.labels[$0].labels }
    let border: PeakBorder = c.border.map { .width(row: $0[0], column: $0[1]) } ?? .minDistance
    let p = c.p_norm ?? .infinity
    let got: [PeakCoordinate]
    if c.dtype == "float32" {
        got = try PeakLocalMax.find(image.x.map(Float.init), rows: image.h, columns: image.w, minDistance: c.min_distance,
                                    thresholdAbs: c.threshold_abs, thresholdRel: c.threshold_rel, excludeBorder: border,
                                    numPeaks: c.num_peaks, footprint: footprint, labels: labels,
                                    numPeaksPerLabel: c.num_peaks_per_label, pNorm: p)
    } else {
        got = try PeakLocalMax.find(image.x, rows: image.h, columns: image.w, minDistance: c.min_distance,
                                    thresholdAbs: c.threshold_abs, thresholdRel: c.threshold_rel, excludeBorder: border,
                                    numPeaks: c.num_peaks, footprint: footprint, labels: labels,
                                    numPeaksPerLabel: c.num_peaks_per_label, pNorm: p)
    }
    #expect(got.map { [$0.row, $0.column] } == c.peaks, "\(c)")
}
