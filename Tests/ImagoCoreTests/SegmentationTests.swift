// Segmentation against tools/segmentation/pins.py: scikit-image 0.26.0 label and regionprops (labels, area, bbox exact;
// moments to a relative 1e-12) and the application's own extents routine (Float32, so 2e-5 relative).
import Foundation
import ImagoCore
import Testing

private struct SegPins: Decodable, Sendable {
    struct Props: Decodable, Sendable {
        let label: Int32, area: Int
        let bbox: [Int]
        let centroid: [Double], eigvals: [Double], inertia: [Double]
        let major: Double, minor: Double, orientation: Double
    }
    struct App: Decodable, Sendable {
        let area: Int
        let cx: Double, cy: Double, length: Double, width: Double, orientation: Double
        let edge: Bool
    }
    struct Mask: Decodable, Sendable {
        let width: Int, height: Int
        let mask: String
        let labels1: [Int32], labels2: [Int32]
        let count1: Int, count2: Int
        let props1: [Props], props2: [Props]
        let app: [App]
    }
    struct Values: Decodable, Sendable {
        let width: Int, height: Int
        let values: [Int32]
        let bg0_1: [Int32], bg0_2: [Int32], all_1: [Int32], all_2: [Int32]
        let bg0count_1: Int, bg0count_2: Int, allcount_1: Int, allcount_2: Int
    }
    let masks: [Mask]
    let values: [Values]
    let appChecked: Bool

    static func load() throws -> SegPins {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/segmentation-pins.json")
        return try JSONDecoder().decode(SegPins.self, from: Data(contentsOf: url))
    }
}

/// |a − b| ≤ rel · max(1, |b|)
private func close(_ a: Double, _ b: Double, rel: Double) -> Bool { abs(a - b) <= rel * max(1, abs(b)) }

@Suite("Segmentation") struct SegmentationTests {
    @Test func labelsEqualScikitImageExactly() throws {
        let pins = try SegPins.load()
        #expect(pins.appChecked)
        for m in pins.masks {
            let mask = m.mask.map { $0 == "1" }
            let four = try Segmentation.label(mask, width: m.width, height: m.height, connectivity: .four)
            let eight = try Segmentation.label(mask, width: m.width, height: m.height)
            #expect(four.labels == m.labels1 && four.count == m.count1)
            #expect(eight.labels == m.labels2 && eight.count == m.count2)
        }
    }

    @Test func valueMapsEqualScikitImageExactly() throws {
        for v in try SegPins.load().values {
            let b0 = try Segmentation.label(values: v.values, width: v.width, height: v.height, background: 0, connectivity: .four)
            #expect(b0.labels == v.bg0_1 && b0.count == v.bg0count_1)
            let b0e = try Segmentation.label(values: v.values, width: v.width, height: v.height, background: 0)
            #expect(b0e.labels == v.bg0_2 && b0e.count == v.bg0count_2)
            let all = try Segmentation.label(values: v.values, width: v.width, height: v.height, background: nil, connectivity: .four)
            #expect(all.labels == v.all_1 && all.count == v.allcount_1)
            let alle = try Segmentation.label(values: v.values, width: v.width, height: v.height, background: nil)
            #expect(alle.labels == v.all_2 && alle.count == v.allcount_2)
        }
    }

    @Test func momentsEqualRegionpropsAndTheApplicationExtents() throws {
        var objects = 0
        for m in try SegPins.load().masks {
            let image = try Segmentation.label(m.mask.map { $0 == "1" }, width: m.width, height: m.height)
            let got = try Segmentation.objectMoments(image)
            #expect(got.count == m.props2.count && got.count == m.app.count)
            for ((o, p), a) in zip(zip(got, m.props2), m.app) {
                objects += 1
                #expect(o.label == p.label && o.area == p.area)
                #expect([o.minRow, o.minCol, o.maxRow, o.maxCol] == p.bbox)
                #expect(close(o.centroidRow, p.centroid[0], rel: 1e-12) && close(o.centroidCol, p.centroid[1], rel: 1e-12))
                // inertia tensor [[cov_cc, −cov_rc], [−cov_rc, cov_rr]]
                #expect(close(o.covarianceCC, p.inertia[0], rel: 1e-12) && close(-o.covarianceRC, p.inertia[1], rel: 1e-12))
                #expect(close(o.covarianceRR, p.inertia[3], rel: 1e-12))
                #expect(close(o.eigenvalueMajor, p.eigvals[0], rel: 1e-12) && close(o.eigenvalueMinor, p.eigvals[1], rel: 1e-12))
                #expect(close(o.axisMajorLength, p.major, rel: 1e-12) && close(o.axisMinorLength, p.minor, rel: 1e-12))
                #expect(close(o.orientation, p.orientation, rel: 1e-12), "label \(o.label)")
                // the application's Float32 routine
                #expect(o.area == a.area && o.touchesEdge == a.edge)
                #expect(close(o.centroidCol, a.cx, rel: 2e-5) && close(o.centroidRow, a.cy, rel: 2e-5))
                #expect(close(o.lengthPx, a.length, rel: 2e-5) && close(o.widthPx, a.width, rel: 2e-5), "label \(o.label)")
                // degrees are compared as angles on the 180-degree circle; a near-circular object's axis is Float32 noise
                if abs(p.major - p.minor) > 0.05 * p.major {
                    var d = abs(o.orientationDegrees - a.orientation).truncatingRemainder(dividingBy: 180)
                    d = min(d, 180 - d)
                    #expect(d < 2e-3, "label \(o.label): \(o.orientationDegrees) vs \(a.orientation)")
                }
            }
        }
        #expect(objects == 888)
    }

    @Test func plantedShapes() throws {
        // a 3x5 rectangle at rows 2...4, columns 1...5 of a 8x9 mask
        var mask = [Bool](repeating: false, count: 72)
        for r in 2...4 { for c in 1...5 { mask[r * 9 + c] = true } }
        let image = try Segmentation.label(mask, width: 9, height: 8)
        let o = try #require(Segmentation.objectMoments(image).first)
        #expect(image.count == 1 && o.area == 15)
        #expect(o.centroidRow == 3 && o.centroidCol == 3)
        #expect(o.covarianceCC == 2 && abs(o.covarianceRR - 2.0 / 3) < 1e-15 && o.covarianceRC == 0)
        #expect(o.lengthPx == 5 && o.widthPx == 3 && o.orientationDegrees == 0 && !o.touchesEdge)
        // a one-pixel object: extents 1, zero axes
        let one = try Segmentation.objectMoments(Segmentation.label([true], width: 1, height: 1))
        #expect(one.count == 1 && one[0].lengthPx == 1 && one[0].widthPx == 1 && one[0].axisMajorLength == 0 && one[0].touchesEdge)
        // the checkerboard is one object 8-connected, many 4-connected
        let board = (0..<72).map { ($0 / 9 + $0 % 9) % 2 == 0 }
        #expect(try Segmentation.label(board, width: 9, height: 8).count == 1)
        #expect(try Segmentation.label(board, width: 9, height: 8, connectivity: .four).count == board.filter { $0 }.count)
    }

    @Test func nearlySymmetricObjectsKeepTheirAxis() throws {
        // vertical bars of odd length (inexact centroid), a horizontal one, and a 45-degree diagonal
        for n in [3, 7, 9, 11] {
            var v = [Bool](repeating: false, count: 13 * 13), hz = v, diag = v
            for i in 0..<n { v[(1 + i) * 13 + 5] = true; hz[5 * 13 + 1 + i] = true; diag[(1 + i) * 13 + 1 + i] = true }
            let ov = try #require(Segmentation.objectMoments(Segmentation.label(v, width: 13, height: 13)).first)
            let oh = try #require(Segmentation.objectMoments(Segmentation.label(hz, width: 13, height: 13)).first)
            let od = try #require(Segmentation.objectMoments(Segmentation.label(diag, width: 13, height: 13)).first)
            #expect(abs(abs(ov.orientationDegrees) - 90) < 1e-9 && abs(ov.lengthPx - Double(n)) < 1e-12 && abs(ov.widthPx - 1) < 1e-12)
            #expect(abs(oh.orientationDegrees) < 1e-9 && abs(oh.lengthPx - Double(n)) < 1e-12 && abs(oh.widthPx - 1) < 1e-12)
            #expect(abs(od.orientationDegrees - 45) < 1e-9 && abs(od.lengthPx - (1 + Double(n - 1) * 2.0.squareRoot())) < 1e-9)
        }
    }

    @Test func refusals() throws {
        #expect(throws: Segmentation.SegmentationError.sizeMismatch(expected: 6, got: 5)) {
            try Segmentation.label([Bool](repeating: true, count: 5), width: 2, height: 3)
        }
        #expect(throws: Segmentation.SegmentationError.negativeDimension) {
            try Segmentation.label([], width: -1, height: 0)
        }
        let empty = try Segmentation.label([], width: 0, height: 0)
        #expect(empty.count == 0)
        #expect(try Segmentation.objectMoments(empty).isEmpty)
        let bad = Segmentation.LabelImage(width: 1, height: 1, labels: [3], count: 1)
        #expect(throws: Segmentation.SegmentationError.labelOutOfRange(3)) { try Segmentation.objectMoments(bad) }
        for broken in [Segmentation.LabelImage(width: 1, height: 1, labels: [0], count: -1),
                       Segmentation.LabelImage(width: -1, height: -1, labels: [0], count: 0),
                       Segmentation.LabelImage(width: Int.max, height: 2, labels: [], count: 0),
                       Segmentation.LabelImage(width: 1, height: 1, labels: [0], count: Int.max)] {
            #expect(throws: Segmentation.SegmentationError.invalidLabelImage) { try Segmentation.objectMoments(broken) }
        }
        #expect(throws: Segmentation.SegmentationError.invalidLabelImage) {
            try Segmentation.label([], width: Int.max, height: 2)
        }
    }
}
