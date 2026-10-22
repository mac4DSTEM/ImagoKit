// What the apps carried as "delete at ImagoKit 0.7.0" interims, now in the kit: the window product, the threshold compare and the
// fraction level, the particle size summary, and PeakLocalMax's spectrum symmetry. Each test holds the kit to the interim's own
// definition, written out literally here, so the apps drop their copies at the re-pin without a number moving.
import Foundation
import ImagoKitTestSupport
import ImagoCore
import ImagoFilters
import Testing

@Suite("Window product") struct WindowApplyTests {
    @Test func theProductIsTheWindowTimesThePixelsInTheImagesOwnType() throws {
        let rows = 7, columns = 9
        let pixels: [Float] = (0..<rows * columns).map { Float($0 % 13) * 0.37 - 1 }
        for type in [WindowType.hann, .hamming, .blackman, .tukey(alpha: 0.5)] {
            let w = try WindowFunction.window2D(type, rows: rows, columns: columns)
            var expected = pixels
            for i in expected.indices { expected[i] *= Float(w[i]) }   // the macTEM interim, verbatim
            #expect(try WindowFunction.apply(type, to: pixels, rows: rows, columns: columns) == expected)
            #expect(try WindowFunction.apply(window: w, to: pixels) == expected)
            let doubles = pixels.map(Double.init)
            let product = try WindowFunction.apply(type, to: doubles, rows: rows, columns: columns)
            for i in doubles.indices { #expect(product[i] == doubles[i] * w[i]) }
        }
    }

    @Test func refusals() {
        #expect(throws: FilterError.shape) { try WindowFunction.apply(.hann, to: [Float](repeating: 1, count: 5), rows: 2, columns: 3) }
        #expect(throws: FilterError.shape) { try WindowFunction.apply(window: [1, 2], to: [Float](repeating: 1, count: 3)) }
    }
}

@Suite("Threshold mask and level") struct ThresholdMaskTests {
    @Test func maskIsTheStrictCompareAndNonFiniteIsFalse() {
        let pixels: [Float] = [0, 1, 2, 3, .nan, .infinity, -.infinity, 2]
        #expect(Thresholding.mask(pixels, threshold: 2) == [false, false, false, true, false, false, false, false])
        #expect(Thresholding.mask(pixels, threshold: 2, polarity: .dark) == [true, true, false, false, false, false, false, false])
    }

    @Test func levelIsLowPlusFractionOfTheFiniteRange() throws {
        let pixels: [Float] = [4, .nan, 10, .infinity, 7]
        // The interim: range.lowerBound + Float(min(max(f, 0), 1)) * (upper - lower).
        for f in [0.0, 0.25, 0.5, 1.0, -3.0, 9.0] {
            let expected: Float = 4 + Float(min(max(f, 0), 1)) * (10 - 4)
            #expect(Thresholding.level(pixels, atFraction: f) == expected)
        }
        #expect(Thresholding.level([Float.nan, .infinity], atFraction: 0.5) == nil)
        #expect(Thresholding.level([Double](repeating: 3, count: 4), atFraction: 0.7) == 3)
    }
}

@Suite("Particle size summary") struct SizeSummaryTests {
    @Test func countsAndMediansFollowTheParticleReportsRules() throws {
        // 14 x 6 frame: bars of 3, 6, 2 and 4 pixels (1 high), a 1-pixel speck on the last column, and a bar on the top row.
        var labels = [Int32](repeating: 0, count: 14 * 6)
        func put(_ l: Int32, row: Int, columns: ClosedRange<Int>) { for c in columns { labels[row * 14 + c] = l } }
        put(1, row: 1, columns: 1...3); put(2, row: 3, columns: 1...6); put(3, row: 4, columns: 9...10); put(4, row: 2, columns: 8...11)
        put(5, row: 5 - 1, columns: 13...13); put(6, row: 0, columns: 2...5)
        let image = Segmentation.LabelImage(width: 14, height: 6, labels: labels, count: 6)
        let objects = try Segmentation.objectMoments(image)
        let s = ObjectStatistics.sizeSummary(objects, minimumArea: 3)
        #expect(s.total == 6 && s.edge == 2 && s.small == 1 && s.counted == 3)     // 5 touches column 13, 6 the top row; 3 (2 px) is small
        let lengths = objects.filter { $0.area >= 3 && !$0.touchesEdge }.map(\.lengthPx).sorted()
        #expect(s.medianLength == lengths[1] && lengths.count == 3)
        #expect(s.meanLength == lengths.reduce(0, +) / 3)
        #expect(s.meanWidth == objects.filter { $0.area >= 3 && !$0.touchesEdge }.map(\.widthPx).reduce(0, +) / 3)
        let none = ObjectStatistics.sizeSummary([], minimumArea: 3)
        #expect(none.total == 0 && none.medianLength == nil && none.meanWidth == nil)
    }
}

@Suite("PeakLocalMax spectrum symmetry") struct SpectrumSymmetryTests {
    /// macTEM's `FFTSpotFinder.find`, verbatim: the peaks, then the centre exclusion and one of each mirror pair, strongest first.
    private func interim(_ values: [Float], width: Int, height: Int, minDistance: Int, thresholdAbs: Double,
                         centreExclusion: Int, count: Int) throws -> [PeakCoordinate] {
        let peaks = try PeakLocalMax.find(values, rows: height, columns: width, minDistance: minDistance, thresholdAbs: thresholdAbs)
        let cy = height / 2, cx = width / 2
        var kept: [PeakCoordinate] = []
        for p in peaks {
            let dy = p.row - cy, dx = p.column - cx
            if dx * dx + dy * dy <= centreExclusion * centreExclusion { continue }
            guard dy > 0 || (dy == 0 && dx > 0) else { continue }
            kept.append(p)
            if kept.count == count { break }
        }
        return kept
    }

    @Test func matchesTheAppsInterimOnRandomCentrosymmetricSpectra() throws {
        var rng = SystemRandomNumberGeneratorSeeded(seed: 7)
        let width = 48, height = 40
        for _ in 0..<25 {
            var v = [Float](repeating: 0, count: width * height)
            let cy = height / 2, cx = width / 2
            for _ in 0..<30 {
                let r = Int(rng.next() % UInt64(height - 6)) + 3, c = Int(rng.next() % UInt64(width - 6)) + 3
                let amplitude = Float(rng.next() % 1000) / 1000 + 0.1
                // A peak and its mirror through the centre.
                let mr = 2 * cy - r, mc = 2 * cx - c
                if r >= 0 && r < height && c >= 0 && c < width { v[r * width + c] += amplitude }
                if mr >= 0 && mr < height && mc >= 0 && mc < width { v[mr * width + mc] += amplitude }
            }
            let expected = try interim(v, width: width, height: height, minDistance: 3, thresholdAbs: 0.2, centreExclusion: 4, count: 6)
            let got = try PeakLocalMax.find(
                v, rows: height, columns: width, minDistance: 3, thresholdAbs: 0.2, numPeaks: 6,
                symmetry: SpectrumSymmetry(centreRow: cy, centreColumn: cx, excludedRadius: 4))
            #expect(got == expected)
        }
    }

    @Test func plantedMirrorPairKeepsOneAndTheCentreCoreIsDropped() throws {
        let width = 21, height = 21
        var v = [Float](repeating: 0, count: width * height)
        v[10 * width + 10] = 9                       // the core
        v[14 * width + 12] = 5; v[6 * width + 8] = 5 // a pair, (+4, +2) and its mirror
        v[10 * width + 17] = 4; v[10 * width + 3] = 4 // a pair on the centre row
        let got = try PeakLocalMax.find(v, rows: height, columns: width, minDistance: 1, thresholdAbs: 1,
                                        symmetry: SpectrumSymmetry(centreRow: 10, centreColumn: 10, excludedRadius: 2))
        #expect(got == [PeakCoordinate(row: 14, column: 12), PeakCoordinate(row: 10, column: 17)])
        // Without the symmetry every peak is returned.
        #expect(try PeakLocalMax.find(v, rows: height, columns: width, minDistance: 1, thresholdAbs: 1).count == 5)
        #expect(throws: FilterError.parameter("excludedRadius")) {
            try PeakLocalMax.find(v, rows: height, columns: width, minDistance: 1, symmetry: SpectrumSymmetry(centreRow: 10, centreColumn: 10, excludedRadius: -1))
        }
    }
}

/// A small seeded generator so the random spectra reproduce.
struct SystemRandomNumberGeneratorSeeded {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state >> 11
    }
}
