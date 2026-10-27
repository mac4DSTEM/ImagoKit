// ObjectStatistics.arealDensity and numberDensity: the closed forms of mac4DSTEM's PrecipitateStatistics tests, the
// refusal rule, and planted truth: over every placement of a box in a frame the edge-corrected count averages to exactly one.
import Foundation
import ImagoCore
import Testing

@Suite("Areal density") struct ArealDensityTests {
    /// A label image from (label, pixel indices) pairs on a width x height frame.
    private func frame(_ width: Int, _ height: Int, _ objects: [(Int32, [Int])]) -> Segmentation.LabelImage {
        var labels = [Int32](repeating: 0, count: width * height)
        for (label, pixels) in objects { for i in pixels { labels[i] = label } }
        return Segmentation.LabelImage(width: width, height: height, labels: labels, count: Int(objects.map(\.0).max() ?? 0))
    }

    @Test func weightsEachCountedObjectByItsBoundingBox() throws {
        // 10 x 10 frame, 80 analysed pixels. A: one pixel (row 5, col 5): 100/(8*8). B: rows 2-3, cols 3-6 (4 x 2): 100/(5*7).
        let boxB: [Int] = [23, 24, 25, 26, 33, 34, 35, 36]
        let f = frame(10, 10, [(1, [55]), (2, boxB)])
        let d = try ObjectStatistics.arealDensity(labels: f, analysedPixels: 80, pixelSize: 1, pixelUnit: "nm")
        #expect(d.counted == 2 && d.edgeExcluded == 0)
        let expected: Double = (100.0 / 64 + 100.0 / 35) / 80
        let got: Double = try #require(d.arealDensity)
        #expect(abs(got - expected) <= 1e-12)
        #expect(d.validation == "none" && d.pixelUnit == "nm")
    }

    @Test func edgeObjectsAddNoWeightAndAreReported() throws {
        let f = frame(10, 10, [(1, [55]), (2, [0, 1])])
        let d = try ObjectStatistics.arealDensity(labels: f, pixelSize: 1)
        #expect(d.counted == 1 && d.edgeExcluded == 1)
        let expected: Double = (100.0 / 64) / 100
        let got: Double = try #require(d.arealDensity)
        #expect(abs(got - expected) <= 1e-12)
        // An accepted set that leaves the interior object out counts only the edge exclusion.
        let none = try ObjectStatistics.arealDensity(labels: f, accepted: [2], pixelSize: 1)
        #expect(none.counted == 0 && none.edgeExcluded == 1 && none.arealDensity == 0 && none.meanLength == nil)
    }

    @Test func noCalibrationNoDensity() throws {
        let f = frame(3, 3, [(1, [4])])
        for size in [nil, 0, -2.5, .nan, .infinity] as [Double?] {
            #expect(try ObjectStatistics.arealDensity(labels: f, pixelSize: size).arealDensity == nil, "pixel size \(String(describing: size))")
        }
        // The one interior pixel of a 3 x 3 frame: weight 9 / (1 * 1).
        let calibrated = try ObjectStatistics.arealDensity(labels: f, pixelSize: 3)
        let weight: Double = 9.0 / (9.0 * 9.0)
        #expect(abs(try #require(calibrated.arealDensity) - weight) <= 1e-12)
        // The count and the summaries do not need a scale.
        let uncalibrated = try ObjectStatistics.arealDensity(labels: f, pixelSize: nil)
        #expect(uncalibrated.counted == 1 && uncalibrated.meanLength == 1)
    }

    @Test func lengthAndWidthSummariesOfTheCountedObjects() throws {
        // Horizontal bars of 3, 5 and 4 pixels, one pixel high, inside a 12 x 8 frame, plus one edge bar that is not summarised.
        let bars: [(Int32, [Int])] = [(1, [14, 15, 16]), (2, [38, 39, 40, 41, 42]), (3, [62, 63, 64, 65]), (4, [0, 1, 2, 3, 4, 5, 6, 7])]
        let f = frame(12, 8, bars)
        let d = try ObjectStatistics.arealDensity(labels: f, pixelSize: 1)
        #expect(d.counted == 3 && d.edgeExcluded == 1)
        #expect(d.meanLength == 4 && d.medianLength == 4 && d.meanWidth == 1)
        let two = try ObjectStatistics.arealDensity(labels: f, accepted: [1, 2], pixelSize: 1)
        #expect(two.meanLength == 4 && two.medianLength == 4)
        let odd = try ObjectStatistics.arealDensity(labels: f, accepted: [1, 2, 3], pixelSize: 1)
        #expect(odd.medianLength == 4)
    }

    /// Planted truth: put a bx x by box with its top-left corner on every one of the W*H frame positions (clipped by the frame). The
    /// boxes that stay clear of the border are counted; the Miles–Lantuéjoul weights make the mean count over all placements exactly 1.
    @Test(arguments: [(1, 1), (4, 2), (3, 3), (2, 5)]) func theEdgeCorrectedCountAveragesToOne(_ box: (Int, Int)) throws {
        let (bx, by) = box
        let w = 12, h = 10
        var total = 0.0
        for r in 0..<h {
            for c in 0..<w {
                var pixels: [Int] = []
                for rr in r..<min(r + by, h) { for cc in c..<min(c + bx, w) { pixels.append(rr * w + cc) } }
                let d = try ObjectStatistics.arealDensity(labels: frame(w, h, [(1, pixels)]), pixelSize: 1)
                total += try #require(d.arealDensity) * Double(w * h)
            }
        }
        #expect(abs(total / Double(w * h) - 1) <= 1e-12, "box \(bx) x \(by)")
    }

    @Test func refusals() throws {
        let f = frame(3, 3, [(1, [4])])
        // Nothing analysed (a map with no indexed pixel): the counts stay, the density is nil.
        let none = try ObjectStatistics.arealDensity(labels: f, analysedPixels: 0, pixelSize: 1)
        #expect(none.counted == 1 && none.arealDensity == nil)
        #expect(throws: ArealDensityError.analysedPixels) { try ObjectStatistics.arealDensity(labels: f, analysedPixels: -1, pixelSize: 1) }
        // A hand-built label image that does not match its size is refused before the frame is multiplied.
        let broken = Segmentation.LabelImage(width: -1, height: 3, labels: [], count: 0)
        #expect(throws: Segmentation.SegmentationError.invalidLabelImage) { try ObjectStatistics.arealDensity(labels: broken, pixelSize: 1) }
        #expect(throws: ArealDensityError.analysedPixels) { try ObjectStatistics.arealDensity(labels: f, analysedPixels: 10, pixelSize: 1) }
        // A frame that does not fit a counted object (here an edge weight asked of an object spanning the whole frame).
        let wide = try #require(try Segmentation.objectMoments(frame(3, 3, [(1, [3, 4, 5])])).first)
        #expect(throws: ArealDensityError.frameDoesNotFitObject(label: 1)) { try ObjectStatistics.edgeWeight(wide, frameWidth: 3, frameHeight: 3) }
    }

    @Test func numberDensityPropagatesSigma() throws {
        let nv = try ObjectStatistics.numberDensity(arealDensity: 4e-3, thickness: 50)
        #expect(nv.value == 8e-5 && nv.sigma == nil && nv.validation == "none")
        // 1/nm^2 over nm; sigmas in quadrature: (0.2e-3/50)^2 + (4e-3 * 5 / 2500)^2.
        let s = try ObjectStatistics.numberDensity(arealDensity: 4e-3, arealDensitySigma: 0.2e-3, thickness: 50, thicknessSigma: 5)
        let partA: Double = 0.2e-3 / 50
        let partT: Double = 4e-3 * 5 / 2500
        let expected: Double = (partA * partA + partT * partT).squareRoot()
        #expect(abs(try #require(s.sigma) - expected) <= 1e-15)
        let thicknessOnly = try ObjectStatistics.numberDensity(arealDensity: 4e-3, thickness: 50, thicknessSigma: 5)
        #expect(abs(try #require(thicknessOnly.sigma) - 8e-5 * 0.1) <= 1e-15)
        #expect(try ObjectStatistics.numberDensity(arealDensity: 0, arealDensitySigma: 1e-3, thickness: 10).sigma == 1e-4)
        for bad in [0.0, -1, .nan, .infinity] {
            #expect(throws: ArealDensityError.invalidValue("thickness")) { try ObjectStatistics.numberDensity(arealDensity: 1, thickness: bad) }
        }
        #expect(throws: ArealDensityError.invalidValue("areal density")) { try ObjectStatistics.numberDensity(arealDensity: -1, thickness: 1) }
        #expect(throws: ArealDensityError.invalidValue("thickness sigma")) {
            try ObjectStatistics.numberDensity(arealDensity: 1, thickness: 1, thicknessSigma: -1)
        }
    }
}
