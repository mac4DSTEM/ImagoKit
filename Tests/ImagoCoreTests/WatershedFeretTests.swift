// Closed forms for the K1 additions (suite task 36): the Feret summary of the counted objects, a watershed ridge between two seeds,
// and the distance transform of a line. The pinned comparisons with SciPy and scikit-image are in ImagoKitParityTests.
import Foundation
import ImagoCore
import Testing

@Suite("Watershed, distance transform, Feret summary") struct WatershedFeretTests {
    /// Three needles on a 20 x 12 frame: A 1 x 5 (row 2), B 1 x 9 (row 6), C 1 x 4 touching the right edge (row 10). Feret of a
    /// horizontal run of n pixels is n (the contour of the pixel centres spans n - 1 plus the pixel itself: scikit-image gives n).
    private func regions() throws -> [RegionProperties] {
        var labels = [Int32](repeating: 0, count: 20 * 12)
        for c in 2..<7 { labels[2 * 20 + c] = 1 }
        for c in 3..<12 { labels[6 * 20 + c] = 2 }
        for c in 16..<20 { labels[10 * 20 + c] = 3 }
        return try Segmentation.regionProperties(Segmentation.LabelImage(width: 20, height: 12, labels: labels, count: 3))
    }

    @Test func feretSummaryCountsEdgeAndSmallAndSummarisesTheRest() throws {
        let all = try regions()
        let s = ObjectStatistics.feretSummary(all)
        #expect(s.total == 3 && s.counted == 2 && s.edge == 1 && s.small == 0)
        let a = all[0].feretDiameterMax, b = all[1].feretDiameterMax
        #expect(s.medianFeret == (a + b) / 2 && s.meanFeret == (a + b) / 2, "two counted: the median is the mean of the two")
        let strict = ObjectStatistics.feretSummary(all, minimumArea: 6)
        #expect(strict.counted == 1 && strict.small == 1 && strict.medianFeret == b)
        let none = ObjectStatistics.feretSummary([])
        #expect(none.counted == 0 && none.medianFeret == nil && none.meanFeret == nil)
    }

    @Test func aRidgeBetweenTwoSeedsSplitsTheLine() throws {
        // Values 0 1 2 3 2 1 0 on one row; seeds at the two minima: the ridge pixel (value 3) goes to the first flooded side.
        let image: [Double] = [0, 1, 2, 3, 2, 1, 0]
        let labels = try Segmentation.watershed(image: image, markers: [1, 0, 0, 0, 0, 0, 2], width: 7, height: 1)
        #expect(Array(labels[0..<3]) == [1, 1, 1] && Array(labels[4...]) == [2, 2, 2])
        #expect(labels[3] == 1 || labels[3] == 2)
    }

    @Test func aMaskKeepsTheFloodInside() throws {
        let labels = try Segmentation.watershed(image: [0, 0, 0, 0], markers: [1, 0, 0, 0], width: 4, height: 1,
                                                mask: [true, true, false, true])
        #expect(labels == [1, 1, 0, 0], "the pixel behind the masked one is unreachable")
    }

    @Test func theDistanceOfALineOfBackgroundIsTheColumnOffset() throws {
        let mask = (0..<5).flatMap { _ in [false, true, true, true, true] }
        let d = try DistanceTransform.euclidean(foreground: mask, width: 5, height: 5)
        #expect(Array(d[0..<5]) == [0, 1, 2, 3, 4])
        let scaled = try DistanceTransform.euclidean(foreground: mask, width: 5, height: 5, sampling: (rows: 1, columns: 0.5))
        #expect(Array(scaled[0..<5]) == [0, 0.5, 1, 1.5, 2])
    }
}

@Suite("0.8.0 review fixes") struct ReviewFixTests {
    @Test func extremeParametersAreRefusedNotTrapped() throws {
        #expect(throws: MorphologyError.self) { try DistanceTransform.euclidean(foreground: [true, false], width: 2, height: 1, sampling: (1e-170, 1)) }
        #expect(throws: MorphologyError.self) { try DistanceTransform.euclidean(foreground: [true, false], width: 2, height: 1, sampling: (1, 1e155)) }
        #expect(throws: RestorationError.self) { try RollingBall.ballKernel(radius: 1e19) }
        #expect(throws: RestorationError.self) { try RollingBall.ellipsoidKernel(rows: 1 << 40, columns: 3, intensity: 1) }
    }

    @Test func aHugeRadiusOnASmallImageIsTheFlatCaseWithoutAllocatingTheKernel() throws {
        // radius beyond the kernel cap: every cell inside the image has dr, dc small relative to the radius, so the ball is
        // flat there to within r - sqrt(r^2 - d^2) ~ d^2 / 2r, which is below 1e-3 for a 4 x 4 image at radius 1e5.
        let image: [Double] = (0..<16).map { Double($0 % 4) }
        let bg = try RollingBall.background(image: image, width: 4, height: 4, radius: 1e5)
        #expect(bg.count == 16 && bg.allSatisfy { $0.isFinite && $0 <= 3.0 + 1e-3 })
        #expect(bg.min()! >= -1e-3)
    }
}
