// Public ImagoCore calls refuse a bad parameter with a typed error, or answer without trapping, instead of crashing
// (macEMSuite/ImagoKit#78). One test per site; each trapped, looped or over-allocated before.
import Foundation
import ImagoCore
import Testing

@Suite struct CoreTrapRefusalTests {
    @Test func aLineProfileWhoseWidthWouldHoldGigabytesIsRefused() throws {
        let image = try Micrograph(width: 4, height: 4, pixels: [Float](repeating: 1, count: 16))
        #expect(throws: LineProfileError.invalidWidth(Int.max)) {
            try LineProfile.sample(image, from: PixelPoint(x: 0, y: 0), to: PixelPoint(x: 3, y: 3), linewidth: Int.max)
        }
        #expect(throws: LineProfileError.invalidWidth(1 << 26)) {
            try LineProfile.sample(image, from: PixelPoint(x: 0, y: 0), to: PixelPoint(x: 3, y: 3), linewidth: 1 << 26)
        }
        _ = try LineProfile.sample(image, from: PixelPoint(x: 0, y: 0), to: PixelPoint(x: 3, y: 3), linewidth: 3)
    }

    /// A radius far wider than a small image takes the image-sized walk (a 16 385² kernel of Doubles, 2 GB, before) and gives the
    /// background the kernel gives.
    @Test func aRollingBallFarWiderThanTheImageBuildsNoGiantKernel() throws {
        let image: [Double] = [3, 1, 4, 1, 5, 9, 2, 6, 5]
        let wide = try RollingBall.background(image: image, rows: 3, columns: 3, radius: 8000)
        let kernel = try RollingBall.background(image: image, rows: 3, columns: 3, kernel: RollingBall.ballKernel(radius: 8))
        let viaRadius8 = try RollingBall.background(image: image, rows: 3, columns: 3, radius: 8)
        #expect(viaRadius8 == kernel)
        #expect(wide.count == 9)
        #expect(wide.allSatisfy { $0.isFinite })
    }

    @Test func nextPowerOfTwoAbove2To62EndsInsteadOfLooping() {
        #expect(FFT2D.nextPow2(Int.max) == 1 << 62)
        #expect(FFT2DDouble.nextPow2((1 << 62) + 1) == 1 << 62)
        #expect(FFT2D.nextPow2(5) == 8)
    }

    @Test func anIntensityHistogramOfNoBinsIsNil() {
        #expect(IntensityHistogram.compute([1, 2, 3], bins: 0) == nil)
        #expect(IntensityHistogram.compute([1, 2, 3], bins: -4) == nil)
        #expect(IntensityHistogram.compute([1, 2, 3], bins: 4) != nil)
    }

    @Test func aLabelImageClaimingMoreObjectsThanPixelsIsRefused() {
        let labels = Segmentation.LabelImage(width: 2, height: 1, labels: [1, 0], count: Int.max)
        #expect(throws: Segmentation.SegmentationError.invalidLabelImage) { try ObjectStatistics.measure(image: [1, 2] as [Double], labels: labels) }
    }

    @Test func watershedMarkersOnAnOverflowingGridAreRefused() {
        #expect(throws: MorphologyError.shape) {
            try Segmentation.markers(peakRows: [0], peakColumns: [0], rows: Int.max, columns: 3)
        }
    }

    @Test func aThumbnailOrStructuringElementOfAnOverflowingSizeIsRefused() {
        #expect(throws: (any Error).self) { try StructuringElement(rows: Int.max, columns: 3, cells: [true]) }
        #expect(throws: (any Error).self) { try RollingBall.Kernel(rows: Int.max, columns: 3, values: [1]) }
    }

    @Test func aTidyCSVsDescriptionAndLineageLinesStayOneLineEach() throws {
        let source = TidyResultsCSV.Source(software: "t", fileName: "f", object: "o", width: 1, height: 1, x: nil, y: nil, notes: [])
        let row = TidyResultsCSV.Row(group: "g", id: "1", label: "l", product: "p", entries: [.init("area", 1)])
        let text = try #require(try TidyResultsCSV.text([row], source: source, title: "t", session: "s",
                                                        descriptionLines: ["# points: a\nb,c"], validations: [("p", "ok")],
                                                        lineageLine: "# lineage: x\r\ny"))
        let header = text.split(separator: "\n").filter { $0.hasPrefix("#") }
        #expect(header.contains("# points: a b,c"))
        #expect(header.contains("# lineage: x  y"))
        #expect(!text.contains("\nb,c"))
    }
}
