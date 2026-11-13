// Public ImagoFilters and ImagoFFT calls refuse a bad parameter with a typed error, or answer without trapping, instead of
// crashing (macEMSuite/ImagoKit#78). One test per site; each trapped or over-allocated before.
import Foundation
import ImagoCore
import ImagoFFT
import ImagoFilters
import Testing

@Suite struct FilterTrapRefusalTests {
    @Test func aTukeyWindowWithANaNAlphaIsRefused() throws {
        #expect(throws: FilterError.parameter("alpha")) { try WindowFunction.window1D(.tukey(alpha: .nan), count: 8) }
        #expect(try WindowFunction.window1D(.tukey(alpha: 0.5), count: 8).count == 8)
    }

    @Test func aWindowOfAnOverflowingShapeIsRefused() {
        #expect(throws: FilterError.parameter("shape")) { try WindowFunction.window2D(.hann, rows: Int.max / 2, columns: 3) }
    }

    /// A constant image beyond 2^24 (Float) or 2^53 (Double) loses the ±0.5 the histogram widens it by: no width to bin.
    @Test func histogramThresholdsOnALargeConstantImageAreRefused() {
        let floats = ThresholdPixels.float32([Float](repeating: 0x1p25, count: 16))
        let doubles = ThresholdPixels.float64([Double](repeating: 0x1p54, count: 16))
        for image in [floats, doubles] {
            #expect(throws: ThresholdError.self) { try Thresholding.yen(image) }
            #expect(throws: ThresholdError.self) { try Thresholding.isodata(image) }
            #expect(throws: ThresholdError.self) { try Thresholding.triangle(image) }
        }
    }

    @Test func integerThresholdsOverAnOverflowingRangeAreRefused() throws {
        let extreme = ThresholdPixels.integers([Int.min, Int.max, 0])
        #expect(throws: ThresholdError.parameter("integer range")) { try Thresholding.li(extreme) }
        #expect(throws: ThresholdError.parameter("integer range")) { try Thresholding.yen(extreme) }
        #expect(try Thresholding.li(.integers([4, 4, 4])) == 4)
    }

    @Test func aButterworthPaddingOfGigabytesIsRefused() {
        let image = [Double](repeating: 1, count: 4)
        #expect(throws: FilterError.cutoff) {
            try BandpassFilters.butterworth(image, rows: 2, columns: 2, npad: (1 << 20) - 1)
        }
    }

    @Test func theHRTEMRadialProfileOfFewerThanTwoPixelsIsEmpty() {
        #expect(HRTEMFilter.radialProfile([1], n: 1).isEmpty)
        #expect(HRTEMFilter.radialProfile([], n: -3).isEmpty)
        #expect(HRTEMFilter.rotationalAverage([7], n: 1) == [7])
        #expect(HRTEMFilter.radialProfile([Double](repeating: 1, count: 16), n: 4).count == 2)
    }

    @Test func aHanningWindowOfNoPointsIsEmpty() {
        #expect(SpotMeasurement.hanning(-1).isEmpty)
        #expect(SpotMeasurement.hanning(0).isEmpty)
        #expect(SpotMeasurement.hanning(1) == [1])
    }

    @Test func separableFiltersOnAnOverflowingShapeAreRefused() {
        #expect(throws: (any Error).self) { try Thresholding.hysteresis([1], width: Int.max, height: 2, low: 0, high: 1) }
    }
}
