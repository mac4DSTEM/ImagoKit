// RadialChartScale.samples, drawable, nearest and powerAxisMarks against tools/radialchart/pins.py (an independent NumPy
// reading of the definition). Selection and one multiplication: compared bit for bit.
import Foundation
import ImagoFFT
import Testing

private struct ChartPins: Decodable, Sendable {
    struct Case: Decodable, Sendable {
        let width: Double, factor: Double
        let units: String
        let meanBits: [UInt64]
        let count: [Int]
        let binsOut: [Int]
        let qBits: [UInt64], meanOutBits: [UInt64]
        let countOut: [Int]
        let queries: [Double]
        let nearestIndex: [Int]
        let marks: [UInt64]
    }
    let cases: [Case]
    static func load() throws -> ChartPins {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/radial-chart-pins.json")
        return try JSONDecoder().decode(ChartPins.self, from: Data(contentsOf: url))
    }
}

@Suite("RadialChartSamples") struct RadialChartSamplesTests {
    @Test func samplesNearestAndMarksEqualThePins() throws {
        for c in try ChartPins.load().cases {
            let profile = try #require(RadialProfile(width: c.width, units: c.units,
                                                     mean: c.meanBits.map(Double.init(bitPattern:)), count: c.count))
            let got = RadialChartScale.samples(profile, factor: c.factor)
            #expect(got.map(\.bin) == c.binsOut)
            #expect(got.map { $0.q.bitPattern } == c.qBits)
            #expect(got.map { $0.mean.bitPattern } == c.meanOutBits)
            #expect(got.map(\.count) == c.countOut)
            for (x, index) in zip(c.queries, c.nearestIndex) {
                let near = RadialChartScale.nearest(got, toX: x)
                if index < 0 { #expect(near == nil) } else { #expect(near == got[index], "x = \(x)") }
            }
            let marks = RadialChartScale.powerAxisMarks(got, plotHeight: 48, minimumSpacing: 16)
            #expect(marks.map(\.bitPattern) == c.marks)
        }
    }

    @Test func planted() throws {
        let profile = try #require(RadialProfile(width: 0.5, units: "1/nm", mean: [9, 4, 0, 2, .nan, 1], count: [1, 3, 5, 0, 7, 2]))
        let s = RadialChartScale.samples(profile, factor: 2)
        // bin 1 (q 0.75 * 2), bin 2 has zero power, bin 3 no pixel, bin 4 NaN, bin 5 (q 2.75 * 2)
        #expect(s.map(\.bin) == [1, 5] && s.map(\.q) == [1.5, 5.5] && s.map(\.mean) == [4, 1] && s.map(\.count) == [3, 2])
        #expect(RadialChartScale.nearest(s, toX: 3.5)?.bin == 1)               // equal distances: the first
        #expect(RadialChartScale.nearest(s, toX: 4)?.bin == 5)
        #expect(RadialChartScale.nearest([], toX: 1) == nil && RadialChartScale.nearest(s, toX: .nan) == nil)
        #expect(RadialChartScale.powerAxisMarks([], plotHeight: 48, minimumSpacing: 16).isEmpty)
        #expect(RadialProfile(width: 0.5, units: "1/nm", mean: [1], count: [1, 2]) == nil)
        #expect(RadialProfile(width: 0, units: "1/nm", mean: [1], count: [1]) == nil)
        #expect(FFTMask.defaultRadius == 3 && FFTMask.defaultEdge == 2 && FFTMask.radiusRange == 1...32 && FFTMask.edgeRange == 0...16)
    }
}
