// SPDX-License-Identifier: GPL-3.0-only
//
//  NativeHistogramTests.swift
//  ImagoKit#85: the histogram of an 8-bit, 16-bit or signed 16-bit image, counted on the stored values, equals the Float32
//  route on the same pixels widened, bin for bin, with and without a range (macEM#566 keeps images at their own bit depth).
//

import Testing
import ImagoCore

@Suite("Native-type histogram") struct NativeHistogramTests {

    /// A seeded pseudo-random sequence (SplitMix64), so the planted images are the same on every run.
    private static func values(_ count: Int, seed: UInt64) -> [UInt64] {
        var state = seed
        return (0..<count).map { _ in
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    private static func expectSame(_ native: IntensityHistogram, _ widened: IntensityHistogram) {
        #expect(native.counts == widened.counts)
        #expect(native.edges == widened.edges)
        #expect(native.finiteCount == widened.finiteCount)
        #expect(native.minimum == widened.minimum)
        #expect(native.maximum == widened.maximum)
        #expect(native.mean == widened.mean)
    }

    private static let ranges: [(Float, Float)?] = [nil, (10, 200), (-3.5, 70_000.25)]
    private static let binCounts = [1, 7, 10, 256, 1000]

    @Test func eightBitEqualsTheFloat32Route() throws {
        let pixels = Self.values(50_000, seed: 85).map { UInt8(truncatingIfNeeded: $0) }
        let widened = pixels.map(Float.init)
        for bins in Self.binCounts {
            for range in Self.ranges {
                Self.expectSame(try IntensityHistogram.histogram(stored: pixels, bins: bins, range: range),
                                try IntensityHistogram.histogram(widened, bins: bins, range: range))
            }
        }
        // One bin per level counts each level exactly.
        let levels = try IntensityHistogram.histogram(stored: pixels, bins: 256, range: (-0.5, 255.5))
        for level in 0..<256 { #expect(levels.counts[level] == pixels.lazy.filter { Int($0) == level }.count) }
    }

    @Test func sixteenBitEqualsTheFloat32Route() throws {
        let pixels = Self.values(50_000, seed: 8516).map { UInt16(truncatingIfNeeded: $0) }
        let widened = pixels.map(Float.init)
        for bins in Self.binCounts {
            for range in Self.ranges {
                Self.expectSame(try IntensityHistogram.histogram(stored: pixels, bins: bins, range: range),
                                try IntensityHistogram.histogram(widened, bins: bins, range: range))
            }
        }
    }

    @Test func signedSixteenBitEqualsTheFloat32Route() throws {
        let pixels = Self.values(50_000, seed: 8517).map { Int16(truncatingIfNeeded: $0) }
        let widened = pixels.map(Float.init)
        for bins in Self.binCounts {
            for range in Self.ranges {
                Self.expectSame(try IntensityHistogram.histogram(stored: pixels, bins: bins, range: range),
                                try IntensityHistogram.histogram(widened, bins: bins, range: range))
            }
        }
    }

    @Test func aConstantImageAndTheRefusalsMatchToo() throws {
        let constant = [UInt16](repeating: 4095, count: 100)
        Self.expectSame(try IntensityHistogram.histogram(stored: constant, bins: 10),
                        try IntensityHistogram.histogram(constant.map(Float.init), bins: 10))
        #expect(throws: HistogramError.invalidBins(0)) { try IntensityHistogram.histogram(stored: constant, bins: 0) }
        #expect(throws: HistogramError.rangeOrder) { try IntensityHistogram.histogram(stored: constant, range: (5, 1)) }
        #expect(IntensityHistogram.compute(stored: [UInt8](), bins: 4) == nil)
        let a = try #require(IntensityHistogram.compute(stored: [UInt8](arrayLiteral: 0, 1, 2, 3), bins: 3))
        let b = try #require(IntensityHistogram.compute([Float](arrayLiteral: 0, 1, 2, 3), bins: 3))
        Self.expectSame(a, b)
    }
}
