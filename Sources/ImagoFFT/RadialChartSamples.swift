// SPDX-License-Identifier: GPL-3.0-only
//
//  RadialChartSamples.swift (ImagoFFT)
//  The radial profile as a chart draws it, and the defaults of the spot mask.
//
//  SAMPLES: the profile's drawn bins (`RadialProfile.points`: k ≥ 1 with a pixel, at their centres), |q| multiplied by the
//  shown unit's factor (`FFTUnits.spectrumUnit`; one multiplication), the mean keeping its value (a power). A log axis can
//  draw a bin only with a positive, finite mean, so `drawable` drops the others (a noise-free plant has bins of zero power).
//  `nearest` is the bin whose centre is closest to a pointer's |q| (the first of equal distances; nil without a bin).
//  All of it is selection and one multiplication: tested exact.
//

import Foundation

extension RadialChartScale {
    /// One bin as charted.
    public struct Sample: Sendable, Equatable {
        public let bin: Int
        /// The bin's centre, (k + ½)·width, in the shown unit.
        public let q: Double
        /// The mean |F|² of the bin.
        public let mean: Double
        public let count: Int

        public init(bin: Int, q: Double, mean: Double, count: Int) {
            self.bin = bin; self.q = q; self.mean = mean; self.count = count
        }
    }

    /// The profile's drawn bins as charted: |q| times `factor`, then only those a log axis can draw (`drawable`).
    public static func samples(_ profile: RadialProfile, factor: Double) -> [Sample] {
        drawable(profile.points.map { Sample(bin: $0.bin, q: $0.q * factor, mean: $0.mean, count: $0.count) })
    }

    /// The samples with a positive, finite mean.
    public static func drawable(_ samples: [Sample]) -> [Sample] {
        samples.filter { $0.mean > 0 && $0.mean.isFinite }
    }

    /// The decades the log power axis is labelled at (`decadeMarks` of the samples' smallest and largest mean); empty without a sample.
    public static func powerAxisMarks(_ samples: [Sample], plotHeight: Double, minimumSpacing: Double) -> [Double] {
        guard let low = samples.map(\.mean).min(), let high = samples.map(\.mean).max() else { return [] }
        return decadeMarks(lowest: low, highest: high, plotHeight: plotHeight, minimumSpacing: minimumSpacing)
    }

    /// The sample whose centre is nearest to `x`; the first of equal distances; nil without a sample or when `x` is NaN.
    public static func nearest(_ samples: [Sample], toX x: Double) -> Sample? {
        guard !x.isNaN else { return nil }
        var best: Sample?
        var bestDistance = Double.infinity
        for s in samples {
            let d = abs(s.q - x)
            if best == nil || d < bestDistance { best = s; bestDistance = d }
        }
        return best
    }
}

/// The defaults and ranges of the masked inverse FFT's spot mask, in bins (the mask itself is
/// `SpotMeasurement.maskSpot(radius:edge:)`).
public nonisolated enum FFTMask {
    public static let defaultRadius: Double = 3
    public static let defaultEdge: Double = 2
    public static let radiusRange: ClosedRange<Double> = 1...32
    public static let edgeRange: ClosedRange<Double> = 0...16
}
