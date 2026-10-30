// SPDX-License-Identifier: GPL-3.0-only
//
//  SpectrumDisplay.swift (ImagoFFT)
//  How a power spectrum is drawn: v = log₁₀(|F|² / (W·H)²), log because the power spans six to ten decades; the divisor makes
//  a lattice read the same in a region of any size, so a hand-set window survives a resize. The DC bin and every bin whose
//  power is zero or not finite are NaN, never −∞. Presentation: no measurement reads these values.
//
//  THE DEFAULT WINDOW: black at the median of the finite v, white at the largest finite v more than 3 bins (index distance)
//  from the centre, so the DC's skirt never sets the white. The median is read from a 2048-bin histogram of the finite v over
//  its range, linearly interpolated inside the bin that crosses half: within one bin width, (highest − lowest)/2048, of the
//  sorted median, in linear time.
//

import Foundation

public nonisolated struct SpectrumDisplay: Sendable {
    public struct Window: Sendable, Equatable {
        public let low: Float
        public let high: Float

        public init(low: Float, high: Float) {
            self.low = low
            self.high = high
        }
    }

    public let width: Int
    public let height: Int
    /// v per bin, row-major, the spectrum's centred layout; NaN where there is nothing to draw.
    public let values: [Float]
    /// nil when no bin is finite, or when nothing finite lies beyond 3 bins of the centre.
    public let defaultWindow: Window?

    public static let medianBins = 2048
    /// Bins within this index distance of the centre never set the default window's white.
    public static let centreRadius = 3.0

    /// v for every bin of `spectrum`, then the default window.
    public static func make(_ spectrum: PowerSpectrum) -> SpectrumDisplay {
        let power: [Float]
        switch spectrum.power {
        case .float32(let p): power = p
        case .float64(let p): power = p.map { Float($0) }
        }
        return make(power: power, width: spectrum.width, height: spectrum.height)
    }

    /// v for every bin of the centred Float32 `power` (`width` × `height`, row-major, zero frequency at (h/2, w/2)), then the
    /// default window. One pass finds the range of the finite v and the white; a second fills the 2048-bin histogram. Scalar
    /// Double; the log is the only transcendental. An array of the wrong length gives an all-NaN display and no window.
    public static func make(power: [Float], width w: Int, height h: Int) -> SpectrumDisplay {
        guard w >= 1, h >= 1, power.count == w * h else {
            return SpectrumDisplay(width: max(w, 0), height: max(h, 0), values: [], defaultWindow: nil)
        }
        let area = Double(w) * Double(h)
        let scale = area * area
        let centreRow = h / 2, centreCol = w / 2
        let discSquared = Int(centreRadius * centreRadius)          // index distance > 3  ⇔  dx² + dy² > 9

        var values = [Float](repeating: .nan, count: w * h)
        var lowest = Float.infinity, highest = -Float.infinity      // the range of the finite v
        var white = -Float.infinity                                  // the largest finite v beyond the disc
        var finiteCount = 0
        power.withUnsafeBufferPointer { power in
            values.withUnsafeMutableBufferPointer { v in
                for r in 0..<h {
                    let dy = r - centreRow
                    for c in 0..<w {
                        if r == centreRow && c == centreCol { continue }           // the DC stays NaN
                        let p = power[r * w + c]
                        guard p > 0, p.isFinite else { continue }                 // zero or not finite: NaN, never −∞
                        let x = Float(log10(Double(p) / scale))
                        v[r * w + c] = x
                        finiteCount += 1
                        lowest = min(lowest, x)
                        highest = max(highest, x)
                        let dx = c - centreCol
                        if dx * dx + dy * dy > discSquared { white = max(white, x) }
                    }
                }
            }
        }
        guard finiteCount > 0, white > -Float.infinity else {
            return SpectrumDisplay(width: w, height: h, values: values, defaultWindow: nil)
        }
        return SpectrumDisplay(width: w, height: h, values: values,
                               defaultWindow: Window(low: median(of: values, count: finiteCount, lowest: lowest, highest: highest),
                                                     high: white))
    }

    /// THE MEDIAN RULE. Over the `count` finite entries of `values`, which span `lowest`…`highest`: a histogram of
    /// `medianBins` equal bins over that range (the maximum counted in the last bin); the first bin whose cumulative count
    /// reaches n/2 is the one that holds the (⌈n/2⌉-th) smallest value; the median is placed inside it by linear
    /// interpolation, at the fraction (n/2 − count before the bin) / (count in the bin) of its width. A constant range
    /// (lowest = highest) is its value.
    private static func median(of values: [Float], count: Int, lowest: Float, highest: Float) -> Float {
        guard lowest < highest else { return lowest }
        let lo = Double(lowest)
        let binWidth = (Double(highest) - lo) / Double(medianBins)
        var counts = [Int](repeating: 0, count: medianBins)
        for x in values where !x.isNaN {
            counts[min(Int((Double(x) - lo) / binWidth), medianBins - 1)] += 1
        }
        let half = Double(count) / 2
        var below = 0
        for bin in 0..<medianBins {
            if Double(below + counts[bin]) >= half {
                let median = lo + (Double(bin) + (half - Double(below)) / Double(counts[bin])) * binWidth
                return Float(min(max(median, lo), Double(highest)))
            }
            below += counts[bin]
        }
        return highest
    }
}
