//
//  IntensityHistogram.swift (ImagoCore)
//  `numpy.histogram(f, bins, range)` of an image's finite pixels, bin for bin, in Float32 like NumPy 2.5.4 with float32 data.
//
//  NUMPY'S ALGORITHM (numpy/lib/_histograms_impl.py `_get_outer_edges`, `_get_bin_edges` and the uniform-bins path of
//  `histogram`), every step in FLOAT32 because a float32 array hands NumPy float32 scalars for its range:
//    1. a range with first == last is widened to (first − 0.5, last + 0.5), in float32; at a magnitude where ±0.5 is lost
//       to rounding (≥ 2^24) the range stays a point and step 3 refuses it.
//    2. the edges are `np.linspace(first, last, bins + 1, dtype=float32)`: step = (last − first)/bins, edge i =
//       Float(i)·step + first, the last edge set to `last` exactly; when step underflows to 0 (a denormal range)
//       edge i = Float(i)/bins·(last − first) + first instead.
//    3. edges that are not strictly increasing in float32 are refused: NumPy raises ValueError("Too many bins for data
//       range").
//    4. a pixel's bin is the float32 index ((p − first)/(last − first))·bins, truncated, then corrected against the edges:
//       a value on `last` moves to bins − 1; ONE decrement if p < edges[i]; then ONE increment if p ≥ edges[i + 1] and i is
//       not the last bin. NumPy ignores a value outside [first, last]. In a denormal range of some hundreds of units one
//       correction each way is not always enough and NumPy's counts disagree with its own edges; the port reproduces that.
//
//  The input is `a[np.isfinite(a)]`: NaN and ±inf are not counted anywhere (NumPy raises on a non-finite autodetected
//  range; with no finite pixel it returns the all-zero histogram on (0, 1), as for an empty array). The extremes order −0.0 below +0.0 as NumPy's `min` and `max` do.
//
//  DEVIATION (overflow): an overflowing `last − first` makes NumPy's edges non-finite and it carries on with them; here it
//  is refused like a range that is too fine for its bins.
//  DEVIATION (mean): `mean` is not part of NumPy's histogram; it is the Double mean of the finite pixels, summed in blocks of
//  4096 (error ≤ ~(4096 + n/4096)·2⁻⁵³ of Σ|p|; tests hold it to 1e-9 of the larger of |mean| and mean |p|).
//

import Foundation

/// Why a histogram cannot be made (NumPy raises the same conditions as `ValueError`).
public nonisolated enum HistogramError: Error, Equatable, Sendable {
    /// `bins` below 1.
    case invalidBins(Int)
    /// A supplied range with a non-finite end.
    case nonFiniteRange
    /// A supplied range whose first value is above its last.
    case rangeOrder
    /// The edges are not strictly increasing in float32 ("Too many bins for data range").
    case tooManyBins
}

public nonisolated struct IntensityHistogram: Sendable, Equatable {
    /// Pixels per bin: `bins` entries, bin k holding [edges[k], edges[k + 1]) and the last bin closed. With the default
    /// range the counts add up to `finiteCount`. EMPTY when `compute` was given a range NumPy refuses (see `compute`).
    public let counts: [Int]
    /// NumPy's `bin_edges`: `bins + 1` float32 values, strictly increasing. Empty exactly when `counts` is.
    public let edges: [Float]
    /// The smallest and largest finite pixel, as the data holds them.
    public let minimum: Float
    public let maximum: Float
    /// How many pixels are finite.
    public let finiteCount: Int
    /// The mean of the finite pixels, summed in Double.
    public let mean: Double

    /// The histogram's range after any widening of a constant image; the data's extremes when there are no edges.
    public var firstEdge: Float { edges.first ?? minimum }
    public var lastEdge: Float { edges.last ?? maximum }

    /// `numpy.histogram(a[np.isfinite(a)], bins=bins, range=range)`, with NumPy's own default of 10 bins; `range` nil is
    /// (min, max) of the finite pixels. Throws what NumPy raises.
    public static func histogram(_ pixels: [Float], bins: Int = 10, range: (Float, Float)? = nil) throws -> IntensityHistogram {
        guard bins >= 1 else { throw HistogramError.invalidBins(bins) }
        let stats = statistics(pixels)
        // No finite pixel: NumPy's empty array has the range (0, 1) and all-zero counts; the extremes are NaN here.
        var first = stats.finite > 0 ? stats.lowest : 0, last = stats.finite > 0 ? stats.highest : 1
        if let range {
            guard range.0.isFinite, range.1.isFinite else { throw HistogramError.nonFiniteRange }
            guard range.0 <= range.1 else { throw HistogramError.rangeOrder }
            (first, last) = range
        }
        if first == last {                              // `_get_outer_edges`: a point range is widened, in float32
            first = first - 0.5
            last = last + 0.5
        }
        guard let edges = linspace(first, last, bins) else { throw HistogramError.tooManyBins }
        // `last − first` is finite and positive here (the edges are strictly increasing and the last is `last`), and a
        // kept p lies in [first, last], so the quotient is in [0, 1] and the index in [0, bins]: `Int(_)` cannot trap.
        let denominator = last - first
        let scale = Float(bins)
        var counts = [Int](repeating: 0, count: bins)
        for p in pixels where p.isFinite && p >= first && p <= last {
            var i = Int(((p - first) / denominator) * scale)
            if i == bins { i -= 1 }                     // a value on the last edge
            if p < edges[i] { i -= 1 }                  // the index is not exact within an ulp of an edge
            if p >= edges[i + 1] && i != bins - 1 { i += 1 }
            counts[i] += 1
        }
        return IntensityHistogram(counts: counts, edges: edges, minimum: stats.finite > 0 ? stats.lowest : .nan,
                                  maximum: stats.finite > 0 ? stats.highest : .nan, finiteCount: stats.finite,
                                  mean: stats.mean)
    }

    /// The same histogram without the exceptions, for a caller that wants the statistics even when NumPy refuses the range:
    /// `counts` and `edges` are EMPTY when `histogram` would throw `tooManyBins`, and nil is returned when no pixel is
    /// finite (NumPy returns an all-zero histogram on (0, 1) there). Other errors (`bins` below 1, a bad supplied range) are programming errors and trap.
    public static func compute(_ pixels: [Float], bins: Int, range: (Float, Float)? = nil) -> IntensityHistogram? {
        precondition(bins >= 1, "IntensityHistogram: \(bins) bins")
        guard pixels.contains(where: \.isFinite) else { return nil }
        do {
            return try histogram(pixels, bins: bins, range: range)
        } catch HistogramError.tooManyBins {
            let s = statistics(pixels)
            return IntensityHistogram(counts: [], edges: [], minimum: s.lowest, maximum: s.highest,
                                      finiteCount: s.finite, mean: s.mean)
        } catch {
            preconditionFailure("IntensityHistogram: \(error)")
        }
    }

    private static func statistics(_ pixels: [Float]) -> (lowest: Float, highest: Float, finite: Int, mean: Double) {
        var lowest = Float.infinity, highest = -Float.infinity
        var finite = 0
        var total = 0.0, block = 0.0
        for p in pixels where p.isFinite {
            if p < lowest || (p == lowest && p.sign == .minus) { lowest = p }
            if p > highest || (p == highest && p.sign == .plus) { highest = p }
            block += Double(p)
            finite += 1
            if finite & 0xFFF == 0 { total += block; block = 0 }
        }
        total += block
        return (lowest, highest, finite, finite > 0 ? total / Double(finite) : 0)
    }

    /// `np.linspace(first, last, bins + 1, endpoint=True, dtype=float32)` for float32 `first` and `last`, nil when the
    /// edges are not strictly increasing (NumPy's refusal; a NaN edge is refused too).
    private static func linspace(_ first: Float, _ last: Float, _ bins: Int) -> [Float]? {
        let division = Float(bins)
        let delta = last - first
        let step = delta / division
        var edges = [Float](repeating: 0, count: bins + 1)
        for i in 0...bins {
            let y = Float(i)
            edges[i] = (step == 0 ? y / division * delta : y * step) + first
        }
        edges[bins] = last
        for i in 0..<bins where !(edges[i] < edges[i + 1]) { return nil }
        return edges
    }
}
