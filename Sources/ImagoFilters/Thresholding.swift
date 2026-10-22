//
//  Thresholding.swift (ImagoFilters)
//  scikit-image 0.26.0's `filters` thresholds on a 2D image: `threshold_otsu`, `_li`, `_yen`, `_triangle`, `_mean`, `_isodata`,
//  `_local` (the Gaussian and mean surfaces) and `apply_hysteresis_threshold`.
//
//  DTYPES. scikit-image computes in the image's dtype and the kit follows: an integer image (`.integers`) histograms by
//  `bincount` over its own range and returns an exact integer threshold; a Float32 image (`.float32`) histograms, subtracts and
//  divides in Float32; a Float64 image (`.float64`) in Double. The histogram is `exposure.histogram(image, nbins)`: for a float
//  image NumPy's `histogram` over [min, max] (`linspace` edges, the fast uniform-bin index with its edge corrections, the last bin
//  closed), centres `(edge[i] + edge[i+1]) / 2`; counts are cast to Float32 before the Otsu, Yen and ISODATA sums, as scikit-image
//  does, so the class weights are Float32 cumulative sums (exact for fewer than 2^24 pixels) and the rest is the dtype's arithmetic.
//
//  DEVIATIONS. A non-finite pixel is refused for every threshold but `li` (NaN is dropped and ±inf handled as scikit-image does);
//  scikit-image feeds NaN to its histogram and raises from NumPy. ISODATA with no fixed point throws where scikit-image raises an
//  IndexError. `threshold_mean` and the means inside `li` accumulate in Double and round once to the dtype (NumPy sums Float32 in
//  Float32 pairwise); the results agree within the rtol 1e-5 that scikit-image's Float32 numbers carry.
//

import Foundation
import ImagoCore

/// The pixels of a grey image in the dtype whose arithmetic scikit-image would use.
public nonisolated enum ThresholdPixels: Sendable, Equatable {
    case float32([Float])
    case float64([Double])
    /// Integer pixels (uint8, uint16, int16 ...): exact in Int, thresholds returned as exact integer values.
    case integers([Int])

    public var count: Int {
        switch self {
        case .float32(let v): v.count
        case .float64(let v): v.count
        case .integers(let v): v.count
        }
    }
}

public nonisolated enum ThresholdError: Error, Equatable, Sendable {
    /// No pixels, or a width and height that do not match the count.
    case shape
    /// A pixel is NaN or infinite (`li` handles them as scikit-image does).
    case nonFinite
    /// A parameter is out of range (`nbins` below 1, an even `blockSize`, a negative sigma ...).
    case parameter(String)
    /// ISODATA found no fixed point (scikit-image raises an IndexError).
    case noThreshold
}

public nonisolated enum Thresholding {
    // MARK: Histogram

    /// `exposure.histogram(image, nbins, source_range='image')`: counts and bin centres (centres exact in Double for Float32).
    package struct Histogram: Sendable, Equatable {
        package var counts: [Int]
        package var centers: [Double]
    }

    package static func histogram(_ image: ThresholdPixels, bins: Int) throws -> Histogram {
        guard image.count > 0 else { throw ThresholdError.shape }
        switch image {
        case .integers(let v):
            let lo = v.min()!, hi = v.max()!
            guard hi - lo < 1 << 26 else { throw ThresholdError.parameter("integer range") }
            var counts = [Int](repeating: 0, count: hi - lo + 1)
            for x in v { counts[x - lo] += 1 }
            return Histogram(counts: counts, centers: (lo...hi).map(Double.init))
        case .float32(let v):
            guard bins >= 1 else { throw ThresholdError.parameter("nbins") }
            let h = try floatHistogram(v, bins: bins)
            return Histogram(counts: h.counts, centers: h.centers.map(Double.init))
        case .float64(let v):
            guard bins >= 1 else { throw ThresholdError.parameter("nbins") }
            let h = try floatHistogram(v, bins: bins)
            return Histogram(counts: h.counts, centers: h.centers)
        }
    }

    /// NumPy's `histogram(a, bins=n)` over [min, max] for a float array, in the array's own float type.
    private static func floatHistogram<T: BinaryFloatingPoint>(_ a: [T], bins n: Int) throws -> (counts: [Int], centers: [T]) {
        guard a.allSatisfy(\.isFinite) else { throw ThresholdError.nonFinite }
        var first = a.min()!, last = a.max()!
        guard (last - first).isFinite else { throw ThresholdError.parameter("the pixel range overflows") }
        if first == last { first -= 0.5; last += 0.5 }
        // np.linspace(first, last, n + 1, dtype): y = arange * step + start, the last edge set to `last`.
        let step = (last - first) / T(n)
        var edges = (0...n).map { T($0) * step + first }
        edges[n] = last
        var counts = [Int](repeating: 0, count: n)
        let denominator = last - first
        let scale = T(n)
        for x in a where x >= first && x <= last {
            var index = Int(((x - first) / denominator) * scale)
            if index == n { index -= 1 }
            if x < edges[index] { index -= 1 }
            if x >= edges[index + 1] && index != n - 1 { index += 1 }
            counts[index] += 1
        }
        let centers = (0..<n).map { (edges[$0] + edges[$0 + 1]) / 2 }
        return (counts, centers)
    }

    /// `_validate_image_histogram` from a computed histogram: counts as Float32 (the image path trims nothing for a float image).
    private static func floatCounts(_ h: Histogram) -> [Float] { h.counts.map { Float($0) } }

    private static func firstPixel(_ image: ThresholdPixels) -> Double {
        switch image {
        case .float32(let v): Double(v[0])
        case .float64(let v): v[0]
        case .integers(let v): Double(v[0])
        }
    }

    private static func isConstant(_ image: ThresholdPixels) -> Bool {
        switch image {
        case .float32(let v): v.allSatisfy { $0 == v[0] }
        case .float64(let v): v.allSatisfy { $0 == v[0] }
        case .integers(let v): v.allSatisfy { $0 == v[0] }
        }
    }

    /// Whether the weights and class means of the Otsu, Yen and ISODATA formulas are Float32 (a Float32 image) or Double.
    private static func isSingle(_ image: ThresholdPixels) -> Bool {
        if case .float32 = image { return true }
        return false
    }

    // MARK: Otsu

    /// `threshold_otsu(image, nbins)`. A constant image returns its value.
    public static func otsu(_ image: ThresholdPixels, bins: Int = 256) throws -> Double {
        guard image.count > 0 else { throw ThresholdError.shape }
        if isConstant(image) { return firstPixel(image) }
        let h = try histogram(image, bins: bins)
        // scikit-image's argmax over an empty class-variance array raises for a single bin.
        guard h.centers.count > 1 else { throw ThresholdError.parameter("nbins") }
        return isSingle(image) ? otsuCore(floatCounts(h), h.centers.map(Float.init)) : otsuCore(floatCounts(h), h.centers)
    }

    private static func otsuCore<C: BinaryFloatingPoint>(_ counts: [Float], _ centers: [C]) -> Double {
        let n = counts.count
        guard n > 1 else { return Double(centers[0]) }
        var weight1 = [Float](repeating: 0, count: n), weight2 = [Float](repeating: 0, count: n)
        var run: Float = 0
        for i in 0..<n { run += counts[i]; weight1[i] = run }
        run = 0
        for i in stride(from: n - 1, through: 0, by: -1) { run += counts[i]; weight2[i] = run }
        var cum1 = [C](repeating: 0, count: n), cum2 = [C](repeating: 0, count: n)
        var acc: C = 0
        for i in 0..<n { acc += C(counts[i]) * centers[i]; cum1[i] = acc }
        acc = 0
        for i in stride(from: n - 1, through: 0, by: -1) { acc += C(counts[i]) * centers[i]; cum2[i] = acc }
        var best = -C.infinity, bestIndex = 0
        for i in 0..<(n - 1) {
            let mean1 = cum1[i] / C(weight1[i]), mean2 = cum2[i + 1] / C(weight2[i + 1])
            let d = mean1 - mean2
            let variance = C(weight1[i] * weight2[i + 1]) * (d * d)
            if i == 0 || variance > best { best = variance; bestIndex = i }
        }
        return Double(centers[bestIndex])
    }

    // MARK: Yen

    /// `threshold_yen(image, nbins)`.
    public static func yen(_ image: ThresholdPixels, bins: Int = 256) throws -> Double {
        guard image.count > 0 else { throw ThresholdError.shape }
        let h = try histogram(image, bins: bins)
        if h.centers.count == 1 { return h.centers[0] }
        let counts = floatCounts(h)
        let n = counts.count
        let total = Float(h.counts.reduce(0, +))
        let pmf = counts.map { $0 / total }
        var p1 = [Float](repeating: 0, count: n), p1sq = [Float](repeating: 0, count: n), p2sq = [Float](repeating: 0, count: n)
        var a: Float = 0, b: Float = 0
        for i in 0..<n { a += pmf[i]; p1[i] = a; b += pmf[i] * pmf[i]; p1sq[i] = b }
        b = 0
        for i in stride(from: n - 1, through: 0, by: -1) { b += pmf[i] * pmf[i]; p2sq[i] = b }
        var best = -Float.infinity, bestIndex = 0
        for i in 0..<(n - 1) {
            let product = p1sq[i] * p2sq[i + 1]
            let inverse = 1 / product
            let q = p1[i] * (1 - p1[i])
            let crit = Float(logf(inverse * (q * q)))
            if i == 0 || crit > best { best = crit; bestIndex = i }
        }
        return h.centers[bestIndex]
    }

    // MARK: ISODATA

    /// `threshold_isodata(image, nbins)`: the first fixed point of the iteration.
    public static func isodata(_ image: ThresholdPixels, bins: Int = 256) throws -> Double {
        try isodataAll(image, bins: bins)[0]
    }

    /// `threshold_isodata(image, nbins, return_all=True)`.
    public static func isodataAll(_ image: ThresholdPixels, bins: Int = 256) throws -> [Double] {
        guard image.count > 0 else { throw ThresholdError.shape }
        let h = try histogram(image, bins: bins)
        if h.centers.count == 1 { return h.centers }
        let r = isSingle(image) ? isodataCore(floatCounts(h), h.centers.map(Float.init)) : isodataCore(floatCounts(h), h.centers)
        guard !r.isEmpty else { throw ThresholdError.noThreshold }
        return r
    }

    private static func isodataCore<C: BinaryFloatingPoint>(_ counts: [Float], _ centers: [C]) -> [Double] {
        let n = counts.count
        var csuml = [Float](repeating: 0, count: n)
        var run: Float = 0
        for i in 0..<n { run += counts[i]; csuml[i] = run }
        let csumh = csuml.map { csuml[n - 1] - $0 }
        var csum = [C](repeating: 0, count: n)
        var acc: C = 0
        for i in 0..<n { acc += C(counts[i]) * centers[i]; csum[i] = acc }
        let binWidth = centers[1] - centers[0]
        var out: [Double] = []
        for i in 0..<(n - 1) {
            let lower = csum[i] / C(csuml[i])
            let higher = (csum[n - 1] - csum[i]) / C(csumh[i])
            let allMean = (lower + higher) / 2
            let distance = allMean - centers[i]
            if distance >= 0 && distance < binWidth { out.append(Double(centers[i])) }
        }
        return out
    }

    // MARK: Triangle

    /// `threshold_triangle(image, nbins)`.
    public static func triangle(_ image: ThresholdPixels, bins: Int = 256) throws -> Double {
        guard image.count > 0 else { throw ThresholdError.shape }
        let h = try histogram(image, bins: bins)
        let n = h.counts.count
        var hist = h.counts
        var peak = 0
        for i in 0..<n where hist[i] > hist[peak] { peak = i }
        let peakHeight = hist[peak]
        let occupied = hist.indices.filter { hist[$0] != 0 }
        var low = occupied.first!, high = occupied.last!
        if low == high { return firstPixel(image) }
        let flip = peak - low < high - peak
        var argPeak = peak
        if flip {
            hist.reverse()
            low = n - high - 1
            argPeak = n - peak - 1
        }
        high = 0
        let width = argPeak - low
        let norm = (Double(peakHeight) * Double(peakHeight) + Double(width) * Double(width)).squareRoot()
        let ph = Double(peakHeight) / norm, w = Double(width) / norm
        var best = -Double.infinity, bestIndex = 0
        for x in 0..<width {
            let length = ph * Double(x) - w * Double(hist[x + low])
            if length > best { best = length; bestIndex = x }
        }
        var level = bestIndex + low
        if flip { level = n - level - 1 }
        return h.centers[level]
    }

    // MARK: Mean

    /// `threshold_mean(image)`: the mean, accumulated in Double.
    public static func mean(_ image: ThresholdPixels) throws -> Double {
        guard image.count > 0 else { throw ThresholdError.shape }
        switch image {
        case .float32(let v): return Double(Float(v.reduce(0.0) { $0 + Double($1) } / Double(v.count)))
        case .float64(let v): return v.reduce(0.0, +) / Double(v.count)
        case .integers(let v): return Double(v.reduce(0, +)) / Double(v.count)
        }
    }

    // MARK: Li

    /// `threshold_li(image)` (default tolerance and initial guess): NaN pixels dropped, a constant image returns its value,
    /// an image with only ±inf returns 0. `nan` for an image of NaN only.
    public static func li(_ image: ThresholdPixels) throws -> Double {
        guard image.count > 0 else { throw ThresholdError.shape }
        switch image {
        case .float32(let v): return try liFloat(v)
        case .float64(let v): return try liFloat(v)
        case .integers(let v): return liInteger(v)
        }
    }

    private static func liFloat<T: BinaryFloatingPoint>(_ input: [T]) throws -> Double {
        var v = input.filter { !$0.isNaN }
        if v.isEmpty { return .nan }
        if v.allSatisfy({ $0 == v[0] }) { return Double(v[0]) }
        v = v.filter { $0.isFinite }
        if v.isEmpty { return 0 }
        let minimum = v.min()!
        for i in v.indices { v[i] -= minimum }
        let unique = Array(Set(v)).sorted()
        // np.min(np.diff(np.unique(image))) raises on an image whose finite pixels are all equal (it still holds an inf).
        guard unique.count > 1 else { throw ThresholdError.noThreshold }
        var smallest = T.infinity
        for i in 1..<unique.count { smallest = min(smallest, unique[i] - unique[i - 1]) }
        let tolerance = smallest / 2
        func average(_ values: [T]) -> T {
            values.isEmpty ? T.nan : T(values.reduce(0.0) { $0 + Double($1) } / Double(values.count))
        }
        var next = average(v)
        var current = -2 * tolerance
        while abs(next - current) > tolerance {
            current = next
            let foreground = v.filter { $0 > current }
            let background = v.filter { !($0 > current) }
            let meanFore = average(foreground), meanBack = average(background)
            if meanBack == 0 { break }
            next = T((Double(meanBack) - Double(meanFore)) / (log(Double(meanBack)) - log(Double(meanFore))))
        }
        return Double(next + minimum)
    }

    private static func liInteger(_ input: [Int]) -> Double {
        let minimum = input.min()!
        if input.allSatisfy({ $0 == input[0] }) { return Double(input[0]) }
        let shifted = input.map { $0 - minimum }
        let tolerance = 0.5
        var hist = [Float](repeating: 0, count: shifted.max()! + 1)
        for x in shifted { hist[x] += 1 }
        let centers = (0..<hist.count).map(Double.init)
        var next = Double(shifted.reduce(0, +)) / Double(shifted.count)
        var current = -2 * tolerance
        while abs(next - current) > tolerance {
            current = next
            var sumF = 0.0, weightF = 0.0, sumB = 0.0, weightB = 0.0
            for i in 0..<hist.count {
                if centers[i] > current { sumF += centers[i] * Double(hist[i]); weightF += Double(hist[i]) }
                else { sumB += centers[i] * Double(hist[i]); weightB += Double(hist[i]) }
            }
            let meanFore = sumF / weightF, meanBack = sumB / weightB
            if meanBack == 0 { break }
            next = (meanBack - meanFore) / (log(meanBack) - log(meanFore))
        }
        return next + Double(minimum)
    }

    // MARK: Local

    /// How `threshold_local` smooths: `gaussian` (sigma (block − 1) / 6 per axis, or `param`) or the `mean` box.
    public enum LocalMethod: Sendable, Equatable {
        case gaussian(sigma: Double? = nil)
        case mean
    }

    /// `threshold_local(image, block_size, method, offset, mode='reflect', cval)`: the threshold surface, `smoothed − offset`, in the
    /// image's float type (integers compute in Double). The mask is `image > surface`.
    public static func local(_ image: ThresholdPixels, width: Int, height: Int, blockSize: Int, method: LocalMethod = .gaussian(),
                             offset: Double = 0, mode: BoundaryMode = .reflect, cval: Double = 0) throws -> [Double] {
        guard width > 0, height > 0, image.count == width * height else { throw ThresholdError.shape }
        guard blockSize >= 1, blockSize % 2 == 1 else { throw ThresholdError.parameter("block_size must be odd") }
        func surface<S: BinaryFloatingPoint & Sendable>(_ pixels: [S]) throws -> [Double] {
            let smoothed: [S]
            switch method {
            case .gaussian(let sigma):
                let s = sigma ?? Double(blockSize - 1) / 6
                smoothed = try SeparableFilters.gaussianFilter(pixels, rows: height, columns: width, sigma: (s, s), mode: mode, cval: cval)
            case .mean:
                smoothed = try SeparableFilters.uniformFilter(pixels, rows: height, columns: width,
                                                              size: (blockSize, blockSize), mode: mode, cval: cval)
            }
            return smoothed.map { Double($0 - S(offset)) }
        }
        do {
            switch image {
            case .float32(let v): return try surface(v)
            case .float64(let v): return try surface(v)
            case .integers(let v): return try surface(v.map(Double.init))
            }
        } catch let e as FilterError {
            throw ThresholdError.parameter("\(e)")
        }
    }

    // MARK: Hysteresis

    /// `apply_hysteresis_threshold(image, low, high)` on an image of any dtype: `low` and `high` are cast to the image's type before the
    /// comparison (a Float32 image compares in Float32), as NumPy does.
    public static func hysteresis(_ image: ThresholdPixels, width: Int, height: Int, low: Double, high: Double) throws -> [Bool] {
        switch image {
        case .float64(let v): return try hysteresis(v, width: width, height: height, low: low, high: high)
        case .integers(let v): return try hysteresis(v.map(Double.init), width: width, height: height, low: low, high: high)
        case .float32(let v):
            // Compare in Float32: cast the pixels to Double (exact) and the thresholds through Float.
            return try hysteresis(v.map(Double.init), width: width, height: height, low: Double(Float(low)), high: Double(Float(high)))
        }
    }

    /// `apply_hysteresis_threshold(image, low, high)`: the pixels above `low` connected (4-neighbour, `ndimage.label`'s default
    /// structure) to a pixel above `high`; `low` is clipped to at most `high`.
    public static func hysteresis(_ values: [Double], width: Int, height: Int, low: Double, high: Double) throws -> [Bool] {
        guard width > 0, height > 0, values.count == width * height else { throw ThresholdError.shape }
        let lo = min(low, high)
        let maskLow = values.map { $0 > lo }
        let labels = try Segmentation.label(maskLow, width: width, height: height, connectivity: .four)
        var connected = [Bool](repeating: false, count: labels.count + 1)
        for (i, l) in labels.labels.enumerated() where l > 0 && values[i] > high { connected[Int(l)] = true }
        return labels.labels.map { $0 > 0 && connected[Int($0)] }
    }
}
