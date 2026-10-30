// SPDX-License-Identifier: GPL-3.0-only
//
//  ObjectStatistics.swift (ImagoCore)
//  SciPy 1.18.1's labelled measurements over a `LabelImage` and any same-sized image: `ndimage.sum`, `mean`,
//  `standard_deviation`, `minimum`, `maximum`, `minimum_position`, `maximum_position`, `center_of_mass` and `find_objects`, for
//  every object 1...count at once.
//
//  DEFINITIONS (SciPy's, in Double, accumulated in row-major pixel order as `np.bincount` does): `sum` = Σ value; `mean` = sum / count;
//  `standardDeviation` = √(Σ (value − mean)² / count), the two-pass population form `ndimage.variance` uses (`_stats(centered=True)`),
//  so there is no cancellation; `centerOfMass` = (Σ value·row, Σ value·column) / Σ value in image coordinates; the positions are the
//  (row, column) of the first row-major pixel holding the minimum or maximum (SciPy's tie order depends on an unstable sort; the
//  value is the same). A label with no pixel has no entry; `findObjects` keeps a nil slot for it, as SciPy's `None`.
//
//  DEVIATION: `regionprops` forms `centroid_weighted` over the whole bounding-box crop (a NaN or infinity outside the object, inside
//  its box, turns the centroid NaN); the kit sums the object's own pixels only, so such a pixel does not reach it.
//
//  ACCURACY: counts, positions and bounding boxes exact; sums, means and deviations are SciPy's float64 values to rounding
//  (SciPy's own tests use decimal 6).
//

import Foundation

public nonisolated enum ObjectStatistics {
    /// One object's statistics of `image` under its pixels.
    public struct Result: Sendable, Equatable {
        public let label: Int32
        /// Pixel count.
        public let count: Int
        public let sum: Double
        public let mean: Double
        /// Population standard deviation (ddof 0).
        public let standardDeviation: Double
        public let minimum: Double
        public let maximum: Double
        /// (row, column) of the first row-major pixel holding the minimum.
        public let minimumPosition: (row: Int, column: Int)
        public let maximumPosition: (row: Int, column: Int)
        /// (row, column) of the intensity-weighted centroid in image coordinates; NaN or infinite when `sum` is zero.
        public let centerOfMass: (row: Double, column: Double)

        public static func == (a: Result, b: Result) -> Bool {
            a.label == b.label && a.count == b.count && a.sum == b.sum && a.mean == b.mean
                && a.standardDeviation == b.standardDeviation && a.minimum == b.minimum && a.maximum == b.maximum
                && a.minimumPosition == b.minimumPosition && a.maximumPosition == b.maximumPosition
                && a.centerOfMass.row == b.centerOfMass.row && a.centerOfMass.column == b.centerOfMass.column
        }
    }

    /// `find_objects`: the half-open rows and columns of an object's bounding box (SciPy's slices).
    public struct BoundingBox: Sendable, Equatable {
        public let rows: Range<Int>
        public let columns: Range<Int>
    }

    /// The statistics of `image` (row-major, `labels.width * labels.height` values) under every present object, in label order.
    public static func measure(image: [Double], labels: Segmentation.LabelImage) throws -> [Result] {
        try validate(labels, imageCount: image.count)
        let n = labels.count, w = labels.width
        var count = [Int](repeating: 0, count: n + 1)
        var sum = [Double](repeating: 0, count: n + 1)
        var weightedRow = [Double](repeating: 0, count: n + 1), weightedColumn = [Double](repeating: 0, count: n + 1)
        var minimum = [Double](repeating: .infinity, count: n + 1), maximum = [Double](repeating: -.infinity, count: n + 1)
        var minimumAt = [Int](repeating: -1, count: n + 1), maximumAt = [Int](repeating: -1, count: n + 1)
        for (i, l32) in labels.labels.enumerated() where l32 > 0 {
            let l = Int(l32), v = image[i]
            count[l] += 1
            sum[l] += v
            weightedRow[l] += v * Double(i / w)
            weightedColumn[l] += v * Double(i % w)
            // A NaN in the object makes its minimum and maximum NaN, as `np.min` and `np.max` do (it wins wherever it sits).
            if minimumAt[l] < 0 || (!minimum[l].isNaN && (v < minimum[l] || v.isNaN)) { minimum[l] = v; minimumAt[l] = i }
            if maximumAt[l] < 0 || (!maximum[l].isNaN && (v > maximum[l] || v.isNaN)) { maximum[l] = v; maximumAt[l] = i }
        }
        var centred = [Double](repeating: 0, count: n + 1)
        for (i, l32) in labels.labels.enumerated() where l32 > 0 {
            let l = Int(l32)
            let d = image[i] - sum[l] / Double(count[l])
            centred[l] += d * d
        }
        return (1...max(n, 1)).compactMap { l in
            guard l <= n, count[l] > 0 else { return nil }
            let c = Double(count[l])
            return Result(
                label: Int32(l), count: count[l], sum: sum[l], mean: sum[l] / c, standardDeviation: (centred[l] / c).squareRoot(),
                minimum: minimum[l], maximum: maximum[l],
                minimumPosition: (minimumAt[l] / w, minimumAt[l] % w), maximumPosition: (maximumAt[l] / w, maximumAt[l] % w),
                centerOfMass: (weightedRow[l] / sum[l], weightedColumn[l] / sum[l]))
        }
    }

    /// The same for a Float32 image (the values are widened exactly; the arithmetic is Double).
    public static func measure(image: [Float], labels: Segmentation.LabelImage) throws -> [Result] {
        try measure(image: image.map(Double.init), labels: labels)
    }

    /// `ndimage.find_objects(labels)`: the bounding box of each label 1...count, nil for a label with no pixel.
    public static func findObjects(_ labels: Segmentation.LabelImage) throws -> [BoundingBox?] {
        try validate(labels, imageCount: labels.labels.count)
        let n = labels.count, w = labels.width
        var minRow = [Int](repeating: Int.max, count: n + 1), maxRow = [Int](repeating: -1, count: n + 1)
        var minColumn = [Int](repeating: Int.max, count: n + 1), maxColumn = [Int](repeating: -1, count: n + 1)
        for (i, l32) in labels.labels.enumerated() where l32 > 0 {
            let l = Int(l32), r = i / w, c = i % w
            minRow[l] = min(minRow[l], r); maxRow[l] = max(maxRow[l], r)
            minColumn[l] = min(minColumn[l], c); maxColumn[l] = max(maxColumn[l], c)
        }
        guard n > 0 else { return [] }
        return (1...n).map { l in
            maxRow[l] < 0 ? nil : BoundingBox(rows: minRow[l]..<(maxRow[l] + 1), columns: minColumn[l]..<(maxColumn[l] + 1))
        }
    }

    private static func validate(_ labels: Segmentation.LabelImage, imageCount: Int) throws {
        let (product, overflow) = labels.width.multipliedReportingOverflow(by: labels.height)
        guard labels.width > 0, labels.height > 0, labels.count >= 0, !overflow else { throw Segmentation.SegmentationError.invalidLabelImage }
        guard labels.labels.count == product else {
            throw Segmentation.SegmentationError.sizeMismatch(expected: product, got: labels.labels.count)
        }
        guard imageCount == product else { throw Segmentation.SegmentationError.sizeMismatch(expected: product, got: imageCount) }
        for l in labels.labels where l < 0 || Int(l) > labels.count { throw Segmentation.SegmentationError.labelOutOfRange(l) }
    }
}

extension ObjectStatistics {
    /// How many objects there are of each kind, and the size summary of the counted ones (a particle report's numbers).
    public struct SizeSummary: Sendable, Equatable {
        public let total: Int
        /// Not on the image edge and not below the minimum area.
        public let counted: Int
        /// Touching the image edge, whatever their size.
        public let edge: Int
        /// Below the minimum area and not on the edge.
        public let small: Int
        /// Over the counted objects, in pixels (`lengthPx`, `widthPx` of the moments); nil with none counted.
        public let medianLength: Double?
        public let meanLength: Double?
        public let meanWidth: Double?
    }

    /// The counts and the length and width summaries of `objects` (from `Segmentation.objectMoments`): an object of fewer than
    /// `minimumArea` pixels is small, one touching the image edge is an edge object, and only the rest are counted. The median is the
    /// middle length (the mean of the two middle ones for an even count), the means are sums over the counted objects in label order
    /// (lengths summed in ascending order, as `arealDensity` does).
    public static func sizeSummary(_ objects: [Segmentation.ObjectMoments], minimumArea: Int = 1) -> SizeSummary {
        let edge = objects.filter(\.touchesEdge)
        let small = objects.filter { $0.area < minimumArea && !$0.touchesEdge }
        let counted = objects.filter { $0.area >= minimumArea && !$0.touchesEdge }
        guard !counted.isEmpty else {
            return SizeSummary(total: objects.count, counted: 0, edge: edge.count, small: small.count,
                               medianLength: nil, meanLength: nil, meanWidth: nil)
        }
        let lengths = counted.map(\.lengthPx).sorted()
        let mid = lengths.count / 2
        let median = lengths.count % 2 == 1 ? lengths[mid] : (lengths[mid - 1] + lengths[mid]) / 2
        return SizeSummary(total: objects.count, counted: counted.count, edge: edge.count, small: small.count, medianLength: median,
                           meanLength: lengths.reduce(0, +) / Double(lengths.count),
                           meanWidth: counted.map(\.widthPx).reduce(0, +) / Double(counted.count))
    }
}

extension ObjectStatistics {
    /// The counts and the Feret-length summary of the objects of `regionProperties`.
    public struct FeretSummary: Sendable, Equatable {
        public let total: Int
        /// Not on the image edge and not below the minimum area.
        public let counted: Int
        /// Touching the image edge, whatever their size.
        public let edge: Int
        /// Below the minimum area and not on the edge.
        public let small: Int
        /// The median and mean of the counted objects' `feretDiameterMax`, in pixels; nil with none counted.
        public let medianFeret: Double?
        public let meanFeret: Double?
    }

    /// `sizeSummary` on the Feret diameter (`regionprops`' `feret_diameter_max`) instead of the extent along the principal axis: the
    /// needle length of a precipitate study. The same counting rule (edge, small, counted), the median the middle length (the mean of
    /// the two middle ones for an even count) and the mean a sum over the counted lengths in ascending order.
    public static func feretSummary(_ regions: [RegionProperties], minimumArea: Int = 1) -> FeretSummary {
        let edge = regions.filter(\.moments.touchesEdge)
        let small = regions.filter { $0.moments.area < minimumArea && !$0.moments.touchesEdge }
        let counted = regions.filter { $0.moments.area >= minimumArea && !$0.moments.touchesEdge }
        guard !counted.isEmpty else {
            return FeretSummary(total: regions.count, counted: 0, edge: edge.count, small: small.count, medianFeret: nil, meanFeret: nil)
        }
        let lengths = counted.map(\.feretDiameterMax).sorted()
        let mid = lengths.count / 2
        let median = lengths.count % 2 == 1 ? lengths[mid] : (lengths[mid - 1] + lengths[mid]) / 2
        return FeretSummary(total: regions.count, counted: counted.count, edge: edge.count, small: small.count, medianFeret: median,
                            meanFeret: lengths.reduce(0, +) / Double(lengths.count))
    }
}
