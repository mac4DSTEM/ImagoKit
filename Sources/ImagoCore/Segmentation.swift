//
//  Segmentation.swift (ImagoCore)
//  Connected components of a mask or a label map, and the moments of each object: scikit-image's `label` and the
//  `regionprops` that follow from the pixel coordinates (area, bounding box, centroid, inertia tensor, axis lengths,
//  orientation), plus the object's extents along and across its principal axis.
//
//  LABELLING: `skimage.measure.label(mask, connectivity:)`. Pixels are visited in row-major order; each unlabelled
//  foreground pixel starts a component that takes the next number (1, 2, ...) and grows over its 4-connected
//  (`connectivity: .four`, skimage connectivity 1) or 8-connected (`.eight`, connectivity 2) neighbours. Components are
//  therefore numbered by their first pixel in row-major order, as scikit-image numbers them. Background is 0.
//  `label(values:background:)` is `skimage.measure.label(values, background:)`: neighbours join only when equal.
//
//  MOMENTS, in Double, from the pixel centres (row i, column j at integers): area = pixel count; centroid = mean row and
//  column; the central second moments over the area (the population covariance, as `regionprops`' inertia tensor)
//  cov_rr, cov_cc, cov_rc; inertia tensor [[cov_cc, −cov_rc], [−cov_rc, cov_rr]]; its eigenvalues λ1 ≥ λ2 ≥ 0;
//  `axisMajorLength = 4√λ1`, `axisMinorLength = 4√λ2`; `orientation` as scikit-image (the angle between the row axis and the
//  major axis, counter-clockwise, in (−π/2, π/2]; π/4 or −π/4 when the two diagonal terms of the tensor are equal).
//
//  The object's direction in image coordinates (x = column, y = row, +y down) is `orientationDegrees`: the angle of the
//  larger eigenvector of [[cov_xx, cov_xy], [cov_xy, cov_yy]] from +x toward +y, folded into (−90, 90]. `lengthPx` and
//  `widthPx` are the ranges of the pixel centres projected on that axis and across it, plus 1 (1 for a single line of
//  pixels). They are not moment estimates: 4√λ is skimage's.
//
//  ACCURACY: labels, area, bounding box and `touchesEdge` are exact. The moments are sums of integers shifted by the
//  centroid, in Double: they agree with scikit-image's float64 values to 1e-12 (relative above 1, absolute below).
//

import Foundation

public nonisolated enum Segmentation {
    public enum Connectivity: Int, Sendable, Equatable, CaseIterable {
        /// Edge neighbours only (skimage connectivity 1).
        case four = 4
        /// Edge and corner neighbours (skimage connectivity 2).
        case eight = 8
    }

    /// A label map: `labels[row * width + col]` is 0 for background, else the component number (1...count).
    public struct LabelImage: Sendable, Equatable {
        public let width: Int
        public let height: Int
        public let labels: [Int32]
        public let count: Int

        public init(width: Int, height: Int, labels: [Int32], count: Int) {
            self.width = width; self.height = height; self.labels = labels; self.count = count
        }
    }

    /// One object's measurements; see the file header for every definition.
    public struct ObjectMoments: Sendable, Equatable {
        public let label: Int32
        public let area: Int
        /// The bounding box in pixels: rows `minRow..<maxRow`, columns `minCol..<maxCol` (skimage's `bbox`).
        public let minRow: Int, minCol: Int, maxRow: Int, maxCol: Int
        public let centroidRow: Double
        public let centroidCol: Double
        public let covarianceRR: Double
        public let covarianceCC: Double
        public let covarianceRC: Double
        /// λ1 ≥ λ2 ≥ 0 of the inertia tensor.
        public let eigenvalueMajor: Double
        public let eigenvalueMinor: Double
        public let axisMajorLength: Double
        public let axisMinorLength: Double
        /// scikit-image's `orientation`, radians.
        public let orientation: Double
        /// The long axis from +x (column) toward +y (row), degrees in (−90, 90].
        public let orientationDegrees: Double
        /// Extent of the pixel centres along the long axis, plus 1.
        public let lengthPx: Double
        /// Extent of the pixel centres across it, plus 1.
        public let widthPx: Double
        /// True when a member lies in the first or last row or column.
        public let touchesEdge: Bool
    }

    /// Why a call is refused.
    public enum SegmentationError: Error, Equatable, Sendable {
        case sizeMismatch(expected: Int, got: Int)
        case negativeDimension
        case labelOutOfRange(Int32)
        /// A `LabelImage` with a negative count, a negative dimension or a size that overflows.
        case invalidLabelImage
    }

    // MARK: Labelling

    /// `skimage.measure.label` of a boolean mask.
    public static func label(_ mask: [Bool], width: Int, height: Int,
                             connectivity: Connectivity = .eight) throws -> LabelImage {
        try grow(count: mask.count, width: width, height: height, connectivity: connectivity,
                 isForeground: { mask[$0] }, same: { _, _ in true })
    }

    /// `skimage.measure.label(values, background:)`: connected regions of equal value; pixels equal to `background` are 0.
    /// A nil `background` labels every value (scikit-image's default is 0; pass 0 for it).
    public static func label(values: [Int32], width: Int, height: Int, background: Int32? = 0,
                             connectivity: Connectivity = .eight) throws -> LabelImage {
        try grow(count: values.count, width: width, height: height, connectivity: connectivity,
                 isForeground: { i in background.map { values[i] != $0 } ?? true }, same: { values[$0] == values[$1] })
    }

    private static func grow(count: Int, width: Int, height: Int, connectivity: Connectivity,
                             isForeground: (Int) -> Bool, same: (Int, Int) -> Bool) throws -> LabelImage {
        guard width >= 0, height >= 0 else { throw SegmentationError.negativeDimension }
        let (product, overflow) = width.multipliedReportingOverflow(by: height)
        guard !overflow else { throw SegmentationError.invalidLabelImage }
        guard count == product else { throw SegmentationError.sizeMismatch(expected: product, got: count) }
        var labels = [Int32](repeating: 0, count: count)
        var next: Int32 = 0
        var stack: [Int] = []
        let diagonal = connectivity == .eight
        for start in 0..<count where labels[start] == 0 && isForeground(start) {
            next += 1
            labels[start] = next
            stack.append(start)
            while let i = stack.popLast() {
                let row = i / width, col = i % width
                for dy in -1...1 {
                    for dx in -1...1 where !(dx == 0 && dy == 0) && (diagonal || dx == 0 || dy == 0) {
                        let r = row + dy, c = col + dx
                        guard r >= 0, r < height, c >= 0, c < width else { continue }
                        let j = r * width + c
                        if labels[j] == 0, isForeground(j), same(i, j) {
                            labels[j] = next
                            stack.append(j)
                        }
                    }
                }
            }
        }
        return LabelImage(width: width, height: height, labels: labels, count: Int(next))
    }

    // MARK: Moments

    /// The measurements of every object 1...count of a label map, in label order. A label outside 0...count is refused.
    public static func objectMoments(_ image: LabelImage) throws -> [ObjectMoments] {
        let w = image.width, h = image.height
        let (product, overflow) = w.multipliedReportingOverflow(by: h)
        guard w >= 0, h >= 0, image.count >= 0, !overflow else { throw SegmentationError.invalidLabelImage }
        guard image.labels.count == product else { throw SegmentationError.sizeMismatch(expected: product, got: image.labels.count) }
        let n = image.count
        guard n <= image.labels.count else { throw SegmentationError.invalidLabelImage }
        for l in image.labels where l < 0 || Int(l) > n { throw SegmentationError.labelOutOfRange(l) }
        var members = [[Int]](repeating: [], count: n)
        for (i, l) in image.labels.enumerated() where l > 0 { members[Int(l) - 1].append(i) }
        return members.enumerated().compactMap { index, pixels in
            pixels.isEmpty ? nil : moments(label: Int32(index + 1), pixels: pixels, width: w, height: h)
        }
    }

    private static func moments(label: Int32, pixels: [Int], width: Int, height: Int) -> ObjectMoments {
        let area = pixels.count
        let a = Double(area)
        var sumR = 0.0, sumC = 0.0
        var minR = Int.max, minC = Int.max, maxR = -1, maxC = -1
        var edge = false
        for i in pixels {
            let r = i / width, c = i % width
            sumR += Double(r); sumC += Double(c)
            minR = min(minR, r); maxR = max(maxR, r); minC = min(minC, c); maxC = max(maxC, c)
            if r == 0 || r == height - 1 || c == 0 || c == width - 1 { edge = true }
        }
        let cr = sumR / a, cc = sumC / a
        var srr = 0.0, scc = 0.0, src = 0.0
        for i in pixels {
            let dr = Double(i / width) - cr, dc = Double(i % width) - cc
            srr += dr * dr; scc += dc * dc; src += dr * dc
        }
        let crr = srr / a, ccc = scc / a, crc = src / a

        // scikit-image: the inertia tensor [[ccc, −crc], [−crc, crr]], eigenvalues of the symmetric 2x2.
        let trace = crr + ccc
        let root = ((crr - ccc) * (crr - ccc) + 4 * crc * crc).squareRoot()
        let l1 = max((trace + root) / 2, 0), l2 = max((trace - root) / 2, 0)
        let ta = ccc, tb = -crc, tc = crr
        let orientation: Double
        if ta - tc == 0 { orientation = tb < 0 ? Double.pi / 4 : -Double.pi / 4 }
        else { orientation = 0.5 * atan2(-2 * tb, tc - ta) }

        // The long axis in image coordinates (x = column, y = row): the larger eigenvector of [[ccc, crc], [crc, crr]].
        // Half the angle of the doubled-angle form: no cancellation when the covariance is nearly diagonal; in (−90, 90].
        let degrees = 0.5 * atan2(2 * crc, ccc - crr) * 180 / Double.pi
        let ax = cos(degrees * Double.pi / 180), ay = sin(degrees * Double.pi / 180)
        var uMin = Double.greatestFiniteMagnitude, uMax = -Double.greatestFiniteMagnitude
        var vMin = Double.greatestFiniteMagnitude, vMax = -Double.greatestFiniteMagnitude
        for i in pixels {
            let dx = Double(i % width) - cc, dy = Double(i / width) - cr
            let u = dx * ax + dy * ay, v = -dx * ay + dy * ax
            uMin = min(uMin, u); uMax = max(uMax, u); vMin = min(vMin, v); vMax = max(vMax, v)
        }
        return ObjectMoments(
            label: label, area: area, minRow: minR, minCol: minC, maxRow: maxR + 1, maxCol: maxC + 1,
            centroidRow: cr, centroidCol: cc, covarianceRR: crr, covarianceCC: ccc, covarianceRC: crc,
            eigenvalueMajor: l1, eigenvalueMinor: l2, axisMajorLength: 4 * l1.squareRoot(), axisMinorLength: 4 * l2.squareRoot(),
            orientation: orientation, orientationDegrees: degrees,
            lengthPx: uMax > uMin ? uMax - uMin + 1 : 1, widthPx: vMax > vMin ? vMax - vMin + 1 : 1, touchesEdge: edge)
    }
}
