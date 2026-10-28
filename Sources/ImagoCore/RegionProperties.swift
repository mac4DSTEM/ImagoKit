//
//  RegionProperties.swift (ImagoCore)
//  The rest of scikit-image 0.26.0's `measure.regionprops` that a particle study reads, beside `Segmentation.objectMoments`:
//  `perimeter` (neighbourhood 4), `perimeter_crofton` (4 directions), `eccentricity`, `solidity`, `equivalent_diameter_area`,
//  `extent`, `euler_number`, `area_filled`, `area_convex`, `feret_diameter_max` and, with an intensity image, `intensity_mean`,
//  `intensity_min`, `intensity_max`, `intensity_std` and `centroid_weighted`. Every property is computed on the object's own
//  bounding-box crop, as `regionprops` does, in Double.
//
//  DEFINITIONS, from `skimage/measure/_regionprops.py` and `_regionprops_utils.py`:
//  * `perimeter`: boundary pixels (the mask minus its 4-neighbour erosion with a zero border) weighted by their 3 x 3 neighbourhood code
//    (`[[10, 2, 10], [2, 1, 2], [10, 2, 10]]` convolution; weights 1, √2, (1 + √2)/2 by code).
//  * `perimeter_crofton`: the 2 x 2 configuration histogram of the zero-padded mask with scikit-image's four-direction coefficients.
//  * `euler_number`: the same histogram with the 8-connectivity coefficients (`regionprops` passes connectivity = ndim).
//  * `area_convex` / `solidity`: the pixels inside or on the convex hull of the diamonds (±½ along each axis) around every object
//    pixel, `convex_hull_image(image)` with `include_borders=True`; here the hull is built and tested in exact integer arithmetic on
//    doubled coordinates, so a pixel centre on a hull edge is inside.
//  * `feret_diameter_max`: the largest distance between the contour vertices `find_contours(…, 0.5, fully_connected='high')` finds on
//    the 2-pixel zero-padded convex hull image; those vertices are the midpoints between every in/out pair of 4-neighbour pixels, so
//    the kit takes the maximum over their convex hull. This is the maximum caliper length (the needle length), unlike
//    `axisMajorLength`, which is 4√λ1 of the inertia tensor.
//  * `area_filled`: `binary_fill_holes` with the full 3 x 3 structure on the crop.
//  * `eccentricity` √(1 − λ2/λ1) (0 when λ1 = 0); `equivalent_diameter_area` √(4·area/π); `extent` area / bounding-box area.
//
//  ACCURACY: areas, Euler numbers and integer counts exact; floating properties agree with scikit-image's float64 values to rounding
//  (the tests hold them to assert_allclose rtol 1e-7).
//

import Foundation

/// The intensity properties of one object under an intensity image.
public nonisolated struct RegionIntensity: Sendable, Equatable {
    public let mean: Double
    public let minimum: Double
    public let maximum: Double
    /// Population standard deviation (`np.std`, ddof 0).
    public let standardDeviation: Double
    /// `centroid_weighted`, (row, column) in image coordinates: Σ I·r / Σ I over the object (NaN when ΣI = 0).
    public let weightedCentroidRow: Double
    public let weightedCentroidColumn: Double
}

/// One object's `regionprops` beyond the moments.
public nonisolated struct RegionProperties: Sendable, Equatable {
    /// Area, bounding box, centroid, inertia and axes, orientation (`Segmentation.objectMoments`).
    public let moments: Segmentation.ObjectMoments
    public let perimeter: Double
    public let perimeterCrofton: Double
    public let eccentricity: Double
    /// area / area_convex.
    public let solidity: Double
    public let equivalentDiameterArea: Double
    public let extent: Double
    public let eulerNumber: Int
    /// Pixels after filling the object's holes (8-connected background rule, `binary_fill_holes` with the full 3 x 3 structure).
    public let areaFilled: Int
    /// Pixels of the convex hull image.
    public let areaConvex: Int
    /// The maximum caliper (Feret) diameter in pixels.
    public let feretDiameterMax: Double
    /// Nil without an intensity image.
    public let intensity: RegionIntensity?
}

extension Segmentation {
    /// `regionprops(labels, intensity_image:)` for every present object in label order.
    public static func regionProperties(_ image: LabelImage, intensity: [Double]? = nil) throws -> [RegionProperties] {
        let moments = try objectMoments(image)
        let w = image.width
        if let intensity {
            guard intensity.count == image.labels.count else {
                throw SegmentationError.sizeMismatch(expected: image.labels.count, got: intensity.count)
            }
        }
        let stats = try intensity.map { try ObjectStatistics.measure(image: $0, labels: image) }
        let statsByLabel = Dictionary(uniqueKeysWithValues: (stats ?? []).map { ($0.label, $0) })
        var members = [[Int]](repeating: [], count: image.count)
        for (i, l) in image.labels.enumerated() where l > 0 { members[Int(l) - 1].append(i) }
        return moments.enumerated().map { index, m in
            let pixels = members[Int(m.label) - 1]
            let h = m.maxRow - m.minRow, cw = m.maxCol - m.minCol
            var crop = [Bool](repeating: false, count: h * cw)
            for i in pixels { crop[(i / w - m.minRow) * cw + (i % w - m.minCol)] = true }
            let convex = RegionShape.convexHull(crop, width: cw, height: h)
            let areaConvex = convex.mask.filter { $0 }.count
            let l1 = m.eigenvalueMajor, l2 = m.eigenvalueMinor
            var intensityProps: RegionIntensity?
            if let s = statsByLabel[m.label] {
                intensityProps = RegionIntensity(
                    mean: s.mean, minimum: s.minimum, maximum: s.maximum, standardDeviation: s.standardDeviation,
                    weightedCentroidRow: s.centerOfMass.row, weightedCentroidColumn: s.centerOfMass.column)
            }
            return RegionProperties(
                moments: m,
                perimeter: RegionShape.perimeter(crop, width: cw, height: h),
                perimeterCrofton: RegionShape.perimeterCrofton(crop, width: cw, height: h),
                eccentricity: l1 == 0 ? 0 : (1 - l2 / l1).squareRoot(),
                solidity: Double(m.area) / Double(areaConvex),
                equivalentDiameterArea: (4 * Double(m.area) / Double.pi).squareRoot(),
                extent: Double(m.area) / Double(h * cw),
                eulerNumber: RegionShape.eulerNumber(crop, width: cw, height: h, connectivity: .eight),
                areaFilled: ((try? BinaryMorphology.fillHoles(crop, width: cw, height: h, structure: .square)) ?? crop).filter { $0 }.count,
                areaConvex: areaConvex,
                feretDiameterMax: RegionShape.feretDiameterMax(convex.mask, width: cw, height: h),
                intensity: intensityProps)
        }
    }

    /// `measure.euler_number(mask, connectivity)` of a 2D mask (the Euler characteristic: objects minus holes).
    public static func eulerNumber(_ mask: [Bool], width: Int, height: Int, connectivity: Connectivity = .eight) throws -> Int {
        let (product, overflow) = width.multipliedReportingOverflow(by: height)
        guard width > 0, height > 0, !overflow else { throw SegmentationError.negativeDimension }
        guard mask.count == product else { throw SegmentationError.sizeMismatch(expected: product, got: mask.count) }
        return RegionShape.eulerNumber(mask, width: width, height: height, connectivity: connectivity)
    }
}

/// The per-object shape routines on a bounding-box crop.
nonisolated enum RegionShape {
    /// `[[0, 0, 0], [0, 1, 4], [0, 2, 8]]` convolved over the zero-padded crop: the 2 x 2 configuration code at each padded pixel.
    private static func configurationHistogram(_ m: [Bool], width w: Int, height h: Int) -> [Int] {
        var histogram = [Int](repeating: 0, count: 16)
        func at(_ r: Int, _ c: Int) -> Int { r >= 0 && r < h && c >= 0 && c < w && m[r * w + c] ? 1 : 0 }
        // Padded coordinates run -1...h and -1...w (the crop is padded by one on every side).
        for r in -1...h {
            for c in -1...w {
                histogram[at(r, c) + 4 * at(r, c - 1) + 2 * at(r - 1, c) + 8 * at(r - 1, c - 1)] += 1
            }
        }
        return histogram
    }

    static func eulerNumber(_ m: [Bool], width: Int, height: Int, connectivity: Segmentation.Connectivity) -> Int {
        let four = [0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0]
        let eight = [0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, -1, 0]
        let coefficients = connectivity == .four ? four : eight
        let histogram = configurationHistogram(m, width: width, height: height)
        return zip(coefficients, histogram).reduce(0) { $0 + $1.0 * $1.1 }
    }

    static func perimeterCrofton(_ m: [Bool], width: Int, height: Int) -> Double {
        let pi = Double.pi, root2 = 2.0.squareRoot()
        let quarter = pi / 4, diagonal = pi / (4 * root2)
        let a = quarter * (1 + 1 / root2)
        // scikit-image's four-direction look-up table, one coefficient per 2 x 2 configuration code.
        let coefficients: [Double] = [0, a, diagonal, pi / (2 * root2), 0, a, 0, diagonal,
                                      quarter, pi / 2, diagonal, diagonal, quarter, pi / 2, 0, 0]
        let histogram = configurationHistogram(m, width: width, height: height)
        var total = 0.0
        for k in 0..<16 { total += coefficients[k] * Double(histogram[k]) }
        return total
    }

    static func perimeter(_ m: [Bool], width w: Int, height h: Int) -> Double {
        func at(_ r: Int, _ c: Int) -> Bool { r >= 0 && r < h && c >= 0 && c < w && m[r * w + c] }
        // The boundary: object pixels with a 4-neighbour missing (a zero border beyond the crop).
        var border = [Int](repeating: 0, count: w * h)
        for r in 0..<h {
            for c in 0..<w where m[r * w + c] {
                border[r * w + c] = at(r - 1, c) && at(r + 1, c) && at(r, c - 1) && at(r, c + 1) ? 0 : 1
            }
        }
        func b(_ r: Int, _ c: Int) -> Int { r >= 0 && r < h && c >= 0 && c < w ? border[r * w + c] : 0 }
        let root2 = 2.0.squareRoot()
        var total = 0.0
        for r in 0..<h {
            for c in 0..<w {
                let code = b(r, c)
                    + 2 * (b(r - 1, c) + b(r + 1, c) + b(r, c - 1) + b(r, c + 1))
                    + 10 * (b(r - 1, c - 1) + b(r - 1, c + 1) + b(r + 1, c - 1) + b(r + 1, c + 1))
                switch code {
                case 5, 7, 15, 17, 25, 27: total += 1
                case 21, 33: total += root2
                case 13, 23: total += (1 + root2) / 2
                default: break
                }
            }
        }
        return total
    }

    // MARK: Convex hull

    private struct IntPoint: Hashable { var x: Int; var y: Int }

    private static func cross(_ o: IntPoint, _ a: IntPoint, _ b: IntPoint) -> Int {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }

    /// Andrew's monotone chain on integer points: the strict convex hull, counter-clockwise, no collinear points.
    private static func hull(_ points: [IntPoint]) -> [IntPoint] {
        var sorted = Array(Set(points))
        sorted.sort { (a: IntPoint, b: IntPoint) -> Bool in
            if a.x != b.x { return a.x < b.x }
            return a.y < b.y
        }
        guard sorted.count > 2 else { return sorted }
        var lower: [IntPoint] = []
        for p in sorted {
            while lower.count >= 2, cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 { lower.removeLast() }
            lower.append(p)
        }
        var upper: [IntPoint] = []
        for p in sorted.reversed() {
            while upper.count >= 2, cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 { upper.removeLast() }
            upper.append(p)
        }
        lower.removeLast(); upper.removeLast()
        return lower + upper
    }

    /// `convex_hull_image(mask)` on a crop: the mask of pixels whose centres lie inside or on the hull of the diamonds around the
    /// object pixels. Coordinates are doubled so the diamond tips and pixel centres are integers.
    static func convexHull(_ m: [Bool], width w: Int, height h: Int) -> (mask: [Bool], polygon: Int) {
        // Only the first and last pixel of every row and column can be a hull vertex (`possible_hull`).
        var extreme = Set<Int>()
        for r in 0..<h {
            if let first = (0..<w).first(where: { m[r * w + $0] }), let last = (0..<w).last(where: { m[r * w + $0] }) {
                extreme.insert(r * w + first); extreme.insert(r * w + last)
            }
        }
        for c in 0..<w {
            if let first = (0..<h).first(where: { m[$0 * w + c] }), let last = (0..<h).last(where: { m[$0 * w + c] }) {
                extreme.insert(first * w + c); extreme.insert(last * w + c)
            }
        }
        var points: [IntPoint] = []
        for i in extreme {
            let r = i / w, c = i % w
            points.append(IntPoint(x: 2 * c - 1, y: 2 * r)); points.append(IntPoint(x: 2 * c + 1, y: 2 * r))
            points.append(IntPoint(x: 2 * c, y: 2 * r - 1)); points.append(IntPoint(x: 2 * c, y: 2 * r + 1))
        }
        guard !points.isEmpty else { return ([Bool](repeating: false, count: w * h), 0) }
        let polygon = hull(points)
        var mask = [Bool](repeating: false, count: w * h)
        for r in 0..<h {
            for c in 0..<w {
                let p = IntPoint(x: 2 * c, y: 2 * r)
                var inside = true
                for i in 0..<polygon.count where cross(polygon[i], polygon[(i + 1) % polygon.count], p) < 0 { inside = false; break }
                mask[r * w + c] = inside
            }
        }
        return (mask, polygon.count)
    }

    /// The maximum distance between the contour vertices of the 2-pixel zero-padded convex hull image: the midpoints of every
    /// 4-neighbour in/out pair, reduced to their convex hull.
    static func feretDiameterMax(_ convex: [Bool], width w: Int, height h: Int) -> Double {
        func at(_ r: Int, _ c: Int) -> Bool { r >= 0 && r < h && c >= 0 && c < w && convex[r * w + c] }
        var vertices: [IntPoint] = []   // doubled coordinates: midpoints are integers
        for r in -1...h {
            for c in -1...w {
                if at(r, c) != at(r, c + 1) { vertices.append(IntPoint(x: 2 * c + 1, y: 2 * r)) }
                if at(r, c) != at(r + 1, c) { vertices.append(IntPoint(x: 2 * c, y: 2 * r + 1)) }
            }
        }
        let outline = hull(vertices)
        var best = 0
        for i in 0..<outline.count {
            for j in (i + 1)..<max(outline.count, i + 1) {
                let dx = outline[i].x - outline[j].x, dy = outline[i].y - outline[j].y
                best = max(best, dx * dx + dy * dy)
            }
        }
        return (Double(best) / 4).squareRoot()   // doubled coordinates: the squared distance is four times too large
    }
}

// MARK: regionprops_table through the tidy CSV

extension RegionProperties {
    /// This object's quantities named as `regionprops_table` names them (`area`, `bbox-0`…`bbox-3`, `centroid-0`, `centroid-1`,
    /// `axis_major_length`, `feret_diameter_max`, …), lengths in `lengthUnit` (default pixels) and areas in its square. Intensity
    /// columns (`intensity_mean` …, `centroid_weighted-0/1`) follow when an intensity image was given; `intensityUnit` labels them.
    public func tidyEntries(lengthUnit: String = "px", intensityUnit: String = "") -> [TidyResultsCSV.Entry] {
        let areaUnit = lengthUnit + "^2"
        let m = moments
        typealias E = TidyResultsCSV.Entry
        var entries: [E] = [
            E("label", Double(m.label)), E("area", Double(m.area), unit: areaUnit),
            E("bbox-0", Double(m.minRow), unit: "px"), E("bbox-1", Double(m.minCol), unit: "px"),
            E("bbox-2", Double(m.maxRow), unit: "px"), E("bbox-3", Double(m.maxCol), unit: "px"),
            E("centroid-0", m.centroidRow, unit: "px"), E("centroid-1", m.centroidCol, unit: "px"),
            E("orientation", m.orientation, unit: "rad"),
            E("axis_major_length", m.axisMajorLength, unit: lengthUnit), E("axis_minor_length", m.axisMinorLength, unit: lengthUnit),
            E("feret_diameter_max", feretDiameterMax, unit: lengthUnit),
            E("perimeter", perimeter, unit: lengthUnit), E("perimeter_crofton", perimeterCrofton, unit: lengthUnit),
            E("eccentricity", eccentricity), E("solidity", solidity),
            E("equivalent_diameter_area", equivalentDiameterArea, unit: lengthUnit), E("extent", extent),
            E("euler_number", Double(eulerNumber)), E("area_filled", Double(areaFilled), unit: areaUnit),
            E("area_convex", Double(areaConvex), unit: areaUnit),
        ]
        if let i = intensity {
            entries += [E("intensity_mean", i.mean, unit: intensityUnit), E("intensity_min", i.minimum, unit: intensityUnit),
                        E("intensity_max", i.maximum, unit: intensityUnit), E("intensity_std", i.standardDeviation, unit: intensityUnit),
                        E("centroid_weighted-0", i.weightedCentroidRow, unit: "px"),
                        E("centroid_weighted-1", i.weightedCentroidColumn, unit: "px")]
        }
        return entries
    }

    /// One tidy row per object for `TidyResultsCSV.text` (id = the label; `product` names the caller's validation sentence).
    public static func tidyRows(_ objects: [RegionProperties], group: String, product: String, lengthUnit: String = "px",
                                intensityUnit: String = "", lineage: String? = nil) -> [TidyResultsCSV.Row] {
        objects.map { o in
            TidyResultsCSV.Row(group: group, id: String(o.moments.label), label: "object \(o.moments.label)", product: product,
                               entries: o.tidyEntries(lengthUnit: lengthUnit, intensityUnit: intensityUnit), lineage: lineage)
        }
    }
}
