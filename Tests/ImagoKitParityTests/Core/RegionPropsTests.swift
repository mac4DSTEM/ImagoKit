// regionprops completion and labelled statistics against the pins of tools/precipitation/pins.py.
// Tolerance: scikit-image 0.26.0 ships no tests in its wheel, so its floats use assert_allclose dtype defaults (rtol 1e-7 Float64);
// area, bbox, Euler number, filled and convex areas are exact. SciPy's ndimage measurement tests use decimal 6 (positions and
// bounding slices exact).
import Foundation
import ImagoKitTestSupport
import ImagoCore
import Testing

@Suite("regionprops (scikit-image 0.26.0)") struct RegionPropsTests {
    static let cases = RegionPropsPins.all.cases

    @Test func versions() {
        let p = RegionPropsPins.all
        #expect(p.skimage == "0.26.0" && p.numpy == "2.5.4" && p.scipy == "1.18.1")
    }

    private func rel(_ a: Double, _ b: Double, _ what: String, _ rtol: Double = 1e-7) {
        #expect(abs(a - b) <= rtol * abs(b) + 1e-12, "\(what): \(a) vs \(b)")
    }

    @Test(arguments: cases) func connectivity2(_ c: RegionPropsPins.Case) throws { try check(c, c.labels2, c.count2, c.props2, .eight) }
    @Test(arguments: cases) func connectivity1(_ c: RegionPropsPins.Case) throws { try check(c, c.labels1, c.count1, c.props1, .four) }

    private func check(_ c: RegionPropsPins.Case, _ labels: [Int], _ count: Int, _ pins: [RegionPropsPins.Props],
                       _ connectivity: Segmentation.Connectivity) throws {
        let image = Segmentation.LabelImage(width: c.width, height: c.height, labels: labels.map { Int32($0) }, count: count)
        // The kit's own labelling is the same partition (labels exact).
        let relabelled = try Segmentation.label(maskFrom(c.mask), width: c.width, height: c.height, connectivity: connectivity)
        #expect(relabelled == image, "\(c.name): labelling")
        let got = try Segmentation.regionProperties(image)
        #expect(got.count == pins.count, "\(c.name): object count")
        for (g, p) in zip(got, pins) {
            let m = g.moments
            let tag = "\(c.name) conn \(connectivity.rawValue) label \(p.label)"
            #expect(Int(m.label) == p.label && m.area == p.area, "\(tag): label, area")
            #expect([m.minRow, m.minCol, m.maxRow, m.maxCol] == p.bbox, "\(tag): bbox")
            rel(m.centroidRow, p.centroid[0], "\(tag) centroid row"); rel(m.centroidCol, p.centroid[1], "\(tag) centroid col")
            // The orientation jumps by π across ±π/2 (a blob with no tilt): the sign of a rounding-level covariance decides which side
            // NumPy's moment arithmetic and the kit's two-pass sums land on. The axis is the same; held modulo π there.
            let wrapped = abs(abs(p.orientation) - Double.pi / 2) < 1e-6 && abs(abs(m.orientation - p.orientation) - Double.pi) < 1e-6
            if !wrapped { rel(m.orientation, p.orientation, "\(tag) orientation") }
            rel(m.axisMajorLength, p.axis_major_length, "\(tag) axis_major_length")
            rel(m.axisMinorLength, p.axis_minor_length, "\(tag) axis_minor_length")
            rel(g.feretDiameterMax, p.feret_diameter_max, "\(tag) feret_diameter_max")
            rel(g.perimeter, p.perimeter, "\(tag) perimeter")
            rel(g.perimeterCrofton, p.perimeter_crofton, "\(tag) perimeter_crofton")
            rel(g.eccentricity, p.eccentricity, "\(tag) eccentricity")
            rel(g.solidity, p.solidity, "\(tag) solidity")
            rel(g.equivalentDiameterArea, p.equivalent_diameter_area, "\(tag) equivalent_diameter_area")
            rel(g.extent, p.extent, "\(tag) extent")
            #expect(g.eulerNumber == p.euler_number, "\(tag): euler_number \(g.eulerNumber) vs \(p.euler_number)")
            #expect(g.areaFilled == p.area_filled, "\(tag): area_filled")
            #expect(g.areaConvex == p.area_convex, "\(tag): area_convex")
        }
    }

    @Test func eulerNumberOfAWholeMask() throws {
        for c in Self.cases {
            let m = maskFrom(c.mask)
            let four = try Segmentation.eulerNumber(m, width: c.width, height: c.height, connectivity: .four)
            let eight = try Segmentation.eulerNumber(m, width: c.width, height: c.height, connectivity: .eight)
            // Objects minus holes, per connectivity; for the planted ring with a cut channel there is one object and no hole.
            if c.name == "full" || c.name == "empty" { #expect(four == (c.name == "full" ? 1 : 0) && eight == four) }
        }
        // A ring: one object, one hole, Euler number 0 under both connectivities.
        var ring = [Bool](repeating: true, count: 25)
        ring[12] = false
        #expect(try Segmentation.eulerNumber(ring, width: 5, height: 5, connectivity: .eight) == 0)
        #expect(try Segmentation.eulerNumber(ring, width: 5, height: 5, connectivity: .four) == 0)
    }

    @Test func intensityStatisticsMatchSciPy() throws {
        let p = RegionPropsPins.all.intensity
        let image = Segmentation.LabelImage(width: p.width, height: p.height, labels: p.labels.map { Int32($0) }, count: p.count)
        let got = try ObjectStatistics.measure(image: p.image, labels: image)
        #expect(got.count == p.count)
        for (i, r) in got.enumerated() {
            func d6(_ a: Double, _ b: Double, _ what: String) { #expect(abs(a - b) <= 1.5e-6, "label \(i + 1) \(what): \(a) vs \(b)") }
            d6(r.sum, p.sum[i], "sum"); d6(r.mean, p.mean[i], "mean"); d6(r.standardDeviation, p.std[i], "std")
            d6(r.minimum, p.min[i], "min"); d6(r.maximum, p.max[i], "max")
            d6(r.centerOfMass.row, p.center_of_mass[i][0], "com row"); d6(r.centerOfMass.column, p.center_of_mass[i][1], "com col")
            #expect([r.minimumPosition.row, r.minimumPosition.column] == p.min_position[i], "label \(i + 1) min position")
            #expect([r.maximumPosition.row, r.maximumPosition.column] == p.max_position[i], "label \(i + 1) max position")
            #expect(Int(r.label) == i + 1)
        }
        let boxes = try ObjectStatistics.findObjects(image)
        #expect(boxes.count == p.find_objects.count)
        for (b, e) in zip(boxes, p.find_objects) {
            #expect(b.map { [$0.rows.lowerBound, $0.rows.upperBound, $0.columns.lowerBound, $0.columns.upperBound] } == e)
        }
    }

    @Test func regionpropsIntensityMatchesScikitImage() throws {
        let p = RegionPropsPins.all.intensity
        let image = Segmentation.LabelImage(width: p.width, height: p.height, labels: p.labels.map { Int32($0) }, count: p.count)
        let got = try Segmentation.regionProperties(image, intensity: p.image)
        #expect(got.count == p.regionprops.count)
        for (g, e) in zip(got, p.regionprops) {
            let i = try #require(g.intensity)
            rel(i.mean, e.intensity_mean, "intensity_mean"); rel(i.minimum, e.intensity_min, "intensity_min")
            rel(i.maximum, e.intensity_max, "intensity_max"); rel(i.standardDeviation, e.intensity_std, "intensity_std")
            rel(i.weightedCentroidRow, e.centroid_weighted[0], "centroid_weighted row")
            rel(i.weightedCentroidColumn, e.centroid_weighted[1], "centroid_weighted col")
        }
    }

    @Test func regionpropsTableThroughTheTidyCSV() throws {
        let c = try #require(Self.cases.first { $0.name == "needles-diagonal" })
        let image = Segmentation.LabelImage(width: c.width, height: c.height, labels: c.labels2.map { Int32($0) }, count: c.count2)
        let props = try Segmentation.regionProperties(image, intensity: [Double](repeating: 2, count: c.width * c.height))
        let rows = RegionProperties.tidyRows(props, group: "particles", product: "particles", lengthUnit: "nm", intensityUnit: "counts")
        #expect(rows.count == c.count2)
        let first = rows[0].entries
        #expect(first.first { $0.name == "feret_diameter_max" }?.unit == "nm")
        #expect(first.first { $0.name == "area" }?.unit == "nm^2")
        #expect(first.first { $0.name == "intensity_mean" }?.value == 2)
        let source = TidyResultsCSV.Source(software: "test", fileName: "f.dm4", object: "o", width: c.width, height: c.height,
                                           x: AxisCalibration(scale: 0.5, unit: "nm"), y: AxisCalibration(scale: 0.5, unit: "nm"), notes: [])
        let text = try #require(try TidyResultsCSV.text(rows, source: source, title: "t", session: "s", descriptionLines: [],
                                                    validations: [("particles", "validation: none")], lineageLine: "# lineage: none"))
        #expect(text.contains(",feret_diameter_max,"))
        #expect(text.split(separator: "\n").filter { $0.contains(",area,") }.count == c.count2)
    }

    @Test func aNaNInTheObjectMakesItsExtremaNaNWhereverItSits() throws {
        // np.min / np.max over the object's pixels return NaN whenever one is NaN (review 2026-10-22).
        let labels = Segmentation.LabelImage(width: 2, height: 2, labels: [1, 1, 0, 1], count: 1)
        for image in [[Double.nan, 1, 5, 3], [1, .nan, 5, 3], [1, 2, 5, .nan]] {
            let r = try #require(try ObjectStatistics.measure(image: image, labels: labels).first)
            #expect(r.minimum.isNaN && r.maximum.isNaN && r.mean.isNaN)
        }
    }

    @Test func refusals() throws {
        let image = Segmentation.LabelImage(width: 2, height: 2, labels: [1, 1, 0, 2], count: 2)
        #expect(throws: Segmentation.SegmentationError.sizeMismatch(expected: 4, got: 3)) { try ObjectStatistics.measure(image: [1, 2, 3] as [Double], labels: image) }
        #expect(throws: Segmentation.SegmentationError.sizeMismatch(expected: 4, got: 3)) { try Segmentation.regionProperties(image, intensity: [1, 2, 3]) }
        // A label with no pixel has no statistics entry and a nil bounding box.
        let gap = Segmentation.LabelImage(width: 2, height: 2, labels: [1, 0, 0, 3], count: 3)
        #expect(try ObjectStatistics.measure(image: [4, 5, 6, 7] as [Double], labels: gap).map(\.label) == [1, 3])
        #expect(try ObjectStatistics.findObjects(gap).map { $0 == nil } == [false, true, false])
    }
}
