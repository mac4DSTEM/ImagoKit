// distance_transform_edt, watershed and rolling_ball against the pins of tools/segmentation-k1/pins.py.
// Tolerances: SciPy's ndimage tests use decimal 6; the EDT values are square roots of exact squared distances so 1e-12 is held.
// scikit-image 0.26.0 ships no tests in its wheel: rolling_ball floats use assert_allclose dtype defaults (rtol 1e-7 Float64, 1e-5
// Float32); watershed labels are exact.
import Foundation
import ImagoKitTestSupport
import ImagoCore
import Testing

@Suite("distance transform, watershed, rolling ball (SciPy 1.18.1, scikit-image 0.26.0)") struct WatershedEdtRollingBallTests {
    static let pins = WatershedEdtRollingBallPins.all

    @Test func versions() {
        #expect(Self.pins.meta.skimage == "0.26.0" && Self.pins.meta.numpy == "2.5.4" && Self.pins.meta.scipy == "1.18.1")
    }

    // MARK: distance_transform_edt

    @Test(arguments: pins.edt) func distanceTransform(_ c: WatershedEdtRollingBallPins.EDT) throws {
        let mask = maskFrom(c.mask)
        let sampling = c.sampling.map { (rows: $0[0], columns: $0[1]) } ?? (1, 1)
        guard c.hasBackground else {
            #expect(throws: MorphologyError.noBackground) { try DistanceTransform.euclidean(foreground: mask, rows: c.height, columns: c.width, sampling: sampling) }
            return
        }
        let got = try DistanceTransform.euclidean(foreground: mask, rows: c.height, columns: c.width, sampling: sampling)
        #expect(got.count == c.distance.count)
        for (i, (g, e)) in zip(got, c.distance).enumerated() { #expect(abs(g - e) <= 1e-12 * max(1, e), "\(c.name) pixel \(i): \(g) vs \(e)") }
    }

    @Test func distanceTransformRefusals() {
        #expect(throws: MorphologyError.shape) { try DistanceTransform.euclidean(foreground: [true], rows: 1, columns: 2) }
        #expect(throws: MorphologyError.self) { try DistanceTransform.euclidean(foreground: [true, false], rows: 1, columns: 2, sampling: (0, 1)) }
        #expect(throws: MorphologyError.self) { try DistanceTransform.euclidean(foreground: [true, false], rows: 1, columns: 2, sampling: (1, .nan)) }
    }

    // MARK: watershed

    @Test(arguments: pins.watershed) func watershed(_ c: WatershedEdtRollingBallPins.Watershed) throws {
        let got = try Segmentation.watershed(
            image: c.image, markers: c.markers.map { Int32($0) }, rows: c.height, columns: c.width, mask: c.mask.map(maskFrom),
            connectivity: c.connectivity == 1 ? .four : .eight)
        let expected = c.labels.map { Int32($0) }
        #expect(got == expected, "\(c.name): \(zip(got, expected).filter { $0 != $1 }.count) of \(got.count) labels differ")
    }

    @Test func watershedRefusals() {
        #expect(throws: MorphologyError.shape) { try Segmentation.watershed(image: [1, 2], markers: [1], rows: 1, columns: 2) }
        #expect(throws: MorphologyError.self) { try Segmentation.watershed(image: [1, .nan], markers: [1, 0], rows: 1, columns: 2) }
        #expect(throws: MorphologyError.self) { try Segmentation.markers(peakRows: [5], peakColumns: [0], rows: 2, columns: 2) }
    }

    @Test func markersFromPeaksNumberTheCoordinatesInOrder() throws {
        #expect(try Segmentation.markers(peakRows: [0, 1], peakColumns: [1, 0], rows: 2, columns: 2) == [0, 1, 2, 0])
    }

    // MARK: rolling_ball

    @Test(arguments: pins.rollingBall) func rollingBall(_ c: WatershedEdtRollingBallPins.RollingBall) throws {
        let kernel = try c.kernel.map { k in try RollingBall.Kernel(rows: k.height, columns: k.width, values: k.values.map { $0 ?? .infinity }) }
        if c.dtype == "float32" {
            let image = c.image.map { Float($0) }
            let got = kernel == nil ? try RollingBall.background(image: image, rows: c.height, columns: c.width, radius: c.radius ?? 0)
                : try RollingBall.background(image: image, rows: c.height, columns: c.width, kernel: kernel!)
            for (i, (g, e)) in zip(got, c.background).enumerated() { #expect(abs(Double(g) - e) <= 1e-5 * abs(e) + 1e-6, "\(c.name) pixel \(i): \(g) vs \(e)") }
        } else {
            let got = kernel == nil ? try RollingBall.background(image: c.image, rows: c.height, columns: c.width, radius: c.radius ?? 0)
                : try RollingBall.background(image: c.image, rows: c.height, columns: c.width, kernel: kernel!)
            for (i, (g, e)) in zip(got, c.background).enumerated() { #expect(abs(g - e) <= 1e-7 * abs(e) + 1e-12, "\(c.name) pixel \(i): \(g) vs \(e)") }
        }
    }

    @Test func ballAndEllipsoidKernelsMatchScikitImagesDefinition() throws {
        let k = try RollingBall.ballKernel(radius: 2)
        #expect(k.rows == 5 && k.columns == 5)
        #expect(k.values[2 * 5 + 2] == 2, "the apex")
        #expect(k.values[0] == .infinity, "outside the ball")
        #expect(k.values[2 * 5 + 4] == 0, "on the rim the height is zero")
        let e = try RollingBall.ellipsoidKernel(rows: 7, columns: 11, intensity: 6)
        #expect(e.rows == 7 && e.columns == 11 && e.values[3 * 11 + 5] == 6)
    }

    @Test func rollingBallRefusals() throws {
        #expect(throws: RestorationError.nonFinite) { try RollingBall.background(image: [1.0, .nan], rows: 1, columns: 2, radius: 1) }
        #expect(throws: RestorationError.shape) { try RollingBall.background(image: [1.0], rows: 1, columns: 2, radius: 1) }
        #expect(throws: RestorationError.self) { try RollingBall.ballKernel(radius: -1) }
    }
}
