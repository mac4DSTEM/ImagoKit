// ndimage.zoom and ndimage.rotate against the pins of tools/resampling/pins.py. Tolerance: SciPy's interpolation tests
// use decimal 6; the pins are float32, so 1.5e-6 absolute is held in both outputs' range [0, 1) (as the kit's shift pins are).
// Shapes and the cval pixels (a source coordinate outside the image in constant mode) are exact.
import Foundation
import ImagoKitTestSupport
import ImagoAlignment
import ImagoCore
import Testing

@Suite("zoom and rotate (SciPy 1.18.1)") struct ResamplingTests {
    static let pins = ResamplingPins.all

    @Test func versions() { #expect(Self.pins.meta.numpy == "2.5.4" && Self.pins.meta.scipy == "1.18.1") }

    private func mode(_ s: String) -> ShiftMode {
        switch s {
        case "constant": .constant
        case "grid-constant": .gridConstant
        case "nearest": .nearest
        case "reflect": .reflect
        case "mirror": .mirror
        case "wrap": .wrap
        default: .gridWrap
        }
    }

    @Test(arguments: pins.cases) func matchesSciPy(_ c: ResamplingPins.Case) throws {
        let image = try Micrograph(width: c.width, height: c.height, pixels: c.pixels)
        let order = try #require(ShiftOrder(rawValue: c.order))
        let got: Micrograph
        if c.kind == "zoom" {
            got = try ImageResampling.zoom(image, zoom: (c.zoomRows!, c.zoomColumns!), order: order, mode: mode(c.mode), cval: c.cval,
                                           gridMode: c.gridMode!)
        } else {
            got = try ImageResampling.rotate(image, degrees: c.angle!, reshape: c.reshape!, order: order, mode: mode(c.mode), cval: c.cval)
        }
        #expect(got.height == c.outHeight && got.width == c.outWidth, "\(c): shape \(got.height) x \(got.width) vs \(c.outHeight) x \(c.outWidth)")
        guard got.pixels.count == c.result.count else { return }
        var worst: Float = 0, worstAt = 0
        for (i, (g, e)) in zip(got.pixels, c.result).enumerated() where abs(g - e) > worst { worst = abs(g - e); worstAt = i }
        #expect(worst <= 1.5e-6, "\(c): worst difference \(worst) at pixel \(worstAt): \(got.pixels[worstAt]) vs \(c.result[worstAt])")
    }
}
