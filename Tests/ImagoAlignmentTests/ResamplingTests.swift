// Closed forms for zoom and rotate (suite task 20): a constant image stays constant, quarter turns permute the pixels exactly,
// the calibration follows the zoom, refusals. The pinned comparison with SciPy is in ImagoKitParityTests.
import Foundation
import ImagoAlignment
import ImagoCore
import Testing

@Suite("Resampling closed forms") struct ResamplingClosedFormTests {
    private func ramp(_ w: Int, _ h: Int) throws -> Micrograph {
        try Micrograph(width: w, height: h, pixels: (0..<w * h).map { Float($0) })
    }

    @Test func aConstantImageStaysConstantUnderZoomAtEveryOrder() throws {
        let image = try Micrograph(width: 5, height: 4, pixels: [Float](repeating: 3, count: 20))
        for order in ShiftOrder.allCases {
            let z = try ImageResampling.zoom(image, zoom: (2.5, 1.5), order: order, mode: .mirror)
            #expect(z.width == 8 && z.height == 10)   // round(7.5) = 8 (half to even), round(10)
            #expect(z.pixels.allSatisfy { abs($0 - 3) < 1e-5 }, "order \(order.rawValue)")
        }
    }

    @Test func aZoomOfOneIsTheImage() throws {
        let image = try ramp(4, 3)
        #expect(try ImageResampling.zoom(image, zoom: (1, 1)) == image)
    }

    @Test func zoomKeepsTheCornersAndRescalesTheCalibration() throws {
        let image = try Micrograph(width: 3, height: 3, pixels: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                   x: AxisCalibration(scale: 2, unit: "nm", origin: 1), y: AxisCalibration(scale: 4, unit: "nm"))
        let z = try ImageResampling.zoom(image, zoom: (2, 2), order: .linear)   // 3 -> 6, corners of the grid coincide
        #expect(z.pixels[0] == 0 && z.pixels[5] == 2 && z.pixels[30] == 6 && z.pixels[35] == 8)
        #expect(z.x == AxisCalibration(scale: 1, unit: "nm", origin: 2) && z.y == AxisCalibration(scale: 2, unit: "nm", origin: 0))
    }

    @Test func quarterTurnsPermuteThePixelsExactly() throws {
        let image = try ramp(4, 3)       // rows 0 1 2 3 / 4 5 6 7 / 8 9 10 11
        let r90 = try ImageResampling.rotate(image, degrees: 90, order: .nearest)
        #expect(r90.width == 3 && r90.height == 4)
        let r180 = try ImageResampling.rotate(image, degrees: 180, order: .nearest)
        #expect(r180.pixels == image.pixels.reversed())
        let r360 = try ImageResampling.rotate(image, degrees: 360, order: .cubic)
        #expect(zip(r360.pixels, image.pixels).allSatisfy { abs($0 - $1) < 1e-4 })
    }

    @Test func rotationOutsideTheImageIsCval() throws {
        let image = try Micrograph(width: 6, height: 6, pixels: [Float](repeating: 1, count: 36))
        let r = try ImageResampling.rotate(image, degrees: 45, order: .linear, mode: .constant, cval: -7)
        #expect(r.pixels[0] == -7 && r.pixels.contains(1))
        let kept = try ImageResampling.rotate(image, degrees: 45, reshape: false, order: .linear, mode: .constant, cval: -7)
        #expect(kept.width == 6 && kept.height == 6)
    }

    @Test func refusals() throws {
        let image = try ramp(3, 3)
        #expect(throws: AlignmentError.self) { try ImageResampling.zoom(image, zoom: (0, 1)) }
        #expect(throws: AlignmentError.self) { try ImageResampling.zoom(image, zoom: (.nan, 1)) }
        #expect(throws: AlignmentError.self) { try ImageResampling.zoom(image, zoom: (1e9, 1)) }
        #expect(throws: AlignmentError.self) { try ImageResampling.rotate(image, degrees: .infinity) }
        #expect(throws: AlignmentError.splineMode(.wrap)) { try ImageResampling.rotate(image, degrees: 30, order: .cubic, mode: .wrap) }
    }
}
