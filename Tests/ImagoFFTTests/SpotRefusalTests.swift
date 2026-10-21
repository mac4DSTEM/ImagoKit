// A powerless or non-finite window is refused by name instead of answering NaN (ROADMAP ImagoKit study row 1).
// Truth: NumPy's reading of the estimator gives 0/0 for a constant window (hanning × (x − mean) = 0), so there is no number to match.
import Foundation
import ImagoCore
import ImagoFFT
import Testing

private func image(_ w: Int, _ h: Int, _ f: (Int, Int) -> Float) -> Micrograph {
    var p = [Float](repeating: 0, count: w * h)
    for r in 0..<h { for c in 0..<w { p[r * w + c] = f(r, c) } }
    return try! Micrograph(width: w, height: h, pixels: p, x: AxisCalibration(scale: 0.1, unit: "nm"),
                           y: AxisCalibration(scale: 0.1, unit: "nm"))
}

@Test(arguments: Precision.allCases)
func constantImageIsRefusedAsPowerless(_ precision: Precision) {
    for value in [Float(0), 7, 0.1, -3.3e6] {
        #expect(throws: SpotMeasurementError.noPower) {
            try SpotMeasurement.measure(image(64, 64) { _, _ in value }, clickRow: 20, clickColumn: 40, precision: precision)
        }
    }
}

@Test(arguments: Precision.allCases)
func constantCropOfATexturedImageIsRefused(_ precision: Precision) throws {
    let textured = image(128, 64) { r, c in c < 64 ? Float(sin(2 * .pi * Double(c) / 8)) : 5 }
    // The right half is constant.
    #expect(throws: SpotMeasurementError.noPower) {
        try SpotMeasurement.measure(textured, crop: PixelCrop(row: 0, col: 64, height: 64, width: 64), clickRow: 20, clickColumn: 40,
                                    precision: precision)
    }
    // The left half is a lattice and still measures (period 8 px × 0.1 nm = 0.8 nm).
    let m = try SpotMeasurement.measure(textured, crop: PixelCrop(row: 0, col: 0, height: 64, width: 64), clickRow: 32,
                                        clickColumn: 32 + 8, precision: precision)
    #expect(abs(m.d - 0.8) / 0.8 < 0.01)
}

@Test(arguments: Precision.allCases)
func nonFinitePixelIsRefused(_ precision: Precision) {
    for bad in [Float.nan, .infinity, -.infinity] {
        #expect(throws: SpotMeasurementError.nonFinite) {
            try SpotMeasurement.measure(image(64, 64) { r, c in r == 3 && c == 4 ? bad : Float(c % 8) },
                                        clickRow: 32, clickColumn: 40, precision: precision)
        }
    }
}
