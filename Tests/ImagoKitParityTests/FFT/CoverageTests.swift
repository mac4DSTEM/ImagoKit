// Tests for the spot's crop path, the case counts, and the
// public members and refusals no other test asserts.
import Foundation
import ImagoKitTestSupport
import ImagoCore
import ImagoFFT
import Testing

@Test func everyPinnedCaseIsLoaded() {
    // A truncated manifest must not pass with fewer cases.
    let expected = FFTPins.full == nil ? (fft: 19, spot: 25, angle: 4) : (fft: 19, spot: 41, angle: 30)
    #expect(FFTPins.fft.count == expected.fft && FFTPins.spot.count == expected.spot && FFTPins.angle.count == expected.angle,
            "fft \(FFTPins.fft.count), spot \(FFTPins.spot.count), angle \(FFTPins.angle.count)")
    #expect(FFTPins.spot.contains { $0.neighbour } && FFTPins.spot.contains { $0.tier == "noisy" })
}

@Test(arguments: Precision.allCases)
func spotOnACropIsTheSpotOfTheCroppedImage(_ precision: Precision) throws {
    let c = try #require(FFTPins.spot.first { $0.name == "noisy256-k10-r0" })
    let image = FFTPins.image(FFTPins.floats(c.input, section: "spot"), width: c.width, height: c.height,
                           scaleX: c.scale_x, scaleY: c.scale_y, units: c.units)
    let region = PixelCrop(row: 40, col: 16, height: 192, width: 224)
    // The planted spot's bin in the crop's spectrum: its frequency in index units scales with the crop's size.
    let kx = Double(c.click_col - c.width / 2) * Double(region.width) / Double(c.width)
    let ky = Double(c.click_row - c.height / 2) * Double(region.height) / Double(c.height)
    let row = region.height / 2 + Int(ky.rounded()), col = region.width / 2 + Int(kx.rounded())
    let onCrop = try SpotMeasurement.measure(image, crop: region, clickRow: row, clickColumn: col, precision: precision)
    let cropped = try #require(image.cropped(region))
    let direct = try SpotMeasurement.measure(cropped, clickRow: row, clickColumn: col, precision: precision)
    // Not bit equality: a first call of a vDSP size can differ by one Float32 ulp, so the two
    // measurements agree to the registered centroid band.
    let band = precision == .float32 ? 1e-6 : 1e-10
    #expect(onCrop.snappedRow == direct.snappedRow && onCrop.snappedCol == direct.snappedCol && onCrop.units == direct.units)
    #expect(abs(onCrop.centroidRow - direct.centroidRow) <= band && abs(onCrop.centroidCol - direct.centroidCol) <= band)
    #expect(onCrop.width == region.width && onCrop.height == region.height)
    // The crop holds 192 × 224 pixels of the same lattice: the spacing stays the planted one within the noisy band's order.
    #expect(abs(onCrop.d - c.d0) / c.d0 < 0.02, "d \(onCrop.d) vs d₀ \(c.d0)")
    let whole = try SpotMeasurement.measure(image, clickRow: c.click_row, clickColumn: c.click_col, precision: precision)
    #expect(onCrop.centroidRow != whole.centroidRow || onCrop.centroidCol != whole.centroidCol, "the crop was not ignored")
    #expect(throws: SpotMeasurementError.crop(PixelCrop(row: 100, col: 0, height: 192, width: 16))) {
        try SpotMeasurement.measure(image, crop: PixelCrop(row: 100, col: 0, height: 192, width: 16), clickRow: 0, clickColumn: 0)
    }
}

@Test func spectrumMembersAndRefusals() throws {
    let image = try Micrograph(width: 40, height: 30, pixels: (0..<1200).map { Float($0 % 7) },
                               x: AxisCalibration(scale: 0.5, unit: "Å"), y: AxisCalibration(scale: 0.25, unit: "Å"))
    let s = try PowerSpectrum.compute(image)
    #expect(s.spacingX == 1 / (40 * 0.5) && s.spacingY == 1 / (30 * 0.25) && s.unitsX == "1/Å")
    #expect(throws: PowerSpectrumError.crop(PixelCrop(row: 0, col: 30, height: 10, width: 11))) {
        try PowerSpectrum.compute(image, crop: PixelCrop(row: 0, col: 30, height: 10, width: 11))
    }
    let p = try RadialProfile.compute(s)
    let points = p.points
    #expect(!points.isEmpty && points.allSatisfy { $0.bin >= 1 && $0.count > 0 })
    #expect(points.allSatisfy { $0.q == (Double($0.bin) + 0.5) * p.width && $0.mean == p.mean[$0.bin] })
    #expect(points.count == p.count.dropFirst().filter { $0 > 0 }.count)
}

@Test func spotUnitsInNanometres() throws {
    let w = 64
    let pixels = (0..<(w * w)).map { Float(cos(2 * Double.pi * 9 * Double($0 % w) / Double(w))) }
    let image = try Micrograph(width: w, height: w, pixels: pixels,
                               x: AxisCalibration(scale: 2, unit: "Å"), y: AxisCalibration(scale: 2, unit: "Å"))
    let m = try SpotMeasurement.measure(image, clickRow: 32, clickColumn: 41)
    #expect(m.units == "Å" && m.reciprocalUnits == "1/Å")
    #expect(m.dNanometres == m.d * 0.1 && m.qPerNanometre == m.q / 0.1)
}
