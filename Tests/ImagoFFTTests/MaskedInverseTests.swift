// The masked inverse FFT against NumPy 2.5.4 (registration-slice-1.md 1.2, section 3, 5.3).
import Foundation
import ImagoCore
import ImagoFFT
import Testing

private let cases = Pins.fft.filter { $0.kind == "ifft" }
private let band: [Precision: Double] = [.float32: 1e-5, .float64: 1e-12]

@Test(arguments: cases, Precision.allCases)
func maskedInverseMatchesNumPy(_ c: FFTPin, _ precision: Precision) throws {
    let image = Pins.image(Pins.floats(c.input, section: "fft"), width: c.width, height: c.height,
                           scaleX: c.scale_x, scaleY: c.scale_y, units: c.units)
    let spots = c.spots!.map { MaskedInverseFFT.Spot(kx: $0.kx, ky: $0.ky, radius: $0.r, edge: $0.e) }
    let result = try MaskedInverseFFT.compute(image, spots: spots, precision: precision)
    let want = Pins.doubles(c.ifft!, section: "fft")
    let top = want.map(abs).max()!
    let worst = Pins.relativeToMaximum(result.image.doubles, want)
    #expect(worst <= band[precision]!, "real image: \(worst) of the maximum, band \(band[precision]!)")
    #expect(result.imaginaryMax <= band[precision]! * top, "imaginary residual \(result.imaginaryMax / top) of the maximum")
    #expect(result.image.precision == precision && result.spots == spots)
    // Without the Friedel partners the result is no longer real: the partners are what make it so.
    let lopsided = try MaskedInverseFFT.compute(image, spots: spots, includeFriedelPartners: false, precision: precision)
    #expect(lopsided.imaginaryMax > 0.1 * top)
}

@Test func maskWeightClosedForms() {
    let s = MaskedInverseFFT.Spot(kx: 3, ky: -2, radius: 2.5, edge: 2)
    #expect(s.weight(atDistance: 0) == 1 && s.weight(atDistance: 2.5) == 1)
    #expect(abs(s.weight(atDistance: 3.5) - 0.5) < 1e-15)                 // r + e/2: ½(1 + cos(π/2))
    #expect(s.weight(atDistance: 4.5) == 0 && s.weight(atDistance: 9) == 0)
    let hard = MaskedInverseFFT.Spot(kx: 0, ky: 0, radius: 3, edge: 0)
    #expect(hard.weight(atDistance: 3) == 1 && hard.weight(atDistance: 3.0000001) == 0)
}

@Test func aMeasuredSpotBecomesAMaskSpot() throws {
    let w = 64, h = 48
    let pixels = (0..<(w * h)).map { Float(cos(2 * Double.pi * (7 * Double($0 % w) / Double(w) + 5 * Double($0 / w) / Double(h)))) }
    let image = try Micrograph(width: w, height: h, pixels: pixels)
    let m = try SpotMeasurement.measure(image, clickRow: h / 2 + 5, clickColumn: w / 2 + 7)
    let spot = m.maskSpot(radius: 3, edge: 2)
    #expect(spot.kx == m.centroidCol - Double(w / 2) && spot.ky == m.centroidRow - Double(h / 2))
    #expect(abs(spot.kx - 7) < 1e-6 && abs(spot.ky - 5) < 1e-6 && spot.radius == 3 && spot.edge == 2)
    // The single wave masked by its own spot comes back: the rebuilt image equals the input to the band.
    let back = try MaskedInverseFFT.compute(image, spots: [spot], precision: .float64)
    let worst = zip(back.image.doubles, pixels).map { abs($0 - Double($1)) }.max()!
    #expect(worst < 1e-6)
}
