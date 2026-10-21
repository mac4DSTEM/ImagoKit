// Spectrum and radial profile against NumPy 2.5.4 and the planted truth (registration-slice-1.md sections 3, 5.1).
import Foundation
import ImagoKitTestSupport
import ImagoCore
import ImagoFFT
import Testing

private let spectra = FFTPins.fft.filter { $0.kind == "spectrum" }
private let profiles = FFTPins.fft.filter { $0.kind == "profile" }
private let bands: [Precision: (spectrum: Double, profile: Double)] = [.float32: (1e-5, 1e-5), .float64: (1e-12, 1e-11)]

private func input(_ c: FFTPin) -> Micrograph {
    FFTPins.image(FFTPins.floats(c.input, section: "fft"), width: c.width, height: c.height,
               scaleX: c.scale_x, scaleY: c.scale_y, units: c.units)
}

private func crop(_ c: FFTPin) -> PixelCrop? {
    c.crop.map { PixelCrop(row: $0.row, col: $0.col, height: $0.height, width: $0.width) }
}

@Test(arguments: spectra, Precision.allCases)
func spectrumMatchesNumPy(_ c: FFTPin, _ precision: Precision) throws {
    let s = try PowerSpectrum.compute(input(c), crop: crop(c), precision: precision)
    let wantPower = FFTPins.doubles(c.power!, section: "fft")
    let band = bands[precision]!.spectrum
    let power = FFTPins.relativeToMaximum(s.power.doubles, wantPower)
    #expect(power <= band, "power: \(power) of the maximum, band \(band)")
    let magnitude = FFTPins.relativeToMaximum(s.magnitude.doubles, wantPower.map { $0.squareRoot() })
    #expect(magnitude <= band, "magnitude: \(magnitude) of the maximum, band \(band)")
    #expect(s.power.precision == precision)
    // The axes are NumPy's fftshift(fftfreq(n, d)); asserted exact today (D12 allows a written tolerance).
    #expect(s.axisX == FFTPins.doubles(c.axis_x!, section: "fft"))
    #expect(s.axisY == FFTPins.doubles(c.axis_y!, section: "fft"))
    #expect(s.unitsX == c.axis_units && s.unitsY == c.axis_units)
}

@Test(arguments: spectra.filter { !($0.peaks ?? []).isEmpty }, Precision.allCases)
func plantedPeaksAreLocalMaxima(_ c: FFTPin, _ precision: Precision) throws {
    let s = try PowerSpectrum.compute(input(c), precision: precision)
    let w = s.width, h = s.height
    for peak in c.peaks! {
        let (r, col) = (peak[0], peak[1])
        let v = s.power[r * w + col]
        for dr in -1...1 { for dc in -1...1 where (dr, dc) != (0, 0) {
            let rr = r + dr, cc = col + dc
            guard (0..<h).contains(rr), (0..<w).contains(cc) else { continue }
            #expect(s.power[rr * w + cc] < v, "planted peak (\(r), \(col)) is not a local maximum")
        } }
    }
}

@Test(arguments: spectra.filter { $0.crop != nil }, Precision.allCases)
func cropEqualsTheCroppedImage(_ c: FFTPin, _ precision: Precision) throws {
    let image = input(c)
    let region = crop(c)!
    let a = try PowerSpectrum.compute(image, crop: region, precision: precision)
    let cropped = try #require(image.cropped(region))
    let b = try PowerSpectrum.compute(cropped, precision: precision)
    // The same computation twice; equal to one Float32 ulp, not bit for bit (a first vDSP call can differ, amendment 8).
    let band = precision == .float32 ? 1e-6 : 1e-14
    #expect(FFTPins.relativeToMaximum(a.power.doubles, b.power.doubles) <= band && a.axisX == b.axisX && a.axisY == b.axisY)
    // The crop's pixels are the harness's cropped input, and its origin moved by the offset.
    #expect(cropped.pixels == FFTPins.floats(c.cropped_input ?? c.name + ".cropped.in.f32", section: "fft"))
    #expect(cropped.x?.origin == -Double(region.col) && cropped.y?.origin == -Double(region.row))
}

@Test(arguments: profiles, Precision.allCases)
func radialProfileMatchesNumPy(_ c: FFTPin, _ precision: Precision) throws {
    let s = try PowerSpectrum.compute(input(c), precision: precision)
    let p = try RadialProfile.compute(s, width: c.bin_width_given)
    let want = FFTPins.doubles(c.pmean!, section: "fft")
    let counts = FFTPins.ints(c.pcount!, section: "fft")
    #expect(p.width == c.bin_width!)
    #expect(p.count == counts)
    try #require(p.mean.count == want.count)
    let worst = zip(p.mean, want).map { abs($0 - $1) }.max()!
    let peak = want.map(abs).max()!
    // The second band is over bins k ≥ 1 against the largest of them (the DC bin, ~10⁶ × a ring, is held by the first).
    let ringTop = want.dropFirst().map(abs).max()!
    let worstAway = zip(p.mean, want).dropFirst().map { abs($0 - $1) }.max()!
    let band = bands[precision]!.profile
    if precision == .float32 { #expect(worst <= 1e-5 * peak, "\(worst / peak) of the maximum") }
    #expect(worstAway <= band * ringTop, "bins ≥ 1: \(worstAway / ringTop) of the largest of them, band \(band)")
    // Truth: the planted ring peaks in the bin holding q0 (bin 0, the DC, excluded).
    let ring = p.mean.indices.dropFirst().max { p.mean[$0] < p.mean[$1] }!
    #expect(ring == c.ring_bin!, "ring peaks in bin \(ring), planted \(c.ring_bin!)")
}

@Test func profileRefusesMixedUnitsAndUnusableWidths() throws {
    let image = try Micrograph(width: 32, height: 32, pixels: [Float](repeating: 1, count: 1024),
                               x: AxisCalibration(scale: 0.1, unit: "nm"), y: AxisCalibration(scale: 0.1, unit: "Å"))
    let s = try PowerSpectrum.compute(image)
    #expect(throws: RadialProfileError.mixedUnits("1/nm", "1/Å")) { try RadialProfile.compute(s) }
    let square = try Micrograph(width: 32, height: 32, pixels: [Float](repeating: 1, count: 1024))
    let t = try PowerSpectrum.compute(square)
    #expect(throws: RadialProfileError.unusableWidth(0)) { try RadialProfile.compute(t, width: 0) }
    #expect(throws: RadialProfileError.self) { try RadialProfile.compute(t, width: 1e-9) }
}
