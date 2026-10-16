// ImagoCore: the unit conversions and the offset form of an axis. The pinned source is the transcription below, copied
// as it stands in the consumer's calibration code at the time of porting (spelling tables unchanged), so a difference
// from it is a deliberate change. The closed forms (1 nm = 10 Å, 1 pm = 0.01 Å, mrad = 1/(1000 λ) Å⁻¹) are checked
// independently.
import Foundation
import ImagoCore
import Testing

private enum Pinned {
    static func normalized(_ unit: String?) -> String {
        (unit ?? "").precomposedStringWithCanonicalMapping
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            .replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "ångström", with: "angstrom")
    }
    static func real(_ value: Double, _ units: String?) -> Double? {
        guard value.isFinite, value > 0 else { return nil }
        switch normalized(units) {
        case "a", "å", "angstrom", "ang": return value
        case "nm", "nanometer", "nanometers", "nanometre", "nanometres": return value * 10
        case "pm", "picometer", "picometers", "picometre", "picometres": return value * 0.01
        default: return nil
        }
    }
    static func canonicalReal(_ units: String?) -> String? {
        switch normalized(units) {
        case "a", "å", "angstrom", "ang": return "Å"
        case "nm", "nanometer", "nanometers", "nanometre", "nanometres": return "nm"
        case "pm", "picometer", "picometers", "picometre", "picometres": return "pm"
        default: return nil
        }
    }
    static func reciprocal(_ value: Double, _ units: String?, _ wavelength: Double?) -> Double? {
        guard value.isFinite, value > 0 else { return nil }
        switch normalized(units) {
        case "1/a", "1/å", "a^-1", "å^-1", "a⁻¹", "å⁻¹", "angstrom^-1", "angstrom⁻¹", "1/angstrom", "ang^-1", "ang⁻¹":
            return value
        case "1/nm", "nm^-1", "nm⁻¹", "1/nanometer", "1/nanometre", "nanometer^-1", "nanometer⁻¹", "nanometre^-1",
             "nanometre⁻¹":
            return value * 0.1
        case "mrad":
            guard let wavelength, wavelength.isFinite, wavelength > 0 else { return nil }
            return value / (1_000 * wavelength)
        default: return nil
        }
    }
    static func canonicalReciprocal(_ units: String?) -> String? {
        switch normalized(units) {
        case "1/a", "1/å", "a^-1", "å^-1", "a⁻¹", "å⁻¹", "angstrom^-1", "angstrom⁻¹", "1/angstrom", "ang^-1", "ang⁻¹":
            return "Å⁻¹"
        case "1/nm", "nm^-1", "nm⁻¹", "1/nanometer", "1/nanometre", "nanometer^-1", "nanometer⁻¹", "nanometre^-1",
             "nanometre⁻¹":
            return "nm⁻¹"
        case "mrad": return "mrad"
        default: return nil
        }
    }
    static func nanometres(_ unit: String) -> Double? {
        switch normalized(unit) {
        case "m", "meter", "meters", "metre", "metres": 1e9
        case "mm", "millimeter", "millimeters", "millimetre", "millimetres": 1e6
        case "\u{00B5}m", "\u{03BC}m", "um", "micron", "microns", "micrometer", "micrometers", "micrometre", "micrometres": 1e3
        case "nm", "nanometer", "nanometers", "nanometre", "nanometres": 1
        case "\u{00E5}", "a", "angstrom", "angstroms", "ang": 0.1
        case "pm", "picometer", "picometers", "picometre", "picometres": 1e-3
        default: nil
        }
    }
    static func isPixel(_ units: String?) -> Bool {
        switch normalized(units) {
        case "px", "pix", "[pix]", "[px]", "pixel", "pixels", "1", "1/px", "1/pixel", "1/pixels": true
        default: false
        }
    }
}

private let spellings: [String?] = [
    nil, "", " ", "Å", "A", "a", "Å", "A\u{030A}", "Ang", "ang", "Angstrom", "angstroms", "Ångström", "ångström", "nm", "NM",
    " nm ", "n m", "nanometer", "Nanometres", "pm", "picometre", "µm", "μm", "um", "micron", "mm", "m", "metres", "px", "pix",
    "[pix]", "[px]", "pixel", "pixels", "1", "1/px", "1/pixel", "1/pixels", "1/Å", "1/A", "1/a", "1/nm", "1/NM", "A^-1",
    "Å^-1", "A⁻¹", "Å⁻¹", "angstrom^-1", "angstrom⁻¹", "1/angstrom", "ang^-1", "ang⁻¹", "nm^-1", "nm⁻¹", "1/nanometer",
    "1/nanometre", "nanometer^-1", "nanometre⁻¹", "mrad", "MRAD", "rad", "1/m", "1/µm", "deg", "keV",
]
private let scales: [Double] = [0.01, 0.5, 1, 2.5, 123.456, 0, -1, .nan, .infinity]

@Suite("UnitConversion") struct UnitConversionTests {
    @Test func realMatchesThePinnedSource() {
        for u in spellings {
            for s in scales {
                #expect(LengthUnit.angstromsPerPixel(scale: s, unit: u) == Pinned.real(s, u), "\(String(describing: u)) \(s)")
            }
            #expect(LengthUnit.canonicalRealUnit(u) == Pinned.canonicalReal(u))
        }
    }

    @Test func reciprocalMatchesThePinnedSource() {
        for u in spellings {
            for s in scales {
                for lambda in [nil, 0.0251, 0.0, -1.0, Double.nan, 0.0197] as [Double?] {
                    let got = LengthUnit.reciprocalAngstromsPerPixel(scale: s, unit: u, wavelengthAngstrom: lambda)
                    #expect(got == Pinned.reciprocal(s, u, lambda), "\(String(describing: u)) \(s) \(String(describing: lambda))")
                }
            }
            #expect(LengthUnit.canonicalReciprocalUnit(u) == Pinned.canonicalReciprocal(u))
            #expect(LengthUnit.isPhysicalReciprocal(u) == (Pinned.reciprocal(1, u, 0.0251) != nil))
            #expect(LengthUnit.isPixel(u) == Pinned.isPixel(u))
        }
    }

    @Test func oneLengthTableServesBothSpellings() {
        // The nanometre table of the spectrum code and of the Å conversion are one function now.
        for u in spellings.compactMap({ $0 }) {
            #expect(LengthUnit.nanometres(per: u) == Pinned.nanometres(u), "\(u)")
        }
    }

    @Test func closedForms() {
        #expect(LengthUnit.angstromsPerPixel(scale: 0.25, unit: "nm") == 2.5)
        #expect(LengthUnit.angstromsPerPixel(scale: 50, unit: "pm") == 0.5)
        #expect(LengthUnit.angstromsPerPixel(scale: 3, unit: "Å") == 3)
        #expect(LengthUnit.angstromsPerPixel(scale: 3, unit: "µm") == nil)
        #expect(LengthUnit.reciprocalAngstromsPerPixel(scale: 5, unit: "1/nm") == 0.5)
        // 1 mrad at 200 kV (λ = 0.02508 Å) is 1/(1000 · 0.02508) Å⁻¹.
        #expect(LengthUnit.reciprocalAngstromsPerPixel(scale: 1, unit: "mrad", wavelengthAngstrom: 0.02508)
                == 1 / (1000 * 0.02508))
        #expect(LengthUnit.reciprocalAngstromsPerPixel(scale: 1, unit: "mrad") == nil)
        #expect(LengthUnit.editableRealUnits.allSatisfy { LengthUnit.canonicalRealUnit($0) == $0 })
        #expect(LengthUnit.editableReciprocalUnits.allSatisfy { LengthUnit.canonicalReciprocalUnit($0) == $0 })
    }

    @Test func axisOffsetForm() {
        let a = AxisCalibration(scale: 0.2, unit: "nm", origin: 10)
        #expect(a.offset == -2)                              // value at pixel 0
        #expect(a.value(at: 0) == a.offset)
        let b = AxisCalibration(offset: a.offset, scale: a.scale, unit: a.unit)
        #expect(b == a)
        #expect(b.value(at: 17.5) == a.value(at: 17.5))
        let c = AxisCalibration(offset: 5, scale: 0, unit: "nm")
        #expect(c.origin == 0 && !c.isCalibrated)
        let d = AxisCalibration(offset: 1.5, scale: 2, unit: "nm")
        #expect(d.value(at: 0) == 1.5 && d.value(at: 4) == 9.5)
    }
}
