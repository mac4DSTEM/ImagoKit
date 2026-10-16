//
//  UnitConversion.swift (ImagoCore)
//  Conversions between the unit spellings image and diffraction files write and the units a calculation needs:
//  angstroms per pixel, reciprocal angstroms per pixel, and the canonical spelling of each. Closed forms (1 nm = 10 Å,
//  1 pm = 0.01 Å, 1 mrad at wavelength λ Å is 1/(1000 λ) Å⁻¹); no display formatting lives here.
//

import Foundation

extension LengthUnit {
    /// Real-space units a calibration control offers, in their canonical spelling.
    public static let editableRealUnits = ["Å", "nm", "pm"]
    /// Reciprocal-space units a calibration control offers, in their canonical spelling.
    public static let editableReciprocalUnits = ["Å⁻¹", "nm⁻¹", "mrad"]

    /// The unit folded for comparison; `nil` reads as the empty string.
    public static func normalized(_ unit: String?) -> String { normalized(unit ?? "") }

    /// Angstroms in one pixel of a real-space sampling of `scale` per pixel in `unit` (Å, nm and pm in their spellings).
    /// nil for a non-finite or non-positive scale, and for every other unit (metres, micrometres, pixels), which this
    /// conversion deliberately does not read; `nanometres(per:)` is the wider length table.
    public static func angstromsPerPixel(scale: Double, unit: String?) -> Double? {
        guard scale.isFinite, scale > 0 else { return nil }
        switch canonicalRealUnit(unit) {
        case "Å": return scale
        case "nm": return scale * 10
        case "pm": return scale * 0.01
        default: return nil
        }
    }

    /// The canonical spelling ("Å", "nm" or "pm") of a real-space unit; nil for any other.
    public static func canonicalRealUnit(_ unit: String?) -> String? {
        switch normalized(unit) {
        case "a", "å", "angstrom", "ang": "Å"
        case "nm", "nanometer", "nanometers", "nanometre", "nanometres": "nm"
        case "pm", "picometer", "picometers", "picometre", "picometres": "pm"
        default: nil
        }
    }

    /// Reciprocal angstroms in one pixel of a reciprocal-space sampling of `scale` per pixel in `unit` (Å⁻¹ and nm⁻¹ in
    /// their spellings, or mrad, which needs the electron `wavelengthAngstrom`). nil for a non-finite or non-positive
    /// scale, an unknown unit, and mrad without a finite positive wavelength.
    public static func reciprocalAngstromsPerPixel(scale: Double, unit: String?,
                                                   wavelengthAngstrom: Double? = nil) -> Double? {
        guard scale.isFinite, scale > 0 else { return nil }
        switch canonicalReciprocalUnit(unit) {
        case "Å⁻¹": return scale
        case "nm⁻¹": return scale * 0.1
        case "mrad":
            guard let wavelengthAngstrom, wavelengthAngstrom.isFinite, wavelengthAngstrom > 0 else { return nil }
            return scale / (1_000 * wavelengthAngstrom)
        default: return nil
        }
    }

    /// The canonical spelling ("Å⁻¹", "nm⁻¹" or "mrad") of a reciprocal-space unit; nil for any other.
    public static func canonicalReciprocalUnit(_ unit: String?) -> String? {
        switch normalized(unit) {
        case "1/a", "1/å", "a^-1", "å^-1", "a⁻¹", "å⁻¹", "angstrom^-1", "angstrom⁻¹", "1/angstrom", "ang^-1", "ang⁻¹":
            "Å⁻¹"
        case "1/nm", "nm^-1", "nm⁻¹", "1/nanometer", "1/nanometre", "nanometer^-1", "nanometer⁻¹", "nanometre^-1",
             "nanometre⁻¹":
            "nm⁻¹"
        case "mrad": "mrad"
        default: nil
        }
    }

    /// Whether `unit` is a reciprocal-space unit this conversion reads (Å⁻¹, nm⁻¹ or mrad).
    public static func isPhysicalReciprocal(_ unit: String?) -> Bool { canonicalReciprocalUnit(unit) != nil }

    /// Whether `unit` says "pixels": "px", "pix", "[pix]", "[px]", "pixel", "pixels", "1", "1/px", "1/pixel", "1/pixels".
    public static func isPixel(_ unit: String?) -> Bool {
        switch normalized(unit) {
        case "px", "pix", "[pix]", "[px]", "pixel", "pixels", "1", "1/px", "1/pixel", "1/pixels": true
        default: false
        }
    }
}
