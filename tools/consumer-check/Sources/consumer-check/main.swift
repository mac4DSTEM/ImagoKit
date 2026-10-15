// ImagoKit consumer check: every symbol named here must be `public`, and each call is checked against a value a
// stranger's app could compute by hand (a planted lattice, a closed form).
import Foundation
import ImagoAlignment
import ImagoCore
import ImagoFFT
import ImagoFilters

var checks = 0
var failures = 0
@MainActor func check(_ condition: Bool, _ label: String) {
    checks += 1
    if !condition { failures += 1; print("FAIL  \(label)") }
}

// A 128 × 96 lattice, 0.0123 nm by 0.0246 nm pixels: one wave at (kx, ky) = (10, −7) bins on DC 5, and one at (−4, 9).
let (w, h) = (128, 96)
let (dx, dy) = (0.0123, 0.0246)
var pixels = [Float](repeating: 0, count: w * h)
for r in 0..<h { for c in 0..<w {
    let a = 2 * Double.pi * (10 * Double(c) / Double(w) - 7 * Double(r) / Double(h))
    let b = 2 * Double.pi * (-4 * Double(c) / Double(w) + 9 * Double(r) / Double(h))
    pixels[r * w + c] = Float(5 + cos(a) + cos(b))
} }
let image = try Micrograph(width: w, height: h, pixels: pixels,
                           x: AxisCalibration(scale: dx, unit: "nm"), y: AxisCalibration(scale: dy, unit: "nm"))

// ImagoCore: the engines and the units.
let fft: FFT2D = try { guard let f = FFT2D(nx: w, ny: h) else { throw MicrographError.emptySize(width: w, height: h) }; return f }()
let fftD: FFT2DDouble? = FFT2DDouble(nx: w, ny: h)
check(fft.nx == w && fftD?.ny == h, "FFT2D and FFT2DDouble are made for any size")
check(FFT2D.fftfreq(3, 4) == -0.25, "fftfreq is NumPy's")
check(LengthUnit.nanometres(per: "Å") == 0.1, "Å is 0.1 nm")
check(image.cropped(PixelCrop(row: 0, col: 0, height: 32, width: 32))?.width == 32, "a crop")

// ImagoFFT: the spectrum, its axes, the profile, two spots, their spacings and angle.
for precision in Precision.allCases {
    let s = try PowerSpectrum.compute(image, precision: precision)
    check(s.power.precision == precision && s.unitsX == "1/nm", "\(precision): the spectrum's unit")
    check(s.spacingX == 1 / (Double(w) * dx), "\(precision): Δq_x = 1/(w·dx)")
    let dc = s.power[(h / 2) * w + w / 2]
    check(abs(dc / pow(5 * Double(w * h), 2) - 1) < 1e-6, "\(precision): the DC bin holds (DC·w·h)²")
    let peak = s.power[(h / 2 - 7) * w + w / 2 + 10]
    check(abs(peak / pow(Double(w * h) / 2, 2) - 1) < 1e-5, "\(precision): a bin-centred wave holds (w·h/2)²")
    let profile = try RadialProfile.compute(s)
    check(profile.count.reduce(0, +) == w * h && !profile.points.isEmpty, "\(precision): the profile counts every pixel")
    let a = try SpotMeasurement.measure(image, clickRow: h / 2 - 7, clickColumn: w / 2 + 10, precision: precision)
    let b = try SpotMeasurement.measure(image, clickRow: h / 2 + 9, clickColumn: w / 2 - 4, precision: precision)
    let qa = (10 / (Double(w) * dx), -7 / (Double(h) * dy)), qb = (-4 / (Double(w) * dx), 9 / (Double(h) * dy))
    check(abs(a.d * (qa.0 * qa.0 + qa.1 * qa.1).squareRoot() - 1) < 2e-3, "\(precision): spot a's d is 1/|q|")
    check(a.dNanometres == a.d && a.qPerNanometre == a.q, "\(precision): nm units")
    let pair = try LatticePair.measure(a, b)
    let planted = try SpotMeasurement.angle(qa: qa, qb: qb)
    check(abs(pair.angle - planted) < 0.1, "\(precision): the angle between the spots is the planted one")
    check(abs(pair.ratio - a.d / b.d) == 0, "\(precision): the spacing ratio")
}
do {
    _ = try SpotMeasurement.measure(image, clickRow: h / 2, clickColumn: w / 2)   // the ±2 snap stays within 2.83 bins
    check(false, "a click at the centre is refused")
} catch let error as SpotMeasurementError {   // bound, then matched: the runner's compiler crashes on `catch E.case`
    check(error == .nearDC, "a click at the centre is refused")
}

// ImagoFFT's masked inverse: lattice a alone, rebuilt from its spot.
let masked = try MaskedInverseFFT.compute(image, spots: [MaskedInverseFFT.Spot(kx: 10, ky: -7, radius: 2, edge: 1)])
let one = (0..<(w * h)).map { i -> Double in
    cos(2 * Double.pi * (10 * Double(i % w) / Double(w) - 7 * Double(i / w) / Double(h)))
}
check(zip(masked.image.doubles, one).map { abs($0 - $1) }.max()! < 1e-4, "the masked inverse keeps lattice a only")
check(masked.imaginaryMax < 1e-3, "the masked inverse is real")

// ImagoFilters: the HRTEM filters on a 64 × 64 crop, both precisions.
let square = image.cropped(PixelCrop(row: 0, col: 0, height: 64, width: 64))!
for precision in Precision.allCases {
    let r = try HRTEMFilter.filter(square, parameters: .defaults, precision: precision)
    let (absf, wiener) = try HRTEMFilter.outputs(r, like: square)
    check(absf.width == 64 && wiener.x == square.x && r.passes >= 1, "\(precision): the HRTEM filters run")
}
do {
    _ = try HRTEMFilter.filter(image)
    check(false, "a non-square image is refused")
} catch let error as HRTEMFilterError {
    check(error == .notSquare(width: 128, height: 96), "a non-square image is refused")
}

// The module still to come names its registration.
check(ImagoAlignmentScope.registration.hasPrefix("docs/registration-slice-2.md"), "ImagoAlignment names its registration")

print("consumer-check summary: \(checks) checks, \(failures) failures")
if failures > 0 { exit(1) }
