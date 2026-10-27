// README.md's program, verbatim (tools/readme-consumer/run.sh checks it line by line), then checks on its numbers.
import Foundation
import ImagoAlignment
import ImagoCore
import ImagoFFT
import ImagoFilters

var real = [Float](repeating: 0, count: 64)
var imaginary = real
real[0] = 1
let fft = FFT1D(n: real.count)!
fft.transform(re: &real, im: &imaginary, forward: true)
fft.transform(re: &real, im: &imaginary, forward: false)
// Both directions are unnormalised: divide these inverse results by Float(fft.n).

// A 256 × 256 image, 0.02 nm pixels: two lattice fringes on a constant background.
let n = 256
var pixels = [Float](repeating: 0, count: n * n)
for row in 0..<n {
    for col in 0..<n {
        let x = Double(col) / Double(n), y = Double(row) / Double(n)
        pixels[row * n + col] = Float(100 + 10 * cos(2 * Double.pi * (40 * x + 12 * y)) + 6 * cos(2 * Double.pi * (-15 * x + 35 * y)))
    }
}
let image = try Micrograph(width: n, height: n, pixels: pixels,
                           x: AxisCalibration(scale: 0.02, unit: "nm"), y: AxisCalibration(scale: 0.02, unit: "nm"))

// The centred spectrum (NumPy's fftshift(fft2(x))) and its radial profile against physical |q|.
let spectrum = try PowerSpectrum.compute(image)
let profile = try RadialProfile.compute(spectrum)
print("spectrum", spectrum.width, "×", spectrum.height, "in", spectrum.unitsX, "; profile bins:", profile.points.count)

// Click near each spot (row, column of the centred spectrum): the estimator snaps and refines to sub-pixel.
let a = try SpotMeasurement.measure(image, clickRow: n / 2 + 12, clickColumn: n / 2 + 40)
let b = try SpotMeasurement.measure(image, clickRow: n / 2 + 35, clickColumn: n / 2 - 15)
let pair = try LatticePair.measure(a, b)
print(String(format: "d₁ = %.4f nm, d₂ = %.4f nm, angle = %.2f°", a.dNanometres!, b.dNanometres!, pair.angle))

// Keep only the first lattice (a Bragg filter), then the HRTEM background-subtraction and Wiener filters.
let bragg = try MaskedInverseFFT.compute(image, spots: [a.maskSpot(radius: 3, edge: 2)])
let filtered = try HRTEMFilter.filter(image)
print("Bragg-filtered image:", bragg.image.count, "pixels; HRTEM filters ran", filtered.passes, "passes")

// Drift: a field of three blobs, displaced by a known sub-pixel amount, is registered back (phase correlation refined to
// 1/20 pixel) and the pair is summed after alignment.
var blobs = [Float](repeating: 5, count: n * n)
for (r0, c0) in [(60.0, 70.0), (120.0, 180.0), (190.0, 90.0)] {
    for row in 0..<n {
        for col in 0..<n {
            let d2 = (Double(row) - r0) * (Double(row) - r0) + (Double(col) - c0) * (Double(col) - c0)
            blobs[row * n + col] += Float(100 * exp(-d2 / 18))
        }
    }
}
let field = try Micrograph(width: n, height: n, pixels: blobs)
let drifted = try ImageShift.fourierShift(field, by: PixelShift(row: -2.5, column: 1.25))
let drift = try PhaseCorrelation.register(reference: field, moving: drifted, upsampleFactor: 20, normalization: .unnormalized)
let summed = try AlignedSum.sum([field, drifted], shifts: [.zero, drift.shift])
print("registered drift:", drift.shift.row, drift.shift.column, "px; aligned sum of", summed.frameCount, "frames")

// The planted truth: d = 1/|q| with q = (40, 12) and (−15, 35) cycles over 5.12 nm; the angle between them.
let qa = (40.0 / 5.12, 12.0 / 5.12), qb = (-15.0 / 5.12, 35.0 / 5.12)
let plantedAngle = try SpotMeasurement.angle(qa: qa, qb: qb)
let ok = real[0] == 64 && real.dropFirst().allSatisfy { $0 == 0 }
    && imaginary.allSatisfy { $0 == 0 }
    && abs(a.dNanometres! * (qa.0 * qa.0 + qa.1 * qa.1).squareRoot() - 1) < 2e-3
    && abs(b.dNanometres! * (qb.0 * qb.0 + qb.1 * qb.1).squareRoot() - 1) < 2e-3
    && abs(pair.angle - plantedAngle) < 0.1
    && bragg.image.count == n * n && filtered.passes >= 1
    && drift.shift == PixelShift(row: 2.5, column: -1.25) && summed.frameCount == 2 && summed.counts.allSatisfy { $0 == 2 }
print("readme-consumer summary: \(ok ? "passed" : "FAILED")")
if !ok { exit(1) }
