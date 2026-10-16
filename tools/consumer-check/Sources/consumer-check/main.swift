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

// The rest of slice 1's public surface (registration gate 7).
let spotA = try SpotMeasurement.measure(image, clickRow: h / 2 - 7, clickColumn: w / 2 + 10)
let maskSpot = spotA.maskSpot(radius: 2, edge: 1)
check(abs(maskSpot.kx - 10) < 0.01 && abs(maskSpot.ky + 7) < 0.01 && maskSpot.weight(atDistance: 0) == 1
      && maskSpot.weight(atDistance: 3) == 0, "maskSpot and Spot.weight")
let oneSided = try MaskedInverseFFT.compute(image, spots: [maskSpot], includeFriedelPartners: false, precision: .float64)
check(oneSided.imaginaryMax > 0.1 && oneSided.precision == .float64, "without Friedel partners the result is complex")
check(MaskedInverseFFTError.fftSetup(width: 1, height: 2).errorDescription?.isEmpty == false, "MaskedInverseFFTError text")
check(HRTEMFilter.edgeWindow(radius: 0, n: 64) == 1 && HRTEMFilter.butterworth(radius: 0, ro: 0.3, order: 4, n: 64) == 1
      && abs(HRTEMFilter.lowFrequencyTaper(radius: 0, taperRadius: 4) - 0.15) < 1e-16, "the filter's closed forms")
var cone = [Double](repeating: 0, count: 64 * 64)
for i in cone.indices {
    let dr = Double(i / 64 - 32), dc = Double(i % 64 - 32)
    cone[i] = (dr * dr + dc * dc).squareRoot()
}
check(HRTEMFilter.radialProfile(cone, n: 64).count == 32 && HRTEMFilter.rotationalAverage(cone, n: 64).count == 64 * 64,
      "radialProfile and rotationalAverage")
let ones = try HRTEMFilter.applyMask([Double](repeating: 1, count: 64 * 64), to: square.pixels, n: 64, precision: .float64)
check(zip(ones.doubles, square.pixels).map { abs($0 - Double($1)) }.max()! < 1e-9, "applyMask with a mask of ones is the identity")
let full = try HRTEMFilter.filter(square, parameters: .defaults, precision: .float64)
let stages: [[Double]] = [full.median, full.initialBackground, full.firstPassC, full.background, full.spectrum,
                          full.absfMask, full.wienerMask]
let sized = stages.allSatisfy { $0.count == 64 * 64 } && full.flagged.count == 64 * 64 && full.firstFlaggedPass.count == 64 * 64
check(sized && full.flagsPerPass.count == full.passes && full.finalDelta >= 0 && full.n == 64 && full.parameters == .defaults,
      "the filter result's fields")
var bad = HRTEMFilter.Parameters(); bad.order = 0
check(bad.outOfRange == "order", "Parameters.outOfRange")
for e in [HRTEMFilterError.oddSize(33), .tooSmall(16), .pixelCount(expected: 4, actual: 3), .maskCount(expected: 4, actual: 3),
          .nonFinitePixel, .parameterOutOfRange("order"), .fftSetup(8), .notSquare(width: 1, height: 2)] {
    check(e.errorDescription?.isEmpty == false, "HRTEMFilterError text: \(e)")
}

// ImagoAlignment (slice 2): shifts, registrations, drift series and aligned sums, on the lattice above.
check(PixelShift.zero == PixelShift(row: 0, column: 0) && PixelShift(row: 1, column: 2).column == 2, "PixelShift")
check(Normalization.allCases.count == 2 && Normalization.phase.rawValue == "phase" && Normalization.unnormalized.rawValue == "unnormalized",
      "Normalization")
check(ShiftOrder.nearest.rawValue == 0 && ShiftOrder.linear.rawValue == 1 && ShiftOrder.allCases.count == 2, "ShiftOrder")
check(ShiftMode.allCases.map(\.rawValue) == ["constant", "nearest", "wrap", "reflect", "mirror", "grid-constant", "grid-wrap"],
      "ShiftMode names are SciPy's")
check(AlignmentMethod.allCases.count == 2 && DriftReference.allCases.map(\.rawValue) == ["first", "mean", "previous"],
      "AlignmentMethod and DriftReference")
check(PhaseCorrelation.maximumUpsampleFactor == 1024 && ImageShift.maximumShift == 1e9, "the limits")

// ImageShift on a row of five pixels: SciPy's modes, by hand.
let row = try Micrograph(width: 5, height: 1, pixels: [10, 20, 30, 40, 50])
func shifted(_ mode: ShiftMode, _ order: ShiftOrder, _ by: Double, cval: Double = -1) throws -> [Float] {
    try ImageShift.shift(row, by: PixelShift(row: 0, column: by), order: order, mode: mode, cval: cval).pixels
}
check(try shifted(.constant, .linear, 2) == [-1, -1, 10, 20, 30], "constant: cval beyond the edge")
check(try shifted(.nearest, .linear, 2) == [10, 10, 10, 20, 30], "nearest")
check(try shifted(.wrap, .linear, 2) == [30, 40, 10, 20, 30], "wrap has SciPy's period n − 1")
check(try shifted(.gridWrap, .linear, 2) == [40, 50, 10, 20, 30], "grid-wrap has period n")
check(try shifted(.reflect, .linear, 2) == [20, 10, 10, 20, 30] && (try shifted(.mirror, .linear, 2)) == [30, 20, 10, 20, 30],
      "reflect and mirror")
check(try shifted(.gridConstant, .linear, 0.5) == [4.5, 15, 25, 35, 45], "grid-constant interpolates toward cval")
check(try shifted(.nearest, .nearest, -0.5) == [20, 30, 40, 50, 50], "order 0 rounds half up")
check(try shifted(.nearest, .linear, 0.25) == [10, 17.5, 27.5, 37.5, 47.5], "order 1 interpolates")
let doubleShift = try ImageShift.shift(row, by: PixelShift(row: 0, column: 0.25), mode: .nearest, precision: .float64)
check(doubleShift.pixels == [10, 17.5, 27.5, 37.5, 47.5] && doubleShift.width == 5, "shift on the Float64 path")

// The Fourier shift of an integer shift is a roll; a half-pixel shift of an even size leaves a Nyquist residual.
let rolled = try ImageShift.fourierShift(image, by: PixelShift(row: 2, column: -3), precision: .float64)
var rollError: Float = 0
for r in 0..<h { for c in 0..<w {
    rollError = max(rollError, abs(rolled.pixels[r * w + c] - image.pixels[((r - 2 + h) % h) * w + (c + 3) % w]))
} }
check(rollError < 1e-4 && rolled.x == image.x, "an integer Fourier shift is a roll and keeps the calibration")
let residual = try ImageShift.fourierShiftWithResidual(image, by: PixelShift(row: 0.5, column: 0), precision: .float64)
check(residual.image.width == w && residual.largestImaginaryPart >= 0 && residual.largestImaginaryPart < 1e-3,
      "fourierShiftWithResidual (a lattice has no Nyquist content)")

// Registration needs structure that is not periodic (a lattice matches itself at every cell): five Gaussian blobs,
// displaced by a known quarter-pixel shift and registered back (the blobs are the reference).
let blobCentres: [(Double, Double, Double)] = [(30, 40, 100), (45, 90, 70), (60, 25, 50), (70, 70, 90), (25, 100, 60)]
var blobPixels = [Float](repeating: 0, count: w * h)
for r in 0..<h { for c in 0..<w {
    var v = 5.0
    for (br, bc, amplitude) in blobCentres {
        let d2 = (Double(r) - br) * (Double(r) - br) + (Double(c) - bc) * (Double(c) - bc)
        v += amplitude * exp(-d2 / 18)
    }
    blobPixels[r * w + c] = Float(v)
} }
let blobs = try Micrograph(width: w, height: h, pixels: blobPixels)
let truth = PixelShift(row: 3.5, column: -2.25)
let moved = try ImageShift.fourierShift(blobs, by: PixelShift(row: -truth.row, column: -truth.column), precision: .float64)
for precision in Precision.allCases {
    let r = try PhaseCorrelation.register(reference: blobs, moving: moved, upsampleFactor: 4, normalization: .unnormalized,
                                          precision: precision)
    check(r.shift == truth, "\(precision): register recovers (3.5, −2.25) on the 1/4 grid, got \(r.shift)")
    check(r.upsampleFactor == 4 && r.normalization == .unnormalized && r.precision == precision && r.error < 0.3
          && abs(r.phaseDifference) < 1e-3, "\(precision): the registration's fields")
}
let whole = try PhaseCorrelation.register(reference: blobs, moving: moved, normalization: .unnormalized, precision: .float64)
check((whole.shift.row == 3 || whole.shift.row == 4) && whole.shift.column == -2,
      "upsample factor 1 gives whole pixels (3.5 is a tie, −2.25 → −2), got \(whole.shift)")
check(ImageRegistration(shift: truth, error: 0.5, phaseDifference: 0.25, upsampleFactor: 2, normalization: .phase,
                        precision: .float32).shift == truth, "ImageRegistration is constructible")

// A drift series and the aligned sum: four frames drifting by known quarter pixels.
let drift = [PixelShift.zero, PixelShift(row: 1, column: 2), PixelShift(row: 2, column: 3.5), PixelShift(row: 3.5, column: 3.5)]
let stack = try drift.map { try ImageShift.fourierShift(blobs, by: PixelShift(row: -$0.row, column: -$0.column), precision: .float64) }
for (reference, name) in [(DriftReference.first, "first"), (.previous, "previous")] {
    let series = try DriftSeries.estimate(stack, reference: reference, upsampleFactor: 4, normalization: .unnormalized,
                                          precision: .float64)
    check(series.map(\.shift) == drift, "DriftSeries .\(name) recovers the drift, got \(series.map(\.shift))")
}
let meanSeries = try DriftSeries.estimate(stack, reference: .mean, upsampleFactor: 4, normalization: .unnormalized)
check(meanSeries.count == 4 && zip(meanSeries, drift).allSatisfy { abs(($0.shift.row - meanSeries[0].shift.row) - $1.row) < 0.5 },
      "DriftSeries .mean follows the drift")
let aligned = try AlignedSum.sum(stack, shifts: drift, method: .fourier, precision: .float64)
var alignedError: Float = 0
for i in 0..<(w * h) { alignedError = max(alignedError, abs(aligned.mean.pixels[i] - blobs.pixels[i])) }
check(alignedError < 1e-3 && aligned.counts.allSatisfy { $0 == 4 } && aligned.frameCount == 4 && aligned.method == .fourier
      && aligned.precision == .float64 && aligned.sum.width == w && aligned.largestImaginaryPart < 1e-3,
      "the Fourier aligned mean is the image")
let overlap = try AlignedSum.sum(stack, shifts: drift, method: .bilinear)
check(overlap.counts[0] == 1 && overlap.counts[(h / 2) * w + w / 2] == 4 && overlap.method == .bilinear
      && overlap.largestImaginaryPart == 0 && overlap.mean.pixels.count == w * h,
      "the bilinear overlap count: one frame covers the corner, four the centre")

// Refusals, by name, with texts: a call that stops throwing fails its check.
@MainActor func refusal(_ label: String, _ expected: AlignmentError, _ body: () throws -> Void) {
    do { try body(); check(false, label + " (no error thrown)") }
    catch let e as AlignmentError { check(e == expected, label) }
    catch { check(false, label + " (another error: \(error))") }
}
let tall = try Micrograph(width: 4, height: 5, pixels: [Float](repeating: 1, count: 20))
refusal("a size mismatch is refused by name", .sizeMismatch(index: 1, width: 4, height: 5, expectedWidth: w, expectedHeight: h)) {
    _ = try PhaseCorrelation.register(reference: image, moving: tall)
}
refusal("an empty stack is refused", .emptyStack) { _ = try DriftSeries.estimate([]) }
refusal("an upsample factor of 0 is refused", .upsampleFactor(0)) {
    _ = try PhaseCorrelation.register(reference: image, moving: image, upsampleFactor: 0)
}
refusal("a NaN shift is refused", .invalidShift(index: 0)) { _ = try ImageShift.shift(image, by: PixelShift(row: .nan, column: 0)) }
refusal("a missing shift is refused", .shiftCount(expected: 1, actual: 0)) { _ = try AlignedSum.sum([image], shifts: []) }
for e in [AlignmentError.emptyStack, .sizeMismatch(index: 1, width: 2, height: 3, expectedWidth: 4, expectedHeight: 5),
          .nonFinitePixel(index: 0), .upsampleFactor(0), .invalidShift(index: 0), .shiftCount(expected: 2, actual: 1),
          .fftSetup(width: 1, height: 2)] {
    check(e.errorDescription?.isEmpty == false, "AlignmentError text: \(e)")
}

// Public radix-2 plans and their split-complex entry points, in both precisions.
let lineFFT = FFT1D(n: 64)!
let lineDouble = FFT1DDouble(n: 64)!
check(lineFFT.n == 64 && lineDouble.n == 64, "public 1D plan lengths")
check(FFT1D.nextPow2(65) == 128 && FFT1DDouble.nextPow2(65) == 128, "public radix-2 padding helpers")
var lineRe = [Float](repeating: 0, count: lineFFT.n)
var lineIm = lineRe
lineRe[0] = 1
lineFFT.transform(re: &lineRe, im: &lineIm, forward: true)
check(lineRe.allSatisfy { $0 == 1 } && lineIm.allSatisfy { $0 == 0 }, "Float impulse FFT")
lineFFT.transform(re: &lineRe, im: &lineIm, forward: false)
check(lineRe[0] == 64 && lineRe.dropFirst().allSatisfy { $0 == 0 }, "Float inverse is unnormalised")
var lineDR = [Double](repeating: 0, count: lineDouble.n)
var lineDI = lineDR
lineDR[0] = 1
lineDouble.transform(re: &lineDR, im: &lineDI, forward: true)
check(lineDR.allSatisfy { $0 == 1 } && lineDI.allSatisfy { $0 == 0 }, "Double impulse FFT")
lineDouble.transform(re: &lineDR, im: &lineDI, forward: false)
check(lineDR[0] == 64 && lineDR.dropFirst().allSatisfy { $0 == 0 }, "Double inverse is unnormalised")

print("consumer-check summary: \(checks) checks, \(failures) failures")
if failures > 0 { exit(1) }
