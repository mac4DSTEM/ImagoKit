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
check(LengthUnit.angstromsPerPixel(scale: 0.25, unit: "nm") == 2.5, "nm to Å per pixel")
check(AxisCalibration(offset: -2, scale: 0.2, unit: "nm").origin == 10, "an axis from an offset")
check(DisplayNormalization.normalized([2, 4, 6]) == [0, 0.5, 1], "display normalisation")
check(ScaleBarMath.plan(x: AxisCalibration(scale: 1, unit: "nm"), y: nil, width: 400, height: 400)?.label == "50 nm", "a scale bar")
check(Thumbnail.make(normalized: [Float](repeating: 0.5, count: 4), width: 2, height: 2)?.width == 2, "a thumbnail")
check(DisplayMapping(low: 0, high: 1, gamma: 1, lut: [UInt8](repeating: 0, count: 1024))?.rgb(-1).0 == DisplayMapping.maskedGrey.0, "a mapping")
check(FFTRegion.centredSquare(width: 100, height: 64) == PixelCrop(row: 0, col: 18, height: 64, width: 64), "a region")
check(FFTUnits.length(1, unit: "Å").value == 0.1, "a shown length")
check(SpectrumDisplay.make(power: [1, 1, 1, 1], width: 2, height: 2).values.count == 4, "a spectrum display")
check(ImageMeasurement.distance(PixelPoint(x: 0, y: 0), PixelPoint(x: 3, y: 4), x: nil, y: nil).pixels == 5, "a distance")
check((try? IntensityHistogram.histogram([0, 1, 2, 3], bins: 3))?.counts == [1, 1, 2], "a histogram")
check((try? LineProfile.sample(image, from: PixelPoint(x: 0, y: 0), to: PixelPoint(x: 5, y: 0), reduce: .max))?.samples.count == 6,
      "a line profile")
check((try? LineProfile.sample(channels: [image, image], from: PixelPoint(x: 0, y: 0), to: PixelPoint(x: 5, y: 0)))?.count == 2,
      "a multichannel line profile")
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

// The rest of the FFT, spot and filter public surface.
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

// ImagoAlignment: shifts, registrations, drift series and aligned sums, on the lattice above.
check(PixelShift.zero == PixelShift(row: 0, column: 0) && PixelShift(row: 1, column: 2).column == 2, "PixelShift")
check(Normalization.allCases.count == 2 && Normalization.phase.rawValue == "phase" && Normalization.unnormalized.rawValue == "unnormalized",
      "Normalization")
check(ShiftOrder.nearest.rawValue == 0 && ShiftOrder.linear.rawValue == 1 && ShiftOrder.allCases.count == 6, "ShiftOrder")
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

// Colour tables, segmentation, chart samples and CSV text (0.3.0).
let inferno = Colormap.inferno.lutRGBA()
check(inferno.count == 1024 && Array(Colormap.gray.lutRGBA()[0..<4]) == [0, 0, 0, 255], "public colour tables")
check(DisplayMapping(low: 0, high: 1, gamma: 1, colormap: .gray).rgb(1) == (255, 255, 255), "mapping through a colormap")
check(DisplayMapping.sampleRGB(inferno, 0) != nil && DisplayMapping.sampleRGB([], 0) == nil, "public table sampler")
var blobMask = [Bool](repeating: false, count: 36)
for r in 1...2 { for c in 1...4 { blobMask[r * 6 + c] = true } }
let blobLabels = try! Segmentation.label(blobMask, width: 6, height: 6)
let blob = try! Segmentation.objectMoments(blobLabels)
check(blobLabels.count == 1 && blob[0].area == 8 && blob[0].lengthPx == 4 && blob[0].widthPx == 2 && !blob[0].touchesEdge,
      "public segmentation")
let chartProfile = RadialProfile(width: 0.5, units: "1/nm", mean: [9, 4, 0], count: [1, 3, 5])!
let chartSamples = RadialChartScale.samples(chartProfile, factor: 2)
check(chartSamples.count == 1 && chartSamples[0].q == 1.5 && RadialChartScale.nearest(chartSamples, toX: 9)?.bin == 1,
      "public chart samples")
check(FFTMask.defaultRadius == 3 && FFTMask.radiusRange == 1...32, "public mask defaults")
check(CSVText.number(0.1) == "0.10000000000000001" && CSVText.quoted("a,b") == "\"a,b\"", "public CSV text")
let tidySource = TidyResultsCSV.Source(software: "s", fileName: "f", object: "o", width: 2, height: 2, x: nil, y: nil, notes: [])
let tidyRow = TidyResultsCSV.Row(group: "g", id: "i", label: "l", product: "p", entries: [.init("v", 1.5, unit: "nm")])
let tidyText = try! TidyResultsCSV.text([tidyRow], source: tidySource, title: "t", session: "s", descriptionLines: [],
                                        validations: [(product: "p", sentence: "closed form")], lineageLine: "# lineage: none")
check(tidyText?.hasSuffix("g,i,l,v,1.5,nm,,closed form,\n") == true, "public tidy CSV")
let csvHeader = FFTExportHeader(software: "s", fileName: "f", object: "o", imageWidth: 8, imageHeight: 8, x: nil, y: nil, notes: [],
                                crop: nil, spectrumWidth: 8, spectrumHeight: 8, spacingX: 0.5, spacingY: 0.5,
                                unitsX: "1/nm", unitsY: "1/nm")
let radialText = RadialProfileCSV.text(chartProfile, header: csvHeader, title: "radial", validation: "v", lineage: "l")
check(radialText?.hasPrefix("# radial\n# software: s\n") == true && radialText?.hasSuffix("\n1,0.75,0.75,4,3\n2,1.25,1.25,0,5\n") == true,
      "public radial CSV")
let csvSpot = SpotMeasurement(clickedRow: 1, clickedCol: 2, snappedRow: 1, snappedCol: 2, centroidRow: 1, centroidCol: 2,
                              width: 8, height: 8, qx: 1, qy: 0, q: 1, d: 1, units: "nm", reciprocalUnits: "1/nm",
                              neighbour: false, precision: .float32)
let spotsText = SpotsCSV.text(rows: [SpotsCSV.Row(number: 1, measured: csvSpot, inMask: true)], omitted: 0, maskRadius: 3,
                              maskEdge: 2, notes: [], header: csvHeader, title: "spots", validation: "v", lineage: "l")
check(spotsText?.hasSuffix("\n1,1,1,1,0,1,1,1,0,2,1,2,1\n") == true, "public spots CSV")

// 0.6.0: the FFT surface, the filters, peaks, masked registration, spline shifts, on hand-computable inputs.
check((try? FFTSurface.fftshift([0, 1, 2, 3, 4], rows: 1, columns: 5)) == [3, 4, 0, 1, 2], "public fftshift")
check((try? FFTSurface.ifftshift([3, 4, 0, 1, 2], rows: 1, columns: 5)) == [0, 1, 2, 3, 4], "public ifftshift")
let flatImage = ComplexImage<Double>(rows: 4, columns: 4, real: [Double](repeating: 1, count: 16))
let onesSpectrum = try FFTSurface.fft2(flatImage, norm: .ortho)
check(abs(onesSpectrum.re[0] - 4) < 1e-12 && abs(onesSpectrum.re[5]) < 1e-12, "public fft2 ortho of a constant")
let half = try FFTSurface.rfft2(real: [Double](repeating: 1, count: 16), rows: 4, columns: 4)
check(half.columns == 3 && abs(half.re[0] - 16) < 1e-12, "public rfft2 shape and DC")
check(abs(try FFTSurface.irfft2(half, columns: 4)[7] - 1) < 1e-12, "public irfft2 round trip")
check((try? SeparableFilters.uniformFilter([Double](repeating: 2, count: 12), rows: 3, columns: 4, size: (3, 3), mode: .nearest))?
    .allSatisfy { abs($0 - 2) < 1e-12 } == true, "public uniform filter keeps a constant")
check((try? SeparableFilters.gaussianFilter([Float](repeating: 5, count: 20), rows: 4, columns: 5, sigma: (1, 1)))?
    .allSatisfy { abs($0 - 5) < 1e-5 } == true, "public Gaussian filter keeps a constant")
check((try? BandpassFilters.butterworth([Double](repeating: 3, count: 64), rows: 8, columns: 8, cutoffFrequencyRatio: 0.2, highPass: true))?
    .allSatisfy { abs($0) < 1e-12 } == true, "public high-pass Butterworth removes a constant")
check((try? BandpassFilters.butterworth([Double](repeating: 3, count: 192), rows: 8, columns: 8, channels: 3, cutoffFrequencyRatio: 0.2,
                                        highPass: true))?.allSatisfy { abs($0) < 1e-12 } == true, "public multichannel Butterworth")
check(abs(((try? WindowFunction.window2D(.hann, rows: 9, columns: 9))?[40] ?? 0) - 1) < 1e-12, "public window")
check((try? WindowFunction.window1D(.dpss(nw: 2.5), count: 9))?.max() == 1 && (try? WindowFunction.window1D(.taylor(), count: 9))?.count == 9
      && (try? WindowFunction.window1D(.kaiserBesselDerived(beta: 4), count: 7)) == nil, "public window families")
let peakImage: [Float] = (0..<49).map { $0 == 24 ? 2 : ($0 == 10 ? 1 : 0) }
check((try? PeakLocalMax.find(peakImage, rows: 7, columns: 7, minDistance: 1))?.map { [$0.row, $0.column] } == [[3, 3], [1, 3]],
      "public peak_local_max")
check(SpotMeasurementError.noPower != SpotMeasurementError.nonFinite, "public noPower and nonFinite")
check((try? SpotMeasurement.measure(try Micrograph(width: 16, height: 16, pixels: [Float](repeating: 1, count: 256)),
                                    clickRow: 3, clickColumn: 9)) == nil, "a flat window is refused")
let squareImage = try Micrograph(width: 8, height: 8, pixels: (0..<64).map { Float((abs(sin(Double($0) * 12.9898) * 43758.5453)).truncatingRemainder(dividingBy: 1)) })
check((try? ImageShift.shift(squareImage, by: .zero, order: .cubic, mode: .mirror))?.pixels.enumerated()
    .allSatisfy { abs($0.element - squareImage.pixels[$0.offset]) < 1e-4 } == true, "public cubic shift by zero")
let maskAll = [Bool](repeating: true, count: 64)
check((try? PhaseCorrelation.registerMasked(reference: squareImage, moving: squareImage, referenceMask: maskAll))
    == PixelShift(row: 0, column: 0), "public masked registration of an image with itself")

let greyBlock: [Double] = (0..<64).map { $0 % 8 < 4 ? 0.2 : 0.8 }
check((try? Thresholding.otsu(.float64(greyBlock))).map { $0 > 0.2 && $0 < 0.8 } == true, "public otsu splits two levels")
check((try? Thresholding.otsu(.integers([10, 10, 200, 200]))) == 10, "public otsu of integers is an integer")
let blockMask = (0..<25).map { ($0 / 5) >= 1 && ($0 / 5) <= 3 && ($0 % 5) >= 1 && ($0 % 5) <= 3 }
check((try? BinaryMorphology.erosion(blockMask, width: 5, height: 5))?.filter { $0 }.count == 1, "public binary erosion of a 3x3 block")
check((try? BinaryMorphology.fillHoles(blockMask.enumerated().map { $0.offset == 12 ? false : $0.element }, width: 5, height: 5))?
    .filter { $0 }.count == 9, "public fill holes")
check((try? StructuringElement.generateStructure(rank: 2, connectivity: 1))?.filter { $0 }.count == 5, "public generate_binary_structure")

let singleBox = Segmentation.LabelImage(width: 10, height: 10, labels: (0..<100).map { $0 == 55 ? 1 : 0 }, count: 1)
check((try? ObjectStatistics.arealDensity(labels: singleBox, pixelSize: 1))?.arealDensity == 100.0 / 64 / 100, "public areal density")
check((try? ObjectStatistics.arealDensity(labels: singleBox, pixelSize: nil))?.arealDensity == nil, "no scale, no density")
check((try? ObjectStatistics.numberDensity(arealDensity: 0.01, thickness: 50))?.value == 0.01 / 50, "public number density")
check((try? Segmentation.regionProperties(singleBox))?.first?.areaConvex == 1, "public regionProperties")
check((try? WindowFunction.apply(.hann, to: [Float](repeating: 1, count: 9), rows: 3, columns: 3))?.count == 9, "public window apply")
check(Thresholding.mask([Float(1), 3], threshold: 2) == [false, true], "public threshold mask")
check(SpectrumSymmetry(centreRow: 1, centreColumn: 1).oneOfEachMirrorPair, "public spectrum symmetry")

// 0.8.0: distance transform, watershed, rolling ball, zoom and rotate, 1D real transforms, denoising, GPA strain.
let edt = try DistanceTransform.euclidean(foreground: [false, true, true, true, false], rows: 1, columns: 5, sampling: (1, 2))
check(edt == [0, 2, 4, 2, 0], "public distance transform with sampling, got \(edt)")
let nearest = try DistanceTransform.euclideanWithIndices(foreground: [false, true, true, true, false], rows: 1, columns: 5)
check(nearest.nearestColumns == [0, 0, 0, 4, 4] && nearest.nearestRows == [0, 0, 0, 0, 0] && nearest.distances == [0, 1, 2, 1, 0],
      "public distance transform indices: the tie goes to the first background pixel")
let seeds = try Segmentation.markers(peakRows: [0, 0], peakColumns: [0, 5], rows: 1, columns: 6)
check(seeds == [1, 0, 0, 0, 0, 2], "public watershed markers from peaks")
check((try? Segmentation.watershed(image: [0, 1, 2, 2, 1, 0], markers: seeds, rows: 1, columns: 6)) == [1, 1, 1, 2, 2, 2],
      "public watershed floods each basin from its marker")
let ball = try RollingBall.ballKernel(radius: 2)
check(ball.rows == 5 && ball.columns == 5 && ball.values.count == 25, "public ball kernel on a (2r + 1) grid")
let ellipsoid = try RollingBall.ellipsoidKernel(rows: 3, columns: 5, intensity: 1)
check(ellipsoid.values.count == ellipsoid.rows * ellipsoid.columns, "public ellipsoid kernel")
check((try? RollingBall.background(image: [Double](repeating: 3, count: 64), rows: 8, columns: 8, radius: 5))?
    .allSatisfy { abs($0 - 3) < 1e-9 } == true, "public rolling ball: a constant is all background")
let nanImage: [Double] = [1, .nan, 1, 1, 1, 1, 1, 1]
let nanBackground = try RollingBall.background(image: nanImage, rows: 1, columns: 8, radius: 1, nanSafe: true)
check(nanBackground[0].isNaN && nanBackground[2].isNaN && nanBackground[4] == 1, "public rolling ball nanSafe: a NaN spreads over the kernel")
let zoomed = try ImageResampling.zoom(image, zoom: (rows: 0.5, columns: 0.5))
check(zoomed.width == w / 2 && zoomed.height == h / 2 && abs(zoomed.x!.scale - 2 * dx) < 1e-12 && abs(zoomed.y!.scale - 2 * dy) < 1e-12,
      "public zoom halves the size and doubles the pixel scale")
let turned = try ImageResampling.rotate(image, degrees: 90, order: .linear)
check(turned.width == h && turned.height == w, "public rotate by 90 degrees swaps the sides")
let ones4: [Float] = [1, 1, 1, 1]
let half1D = FFT1D(n: 4)!.rfft(ones4)
check(half1D.re == [4, 0, 0] && half1D.im == [0, 0, 0], "public 1D rfft of a constant")
check(FFT1D(n: 4)!.irfft(re: half1D.re, im: half1D.im) == ones4, "public 1D irfft round trip")
check(FFT1DDouble.rfft([1, 2, 3], n: 3) == nil, "public 1D rfft refuses a length that is not a power of two")
let flat = [Float](repeating: 0.5, count: 64)
check((try? Denoise.totalVariationChambolle(flat, rows: 8, columns: 8))?.allSatisfy { abs($0 - 0.5) < 1e-6 } == true,
      "public TV Chambolle keeps a constant")
check((try? Denoise.totalVariationBregman(flat.map(Double.init), rows: 8, columns: 8))?.allSatisfy { abs($0 - 0.5) < 1e-9 } == true,
      "public TV Bregman keeps a constant")
check((try? Denoise.nonLocalMeans(flat, rows: 8, columns: 8, patchSize: 3, patchDistance: 2))?.allSatisfy { abs($0 - 0.5) < 1e-6 } == true,
      "public non-local means keeps a constant")
do { _ = try Denoise.totalVariationChambolle(flat, rows: 8, columns: 8, maxNumIter: 0); check(false, "maxNumIter 0 is refused") }
catch is DenoiseError { check(true, "maxNumIter 0 is refused") } catch { check(false, "maxNumIter 0: another error \(error)") }
// A perfect zero-mean two-fringe lattice, 8 cycles across 64 pixels on each axis (a mean would leak through the masks): no strain.
let lattice: [Double] = (0..<4096).map {
    let (r, c) = (Double($0 / 64), Double($0 % 64))
    return cos(2 * Double.pi * c / 8) + 0.6 * cos(2 * Double.pi * r / 8)
}
let found = try GeometricPhaseAnalysis.findReflections(lattice, rows: 64, columns: 64)
let foundSet = [found.g1, found.g2].map { [($0.row * 1e6).rounded() / 1e6, ($0.column * 1e6).rounded() / 1e6] }
check(foundSet == [[0, 0.125], [0.125, 0]], "public GPA finds the two reflections, got \(foundSet)")
let gpa = try GeometricPhaseAnalysis.analyze(lattice, rows: 64, columns: 64, g1: (0, 0.125), g2: (0.125, 0))
let worst = (gpa.strainRR + gpa.strainCC + gpa.strainRC + gpa.rotation).map(abs).max() ?? .nan
check(gpa.validation == GeometricPhaseAnalysis.Result.validationNone && worst < 1e-8,
      "public GPA of a perfect lattice: zero strain and rotation, badged unvalidated (worst \(worst))")
do { _ = try GeometricPhaseAnalysis.analyze(lattice, rows: 64, columns: 64, g1: (0, 0.125), g2: (0, 0.25)); check(false, "collinear g") }
catch let e as GeometricPhaseAnalysisError { check(e == .collinearReflections, "public GPA refuses collinear reflections") }
catch { check(false, "collinear g: another error \(error)") }

print("consumer-check summary: \(checks) checks, \(failures) failures")
if failures > 0 { exit(1) }
