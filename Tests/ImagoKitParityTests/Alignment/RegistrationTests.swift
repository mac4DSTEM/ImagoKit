// Phase correlation: the pinned cases against scikit-image on both precisions, the
// planted shifts against their truth, and the closed forms of the integer stage (the midpoint, the region, a length-1 axis).
import Foundation
import ImagoKitTestSupport
import ImagoAlignment
import ImagoCore
import Testing

/// "BAND" lines are the measured values tools/registration-drift/full-parity.sh summarises (it sets IMAGOKIT_BANDS); otherwise it is silent.
let printsBands = ProcessInfo.processInfo.environment["IMAGOKIT_BANDS"] != nil

func band(_ label: String, _ value: Double, _ limit: Double, _ pin: String) {
    if printsBands { print("BAND \(label) \(value) limit \(limit) \(pin)") }
}

func images(_ pin: RegisterPin) -> (Micrograph, Micrograph) {
    (AlignmentPins.image(AlignmentPins.floats(pin.ref), width: pin.width, height: pin.height),
     AlignmentPins.image(AlignmentPins.floats(pin.mov), width: pin.width, height: pin.height))
}

func difference(_ r: ImageRegistration, _ o: OraclePin) -> Double {
    max(abs(r.shift.row - o.shift[0]), abs(r.shift.column - o.shift[1]))
}

/// `image` displaced by the integers (rows, columns): out[i, j] = in[(i + dr) mod h, (j + dc) mod w], the moving image
/// whose registration onto `image` is (dr, dc) wrapped.
func rolled(_ image: Micrograph, _ dr: Int, _ dc: Int) -> Micrograph {
    let w = image.width, h = image.height
    var out = [Float](repeating: 0, count: w * h)
    for i in 0..<h {
        for j in 0..<w {
            out[i * w + j] = image.pixels[(((i + dr) % h + h) % h) * w + ((j + dc) % w + w) % w]
        }
    }
    return AlignmentPins.image(out, width: w, height: h)
}

/// Whether `shift` lies in the bounding box [row min, row max, column min, column max] (to `slack`).
func inside(_ shift: PixelShift, _ box: [Double], slack: Double) -> Bool {
    shift.row >= box[0] - slack && shift.row <= box[1] + slack && shift.column >= box[2] - slack
        && shift.column <= box[3] + slack
}

/// The distance between two angles on the circle (a phase difference of π and −π are one angle).
func circular(_ a: Double, _ b: Double) -> Double {
    let d = abs(a - b).truncatingRemainder(dividingBy: 2 * Double.pi)
    return min(d, 2 * Double.pi - d)
}

@Suite("Phase correlation against scikit-image 0.26.0")
struct RegistrationAgainstOracle {
    /// One path against its oracle: the shift equal, or (a near-tie) inside the box of shifts the measured noise
    /// of the correlation cannot tell apart; where the shifts are equal the error (as its square, which is what is computed)
    /// and the phase difference within the registered bands plus the measured noise of the last correlation stage.
    private func compare(_ r: ImageRegistration, _ o: OraclePin, _ pin: RegisterPin, tie: Bool, box: [Double]?,
                         eps: [Double], tag: String, slack: Double, errorBand: Double, plainBand: Double, phaseBand: Double) {
        let d = difference(r, o)
        if tie, let box {
            #expect(inside(r.shift, box, slack: slack), "a near-tie case: \(r.shift) outside \(box)")
        } else {
            #expect(d <= slack, "the shift is scikit-image's: kit \(r.shift), oracle \(o.shift)")
        }
        band("\(tag).shift", d, tie ? -1 : slack, pin.name)
        guard d <= slack else { return }
        let noise = pin.u == 1 ? eps[0] : eps[1]
        let e2 = abs(r.error * r.error - o.error * o.error)
        let p = circular(r.phaseDifference, o.phasediff)
        let errorLimit = errorBand + 8 * noise, phaseLimit = phaseBand + 4 * noise
        #expect(e2 <= errorLimit || (r.error.isNaN && o.error.isNaN), "error² \(r.error * r.error) vs \(o.error * o.error)")
        #expect(p <= phaseLimit, "phase difference \(r.phaseDifference) vs \(o.phasediff)")
        band("\(tag).error2", e2, errorLimit, pin.name)
        band("\(tag).phase", p, phaseLimit, pin.name)
        band("\(tag).error", abs(r.error - o.error), o.error >= 0.1 ? plainBand : -1, pin.name)
        if o.error >= 0.1 {
            #expect(abs(r.error - o.error) <= plainBand + 8 * noise, "error \(r.error) vs \(o.error)")
        }
    }

    @Test(arguments: AlignmentPins.register)
    func float64(_ pin: RegisterPin) throws {
        let (ref, mov) = images(pin)
        let r = try PhaseCorrelation.register(reference: ref, moving: mov, upsampleFactor: pin.u,
                                              normalization: pin.norm, precision: .float64)
        compare(r, pin.f64, pin, tie: pin.tie64, box: pin.box64, eps: pin.eps64, tag: "f64", slack: 1e-9,
                errorBand: 1e-12, plainBand: 1e-10, phaseBand: 1e-10)
        #expect(r.upsampleFactor == pin.u && r.normalization == pin.norm && r.precision == .float64)
    }

    @Test(arguments: AlignmentPins.register)
    func float32(_ pin: RegisterPin) throws {
        let (ref, mov) = images(pin)
        let r = try PhaseCorrelation.register(reference: ref, moving: mov, upsampleFactor: pin.u,
                                              normalization: pin.norm, precision: .float32)
        compare(r, pin.f32, pin, tie: pin.tie32, box: pin.box32, eps: pin.eps32, tag: "f32", slack: 1e-4,
                errorBand: 1e-5, plainBand: 1e-5, phaseBand: 1e-5)
        #expect(r.precision == .float32)
    }
}

@Suite("Planted shifts within their truth band")
struct PlantedShifts {
    static let gated = AlignmentPins.register.filter { $0.truth != nil && $0.truth_band != nil }

    @Test(arguments: PlantedShifts.gated, [Precision.float32, Precision.float64])
    func recoversThePlantedShift(_ pin: RegisterPin, _ precision: Precision) throws {
        let (ref, mov) = images(pin)
        let r = try PhaseCorrelation.register(reference: ref, moving: mov, upsampleFactor: pin.u,
                                              normalization: pin.norm, precision: precision)
        let truth = try #require(pin.truth), limit = try #require(pin.truth_band)
        let d = max(abs(r.shift.row - truth[0]), abs(r.shift.column - truth[1]))
        band("truth.\(precision.rawValue)", d, limit, pin.name)
        if precision == .float32, pin.tie32, let box = pin.box32 {
            // The correlation of this case is within the Float32 FFT's noise of a tie: the shift is held in the
            // box of shifts that noise cannot tell apart (the comparison with the oracle), not to the planted band.
            #expect(inside(r.shift, box, slack: 1e-4), "\(r.shift) outside \(box)")
        } else {
            #expect(d <= limit + 1e-9, "|shift − truth| = \(d) beyond \(limit)")
        }
    }

    @Test(arguments: PlantedShifts.gated.filter { $0.u == 20 && $0.norm == .phase && ($0.sigma ?? 0) == 0 })
    func appliedToTheMovingImageTheShiftRegistersItOntoTheReference(_ pin: RegisterPin) throws {
        let (ref, mov) = images(pin)
        let r = try PhaseCorrelation.register(reference: ref, moving: mov, upsampleFactor: 20, precision: .float64)
        let back = try ImageShift.fourierShift(mov, by: r.shift, precision: .float64)
        var worst = 0.0, top = 0.0
        for i in 0..<ref.pixels.count {
            worst = max(worst, Double(abs(back.pixels[i] - ref.pixels[i])))
            top = max(top, Double(abs(ref.pixels[i] - 1000)))
        }
        // The residual of a 1/40 px error on a pattern of this bandwidth is a few percent of its contrast.
        #expect(worst / top < 0.15, "the sign: the shift applied to the moving image must register it, residual \(worst / top)")
        let wrong = try ImageShift.fourierShift(mov, by: PixelShift(row: -r.shift.row, column: -r.shift.column),
                                                precision: .float64)
        var worstWrong = 0.0
        for i in 0..<ref.pixels.count { worstWrong = max(worstWrong, Double(abs(wrong.pixels[i] - ref.pixels[i]))) }
        #expect(worstWrong > 2 * worst || r.shift == .zero, "the opposite sign must register worse")
    }
}

@Suite("Phase correlation, closed forms")
struct RegistrationClosedForms {
    /// A peak index above trunc(n/2) has n subtracted: n/2 itself stays positive, n/2 + 1 becomes −(n/2 − 1) (even n);
    /// (n−1)/2 stays and (n+1)/2 becomes −(n−1)/2 (odd n).
    @Test(arguments: [Precision.float32, Precision.float64])
    func theMidpointWraps(_ precision: Precision) throws {
        for (n, cases) in [(8, [(4, 4), (5, -3), (-4, 4), (3, 3), (-3, -3), (7, -1)]), (7, [(3, 3), (4, -3), (-3, -3), (-4, 3)])] {
            let image = noise(n, n, seed: 11 + UInt64(n))
            for (shift, expected) in cases {
                // rows roll by `shift`, columns by `-shift`: the registration is `expected` on the rows and
                // `expected` wrapped from -shift on the columns (n/2 stays positive, above it wraps).
                let moving = rolled(image, shift, -shift)
                let wrappedColumn = ((-shift % n) + n) % n > n / 2 ? ((-shift % n) + n) % n - n : ((-shift % n) + n) % n
                let r = try PhaseCorrelation.register(reference: image, moving: moving, upsampleFactor: 1, precision: precision)
                #expect(r.shift == PixelShift(row: Double(expected), column: Double(wrappedColumn)),
                        "n \(n) shift \(shift): \(r.shift)")
            }
        }
    }

    @Test(arguments: [Precision.float32, Precision.float64])
    func rectangularImagesWrapEachAxisAtItsOwnMidpoint(_ precision: Precision) throws {
        let image = noise(12, 7, seed: 5)        // 12 columns, 7 rows
        let moving = rolled(image, 4, 7)
        let r = try PhaseCorrelation.register(reference: image, moving: moving, precision: precision)
        #expect(r.shift == PixelShift(row: -3, column: -5), "rows 4 of 7 wrap to −3; columns 7 of 12 wrap to −5")
    }

    /// The refinement region has ceil(1.5·u) samples per axis and its centre sample is trunc(S/2).
    @Test func theRegionSize() {
        let sizes = [1: 2, 2: 3, 3: 5, 4: 6, 5: 8, 7: 11, 10: 15, 20: 30, 100: 150]
        for (u, expected) in sizes {
            #expect(PhaseCorrelation.refinementRegion(for: u) == expected, "u \(u)")
        }
        #expect(PhaseCorrelation.refinementCentre(for: 20) == 15 && PhaseCorrelation.refinementCentre(for: 3) == 2)
    }

    @Test(arguments: [Precision.float32, Precision.float64])
    func aLengthOneAxisHasShiftZero(_ precision: Precision) throws {
        let row = noise(32, 1, seed: 3)
        let moving = rolled(row, 0, 5)
        for u in [1, 20] {
            let r = try PhaseCorrelation.register(reference: row, moving: moving, upsampleFactor: u, precision: precision)
            #expect(r.shift.row == 0 && r.shift.column == 5, "u \(u): \(r.shift)")
        }
        let column = noise(1, 32, seed: 4)
        let r = try PhaseCorrelation.register(reference: column, moving: rolled(column, -6, 0), upsampleFactor: 20,
                                              precision: precision)
        #expect(r.shift.column == 0 && r.shift.row == -6)
        let one = noise(1, 1, seed: 9)
        let single = try PhaseCorrelation.register(reference: one, moving: one, upsampleFactor: 1, precision: precision)
        #expect(single.shift == .zero)
    }

    /// An identical pair: shift 0, phase difference 0, and (unnormalised, u = 1) the error is zero: |CCmax|² = src·target.
    @Test(arguments: [Precision.float32, Precision.float64])
    func anImageRegisteredOntoItself(_ precision: Precision) throws {
        let image = noise(20, 16, seed: 21)
        let r = try PhaseCorrelation.register(reference: image, moving: image, normalization: .unnormalized,
                                              precision: precision)
        #expect(r.shift == .zero && abs(r.phaseDifference) < 1e-6)
        #expect(r.error < (precision == .float64 ? 1e-7 : 2e-3), "error \(r.error)")
    }

    /// The refined branch's amplitudes are the unnormalised sums, and so is its correlation: the error of an identical pair is
    /// the same at every upsample factor (a branch that divided the amplitudes by the pixel count would give a huge one).
    @Test func theRefinedBranchAgreesWithTheIntegerBranch() throws {
        let image = noise(16, 16, seed: 22)
        let one = try PhaseCorrelation.register(reference: image, moving: image, upsampleFactor: 1,
                                                normalization: .unnormalized, precision: .float64)
        let twenty = try PhaseCorrelation.register(reference: image, moving: image, upsampleFactor: 20,
                                                   normalization: .unnormalized, precision: .float64)
        #expect(one.error < 1e-7 && twenty.error < 1e-7, "errors \(one.error), \(twenty.error)")
        // And for a pair that does not match: the same error from both branches.
        let other = noise(16, 16, seed: 25)
        let a = try PhaseCorrelation.register(reference: image, moving: other, upsampleFactor: 1,
                                              normalization: .unnormalized, precision: .float64)
        let b = try PhaseCorrelation.register(reference: image, moving: other, upsampleFactor: 4,
                                              normalization: .unnormalized, precision: .float64)
        if a.shift == b.shift { #expect(abs(a.error - b.error) < 1e-9, "errors \(a.error), \(b.error)") }
    }

    /// The error and the phase difference from a hand computation: moving = 2 × reference (positive scale), u = 1, no
    /// normalisation: CCmax = 2·Σr², src_amp = Σr², target_amp = 4·Σr², so error = √|1 − 4/4| = 0.
    @Test func theErrorOfAScaledCopyIsZeroAndAnInvertedCopyHasPhaseDifferencePi() throws {
        let image = noise(16, 12, seed: 23)
        let doubled = AlignmentPins.image(image.pixels.map { $0 * 2 }, width: 16, height: 12)
        let a = try PhaseCorrelation.register(reference: image, moving: doubled, normalization: .unnormalized,
                                              precision: .float64)
        #expect(a.error < 1e-7 && abs(a.phaseDifference) < 1e-12)
        // The negated image: the correlation peak is −Σr² at the origin, but |CC| peaks at the origin all the same, with
        // the phase difference π (atan2(−0, −x) = π: the imaginary part is rounding noise of either sign).
        let negated = AlignmentPins.image(image.pixels.map { -$0 }, width: 16, height: 12)
        let b = try PhaseCorrelation.register(reference: image, moving: negated, normalization: .unnormalized,
                                              precision: .float64)
        #expect(b.shift == .zero && abs(abs(b.phaseDifference) - Double.pi) < 1e-9, "phase \(b.phaseDifference)")
        #expect(b.error < 1e-7)
    }

    /// The error with `.phase` is near 1 (CCmax ≤ 1 against huge amplitudes) at both upsample factors.
    @Test func thePhaseNormalisedErrorIsNearOne() throws {
        let image = noise(16, 16, seed: 24)
        for u in [1, 20] {
            let r = try PhaseCorrelation.register(reference: image, moving: image, upsampleFactor: u, precision: .float64)
            #expect(abs(r.error - 1) < 1e-3, "u \(u): \(r.error)")
        }
    }

    /// Frames that are identically zero have no amplitude: the error is NaN, as scikit-image's (which warns), not a refusal.
    @Test func zeroImagesGiveAnErrorOfNaN() throws {
        let zero = AlignmentPins.image([Float](repeating: 0, count: 64), width: 8, height: 8)
        let r = try PhaseCorrelation.register(reference: zero, moving: zero, precision: .float64)
        #expect(r.shift == .zero && r.error.isNaN)
    }
}
