// Drift series and aligned sums (registration-slice-2.md section 1, 3.4 and 4): a 12-frame stack of the planted image with a
// random-walk drift, against the planted drift, scikit-image's series, the cumulated pairwise shifts, and SciPy's sums.
import Foundation
import ImagoAlignment
import ImagoCore
import Testing

private func frames(_ file: String, _ pin: DriftPin) -> [Micrograph] {
    let all = Pins.floats(file)
    let n = pin.size * pin.size
    return (0..<pin.frames).map { Pins.image(Array(all[($0 * n)..<(($0 + 1) * n)]), width: pin.size, height: pin.size) }
}

private let kinds: [(String, DriftReference)] = [("first", .first), ("mean", .mean), ("previous", .previous)]

@Suite("Drift series")
struct DriftSeriesTests {
    /// The recovered drift against the planted one: within 0.05 px for `.first` and `.mean` (relative to frame 0's shift),
    /// 0.05·√k for `.previous`.
    @Test(arguments: Pins.drift, [Precision.float32, Precision.float64])
    func recoversThePlantedDrift(_ pin: DriftPin, _ precision: Precision) throws {
        let stack = frames(pin.noisy, pin)
        for (name, reference) in kinds {
            let series = try DriftSeries.estimate(stack, reference: reference, upsampleFactor: pin.u, precision: precision)
            #expect(series.count == pin.frames)
            let s0 = series[0].shift
            var worst = 0.0
            for k in 0..<pin.frames {
                let dr = (series[k].shift.row - s0.row) - pin.truth[k][0]
                let dc = (series[k].shift.column - s0.column) - pin.truth[k][1]
                let limit = try #require(pin.truth_band[name])[k]
                worst = max(worst, max(abs(dr), abs(dc)) / limit)
                #expect(max(abs(dr), abs(dc)) <= limit + 1e-9, "\(name) frame \(k): off by (\(dr), \(dc)), limit \(limit)")
            }
            band("drift.\(precision.rawValue).\(name)", worst, 1, pin.name)
        }
    }

    /// The series against scikit-image's on the same frames: equal, except where the pins flag a near-tie (then within 1/u).
    @Test(arguments: Pins.drift, [Precision.float32, Precision.float64])
    func matchesScikitImagesSeries(_ pin: DriftPin, _ precision: Precision) throws {
        let stack = frames(pin.noisy, pin)
        for (name, reference) in kinds {
            let series = try DriftSeries.estimate(stack, reference: reference, upsampleFactor: pin.u, precision: precision)
            let oracle = try #require(pin.oracle["\(name)-\(precision == .float64 ? "f64" : "f32")"])
            for k in 0..<pin.frames {
                guard let o = oracle[k] else {
                    #expect(k == 0 && reference != .mean && series[0].shift == .zero, "frame 0 is the origin")
                    continue
                }
                let tie = precision == .float64 ? o.tie64 : o.tie32
                let box = precision == .float64 ? o.box64 : o.box32
                let slack = precision == .float64 ? 1e-9 : 1e-4
                // `.previous` entries are cumulated; the oracle's pair shift is what a near-tie box describes.
                let kit = reference == .previous
                    ? PixelShift(row: series[k].shift.row - series[k - 1].shift.row,
                                 column: series[k].shift.column - series[k - 1].shift.column)
                    : series[k].shift
                let want = reference == .previous ? o.pair! : o.shift
                let d = max(abs(kit.row - want[0]), abs(kit.column - want[1]))
                if tie, let box {
                    #expect(inside(kit, box, slack: slack), "\(name) frame \(k): near-tie, kit \(kit) outside \(box)")
                } else {
                    #expect(d <= slack, "\(name) frame \(k): kit \(kit), oracle \(want)")
                }
                band("drift.oracle.\(precision.rawValue).\(name)", d, tie ? -1 : slack, "\(pin.name) frame \(k)")
                if d <= slack {
                    let noise = (precision == .float64 ? o.eps64 : o.eps32).last ?? 0   // u = 20: the refinement stage's
                    let e2 = abs(series[k].error * series[k].error - o.error * o.error)
                    #expect(e2 <= (precision == .float64 ? 1e-12 : 1e-5) + 8 * noise)
                    #expect(circular(series[k].phaseDifference, o.phasediff) <= (precision == .float64 ? 1e-10 : 1e-5) + 4 * noise)
                }
            }
        }
    }

    /// `.previous` is the cumulated pairwise shifts, exactly: the same additions in the same order.
    @Test(arguments: Pins.drift, [Precision.float32, Precision.float64])
    func previousCumulatesThePairwiseShiftsExactly(_ pin: DriftPin, _ precision: Precision) throws {
        let stack = frames(pin.noisy, pin)
        let series = try DriftSeries.estimate(stack, reference: .previous, upsampleFactor: pin.u, precision: precision)
        #expect(series[0].shift == .zero)
        var row = 0.0, column = 0.0
        for k in 1..<pin.frames {
            let pair = try PhaseCorrelation.register(reference: stack[k - 1], moving: stack[k], upsampleFactor: pin.u,
                                                     precision: precision)
            row += pair.shift.row
            column += pair.shift.column
            #expect(series[k].shift == PixelShift(row: row, column: column), "frame \(k)")
            #expect(series[k].error == pair.error || abs(series[k].error - pair.error) < 1e-5, "the pair's error")
        }
    }

    @Test func theCumulationAddsRowThenColumnFromZero() {
        let identity = ImageRegistration(shift: .zero, error: 0, phaseDifference: 0, upsampleFactor: 4,
                                         normalization: .phase, precision: .float64)
        func pair(_ r: Double, _ c: Double, _ e: Double) -> ImageRegistration {
            ImageRegistration(shift: PixelShift(row: r, column: c), error: e, phaseDifference: e / 2, upsampleFactor: 4,
                              normalization: .phase, precision: .float64)
        }
        let out = DriftSeries.cumulated([pair(0.25, -0.5, 0.1), pair(0.5, 0.25, 0.2), pair(-1, 1, 0.3)], identity)
        #expect(out.map(\.shift) == [PixelShift(row: 0.25, column: -0.5), PixelShift(row: 0.75, column: -0.25),
                                     PixelShift(row: -0.25, column: 0.75)])
        #expect(out.map(\.error) == [0.1, 0.2, 0.3] && out.map(\.phaseDifference) == [0.05, 0.1, 0.15])
    }

    /// `.mean` registers every frame (frame 0 included) onto the mean of the unshifted frames.
    @Test(arguments: Pins.drift)
    func meanRegistersOntoTheMeanImage(_ pin: DriftPin) throws {
        let stack = frames(pin.noisy, pin)
        var sum = [Double](repeating: 0, count: pin.size * pin.size)
        for f in stack { for i in 0..<sum.count { sum[i] += Double(f.pixels[i]) } }
        let mean = Pins.image(sum.map { Float($0 / Double(pin.frames)) }, width: pin.size, height: pin.size)
        let series = try DriftSeries.estimate(stack, reference: .mean, upsampleFactor: pin.u, precision: .float32)
        for k in 0..<pin.frames {
            let direct = try PhaseCorrelation.register(reference: mean, moving: stack[k], upsampleFactor: pin.u,
                                                       precision: .float32)
            #expect(series[k].shift == direct.shift, "frame \(k): \(series[k].shift) vs \(direct.shift)")
        }
        #expect(series[0].shift != .zero || pin.truth[0] == [0, 0], "frame 0 is registered, not assumed")
    }

    @Test func aSingleFrameIsItsOwnOrigin() throws {
        let image = noise(8, 8, seed: 61)
        for (_, reference) in kinds {
            let series = try DriftSeries.estimate([image], reference: reference, upsampleFactor: 4, precision: .float64)
            #expect(series.count == 1 && series[0].shift == .zero)
        }
    }

    @Test func aStackOfWholePixelRollsGivesTheCumulatedRolls() throws {
        let base = noise(16, 12, seed: 62)
        let rolls = [(0, 0), (1, 2), (3, 1), (2, -3), (-1, -1)]
        let stack = rolls.map { rolled(base, $0.0, $0.1) }
        let first = try DriftSeries.estimate(stack, reference: .first, precision: .float64)
        #expect(first.map(\.shift) == rolls.map { PixelShift(row: Double($0.0), column: Double($0.1)) })
        let previous = try DriftSeries.estimate(stack, reference: .previous, precision: .float64)
        #expect(previous.map(\.shift) == first.map(\.shift), "the cumulated pairwise rolls are the rolls")
    }
}

@Suite("Aligned sums")
struct AlignedSumTests {
    private func counts(_ shifts: [[Double]], _ size: Int) -> [Int] {
        var out = [Int](repeating: 0, count: size * size)
        for s in shifts {
            for r in 0..<size {
                let sr = Double(r) - s[0]
                for c in 0..<size {
                    let sc = Double(c) - s[1]
                    if sr >= 0, sr <= Double(size - 1), sc >= 0, sc <= Double(size - 1) { out[r * size + c] += 1 }
                }
            }
        }
        return out
    }

    @Test(arguments: Pins.drift, [Precision.float32, Precision.float64])
    func againstScipyAndTheClosedForms(_ pin: DriftPin, _ precision: Precision) throws {
        for method in AlignmentMethod.allCases { try check(pin, precision, method) }
    }

    private func check(_ pin: DriftPin, _ precision: Precision, _ method: AlignmentMethod) throws {
        let stack = frames(pin.clean, pin)
        let shifts = pin.sum_shifts.map { PixelShift(row: $0[0], column: $0[1]) }
        let result = try AlignedSum.sum(stack, shifts: shifts, method: method, precision: precision)
        let oracle = Pins.doubles(method == .fourier ? pin.sum_fourier : pin.sum_bilinear)
        let top = oracle.map(abs).max()!
        let worst = zip(result.sum.pixels, oracle).map { abs(Double($0) - $1) }.max()! / top
        let limit = method == .fourier ? (precision == .float64 ? 1e-7 : 1e-5) : (precision == .float64 ? 1e-7 : 1e-6)
        band("sum.\(method.rawValue).\(precision.rawValue)", worst, limit, pin.name)
        #expect(worst <= limit, "sum \(worst)")
        // The counts: the pin's (SciPy-independent closed form in NumPy) and this test's own re-derivation.
        let want = method == .fourier ? [Int](repeating: pin.frames, count: pin.size * pin.size) : Pins.int32s(pin.sum_count)
        #expect(result.counts == want)
        if method == .bilinear { #expect(result.counts == counts(pin.sum_shifts, pin.size)) }
        // The mean is sum / count, 0 where the count is 0.
        for i in stride(from: 0, to: result.counts.count, by: 7) {
            let expected = result.counts[i] == 0 ? 0 : Double(result.sum.pixels[i]) / Double(result.counts[i])
            #expect(abs(Double(result.mean.pixels[i]) - expected) <= 2e-7 * abs(expected) + 1e-12)
        }
        #expect(result.frameCount == pin.frames && result.method == method && result.precision == precision)
        #expect(method == .bilinear ? result.largestImaginaryPart == 0 : result.largestImaginaryPart < 1e-3 * top)
    }

    /// Frames of ones: a frame's bilinear footprint is all it covers, so the sum IS the count; the counts follow from the
    /// shifts by hand. A 4 × 5 grid (4 rows, 5 columns), shifts (0, 0), (1, 0), (0.5, 0), (0, −2), (−1, −1.5).
    @Test(arguments: [Precision.float32, Precision.float64])
    func frameCountsAtTheFootprintEdges(_ precision: Precision) throws {
        let ones = Pins.image([Float](repeating: 1, count: 20), width: 5, height: 4)
        let shifts = [PixelShift.zero, PixelShift(row: 1, column: 0), PixelShift(row: 0.5, column: 0),
                      PixelShift(row: 0, column: -2), PixelShift(row: -1, column: -1.5)]
        let result = try AlignedSum.sum(Array(repeating: ones, count: 5), shifts: shifts, method: .bilinear, precision: precision)
        // Rows covered (row − shift in [0, 3]): all; 1...3; 1...3; all; 0...2. Columns covered (col − shift in [0, 4]):
        // all; all; all; 0...2; 0...3 (col + 1.5 in [0, 4] needs col ≤ 2.5).
        func covered(_ i: Int, _ j: Int) -> Int {
            let rows = [0...3, 1...3, 1...3, 0...3, 0...2]
            let columns = [0...4, 0...4, 0...4, 0...2, 0...2]
            return (0..<5).filter { rows[$0].contains(i) && columns[$0].contains(j) }.count
        }
        for i in 0..<4 {
            for j in 0..<5 {
                #expect(result.counts[i * 5 + j] == covered(i, j), "pixel (\(i), \(j))")
                #expect(result.sum.pixels[i * 5 + j] == Float(covered(i, j)), "a frame of ones sums to the count")
                #expect(result.mean.pixels[i * 5 + j] == (covered(i, j) > 0 ? 1 : 0))
            }
        }
    }

    @Test func aSumOfIntegerShiftedFourierFramesIsTheSumOfRolls() throws {
        let base = noise(8, 6, seed: 81)
        let stack = [rolled(base, 1, -2), rolled(base, -3, 1), base]
        // Frame k is `base` rolled by (dr, dc); shifting it by the registration (dr, dc) wrapped restores `base`.
        let shifts = [PixelShift(row: 1, column: -2), PixelShift(row: -3, column: 1), PixelShift.zero]
        let result = try AlignedSum.sum(stack, shifts: shifts, method: .fourier, precision: .float64)
        for i in 0..<base.pixels.count {
            #expect(abs(result.sum.pixels[i] - 3 * base.pixels[i]) < 1e-3, "pixel \(i)")
            #expect(result.counts[i] == 3)
        }
    }

    /// Planted truth: the Fourier aligned mean of the noise-free drifted stack is the unshifted image (frame 0's content);
    /// the bilinear mean correlates above 0.999 with it where all frames overlap.
    @Test(arguments: Pins.drift)
    func theAlignedMeanIsTheUnshiftedImage(_ pin: DriftPin) throws {
        let stack = frames(pin.clean, pin)
        let image = stack[0].pixels.map(Double.init)
        let top = image.map(abs).max()!
        let shifts = pin.sum_shifts.map { PixelShift(row: $0[0], column: $0[1]) }
        let fourier = try AlignedSum.sum(stack, shifts: shifts, method: .fourier, precision: .float64)
        let worst = zip(fourier.mean.pixels, image).map { abs(Double($0) - $1) }.max()! / top
        band("sum.truth.fourier", worst, 1e-6, pin.name)
        #expect(worst <= 1e-6, "the Fourier mean differs from the image by \(worst) of its maximum")
        let bilinear = try AlignedSum.sum(stack, shifts: shifts, method: .bilinear, precision: .float64)
        var a = [Double](), b = [Double]()
        for i in 0..<image.count where bilinear.counts[i] == pin.frames {
            a.append(Double(bilinear.mean.pixels[i]))
            b.append(image[i])
        }
        let ma = a.reduce(0, +) / Double(a.count), mb = b.reduce(0, +) / Double(b.count)
        var sab = 0.0, saa = 0.0, sbb = 0.0
        for i in 0..<a.count {
            sab += (a[i] - ma) * (b[i] - mb)
            saa += (a[i] - ma) * (a[i] - ma)
            sbb += (b[i] - mb) * (b[i] - mb)
        }
        let correlation = sab / (saa * sbb).squareRoot()
        band("sum.truth.bilinear.correlation", 1 - correlation, 0.001, pin.name)
        #expect(a.count == pin.full_overlap_pixels && correlation > 0.999, "correlation \(correlation) over \(a.count) px")
    }

    /// Estimating the drift, then summing with the estimate, gives back the structure: the full path.
    @Test(arguments: Pins.drift)
    func theEstimatedDriftAlignsTheNoisyStack(_ pin: DriftPin) throws {
        let noisy = frames(pin.noisy, pin)
        let series = try DriftSeries.estimate(noisy, reference: .first, upsampleFactor: pin.u, precision: .float64)
        let clean = frames(pin.clean, pin)
        let aligned = try AlignedSum.sum(clean, shifts: series.map(\.shift), method: .fourier, precision: .float64)
        let image = clean[0].pixels.map(Double.init)
        let top = image.map { abs($0 - 1000) }.max()!
        let worst = zip(aligned.mean.pixels, image).map { abs(Double($0) - $1) }.max()! / top
        // The shifts are within 0.05 px; the lattice's contrast moves by a few percent of itself at most.
        #expect(worst < 0.2, "worst residual \(worst) of the contrast")
    }
}
