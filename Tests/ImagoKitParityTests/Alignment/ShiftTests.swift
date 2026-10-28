// ndimage.shift (orders 0 and 1, seven modes) and fourier_shift: the pinned
// cases on both paths, the closed forms of the modes and ties, and the Fourier shift's definition.
import Foundation
import ImagoKitTestSupport
import ImagoAlignment
import ImagoCore
import Testing

private func relative(_ got: [Double], _ want: [Double]) -> Double {
    let top = want.map(abs).max() ?? 0
    let worst = zip(got, want).map { abs($0 - $1) }.max() ?? 0
    return top > 0 ? worst / top : worst
}

private func mode(_ name: String) -> ShiftMode { ShiftMode(rawValue: name)! }

@Suite("ndimage.shift against SciPy 1.18.1")
struct ShiftAgainstOracle {
    /// The Float64 engine (package access) within 1e-12 of the case's maximum; the Float32 engine within 1e-6; the public
    /// calls return Float32 pixels, so they equal the Float32 rounding of the reference (half an ulp, 6e-8 of the maximum).
    @Test(arguments: AlignmentPins.shift.combos)
    func everyModeAndOrder(_ combo: ShiftComboPin) throws {
        let pins = AlignmentPins.shift
        let expected = AlignmentPins.doubles(combo.file)
        let order = ShiftOrder(rawValue: combo.order)!
        var offset = 0
        var worst64 = 0.0, worst32 = 0.0, worstPublic64 = 0.0, worstPublic32 = 0.0
        for im in pins.images {
            let pixels = AlignmentPins.floats(im.file)
            let image = AlignmentPins.image(pixels, width: im.width, height: im.height)
            var f64 = [Double](), f32 = [Double](), pub64 = [Double](), pub32 = [Double](), want = [Double]()
            for s in pins.shifts {
                let shift = PixelShift(row: s[0], column: s[1])
                let count = im.width * im.height
                want += expected[offset..<(offset + count)]
                offset += count
                f64 += ImageShift.shifted(pixels.map(Double.init), im.width, im.height, shift, order, mode(combo.mode), pins.cval)
                f32 += ImageShift.shifted(pixels, im.width, im.height, shift, order, mode(combo.mode), Float(pins.cval)).map(Double.init)
                pub64 += try ImageShift.shift(image, by: shift, order: order, mode: mode(combo.mode), cval: pins.cval,
                                              precision: .float64).pixels.map(Double.init)
                pub32 += try ImageShift.shift(image, by: shift, order: order, mode: mode(combo.mode), cval: pins.cval,
                                              precision: .float32).pixels.map(Double.init)
            }
            // The image's own maximum (not the case's) is the scale: the per-image tolerance of the registration.
            let top = max(pixels.map { abs(Double($0)) }.max() ?? 0, abs(pins.cval))
            let scaled = { (got: [Double]) in zip(got, want).map { abs($0 - $1) }.max()! / top }
            worst64 = max(worst64, scaled(f64))
            worst32 = max(worst32, scaled(f32))
            worstPublic64 = max(worstPublic64, scaled(pub64))
            worstPublic32 = max(worstPublic32, scaled(pub32))
        }
        band("shift.f64.\(combo)", worst64, 1e-12, combo.mode)
        band("shift.f32.\(combo)", worst32, 1e-6, combo.mode)
        band("shift.public64.\(combo)", worstPublic64, 6e-8, combo.mode)
        band("shift.public32.\(combo)", worstPublic32, 1e-6, combo.mode)
        #expect(worst64 <= 1e-12)
        #expect(worst32 <= 1e-6)
        #expect(worstPublic64 <= 6.5e-8, "the public Float64 call is the Float32 rounding of the Float64 result")
        #expect(worstPublic32 <= 1e-6)
        #expect(offset == expected.count)
    }

    @Test(arguments: AlignmentPins.fourier.images)
    func fourierShift(_ im: FourierImagePin) throws {
        let pins = AlignmentPins.fourier
        let pixels = AlignmentPins.floats(im.file)
        let image = AlignmentPins.image(pixels, width: im.width, height: im.height)
        let expected = AlignmentPins.doubles(im.out)
        let count = im.width * im.height
        let top = pixels.map { abs(Double($0)) }.max()!
        var worst64 = 0.0, worst32 = 0.0, worstPublic64 = 0.0
        for (k, s) in pins.shifts.enumerated() {
            let shift = PixelShift(row: s[0], column: s[1])
            let want = Array(expected[(k * count)..<((k + 1) * count)])
            let (f64, i64) = try ImageShift.fourierShiftWithResidual(image, by: shift, precision: .float64)
            let (f32, i32) = try ImageShift.fourierShiftWithResidual(image, by: shift, precision: .float32)
            worst64 = max(worst64, zip(f64.pixels, want).map { abs(Double($0) - $1) }.max()! / top)
            worst32 = max(worst32, zip(f32.pixels, want).map { abs(Double($0) - $1) }.max()! / top)
            let plain = try ImageShift.fourierShift(image, by: shift, precision: .float64)
            worstPublic64 = max(worstPublic64, zip(plain.pixels, want).map { abs(Double($0) - $1) }.max()! / top)
            // Two calls of one computation: equal to a Float32 ulp (a vDSP transform is not bit-reproducible).
            #expect(zip(plain.pixels, f64.pixels).allSatisfy { abs($0 - $1) <= 2e-7 * Float(top) })
            // The imaginary residual: NumPy's, to the precision's noise floor (it is exactly rounding where the pin has ~0).
            let floor64 = max(im.imag_max[k] * 1e-6, 1e-12 * top), floor32 = max(im.imag_max[k] * 1e-3, 3e-5 * top)
            #expect(abs(i64 - im.imag_max[k]) <= floor64 + 1e-9 * top, "float64 residual \(i64) vs \(im.imag_max[k])")
            #expect(abs(i32 - im.imag_max[k]) <= floor32, "float32 residual \(i32) vs \(im.imag_max[k])")
        }
        band("fourier.public64.\(im.name)", worstPublic64, 6.5e-8, im.name)
        band("fourier.f32.\(im.name)", worst32, 1e-5, im.name)
        #expect(worstPublic64 <= 6.5e-8, "the Float64 path, rounded to Float32 pixels, is the reference's Float32 rounding")
        #expect(worst32 <= 1e-5)
        _ = worst64
    }
}

@Suite("Shifts, closed forms")
struct ShiftClosedForms {
    private let row = try! Micrograph(width: 5, height: 1, pixels: [10, 20, 30, 40, 50])

    private func apply(_ m: ShiftMode, _ order: ShiftOrder, _ s: Double, cval: Double = -1,
                       _ precision: Precision = .float64) throws -> [Float] {
        try ImageShift.shift(row, by: PixelShift(row: 0, column: s), order: order, mode: m, cval: cval,
                             precision: precision).pixels
    }

    @Test(arguments: [Precision.float32, Precision.float64])
    func anIntegerShiftMovesTheContentAndThePolicyFillsTheRest(_ precision: Precision) throws {
        #expect(try apply(.constant, .linear, 2, precision) == [-1, -1, 10, 20, 30])
        #expect(try apply(.constant, .nearest, -2, precision) == [30, 40, 50, -1, -1])
        #expect(try apply(.nearest, .linear, 2, precision) == [10, 10, 10, 20, 30])
        #expect(try apply(.gridWrap, .linear, 2, precision) == [40, 50, 10, 20, 30])
        #expect(try apply(.reflect, .linear, 2, precision) == [20, 10, 10, 20, 30])
        #expect(try apply(.mirror, .linear, 2, precision) == [30, 20, 10, 20, 30])
        // "wrap" has period n − 1 = 4: the last pixel is the first one's twin.
        #expect(try apply(.wrap, .linear, 2, precision) == [30, 40, 10, 20, 30])
        #expect(try apply(.gridConstant, .linear, 2, precision) == [-1, -1, 10, 20, 30])
    }

    /// Order 0 takes the pixel at floor(coordinate + 1/2) (ties go up): shift +1/2 reads coordinate i − 1/2 → pixel i; shift
    /// −1/2 reads i + 1/2 → pixel i + 1.
    @Test func orderZeroTiesGoUp() throws {
        #expect(try apply(.nearest, .nearest, 0.5) == [10, 20, 30, 40, 50])
        #expect(try apply(.nearest, .nearest, -0.5) == [20, 30, 40, 50, 50])
        #expect(try apply(.nearest, .nearest, 1.5) == [10, 10, 20, 30, 40])
        #expect(try apply(.nearest, .nearest, 0.49) == [10, 20, 30, 40, 50])
        #expect(try apply(.nearest, .nearest, 0.51) == [10, 10, 20, 30, 40])
    }

    @Test func orderOneInterpolatesLinearly() throws {
        #expect(try apply(.nearest, .linear, 0.25) == [10, 17.5, 27.5, 37.5, 47.5])
        #expect(try apply(.nearest, .linear, -0.5) == [15, 25, 35, 45, 50])
        // "constant" does not interpolate toward cval; "grid-constant" does.
        #expect(try apply(.constant, .linear, 0.5) == [-1, 15, 25, 35, 45])
        #expect(try apply(.gridConstant, .linear, 0.5) == [4.5, 15, 25, 35, 45])
    }

    /// Output (1, 1) of a 2 × 2 image shifted by (0.25, 0.5) reads the coordinate (0.75, 0.5): halfway between the columns,
    /// three quarters of the way down the rows: 0.25 · 5 + 0.75 · 105 = 80. Swapped weights would give 55 or 105 − 25.
    @Test func bilinearWeightsBelongToTheirAxes() throws {
        let image = try Micrograph(width: 2, height: 2, pixels: [0, 10, 100, 110])
        let r = try ImageShift.shift(image, by: PixelShift(row: 0.25, column: 0.5), order: .linear, mode: .nearest,
                                     precision: .float64)
        #expect(r.pixels == [0, 5, 75, 80], "\(r.pixels)")
    }

    @Test func theCalibrationIsKept() throws {
        let cal = try Micrograph(width: 3, height: 2, pixels: [1, 2, 3, 4, 5, 6],
                                 x: AxisCalibration(scale: 0.5, unit: "nm", origin: 1), y: AxisCalibration(scale: 0.25, unit: "nm"))
        for r in [try ImageShift.shift(cal, by: PixelShift(row: 1, column: 1)),
                  try ImageShift.fourierShift(cal, by: PixelShift(row: 1, column: 1))] {
            #expect(r.x == cal.x && r.y == cal.y && r.width == 3 && r.height == 2)
        }
    }

    @Test func aZeroShiftIsTheIdentity() throws {
        let image = noise(9, 7, seed: 31)
        for m in ShiftMode.allCases {
            for o in [ShiftOrder.nearest, .linear] {
                #expect(try ImageShift.shift(image, by: .zero, order: o, mode: m, precision: .float64) == image, "\(m) \(o)")
            }
        }
        let f = try ImageShift.fourierShift(image, by: .zero, precision: .float64)
        #expect(zip(f.pixels, image.pixels).allSatisfy { abs($0 - $1) < 1e-4 })
    }

    /// The Fourier shift of an integer shift is a circular roll, exactly to rounding, on both paths and for every size class.
    @Test(arguments: [Precision.float32, Precision.float64], [(8, 6), (7, 9), (10, 1), (1, 10), (17, 5)])
    func anIntegerFourierShiftIsACircularRoll(_ precision: Precision, _ size: (Int, Int)) throws {
        let image = noise(size.0, size.1, seed: 41)
        let r = try ImageShift.fourierShift(image, by: PixelShift(row: 2, column: -3), precision: precision)
        let w = size.0, h = size.1
        var worst: Float = 0
        for i in 0..<h {
            for j in 0..<w {
                let want = image.pixels[(((i - 2) % h + h) % h) * w + (((j + 3) % w) + w) % w]
                worst = max(worst, abs(r.pixels[i * w + j] - want))
            }
        }
        #expect(worst < (precision == .float64 ? 1e-5 : 1e-3), "worst \(worst)")
    }

    /// A half-pixel Fourier shift of an even size leaves an imaginary part at Nyquist (the definition's), an odd size none.
    @Test func theNyquistResidual() throws {
        let even = try ImageShift.fourierShiftWithResidual(noise(8, 8, seed: 51), by: PixelShift(row: 0.5, column: 0.5),
                                                           precision: .float64)
        let odd = try ImageShift.fourierShiftWithResidual(noise(9, 9, seed: 52), by: PixelShift(row: 0.5, column: 0.5),
                                                          precision: .float64)
        let integer = try ImageShift.fourierShiftWithResidual(noise(8, 8, seed: 53), by: PixelShift(row: 2, column: 3),
                                                              precision: .float64)
        #expect(even.largestImaginaryPart > 1e-3, "even size: \(even.largestImaginaryPart)")
        #expect(odd.largestImaginaryPart < 1e-9, "odd size: \(odd.largestImaginaryPart)")
        #expect(integer.largestImaginaryPart < 1e-9)
    }

    /// The Fourier shift's direction: a positive shift moves the content toward larger indices.
    @Test func theFourierShiftsDirection() throws {
        let image = try Micrograph(width: 8, height: 1, pixels: [0, 0, 100, 0, 0, 0, 0, 0])
        let r = try ImageShift.fourierShift(image, by: PixelShift(row: 0, column: 1), precision: .float64)
        #expect(r.pixels[3] > 99.9 && r.pixels[2] < 0.1)
        let c = try ImageShift.shift(image, by: PixelShift(row: 0, column: 1), order: .linear, mode: .constant, precision: .float64)
        #expect(c.pixels[3] == 100)
    }
}
