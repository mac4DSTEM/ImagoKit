// ndimage.shift at spline orders 2 to 5 and the masked phase_cross_correlation against SciPy 1.18.1 / scikit-image 0.26.0
// (tools/alignment/pins.py). Tolerances: SciPy's shift tests compare to decimal 6 (test_interpolation.py); the float32
// outputs here are rounded from Double, so they are held to 1e-5 of the image maximum (the float32 default). The masked
// correlation (test_masked_phase_cross_correlation.py upstream, not shipped in the wheel) is held to assert_allclose
// defaults for float64 (rtol 1e-7, atol 1e-7) and 1e-4 absolute for float32 (the FFT length differs from SciPy's fast length).
import Foundation
import ImagoAlignment
import ImagoCore
import Testing

struct SplineShiftPin: Decodable, Sendable, CustomStringConvertible {
    let h, w, order: Int; let mode: String; let shift: [Double]; let cval: Double; let x, y: [Double]
    var description: String { "\(h)x\(w) o\(order) \(mode) \(shift)" }
    static let all: [SplineShiftPin] = {
        struct File: Decodable { let cases: [SplineShiftPin] }
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/spline-shift-pins.json")
        return try! JSONDecoder().decode(File.self, from: Data(contentsOf: url)).cases
    }()
}

struct MaskedPin: Decodable, Sendable, CustomStringConvertible {
    let dtype: String; let rh, rw, mh, mw: Int; let ratio: Double
    let ref, mov: [Double]; let rmask: [Int]; let mmask: [Int]?
    let shift: [Double]; let cc: [Double]; let ccShape: [Int]; let unique: Bool
    var description: String { "\(dtype) ref \(rh)x\(rw) mov \(mh)x\(mw) r\(ratio)" }
    static let all: [MaskedPin] = {
        struct File: Decodable { let cases: [MaskedPin] }
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/masked-registration-pins.json")
        return try! JSONDecoder().decode(File.self, from: Data(contentsOf: url)).cases
    }()
}

@Test(arguments: SplineShiftPin.all)
func splineShiftMatchesSciPy(_ c: SplineShiftPin) throws {
    let pixels = c.x.map(Float.init)
    let image = try Micrograph(width: c.w, height: c.h, pixels: pixels, x: nil, y: nil)
    let order = ShiftOrder(rawValue: c.order)!
    let mode = ShiftMode(rawValue: c.mode)!
    for precision in Precision.allCases {
        let got = try ImageShift.shift(image, by: PixelShift(row: c.shift[0], column: c.shift[1]), order: order, mode: mode,
                                       cval: c.cval, precision: precision)
        let top = max(c.x.map(abs).max()!, abs(c.cval))
        let worst = zip(got.pixels, c.y).map { abs(Double($0) - $1) }.max()!
        #expect(worst <= 1e-5 * top, "\(c) \(precision): \(worst) vs \(1e-5 * top)")
    }
}

@Test func splineOrdersRefuseTheModesNotPorted() throws {
    let image = try Micrograph(width: 4, height: 4, pixels: (0..<16).map(Float.init), x: nil, y: nil)
    for mode in [ShiftMode.constant, .wrap] {
        #expect(throws: AlignmentError.splineMode(mode)) {
            try ImageShift.shift(image, by: PixelShift(row: 0.5, column: 0.5), order: .cubic, mode: mode)
        }
    }
}

@Test func splineShiftByZeroIsTheIdentity() throws {
    let pixels = (0..<35).map { Float(($0 * 7919) % 101) }
    let image = try Micrograph(width: 7, height: 5, pixels: pixels, x: nil, y: nil)
    for order in [ShiftOrder.quadratic, .cubic, .quartic, .quintic] {
        let got = try ImageShift.shift(image, by: .zero, order: order, mode: .mirror)
        #expect(zip(got.pixels, pixels).allSatisfy { abs($0 - $1) < 1e-4 }, "order \(order)")
    }
}

@Test(arguments: MaskedPin.all)
func maskedRegistrationMatchesSciKitImage(_ c: MaskedPin) throws {
    let precision: Precision = c.dtype == "float32" ? .float32 : .float64
    let ref = try Micrograph(width: c.rw, height: c.rh, pixels: c.ref.map(Float.init), x: nil, y: nil)
    let mov = try Micrograph(width: c.mw, height: c.mh, pixels: c.mov.map(Float.init), x: nil, y: nil)
    let rmask = c.rmask.map { $0 != 0 }, mmask = c.mmask?.map { $0 != 0 }
    let surface = try PhaseCorrelation.maskedCorrelation(reference: ref, moving: mov, referenceMask: rmask, movingMask: mmask ?? rmask,
                                                         overlapRatio: c.ratio, precision: precision)
    #expect(surface.rows == c.ccShape[0] && surface.columns == c.ccShape[1])
    let worst = zip(surface.values, c.cc).map { abs($0 - $1) }.max()!
    #expect(worst <= (precision == .float32 ? 1e-4 : 1e-7), "\(c): surface differs by \(worst)")
    if c.unique {
        let shift = try PhaseCorrelation.registerMasked(reference: ref, moving: mov, referenceMask: rmask, movingMask: mmask,
                                                        overlapRatio: c.ratio, precision: precision)
        #expect(shift.row == c.shift[0] && shift.column == c.shift[1], "\(c): \(shift) vs \(c.shift)")
    }
}

@Test func maskedRegistrationRefusesBadMasks() throws {
    let a = try Micrograph(width: 4, height: 4, pixels: [Float](repeating: 1, count: 16), x: nil, y: nil)
    let b = try Micrograph(width: 5, height: 4, pixels: [Float](repeating: 1, count: 20), x: nil, y: nil)
    #expect(throws: AlignmentError.maskCount(expected: 16, actual: 3)) {
        try PhaseCorrelation.registerMasked(reference: a, moving: a, referenceMask: [true, false, true])
    }
    // Differently sized images need an explicit moving mask (scikit-image raises).
    #expect(throws: AlignmentError.self) {
        try PhaseCorrelation.registerMasked(reference: a, moving: b, referenceMask: [Bool](repeating: true, count: 16))
    }
}
