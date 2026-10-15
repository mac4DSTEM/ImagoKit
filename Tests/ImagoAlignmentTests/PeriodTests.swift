// Every periodic boundary mode repeats with a known period, so a whole-number shift by a multiple of it returns the image
// unchanged: "reflect" (half-sample symmetric) every 2n, "mirror" (whole-sample) every 2n − 2, "grid-wrap" every n. Added
// after the 0.1.0 review: "reflect" returned cval for shifts of 4n, 6n, … (a coordinate reduced to exactly 0 was then
// mapped to −1); the pins' integer shifts never reached a multiple of 2n beyond 2n.
import Foundation
import ImagoAlignment
import ImagoCore
import Testing

@Suite("Periodic modes")
struct PeriodicModes {
    static func period(_ mode: ShiftMode, _ n: Int) -> Int {
        switch mode {
        case .reflect: 2 * n
        case .mirror: 2 * n - 2
        default: n                      // grid-wrap
        }
    }

    @Test(arguments: [2, 3, 5], [ShiftOrder.nearest, ShiftOrder.linear])
    func aShiftByWholePeriodsIsTheIdentity(_ n: Int, _ order: ShiftOrder) throws {
        let pixels = (0..<(n * n)).map { Float(10 + 7 * $0) }
        let image = try Micrograph(width: n, height: n, pixels: pixels)
        for mode in [ShiftMode.reflect, .mirror, .gridWrap] {
            let p = Double(Self.period(mode, n))
            for k in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0] {
                for shift in [PixelShift(row: k * p, column: 0), PixelShift(row: 0, column: k * p), PixelShift(row: k * p, column: k * p)] {
                    let out = try ImageShift.shift(image, by: shift, order: order, mode: mode, cval: -999)
                    #expect(out.pixels == pixels, "\(mode), n \(n), shift (\(shift.row), \(shift.column))")
                }
            }
        }
    }
}
