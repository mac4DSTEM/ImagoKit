// Refusals by name, and the error texts (registration-slice-2.md section 1).
import Foundation
import ImagoAlignment
import ImagoCore
import Testing

@Suite("Refusals")
struct Refusals {
    private let a = noise(8, 8, seed: 71)

    private func refused<T>(_ body: () throws -> T) -> AlignmentError? {
        do {
            _ = try body()
            return nil
        } catch let error as AlignmentError {
            return error
        } catch {
            return nil
        }
    }

    @Test func differentSizes() {
        let b = noise(8, 9, seed: 72)
        let e = refused { try PhaseCorrelation.register(reference: a, moving: b) }
        #expect(e == .sizeMismatch(index: 1, width: 8, height: 9, expectedWidth: 8, expectedHeight: 8))
        let stack = refused { try DriftSeries.estimate([a, a, b]) }
        #expect(stack == .sizeMismatch(index: 2, width: 8, height: 9, expectedWidth: 8, expectedHeight: 8))
        let sum = refused { try AlignedSum.sum([a, b], shifts: [.zero, .zero]) }
        #expect(sum == .sizeMismatch(index: 1, width: 8, height: 9, expectedWidth: 8, expectedHeight: 8))
    }

    @Test func nonFinitePixels() throws {
        var pixels = a.pixels
        pixels[10] = .nan
        let bad = Pins.image(pixels, width: 8, height: 8)
        pixels[10] = .infinity
        let inf = Pins.image(pixels, width: 8, height: 8)
        #expect(refused { try PhaseCorrelation.register(reference: a, moving: bad) } == .nonFinitePixel(index: 1))
        #expect(refused { try PhaseCorrelation.register(reference: inf, moving: a) } == .nonFinitePixel(index: 0))
        #expect(refused { try DriftSeries.estimate([a, bad]) } == .nonFinitePixel(index: 1))
        #expect(refused { try AlignedSum.sum([bad], shifts: [.zero], method: .fourier) } == .nonFinitePixel(index: 0))
        #expect(refused { try ImageShift.fourierShift(bad, by: .zero) } == .nonFinitePixel(index: 0))
        // A shift by interpolation propagates the NaN, as SciPy's does, and is not refused.
        #expect(refused { try ImageShift.shift(bad, by: PixelShift(row: 1, column: 1)) } == nil)
    }

    @Test(arguments: [0, -1, 1025, Int.max])
    func theUpsampleFactor(_ u: Int) {
        #expect(refused { try PhaseCorrelation.register(reference: a, moving: a, upsampleFactor: u) } == .upsampleFactor(u))
        #expect(refused { try DriftSeries.estimate([a, a], upsampleFactor: u) } == .upsampleFactor(u))
    }

    @Test func theLimitsOfTheUpsampleFactorAreAccepted() throws {
        let small = noise(4, 4, seed: 73)
        for u in [1, PhaseCorrelation.maximumUpsampleFactor] {
            _ = try PhaseCorrelation.register(reference: small, moving: small, upsampleFactor: u, precision: .float64)
        }
    }

    @Test func emptyStacksAndShiftCounts() {
        #expect(refused { try DriftSeries.estimate([]) } == .emptyStack)
        #expect(refused { try AlignedSum.sum([], shifts: []) } == .emptyStack)
        #expect(refused { try AlignedSum.sum([a, a], shifts: [.zero]) } == .shiftCount(expected: 2, actual: 1))
    }

    @Test(arguments: [Double.nan, Double.infinity, -Double.infinity, 2e9, -1.0000001e9])
    func invalidShifts(_ value: Double) {
        let bad = PixelShift(row: 0, column: value)
        #expect(refused { try ImageShift.shift(a, by: bad) } == .invalidShift(index: 0))
        #expect(refused { try ImageShift.fourierShift(a, by: bad) } == .invalidShift(index: 0))
        #expect(refused { try AlignedSum.sum([a, a], shifts: [.zero, bad]) } == .invalidShift(index: 1))
        let big = PixelShift(row: ImageShift.maximumShift, column: -ImageShift.maximumShift)
        #expect(refused { try ImageShift.shift(a, by: big) } == nil, "the limit itself is accepted")
    }

    @Test func everyRefusalHasAText() {
        let all: [AlignmentError] = [.emptyStack, .sizeMismatch(index: 1, width: 2, height: 3, expectedWidth: 4, expectedHeight: 5),
                                     .nonFinitePixel(index: 3), .upsampleFactor(0), .invalidShift(index: 2),
                                     .shiftCount(expected: 3, actual: 2), .fftSetup(width: 7, height: 8)]
        let texts = all.map { $0.errorDescription ?? "" }
        #expect(texts.allSatisfy { !$0.isEmpty })
        #expect(Set(texts).count == texts.count)
        #expect(texts[1].contains("2 × 3") && texts[1].contains("4 × 5") && texts[1].contains("Image 1"))
        #expect(texts[3].contains("1024") && texts[5].contains("2 shifts") && texts[5].contains("3 frames"))
        #expect(texts[4].contains("frame 2") && texts[6].contains("7 × 8"))
    }
}
