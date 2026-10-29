// The mixed-radix Float64 plan (lengths of prime factors 2, 3, 5, 7) against the direct O(n²) DFT, alone and inside the 2D
// transform, and the routing: which lengths take which engine is part of the contract (design rule: vDSP_DFT_zop for
// 2ⁿ × {1, 3, 5, 15}, Bluestein for the rest).
import Foundation
import Testing
@testable import ImagoCore
import ImagoKitTestSupport

/// The exact DFT by definition, phases reduced modulo n in integers.
private func direct(_ re: [Double], _ im: [Double], forward: Bool) -> ([Double], [Double]) {
    let n = re.count
    var outRe = [Double](repeating: 0, count: n), outIm = outRe
    let sign = forward ? -1.0 : 1.0
    for k in 0..<n {
        var sr = 0.0, si = 0.0
        for j in 0..<n {
            let angle = sign * 2 * Double.pi * Double((j * k) % n) / Double(n)
            let c = cos(angle), s = sin(angle)
            sr += re[j] * c - im[j] * s
            si += re[j] * s + im[j] * c
        }
        outRe[k] = sr; outIm[k] = si
    }
    return (outRe, outIm)
}

private func signal(_ n: Int, seed: UInt64) -> ([Double], [Double]) {
    var g = SplitMix(state: seed)
    return ((0..<n).map { _ in g.next() }, (0..<n).map { _ in g.next() })
}

private struct SplitMix {
    var state: UInt64
    mutating func next() -> Double {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return Double((z ^ (z >> 31)) >> 11) / Double(1 << 53) - 0.5
    }
}

private let smoothLengths: [Int] = (2...140).filter { MixedRadixPlanD.supports($0) } + [243, 343, 360, 480, 500, 504, 625, 1000, 1008]

@Test func factorizationIsTwoThenFoursThenOddPrimes() {
    #expect(MixedRadixPlanD.factorization(500) == [4, 5, 5, 5])
    #expect(MixedRadixPlanD.factorization(504) == [2, 4, 3, 3, 7])
    #expect(MixedRadixPlanD.factorization(360) == [2, 4, 3, 3, 5])
    #expect(MixedRadixPlanD.factorization(480) == [2, 4, 4, 3, 5])
    #expect(MixedRadixPlanD.factorization(49) == [7, 7])
    for n in [0, 1, 11, 13, 22, 26, 77 * 11, 1009] { #expect(MixedRadixPlanD.factorization(n) == nil, "n = \(n)") }
}

@Test(arguments: smoothLengths)
func mixedRadixLineEqualsTheDirectDFT(_ n: Int) throws {
    let plan = try #require(MixedRadixPlanD(length: n))
    for odd in [0, 1] {                                         // an even and an odd number of lines: the zero-partner pass
        let lines = 3 + odd
        var inRe = [Double](), inIm = [Double]()
        for l in 0..<lines { let (r, i) = signal(n, seed: UInt64(n * 10 + l)); inRe += r; inIm += i }
        for forward in [true, false] {
            var outRe = [Double](repeating: 0, count: n * lines), outIm = outRe
            plan.transformLines(count: lines, forward: forward, inputStride: n, inRe, inIm, &outRe, &outIm)
            for l in 0..<lines {
                let (wr, wi) = direct(Array(inRe[l * n..<(l + 1) * n]), Array(inIm[l * n..<(l + 1) * n]), forward: forward)
                var worst = 0.0
                for k in 0..<n { worst = max(worst, abs(outRe[l * n + k] - wr[k]), abs(outIm[l * n + k] - wi[k])) }
                #expect(worst <= 1e-12 * Double(n), "n = \(n) line \(l) forward \(forward): \(worst)")
            }
        }
    }
}

@Test(arguments: [(6, 7), (14, 9), (21, 20), (60, 35), (35, 60), (49, 8), (45, 33), (500, 3), (3, 504)])
func mixedRadix2DEqualsSeparableDirectDFT(_ shape: (Int, Int)) throws {
    let (w, h) = shape                                          // columns, rows (nx, ny)
    let plan = try #require(FFT2DDouble(nx: w, ny: h))
    var (re, im) = (signal(w * h, seed: 1).0, signal(w * h, seed: 2).0)
    let (inRe, inIm) = (re, im)
    plan.transform(re: &re, im: &im, forward: true, scaleInverse: false)
    // Rows then columns by definition.
    var tr = inRe, ti = inIm
    for r in 0..<h {
        let (a, b) = direct(Array(tr[r * w..<(r + 1) * w]), Array(ti[r * w..<(r + 1) * w]), forward: true)
        tr.replaceSubrange(r * w..<(r + 1) * w, with: a); ti.replaceSubrange(r * w..<(r + 1) * w, with: b)
    }
    for c in 0..<w {
        let (a, b) = direct((0..<h).map { tr[$0 * w + c] }, (0..<h).map { ti[$0 * w + c] }, forward: true)
        for r in 0..<h { tr[r * w + c] = a[r]; ti[r * w + c] = b[r] }
    }
    var worst = 0.0
    for i in 0..<(w * h) { worst = max(worst, abs(re[i] - tr[i]), abs(im[i] - ti[i])) }
    #expect(worst <= 1e-11 * Double(w * h).squareRoot(), "\(w)x\(h): \(worst)")
    plan.transform(re: &re, im: &im, forward: false, scaleInverse: true)
    var trip = 0.0
    for i in 0..<(w * h) { trip = max(trip, abs(re[i] - inRe[i]), abs(im[i] - inIm[i])) }
    #expect(trip <= 1e-13, "round trip \(w)x\(h): \(trip)")
}

/// Which engine serves a length (the kit's design contract), seen through the plan's routing.
@Test func lengthsAreRoutedToTheirEngine() {
    // vDSP_DFT_zop (2ⁿ × {1, 3, 5, 15}) and the radix plans keep their lengths; the mixed-radix plan only takes smooth lengths
    // nothing else serves; Bluestein stays for any other prime factor.
    for n in [16, 48, 80, 240, 960] { #expect(DFTPairD(length: n) != nil, "vDSP serves \(n)") }
    for n in [500, 504, 360, 7, 14, 49, 100, 700] { #expect(DFTPairD(length: n) == nil, "vDSP has no plan for \(n)") }
    for n in [500, 504, 360, 700] { #expect(MixedRadixPlanD.supports(n)) }
    for n in [11, 13, 22, 1009, 2 * 11 * 3] { #expect(!MixedRadixPlanD.supports(n), "\(n) stays on Bluestein") }
}
