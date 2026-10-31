// The 14 get_window families beyond the first 11 (triang, parzen, bohman, barthann, cosine, exponential, taylor, lanczos,
// kaiser_bessel_derived, general_cosine, general_gaussian, general_hamming, dpss, chebwin) against the pinned runs of
// tools/window-families/pins.py (NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0).
// Tolerance, 1D: what scipy/signal/tests/test_windows.py uses for the family (xp_assert_close defaults to rtol 4 sqrt(eps),
// atol 0): barthann rtol 1e-15 atol 1e-15; exponential rtol 1e-14; cosine and general_gaussian atol 1e-12; lanczos atol 1e-9;
// chebwin atol 1e-8 (its chebwin(6, 100) case; the others take the default or decimal 4); dpss atol 1e-7; the rest the default. 2D: scikit-image ships no tests, so assert_allclose float64
// defaults (rtol 1e-7, atol 0) widened by the family tolerance of the 1D window it interpolates.
import Foundation
import ImagoKitTestSupport
import ImagoFilters
import Testing

struct WindowFamilyPins: Decodable, Sendable {
    struct Case: Decodable, Sendable, CustomStringConvertible {
        let family: String; let params: [Double]; let h, w: Int; let window1d, window2d: [Double]
        var description: String { "\(family)\(params) \(h)x\(w)" }
    }
    struct Refusal: Decodable, Sendable, CustomStringConvertible {
        let family: String; let params: [Double]; let count: Int
        var description: String { "\(family)\(params) M=\(count)" }
    }
    struct Long: Decodable, Sendable, CustomStringConvertible {
        let family: String; let params: [Double]; let count: Int; let window1d: [Double]
        var description: String { "\(family)\(params) M=\(count)" }
    }
    let cases: [Case]; let long: [Long]; let refusals: [Refusal]
    static let all: WindowFamilyPins = {
        let url = FixtureDirs.fixtures(of: "ImagoFiltersTests").appendingPathComponent("window-family-pins.json")
        return try! JSONDecoder().decode(WindowFamilyPins.self, from: Data(contentsOf: url))
    }()
}

private func windowType(_ family: String, _ p: [Double]) -> WindowType {
    switch family {
    case "triang": .triang
    case "parzen": .parzen
    case "bohman": .bohman
    case "barthann": .barthann
    case "cosine": .cosine
    case "exponential": .exponential(tau: p[0])
    case "taylor": .taylor(nbar: Int(p[0]), sll: p[1], norm: p[2] != 0)
    case "lanczos": .lanczos
    case "kaiser_bessel_derived": .kaiserBesselDerived(beta: p[0])
    case "general_cosine": .generalCosine(a: p)
    case "general_gaussian": .generalGaussian(p: p[0], sig: p[1])
    case "general_hamming": .generalHamming(alpha: p[0])
    case "dpss": .dpss(nw: p[0])
    default: .chebwin(at: p[0])
    }
}

/// (rtol, atol) of the family's own SciPy test.
private func tolerance(_ family: String) -> (rtol: Double, atol: Double) {
    let standard = 4 * Double.ulpOfOne.squareRoot()
    switch family {
    case "barthann": return (1e-15, 1e-15)
    case "exponential": return (1e-14, 0)
    case "cosine", "general_gaussian": return (standard, 1e-12)
    case "lanczos": return (standard, 1e-9)
    case "chebwin": return (standard, 1e-8)
    case "dpss": return (standard, 1e-7)
    default: return (standard, 0)
    }
}

private func allclose(_ got: [Double], _ want: [Double], rtol: Double, atol: Double, _ label: String) {
    #expect(got.count == want.count, "\(label) count \(got.count) vs \(want.count)")
    var worst = -Double.infinity
    for (g, w) in zip(got, want) { worst = max(worst, abs(g - w) - (atol + rtol * abs(w))) }
    #expect(worst <= 0, "\(label): exceeds atol \(atol) + rtol \(rtol) by \(worst)")
}

@Test(arguments: WindowFamilyPins.all.cases)
func windowFamiliesMatchSciPyAndSciKitImage(_ c: WindowFamilyPins.Case) throws {
    let type = windowType(c.family, c.params)
    let t = tolerance(c.family)
    allclose(try WindowFunction.window1D(type, count: max(c.h, c.w)), c.window1d, rtol: t.rtol, atol: t.atol, "get_window \(c)")
    allclose(try WindowFunction.window2D(type, rows: c.h, columns: c.w), c.window2d, rtol: max(1e-7, t.rtol), atol: t.atol,
             "window \(c)")
}

/// Length 2 for every family, and lengths 1023 to 4096 for the dpss eigensolver and the chebwin DFT.
@Test(arguments: WindowFamilyPins.all.long)
func windowFamiliesMatchSciPyAtTheEdgeAndAtImageLengths(_ c: WindowFamilyPins.Long) throws {
    let t = tolerance(c.family)
    allclose(try WindowFunction.window1D(windowType(c.family, c.params), count: c.count), c.window1d, rtol: t.rtol, atol: t.atol,
             "get_window \(c)")
}

@Test(arguments: WindowFamilyPins.all.refusals)
func windowFamilyRefusalsMatchSciPy(_ r: WindowFamilyPins.Refusal) {
    #expect(throws: FilterError.self) { try WindowFunction.window1D(windowType(r.family, r.params), count: r.count) }
}

@Test func windowFamilyDefaultsAreSciPyDefaults() throws {
    #expect(try WindowFunction.window1D(.taylor(), count: 9) == WindowFunction.window1D(.taylor(nbar: 4, sll: 30, norm: true), count: 9))
    #expect(try WindowFunction.window1D(.exponential(), count: 9) == WindowFunction.window1D(.exponential(tau: 1), count: 9))
    #expect(try WindowFunction.window1D(.kaiserBesselDerived(beta: 4), count: 0).isEmpty)
}
