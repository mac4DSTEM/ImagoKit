// The HRTEM filters against a float64 NumPy/SciPy reading of the same steps, with that reading's checks. The pins in
// Fixtures/hrtem-filter-pins.json are frozen: their generator read macTEM's parity harness from a commit whose history was
// squashed, so it was retired.
import Foundation
import ImagoKitTestSupport
import ImagoCore
import ImagoFFT
import ImagoFilters
import Testing

struct HRTEMParams: Decodable, Sendable, Equatable {
    let delta, step: Double
    let cycles: Int
    let smoothing: Bool
    let order: Int
    let ro, pct: Double
}

struct HRTEMPin: Decodable, Sendable, CustomStringConvertible {
    let name, tier, input: String
    let n, passes: Int
    let params: HRTEMParams
    let files: [String: String]
    var description: String { name }
}

enum HRTEMPins {
    static let fixtures = FixtureDirs.fixtures(of: "ImagoFiltersTests")
    static let full: URL? = ProcessInfo.processInfo.environment["IMAGOKIT_PINS_FULL"].map {
        URL(fileURLWithPath: $0).appendingPathComponent("hrtem").appendingPathComponent("ref")
    }.flatMap { FileManager.default.fileExists(atPath: $0.appendingPathComponent("manifest.json").path) ? $0 : nil }
    private struct File: Decodable { let hrtem: [HRTEMPin] }
    private struct Harness: Decodable { let cases: [HRTEMPin] }
    static let folder = full ?? fixtures.appendingPathComponent("hrtem")
    static let cases: [HRTEMPin] = {
        if let full {
            return try! JSONDecoder().decode(Harness.self, from: Data(contentsOf: full.appendingPathComponent("manifest.json"))).cases
        }
        return try! JSONDecoder().decode(File.self, from: Data(contentsOf: fixtures.appendingPathComponent("hrtem-filter-pins.json"))).hrtem
    }()
    static func doubles(_ name: String) -> [Double] {
        try! Data(contentsOf: folder.appendingPathComponent(name)).withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
    }
    /// max |got − want| / max |want|.
    static func relativeToMaximum(_ got: [Double], _ want: [Double]) -> Double {
        precondition(got.count == want.count)
        let top = want.map(abs).max() ?? 0
        let worst = zip(got, want).map { abs($0 - $1) }.max() ?? 0
        return top > 0 ? worst / top : worst
    }
    static func floats(_ name: String) -> [Float] {
        try! Data(contentsOf: folder.appendingPathComponent(name)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}

private let defaults = HRTEMParams(delta: 5, step: 2, cycles: 99, smoothing: true, order: 4, ro: 0.3, pct: 0)
private let noisyTiers: Set<String> = ["bin", "offbin", "noise", "ramp", "nyquist", "offbin0", "offbin25", "offbin50"]
private let outputBand: [Precision: Double] = [.float32: 1e-5, .float64: 1e-10]

@Test func everyHRTEMCaseIsLoaded() {
    #expect(HRTEMPins.cases.count == (HRTEMPins.full == nil ? 2 : 41))
}

@Test(arguments: HRTEMPins.cases, Precision.allCases)
func filtersMatchTheFloat64Reading(_ c: HRTEMPin, _ precision: Precision) throws {
    let n = c.n, count = n * n, half = n / 2
    let input = HRTEMPins.floats(c.input)
    var p = HRTEMFilter.Parameters()
    p.delta = c.params.delta; p.step = c.params.step; p.cycles = c.params.cycles; p.smoothing = c.params.smoothing
    p.order = c.params.order; p.ro = c.params.ro; p.lowFrequencyPercent = c.params.pct
    let result = try HRTEMFilter.filter(pixels: input, width: n, height: n, parameters: p, precision: precision)
    func ref(_ key: String) -> [Double] { HRTEMPins.doubles(c.files[key]!) }
    func radius(_ i: Int) -> Double { let dr = Double(i / n - half), dc = Double(i % n - half); return (dr * dr + dc * dc).squareRoot() }

    // Outputs.
    for (key, got) in [("absf", result.absf.doubles), ("wiener", result.wiener.doubles)] {
        let e = HRTEMPins.relativeToMaximum(got, ref(key))
        #expect(e <= outputBand[precision]!, "\(key): \(e) of the maximum")
    }
    // Stages, W-weighted over r > 0 (undefined where W off the DC is roundoff: the constant image).
    let refW = ref("W")
    let offDC = (0..<count).filter { radius($0) > 0 }
    let sumW = offDC.reduce(0.0) { $0 + refW[$1] }
    if sumW >= 1e-9 * refW[half * n + half] {
        for (key, got) in [("W", result.spectrum), ("M", result.background), ("a", result.absfMask), ("w", result.wienerMask)] {
            let r = ref(key)
            var num = 0.0, den = 0.0
            for i in offDC { num += abs(got[i] - r[i]) * refW[i]; den += abs(r[i]) * refW[i] }
            let e = den > 0 ? num / den : 0
            #expect(e <= 1e-4, "\(key): W-weighted \(e)")
        }
    }
    // The DC: the ABSF output's mean over the input's against the float64 mask at the DC.
    let meanIn = input.reduce(0.0) { $0 + Double($1) } / Double(count)
    let ratio = result.absf.doubles.reduce(0, +) / Double(count) / meanIn
    #expect(abs(ratio - ref("a")[half * n + half]) <= 1e-6, "ABSF mean ratio \(ratio)")
    // Flags, per bin: at the first pass where any bin differs, each differing bin's float64 margin is within δ.
    let refC1 = ref("c1"), refS = ref("S")
    var deltaC = 0.0
    for i in 0..<count where !result.firstPassC.isEmpty && radius(i) < Double(half) && refS[i] > 0 {
        deltaC = max(deltaC, abs(result.firstPassC[i] - refC1[i]))
    }
    let tolerance = precision == .float32 ? 3 * deltaC : max(3 * deltaC, 1e-9)
    let refFirst = ref("firstpass").map { Int($0) }
    let near = ref("nearmargins")
    var margin: [Int: [Int: Double]] = [:]
    for k in stride(from: 0, to: near.count, by: 3) { margin[Int(near[k]), default: [:]][Int(near[k + 1])] = near[k + 2] }
    var anyDiffer = false
    for j in 1...max(1, min(result.passes, c.passes)) where result.passes > 0 && c.passes > 0 {
        let differing = (0..<count).filter { (1...j).contains(result.firstFlaggedPass[$0]) != (1...j).contains(refFirst[$0]) }
        guard !differing.isEmpty else { continue }
        anyDiffer = true
        let unexplained = differing.filter { (margin[j]?[$0] ?? 5) > tolerance }
        #expect(unexplained.isEmpty, "pass \(j): \(differing.count) bins differ, \(unexplained.count) beyond δ = \(tolerance)")
        break
    }
    if precision == .float64 && !anyDiffer { #expect(result.passes == c.passes, "passes \(result.passes) vs \(c.passes)") }
    // The result's other fields: pass 1's c, the median and, where present, M0 against
    // the float64 reading; the flags against their own first pass; the counts and the final d against the flags.
    let disk = (0..<count).filter { radius($0) < Double(half) && refS[$0] > 0 }
    if precision == .float64 {
        #expect(disk.allSatisfy { abs(result.firstPassC[$0] - refC1[$0]) <= 1e-8 }, "pass 1's c beyond 1e-8 points")
        #expect(HRTEMPins.relativeToMaximum(result.median, refS) <= 1e-10)
        if let m0 = c.files["M0"] { #expect(HRTEMPins.relativeToMaximum(result.initialBackground, HRTEMPins.doubles(m0)) <= 1e-10) }
        #expect(result.firstFlaggedPass == refFirst, "flags differ from the float64 reading")
    }
    #expect((0..<count).allSatisfy { result.flagged[$0] == (result.firstFlaggedPass[$0] > 0) })
    let flaggedCount = result.flagged.filter { $0 }.count
    #expect(result.flagsPerPass.count == result.passes && result.flagsPerPass.last == flaggedCount)
    #expect(result.finalDelta == 100 * Double(flaggedCount) / Double(count))
    // Truth (P2): at the defaults on the noisy tiers, each output's energy outside the lattice bins is below the input's.
    if noisyTiers.contains(c.tier), c.params == defaults, let regionFile = c.files["p2region"] {
        let region = HRTEMPins.doubles(regionFile)
        func energy(_ x: [Double]) throws -> Double {
            let fft = try #require(FFT2DDouble(nx: n, ny: n))
            var re = x, im = [Double](repeating: 0, count: count)
            fft.transform(re: &re, im: &im, forward: true)
            return (0..<count).filter { region[$0] != 0 }.reduce(0.0) { $0 + re[$1] * re[$1] + im[$1] * im[$1] }
        }
        let eIn = try energy(input.map(Double.init))
        #expect(try energy(result.absf.doubles) < eIn && energy(result.wiener.doubles) < eIn)
    }
}

@Test func stepClosedForms() {
    #expect(abs(HRTEMFilter.edgeWindow(radius: 0.4 * 128, n: 128) - 1 / 1.414) < 1e-15)
    #expect(HRTEMFilter.edgeWindow(radius: 0, n: 128) == 1)
    #expect(abs(HRTEMFilter.butterworth(radius: 0.3 * 64, ro: 0.3, order: 4, n: 128) - 1 / 1.414) < 1e-15)
    #expect(abs(HRTEMFilter.butterworth(radius: 2 * 0.3 * 64, ro: 0.3, order: 1, n: 128) - 1 / (1 + 0.414 * 4)) < 1e-15)
    #expect(abs(HRTEMFilter.lowFrequencyTaper(radius: 0, taperRadius: 6) - 0.15) < 1e-16)
    #expect(abs(HRTEMFilter.lowFrequencyTaper(radius: 3, taperRadius: 6) - 0.1) < 1e-16)
}

@Test func rotationalAverageOfARadialFunction() {
    // A field that depends on r only is its own rotational average (to the bilinear sampling), and the profile's last
    // value holds beyond ρ = n/2 − 1.
    let n = 64, half = 32
    let x = (0..<(n * n)).map { i -> Double in
        let dr = Double(i / n - half), dc = Double(i % n - half); return (dr * dr + dc * dc).squareRoot()
    }
    let p = HRTEMFilter.radialProfile(x, n: n)
    #expect(p.count == half && p[0] == 0)
    // Bilinear sampling of the cone r overestimates it, most near the centre (the interpolant is the chord): measured
    // 0.087 at ρ = 1, 0.043 at 2, 0.021 at 4, falling as about 0.087/ρ; the bound is 0.1/ρ.
    #expect(p.indices.allSatisfy { abs(p[$0] - Double($0)) <= 0.1 / Double(max($0, 1)) }, "\(p.prefix(6))")
    let rot = HRTEMFilter.rotationalAverage(x, n: n)
    #expect(rot[0] == p[half - 1])
}

@Test func refusalsByName() throws {
    #expect(throws: HRTEMFilterError.notSquare(width: 32, height: 34)) {
        try HRTEMFilter.filter(pixels: [Float](repeating: 0, count: 32 * 34), width: 32, height: 34)
    }
    #expect(throws: HRTEMFilterError.oddSize(33)) { try HRTEMFilter.filter(pixels: [Float](repeating: 0, count: 33 * 33), width: 33, height: 33) }
    #expect(throws: HRTEMFilterError.tooSmall(30)) { try HRTEMFilter.filter(pixels: [Float](repeating: 0, count: 900), width: 30, height: 30) }
    #expect(throws: HRTEMFilterError.pixelCount(expected: 1024, actual: 3)) { try HRTEMFilter.filter(pixels: [0, 0, 0], width: 32, height: 32) }
    #expect(throws: HRTEMFilterError.maskCount(expected: 1024, actual: 5)) {
        try HRTEMFilter.applyMask([Double](repeating: 1, count: 5), to: [Float](repeating: 1, count: 1024), n: 32)
    }
    var nan = [Float](repeating: 1, count: 1024); nan[5] = .nan
    #expect(throws: HRTEMFilterError.nonFinitePixel) { try HRTEMFilter.filter(pixels: nan, width: 32, height: 32) }
    for (name, edit) in [("delta", { (p: inout HRTEMFilter.Parameters) in p.delta = 51 }),
                         ("step", { $0.step = 0.5 }), ("cycles", { $0.cycles = 100 }), ("order", { $0.order = 0 }),
                         ("ro", { $0.ro = 3.5 }), ("lowFrequencyPercent", { $0.lowFrequencyPercent = 21 })] {
        var p = HRTEMFilter.Parameters(); edit(&p)
        #expect(throws: HRTEMFilterError.parameterOutOfRange(name)) {
            try HRTEMFilter.filter(pixels: [Float](repeating: 1, count: 1024), width: 32, height: 32, parameters: p)
        }
    }
}

@Test func outputsKeepTheCalibration() throws {
    let n = 32
    let image = try Micrograph(width: n, height: n, pixels: (0..<(n * n)).map { Float($0 % 5) },
                               x: AxisCalibration(scale: 0.02, unit: "nm"), y: AxisCalibration(scale: 0.02, unit: "nm"))
    for precision in Precision.allCases {
        let r = try HRTEMFilter.filter(image, precision: precision)
        let (a, w) = try HRTEMFilter.outputs(r, like: image)
        #expect(a.x == image.x && w.y == image.y && a.width == n && r.precision == precision && r.parameters == .defaults)
    }
}

// Covers the cycle cap, the taper radius's rounding and the flag disk's edge, which the two
// committed cases cannot reach (they stop by delta at 36 passes, R = 3.2, and no flag reaches r = n/2).
private func committedBin128() throws -> (pixels: [Float], n: Int) {
    let c = try #require(HRTEMPins.cases.first { $0.name == "bin-128" })
    return (HRTEMPins.floats(c.input), c.n)
}

@Test func theCycleCapStopsTheLoop() throws {
    let (pixels, n) = try committedBin128()
    var p = HRTEMFilter.Parameters(); p.cycles = 3; p.delta = 50
    let r = try HRTEMFilter.filter(pixels: pixels, width: n, height: n, parameters: p)
    #expect(r.passes == 3 && r.flagsPerPass.count == 3)
}

@Test func theTaperRadiusRoundsHalfAwayFromZero() throws {
    let (pixels, n) = try committedBin128()
    var p = HRTEMFilter.Parameters(); p.lowFrequencyPercent = 20; p.smoothing = false   // R = round(64 · 0.2) = round(12.8) = 13
    let r = try HRTEMFilter.filter(pixels: pixels, width: n, height: n, parameters: p)
    let at12 = (n / 2) * n + n / 2 + 12                                                   // r = 12 < 13: tapered
    #expect(abs(r.absfMask[at12] - HRTEMFilter.lowFrequencyTaper(radius: 12, taperRadius: 13)) < 1e-15)
}

@Test func nothingIsFlaggedOnTheDiskEdge() throws {
    let (pixels, n) = try committedBin128()
    var p = HRTEMFilter.Parameters(); p.delta = 50; p.step = 10                             // flag far out, fast
    let r = try HRTEMFilter.filter(pixels: pixels, width: n, height: n, parameters: p)
    let half = n / 2
    let onEdge = [half, half * n]                                                         // (0, n/2) and (n/2, 0): r = n/2
    for i in onEdge { #expect(r.firstPassC[i] == 0 && !r.flagged[i], "pixel \(i) on r = n/2") }
    #expect(r.flagged.contains(true))
}
