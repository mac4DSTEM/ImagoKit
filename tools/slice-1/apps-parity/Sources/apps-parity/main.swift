// apps-parity: macTEM 3d17311's Float32 originals (MacTEMRef) vs ImagoKit's Float32 path, bit for bit as the D12 quick check, on every planted
// case of the full set. READ-ONLY on the planted set. Usage: apps-parity [pins-full dir]
import Foundation
import ImagoCore
import ImagoFFT
import ImagoFilters
import MacTEMRef

typealias RefImage = MacTEMRef.CalibratedImage
typealias RefSpectrum = MacTEMRef.PowerSpectrum
typealias RefProfile = MacTEMRef.RadialProfile
typealias RefSpot = MacTEMRef.SpotMeasure
typealias KitSpectrum = ImagoFFT.PowerSpectrum
typealias KitProfile = ImagoFFT.RadialProfile
typealias KitSpot = ImagoFFT.SpotMeasurement

let pins = CommandLine.arguments.count > 1 ? CommandLine.arguments[1]
    : "../../../.build/pins-full"
let kitRoot = URL(fileURLWithPath: pins).standardizedFileURL.deletingLastPathComponent().deletingLastPathComponent().path   // <kit>/.build/pins-full

struct Tally {
    var name: String
    var cases = 0, compared = 0, identical = 0
    var diffs: [String] = []
    var stringDiffs: [String] = []
    var stringsCompared = 0
    mutating func f32(_ cs: String, _ field: String, _ a: [Float], _ b: [Float]) {
        if a.count != b.count { compared += 1; diffs.append("\(cs) \(field) count \(a.count) vs \(b.count)"); return }
        for i in 0..<a.count {
            compared += 1
            if a[i].bitPattern == b[i].bitPattern { identical += 1 }
            else { diffs.append("\(cs) \(field)[\(i)] ref \(a[i]) (\(String(a[i].bitPattern, radix: 16))) kit \(b[i]) (\(String(b[i].bitPattern, radix: 16)))") }
        }
    }
    mutating func f64(_ cs: String, _ field: String, _ a: [Double], _ b: [Double]) {
        if a.count != b.count { compared += 1; diffs.append("\(cs) \(field) count \(a.count) vs \(b.count)"); return }
        for i in 0..<a.count { f64(cs, "\(field)[\(i)]", a[i], b[i]) }
    }
    mutating func f64(_ cs: String, _ field: String, _ a: Double, _ b: Double) {
        compared += 1
        if a.bitPattern == b.bitPattern { identical += 1 }
        else { diffs.append("\(cs) \(field) ref \(a) (\(String(a.bitPattern, radix: 16))) kit \(b) (\(String(b.bitPattern, radix: 16)))") }
    }
    mutating func int(_ cs: String, _ field: String, _ a: Int, _ b: Int) {
        compared += 1
        if a == b { identical += 1 } else { diffs.append("\(cs) \(field) ref \(a) kit \(b)") }
    }
    mutating func ints(_ cs: String, _ field: String, _ a: [Int], _ b: [Int]) {
        if a.count != b.count { compared += 1; diffs.append("\(cs) \(field) count \(a.count) vs \(b.count)"); return }
        for i in 0..<a.count { int(cs, "\(field)[\(i)]", a[i], b[i]) }
    }
    mutating func bool(_ cs: String, _ field: String, _ a: Bool, _ b: Bool) {
        compared += 1
        if a == b { identical += 1 } else { diffs.append("\(cs) \(field) ref \(a) kit \(b)") }
    }
    mutating func str(_ cs: String, _ field: String, _ a: String, _ b: String) {
        stringsCompared += 1
        if a != b { stringDiffs.append("\(cs) \(field) ref \"\(a)\" kit \"\(b)\"") }
    }
    mutating func text(_ cs: String, _ field: String, _ a: String, _ b: String) {
        compared += 1
        if a == b { identical += 1 } else { diffs.append("\(cs) \(field) ref \(a) kit \(b)") }
    }
}

func readF32(_ path: String, count: Int) -> [Float] {
    guard let d = FileManager.default.contents(atPath: path), d.count == count * 4 else {
        fatalError("bad input \(path): expected \(count * 4) bytes")
    }
    return d.withUnsafeBytes { raw in
        (0..<count).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
    }
}

func json(_ path: String) -> [String: Any] {
    try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as! [String: Any]
}
func num(_ v: Any?) -> Double { (v as! NSNumber).doubleValue }
func inte(_ v: Any?) -> Int { (v as! NSNumber).intValue }

struct Both { let ref: RefImage; let kit: Micrograph }
func load(_ c: [String: Any], base: String) -> Both {
    let w = inte(c["width"]), h = inte(c["height"])
    let units = c["units"] as! String
    let sx = num(c["scale_x"]), sy = num(c["scale_y"])
    var path = c["input"] as! String
    if !path.hasPrefix("/") { path = base + "/" + path }
    let px = readF32(path, count: w * h)
    // units "" is uncalibrated: scale 1 and unit "" for both libraries.
    let (ux, uy, ax, ay) = units.isEmpty ? (1.0, 1.0, "", "") : (sx, sy, units, units)
    let ref = RefImage(width: w, height: h, pixels: px,
                       x: ImageAxis(origin: 0, scale: ux, units: ax), y: ImageAxis(origin: 0, scale: uy, units: ay),
                       fileName: "x", source: "x", voltage: nil)
    let kit = try! Micrograph(width: w, height: h, pixels: px,
                              x: AxisCalibration(scale: ux, unit: ax), y: AxisCalibration(scale: uy, unit: ay))
    return Both(ref: ref, kit: kit)
}

func cropOf(_ c: [String: Any]) -> (MacTEMRef.PixelCrop, ImagoCore.PixelCrop)? {
    guard let d = c["crop"] as? [String: Any] else { return nil }
    let r = inte(d["row"]), co = inte(d["col"]), h = inte(d["height"]), w = inte(d["width"])
    return (MacTEMRef.PixelCrop(row: r, col: co, height: h, width: w), ImagoCore.PixelCrop(row: r, col: co, height: h, width: w))
}

func compareSpectrum(_ t: inout Tally, _ cs: String, _ a: RefSpectrum, _ b: KitSpectrum) {
    t.int(cs, "width", a.width, b.width); t.int(cs, "height", a.height, b.height)
    guard case .float32(let bm) = b.magnitude, case .float32(let bp) = b.power else {
        t.diffs.append("\(cs) kit values not float32"); return
    }
    t.f32(cs, "magnitude", a.magnitude, bm)
    t.f32(cs, "power", a.power, bp)
    t.f64(cs, "axisX", a.axisX, b.axisX); t.f64(cs, "axisY", a.axisY, b.axisY)
    t.f64(cs, "spacingX", a.spacingX, b.spacingX); t.f64(cs, "spacingY", a.spacingY, b.spacingY)
    t.str(cs, "unitsX", a.unitsX, b.unitsX); t.str(cs, "unitsY", a.unitsY, b.unitsY)
    t.bool(cs, "crop-present", a.crop != nil, b.crop != nil)
    if let ac = a.crop, let bc = b.crop {
        t.bool(cs, "crop-equal", ac.row == bc.row && ac.col == bc.col && ac.height == bc.height && ac.width == bc.width, true)
    }
}

func compareProfile(_ t: inout Tally, _ cs: String, _ a: RefProfile?, _ kitResult: Result<KitProfile, Error>) {
    switch (a, kitResult) {
    case (nil, .failure): t.cases += 0; t.bool(cs, "both-refuse", true, true)
    case (nil, .success): t.bool(cs, "refusal agreement (ref refuses, kit does not)", true, false)
    case (.some, .failure(let e)): t.bool(cs, "refusal agreement (kit refuses: \(e), ref does not)", false, true)
    case (.some(let a), .success(let b)):
        t.f64(cs, "width", a.width, b.width)
        t.f64(cs, "mean", a.mean, b.mean)
        t.ints(cs, "count", a.count, b.count)
        t.str(cs, "units", a.units, b.units)
    }
}

// Every family runs twice in one process and the SECOND pass is the one reported: on this OS (macOS 27, vDSP) the first
// call of a transform size in a process can return different bits from every later call, in either library (measured
// 2026-10-15: the same compute() called kit-first or ref-first differs from its own repeat on p128x96). The first
// (cold) pass is printed as a diagnostic.
func runAll() -> (tallies: [Tally], angleSpots: Int) {
// ---- fft family: spectrum and profile ----
let fftBase = pins + "/fft/ref"
let fftManifest = json(fftBase + "/manifest.json")
var spectrumT = Tally(name: "spectrum (PowerSpectrum.compute, float32)")
var profileT = Tally(name: "profile (RadialProfile.compute on the case's spectrum)")
var profileDefT = Tally(name: "profile, default width, on every spectrum case (extra)")
for case let c as [String: Any] in fftManifest["cases"] as! [Any] {
    let kind = c["kind"] as! String, name = c["name"] as! String
    guard kind == "spectrum" || kind == "profile" else { continue }
    let img = load(c, base: kitRoot)
    let crop = cropOf(c)
    let rs = try! RefSpectrum.compute(img.ref, crop: crop?.0)
    let ks = try! KitSpectrum.compute(img.kit, crop: crop?.1, precision: .float32)
    if kind == "spectrum" {
        spectrumT.cases += 1
        compareSpectrum(&spectrumT, name, rs, ks)
        profileDefT.cases += 1
        compareProfile(&profileDefT, name, RefProfile.compute(rs), Result { try KitProfile.compute(ks) })
    } else {
        profileT.cases += 1
        // The profile cases' own spectra are compared too (counted under spectrum, as extra cases).
        spectrumT.cases += 1
        compareSpectrum(&spectrumT, name + "(profile-case spectrum)", rs, ks)
        let given = c["bin_width_given"] as? NSNumber
        let wd = given?.doubleValue
        compareProfile(&profileT, name, RefProfile.compute(rs, width: wd), Result { try KitProfile.compute(ks, width: wd) })
    }
}
// Refusal agreement probes beyond the planted set (not planted cases; reported separately): width 0 and NaN.
var probeT = Tally(name: "profile refusal probes on p128x96 (width 0, -1, NaN, 1e-12) (extra, not planted)")
do {
    let c = (fftManifest["cases"] as! [[String: Any]]).first { ($0["name"] as! String) == "p128x96" }!
    let img = load(c, base: kitRoot)
    let rs = try! RefSpectrum.compute(img.ref), ks = try! KitSpectrum.compute(img.kit, precision: .float32)
    for w in [0.0, -1.0, Double.nan, 1e-12] {
        probeT.cases += 1
        compareProfile(&probeT, "w=\(w)", RefProfile.compute(rs, width: w), Result { try KitProfile.compute(ks, width: w) })
    }
}

// ---- spot ----
func refusalName(_ e: Error) -> String {
    if let f = e as? RefSpot.Failure {
        switch f {
        case .nearDC: return "nearDC"
        case .nearEdge: return "nearEdge"
        case .crop: return "crop"
        case .mixedUnits: return "mixedUnits"
        case .fftSetup: return "fftSetup"
        case .outside: return "outside"
        }
    }
    if let f = e as? SpotMeasurementError {
        switch f {
        case .nearDC: return "nearDC"
        case .nearEdge: return "nearEdge"
        case .crop: return "crop"
        case .mixedUnits: return "mixedUnits"
        case .fftSetup: return "fftSetup"
        case .outside: return "outside"
        case .zeroFrequency: return "zeroFrequency"
        }
    }
    return "other:\(e)"
}

func compareSpot(_ t: inout Tally, _ cs: String, _ img: Both, row: Int, col: Int, crop: (MacTEMRef.PixelCrop, ImagoCore.PixelCrop)? = nil) {
    let a = Result { try RefSpot.measure(img.ref, crop: crop?.0, clickRow: row, clickCol: col) }
    let b = Result { try KitSpot.measure(img.kit, crop: crop?.1, clickRow: row, clickColumn: col, precision: .float32) }
    switch (a, b) {
    case (.failure(let ea), .failure(let eb)): t.text(cs, "refusal", refusalName(ea), refusalName(eb))
    case (.failure(let ea), .success): t.text(cs, "refusal", refusalName(ea), "(kit measured)")
    case (.success, .failure(let eb)): t.text(cs, "refusal", "(ref measured)", refusalName(eb))
    case (.success(let a), .success(let b)):
        t.int(cs, "clickedRow", a.clickedRow, b.clickedRow); t.int(cs, "clickedCol", a.clickedCol, b.clickedCol)
        t.int(cs, "snappedRow", a.snappedRow, b.snappedRow); t.int(cs, "snappedCol", a.snappedCol, b.snappedCol)
        t.f64(cs, "centroidRow", a.centroidRow, b.centroidRow); t.f64(cs, "centroidCol", a.centroidCol, b.centroidCol)
        t.f64(cs, "qx", a.qx, b.qx); t.f64(cs, "qy", a.qy, b.qy); t.f64(cs, "q", a.q, b.q); t.f64(cs, "d", a.d, b.d)
        t.bool(cs, "neighbour", a.neighbour, b.neighbour)
        t.str(cs, "units", a.units, b.units); t.str(cs, "reciprocalUnits", a.reciprocalUnits, b.reciprocalUnits)
    }
}

var spotT = Tally(name: "spot (SpotMeasure.measure, float32)")
let spotManifest = json(pins + "/spot/ref/manifest.json")
for case let c as [String: Any] in spotManifest["cases"] as! [Any] {
    spotT.cases += 1
    compareSpot(&spotT, c["name"] as! String, load(c, base: kitRoot), row: inte(c["click_row"]), col: inte(c["click_col"]))
}

var angleT = Tally(name: "angle (two spots per case, each compared as a spot)")
let angleBase = pins + "/angle"
let angleManifest = json(angleBase + "/manifest.json")
var angleSpots = 0
for case let c as [String: Any] in angleManifest["cases"] as! [Any] {
    angleT.cases += 1
    let img = load(c, base: angleBase)
    for (i, s) in (c["spots"] as! [[String: Any]]).enumerated() {
        angleSpots += 1
        compareSpot(&angleT, "\(c["name"] as! String)#\(i)", img, row: inte(s["click_row"]), col: inte(s["click_col"]))
    }
}

// ---- ifft family (slice 1b): MaskedInverseFFT, Float32, with and without the Friedel partners ----
typealias RefMasked = MacTEMRef.MaskedInverseFFT
typealias KitMasked = ImagoFFT.MaskedInverseFFT
var ifftT = Tally(name: "ifft (MaskedInverseFFT.compute, float32, Friedel partners on and off)")
var ifftSpotsT = Tally(name: "ifft spots echoed back (extra)")
for case let c as [String: Any] in fftManifest["cases"] as! [Any] {
    guard (c["kind"] as! String) == "ifft" else { continue }
    let name = c["name"] as! String
    let img = load(c, base: kitRoot)
    let sp = (c["spots"] as! [[String: Any]]).map { (kx: num($0["kx"]), ky: num($0["ky"]), r: num($0["r"]), e: num($0["e"])) }
    let refSpots = sp.map { RefMasked.Spot(kx: $0.kx, ky: $0.ky, radius: $0.r, edge: $0.e) }
    let kitSpots = sp.map { KitMasked.Spot(kx: $0.kx, ky: $0.ky, radius: $0.r, edge: $0.e) }
    for friedel in [true, false] {
        let cs = "\(name)\(friedel ? "" : " [no Friedel]")"
        ifftT.cases += 1
        let a = try! RefMasked.compute(img.ref, spots: refSpots, includeFriedelPartners: friedel)
        let b = try! KitMasked.compute(img.kit, spots: kitSpots, includeFriedelPartners: friedel, precision: .float32)
        ifftT.int(cs, "width", a.width, b.width); ifftT.int(cs, "height", a.height, b.height)
        guard case .float32(let bi) = b.image else { ifftT.diffs.append("\(cs) kit image not float32"); continue }
        ifftT.f32(cs, "image", a.image, bi)
        ifftT.f64(cs, "imaginaryMax", a.imaginaryMax, b.imaginaryMax)
        ifftSpotsT.cases += 1
        ifftSpotsT.bool(cs, "spots-count", a.spots.count == b.spots.count, true)
        for (i, (x, y)) in zip(a.spots, b.spots).enumerated() {
            ifftSpotsT.bool(cs, "spot[\(i)]", x.kx == y.kx && x.ky == y.ky && x.radius == y.radius && x.edge == y.edge, true)
        }
    }
}

// ---- hrtem family (slice 1b): HRTEMFilter.filter, Float32 ----
typealias RefHRTEM = MacTEMRef.HRTEMFilter
typealias KitHRTEM = ImagoFilters.HRTEMFilter
var hrtemT = Tally(name: "hrtem (HRTEMFilter.filter, float32)")
let hrtemBase = pins + "/hrtem/ref"
let hrtemManifest = json(hrtemBase + "/manifest.json")
for case let c as [String: Any] in hrtemManifest["cases"] as! [Any] {
    let name = c["name"] as! String, n = inte(c["n"])
    let p = c["params"] as! [String: Any]
    let px = readF32(hrtemBase + "/" + (c["input"] as! String), count: n * n)
    var rp = RefHRTEM.Parameters(), kp = KitHRTEM.Parameters()
    rp.delta = num(p["delta"]); rp.step = num(p["step"]); rp.cycles = inte(p["cycles"]); rp.smoothing = (p["smoothing"] as! NSNumber).boolValue
    rp.order = inte(p["order"]); rp.ro = num(p["ro"]); rp.lowFrequencyPercent = num(p["pct"])
    kp.delta = rp.delta; kp.step = rp.step; kp.cycles = rp.cycles; kp.smoothing = rp.smoothing
    kp.order = rp.order; kp.ro = rp.ro; kp.lowFrequencyPercent = rp.lowFrequencyPercent
    hrtemT.cases += 1
    let a = try! RefHRTEM.filter(pixels: px, width: n, height: n, parameters: rp)
    let b = try! KitHRTEM.filter(pixels: px, width: n, height: n, parameters: kp, precision: .float32)
    hrtemT.int(name, "n", a.n, b.n)
    if case .float32(let v) = b.absf { hrtemT.f32(name, "absf", a.absf, v) } else { hrtemT.diffs.append("\(name) kit absf not float32") }
    if case .float32(let v) = b.wiener { hrtemT.f32(name, "wiener", a.wiener, v) } else { hrtemT.diffs.append("\(name) kit wiener not float32") }
    hrtemT.f64(name, "initialBackground", a.initialBackground, b.initialBackground)
    hrtemT.f64(name, "median", a.median, b.median)
    hrtemT.f64(name, "firstPassC", a.firstPassC, b.firstPassC)
    hrtemT.f64(name, "background", a.background, b.background)
    hrtemT.f64(name, "spectrum", a.spectrum, b.spectrum)
    hrtemT.f64(name, "absfMask", a.absfMask, b.absfMask)
    hrtemT.f64(name, "wienerMask", a.wienerMask, b.wienerMask)
    hrtemT.int(name, "flagged.count", a.flagged.count, b.flagged.count)
    if a.flagged.count == b.flagged.count { for i in 0..<a.flagged.count { hrtemT.bool(name, "flagged[\(i)]", a.flagged[i], b.flagged[i]) } }
    hrtemT.ints(name, "firstFlaggedPass", a.firstFlaggedPass, b.firstFlaggedPass)
    hrtemT.int(name, "passes", a.passes, b.passes)
    hrtemT.ints(name, "flagsPerPass", a.flagsPerPass, b.flagsPerPass)
    hrtemT.f64(name, "finalDelta", a.finalDelta, b.finalDelta)
}

    return ([spectrumT, profileT, profileDefT, probeT, spotT, angleT, ifftT, ifftSpotsT, hrtemT], angleSpots)
}

let cold = runAll()
print("cold pass (diagnostic, first call of each size in this process): differing \(cold.tallies.reduce(0) { $0 + $1.diffs.count })")
for t in cold.tallies where !t.diffs.isEmpty { print("   cold: \(t.name): \(t.diffs.count) differing values") }
let warm = runAll()
let angleSpots = warm.angleSpots
// ---- report ----
var total = 0, totalIdentical = 0, totalDiffs = 0
for t in warm.tallies {
    print("== \(t.name)")
    print("   cases \(t.cases), values compared \(t.compared), identical \(t.identical), differing \(t.compared - t.identical)")
    print("   strings compared \(t.stringsCompared), differing \(t.stringDiffs.count)")
    for d in t.diffs { print("   DIFF \(d)") }
    for d in t.stringDiffs { print("   STRING DIFF \(d)") }
    total += t.compared; totalIdentical += t.identical; totalDiffs += t.diffs.count
}
print("angle spots measured: \(angleSpots)")
print("TOTAL values \(total), identical \(totalIdentical), differing \(totalDiffs)")
exit(totalDiffs == 0 ? 0 : 1)
