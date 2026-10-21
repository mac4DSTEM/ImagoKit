// DisplayNormalization, Thumbnail, ScaleBarMath and DisplayMapping against tools/display/pins.py (an independent NumPy /
// Python reading of each definition) and planted cases.
import Foundation
import ImagoCore
import Testing

private struct AxisPin: Decodable, Sendable {
    let scale: Double
    let unit: String
    init(from decoder: any Decoder) throws {
        var c = try decoder.unkeyedContainer()
        scale = try c.decode(Double.self)
        unit = try c.decode(String.self)
    }
    var axis: AxisCalibration { AxisCalibration(scale: scale, unit: unit) }
}

private struct PlanPin: Decodable, Sendable {
    let perPixel: Double
    let unit: String
    let value: Double
    let labelUnit, label: String
    let lengthPixels: Double
    let drawnPixels, overlayScale: Int
}

private struct DisplayCorePins: Decodable, Sendable {
    struct Norm: Decodable, Sendable {
        let kind: String
        let pixels: [UInt32]
        let lo, hi: UInt32
        let log, linear, symmetric: [UInt32]
    }
    struct Thumb: Decodable, Sendable {
        let width, height, w, h: Int
        let pixels, out: String
    }
    struct Bar: Decodable, Sendable {
        let x, y: AxisPin?
        let width, height: Int
        let plan: PlanPin?
    }
    struct Nice: Decodable, Sendable { let v: Double; let nice: Double; let format: String }
    struct Ladder: Decodable, Sendable { let v: Double; let unit: String; let value: Double; let outUnit: String }
    struct Mapping: Decodable, Sendable {
        let lut: [UInt8]
        let low, high: UInt32
        let gamma: Double
        let n: [UInt32]
        let rgb: [[UInt8]]
    }
    let numpy: String
    let normalization: [Norm]
    let thumbnails: [Thumb]
    let bars: [Bar]
    let nice: [Nice]
    let ladder: [Ladder]
    let mapping: [Mapping]

    static func load() throws -> DisplayCorePins {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/display-core-pins.json")
        return try JSONDecoder().decode(DisplayCorePins.self, from: Data(contentsOf: url))
    }
}

private func floats(_ bits: [UInt32]) -> [Float] { bits.map { Float(bitPattern: $0) } }
private func floats(base64: String) -> [Float] {
    let data = Data(base64Encoded: base64)!
    return data.withUnsafeBytes { raw in (0..<(data.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) } }
}

/// The unit spelling a display shows, as the pinned generator states it (px, then the canonical reciprocal, then the real one).
private func displayLabel(_ unit: String) -> String {
    if ["px", "pix", "[pix]", "[px]", "pixel", "pixels"].contains(LengthUnit.normalized(unit)) { return "px" }
    return LengthUnit.canonicalReciprocalUnit(unit) ?? LengthUnit.canonicalRealUnit(unit) ?? unit
}

@Suite("DisplayNormalization") struct DisplayNormalizationTests {
    @Test func matchesNumPyFloat32BitForBit() throws {
        for c in try DisplayCorePins.load().normalization {
            let p = floats(c.pixels)
            let (lo, hi) = DisplayNormalization.finiteMinMax(p)
            #expect(lo.bitPattern == c.lo && hi.bitPattern == c.hi, "\(c.kind) range")
            #expect(DisplayNormalization.normalized(p).map(\.bitPattern) == c.linear, "\(c.kind) linear")
            #expect(DisplayNormalization.normalized(p, symmetric: true).map(\.bitPattern) == c.symmetric, "\(c.kind) symmetric")
            #expect(DisplayNormalization.logContrast(p).map(\.bitPattern) == c.log, "\(c.kind) log")
        }
    }

    @Test func plantedValues() {
        #expect(DisplayNormalization.normalized([2, 4, 6]) == [0, 0.5, 1])
        #expect(DisplayNormalization.normalized([-2, 0, 2], symmetric: true) == [0, 0.5, 1])
        #expect(DisplayNormalization.normalized([1, .nan, .infinity, 3]) == [0, -1, -1, 1])
        #expect(DisplayNormalization.normalized([5, 5, 5]) == [0, 0, 0])
        #expect(DisplayNormalization.normalized([0, 0], symmetric: true) == [0.5, 0.5])
        let (lo, hi) = DisplayNormalization.finiteMinMax([.nan, .infinity])
        #expect(lo == 0 && hi == 1)
        #expect(DisplayNormalization.logContrast([0, 9, -4, .nan]) == [0, 1, 0, 0])
        #expect(DisplayNormalization.normalizedLog([0, 9, 99]) == [0, Float(log10(10.0) / log10(100.0)), 1])
    }
}

@Suite("Thumbnail") struct ThumbnailTests {
    @Test func matchesTheFloorBoxMeans() throws {
        for t in try DisplayCorePins.load().thumbnails {
            let made = try #require(Thumbnail.make(normalized: floats(base64: t.pixels), width: t.width, height: t.height))
            #expect(made.width == t.w && made.height == t.h, "\(t.width)x\(t.height)")
            #expect(made.normalized.map(\.bitPattern) == floats(base64: t.out).map(\.bitPattern), "\(t.width)x\(t.height)")
        }
    }

    @Test func smallImagesAreNotUpscaled() throws {
        let p = (0..<12).map { Float($0) / 12 }
        let t = try #require(Thumbnail.make(normalized: p, width: 4, height: 3))
        #expect(t.width == 4 && t.height == 3 && t.normalized == p)
        #expect(Thumbnail.make(normalized: p, width: 5, height: 3) == nil)
        #expect(Thumbnail.make(normalized: p, width: 4, height: 3, maxSide: 0) == nil)
        #expect(Thumbnail(width: 0, height: 3, normalized: []) == nil)
    }

    @Test func theCappedInitRefusesASideOverTheCap() {
        let p = [Float](repeating: 0.5, count: 97 * 2)
        #expect(Thumbnail(width: 97, height: 2, normalized: p, maxSide: Thumbnail.defaultMaxSide) == nil)
        #expect(Thumbnail(width: 97, height: 2, normalized: p, maxSide: 97)?.width == 97)
        #expect(Thumbnail(width: 2, height: 97, normalized: p, maxSide: 96) == nil)
        #expect(Thumbnail(width: 2, height: 3, normalized: p, maxSide: 96) == nil)       // the count is still checked
    }

    @Test func aBoxWithNoValidPixelIsTheSentinel() throws {
        var p = [Float](repeating: 0.5, count: 200 * 100)
        for y in 0..<100 { for x in 0..<20 { p[y * 200 + x] = -1 } }
        let t = try #require(Thumbnail.make(normalized: p, width: 200, height: 100, maxSide: 10))
        #expect(t.width == 10 && t.height == 5)
        #expect(t.normalized[0] == -1 && t.normalized[1] == 0.5 && t.normalized[2] == 0.5)
    }
}

@Suite("ScaleBarMath") struct ScaleBarMathTests {
    @Test func planMatchesThePinnedDefinition() throws {
        var seen = 0
        for b in try DisplayCorePins.load().bars {
            let got = ScaleBarMath.plan(x: b.x?.axis, y: b.y?.axis, width: b.width, height: b.height, unitLabel: displayLabel)
            guard let want = b.plan else { #expect(got == nil, "\(b.width)x\(b.height)"); continue }
            let p = try #require(got, "\(String(describing: b.x)) \(b.width)x\(b.height)")
            seen += 1
            #expect(p.perPixel == want.perPixel && p.unit == want.unit && p.value == want.value && p.labelUnit == want.labelUnit
                    && p.label == want.label && p.lengthPixels == want.lengthPixels && p.drawnPixels == want.drawnPixels
                    && p.overlayScale == want.overlayScale, "\(p) vs \(want)")
        }
        #expect(seen > 150)
    }

    @Test func nice125AndFormat() throws {
        let pins = try DisplayCorePins.load()
        for n in pins.nice {
            #expect(ScaleBarMath.nice125(n.v) == n.nice, "\(n.v)")
            #expect(ScaleBarMath.format(n.v) == n.format, "\(n.v)")
        }
        #expect(ScaleBarMath.nice125(.infinity) == 1 && ScaleBarMath.nice125(.nan) == 1)
        #expect(ScaleBarMath.nice125(1.4) == 1 && ScaleBarMath.nice125(1.5) == 2 && ScaleBarMath.nice125(7.5) == 10)
    }

    @Test func ladderReadsInNmMicrometreOrMillimetre() throws {
        for l in try DisplayCorePins.load().ladder {
            let got = ScaleBarMath.ladder(l.v, unit: l.unit)
            #expect(got.value == l.value && got.unit == l.outUnit, "\(l.v) \(l.unit)")
        }
        let r = ScaleBarMath.ladder(2000, unit: "nm")
        #expect(r.value == 2 && r.unit == "\u{00B5}m")
        #expect(ScaleBarMath.ladder(200, unit: "Å").unit == "Å")
    }

    @Test func overlayScaleAndSampling() {
        #expect(ScaleBarMath.overlayScale(width: 512, height: 512) == 1)
        #expect(ScaleBarMath.overlayScale(width: 4096, height: 4096) == 8)
        #expect(ScaleBarMath.overlayScale(width: 100, height: 100) == 1)
        let m = AxisCalibration(scale: 2.5e-10, unit: "m")
        let s = ScaleBarMath.sampling(x: m, y: nil, width: 1000, height: 1000)
        #expect(s.unit == "nm" && abs(s.perPixel - 0.25) < 1e-15)
        let noX = ScaleBarMath.sampling(x: nil, y: AxisCalibration(scale: 3, unit: "nm"), width: 10, height: 10)
        #expect(noX.perPixel == 1 && noX.unit == "px")           // x decides; without x there is no bar sampling
        #expect(ScaleBarMath.plan(x: AxisCalibration(scale: 1, unit: "nm"), y: nil, width: 1, height: 1) == nil)   // under half a pixel
        #expect(ScaleBarMath.plan(x: AxisCalibration(scale: 1, unit: "nm"), y: nil, width: 400, height: 400)?.label == "50 nm")
    }
}

@Suite("DisplayMapping") struct DisplayMappingTests {
    @Test func matchesThePinnedDefinition() throws {
        for m in try DisplayCorePins.load().mapping {
            let map = try #require(DisplayMapping(low: Float(bitPattern: m.low), high: Float(bitPattern: m.high), gamma: m.gamma, lut: m.lut))
            for (n, want) in zip(m.n, m.rgb) {
                let got = map.rgb(Float(bitPattern: n))
                #expect([got.0, got.1, got.2] == want, "n \(Float(bitPattern: n)) low \(Float(bitPattern: m.low)) gamma \(m.gamma)")
            }
        }
    }

    @Test func maskedAndEdgeValues() {
        let lut = (0..<1024).map { UInt8($0 / 4) }                  // texel i is grey level i
        let map = DisplayMapping(low: 0, high: 1, gamma: 1, lut: lut)!
        #expect(map.rgb(-1) == DisplayMapping.maskedGrey && map.rgb(.nan) == DisplayMapping.maskedGrey)
        #expect(map.rgb(0).0 == 0 && map.rgb(1).0 == 255)
        #expect(map.rgbBytes([0, -1]).count == 6)
        #expect(DisplayMapping(low: 0, high: 1, gamma: 1, lut: [0]) == nil)
    }
}
