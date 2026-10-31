// skimage.restoration.denoise_tv_chambolle / denoise_tv_bregman / denoise_nl_means on channel-last multichannel images
// (channel_axis=-1) and on integer images (img_as_float, or preserve_range for non-local means), against the pinned runs of
// tools/denoise-channels/pins.py (scikit-image 0.26.0, NumPy 2.5.4, SciPy 1.18.1).
// Tolerance: scikit-image's wheel ships no tests, so the functions are held to assert_allclose's dtype defaults:
// rtol 1e-7 for Float64 (every integer case returns float64) and 1e-5 for Float32, atol 0.
import Foundation
import ImagoFilters
import ImagoKitTestSupport
import Testing

struct DenoiseChannelPinFile: Decodable, Sendable {
    struct Image: Decodable, Sendable {
        let id: Int; let name: String; let h, w, channels: Int; let dtype: String
        let x: [Sample]
    }
    /// A float sample, or an integer sample written as a decimal string (exact for the 64-bit types).
    enum Sample: Decodable, Sendable {
        case float(Double), integer(String)
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) { self = .integer(s) } else { self = .float(try c.decode(Double.self)) }
        }
        var double: Double { if case .float(let v) = self { return v }; return .nan }
        var text: String { if case .integer(let s) = self { return s }; return "" }
    }
    struct Params: Decodable, Sendable {
        let weight, eps: Double?; let max_num_iter: Int?; let isotropic: Bool?
        let patch_size, patch_distance: Int?; let h: Double?; let fast_mode: Bool?; let sigma: Double?; let preserve_range: Bool?
    }
    struct Case: Decodable, Sendable {
        let kind: String; let image: Int; let dtype: String; let params: Params; let y: [Double]
    }
    let images: [Image]
    let cases: [Case]

    static let shared: DenoiseChannelPinFile = {
        let url = FixtureDirs.fixtures(of: "ImagoFiltersTests").appendingPathComponent("denoise-channel-pins.json")
        return try! JSONDecoder().decode(DenoiseChannelPinFile.self, from: Data(contentsOf: url))
    }()
}

struct DenoiseChannelPin: Sendable, CustomStringConvertible {
    let c: DenoiseChannelPinFile.Case
    let image: DenoiseChannelPinFile.Image
    var description: String { "\(c.kind) \(image.name) \(image.dtype) \(c.params)" }

    static func all(_ kind: String) -> [DenoiseChannelPin] {
        let file = DenoiseChannelPinFile.shared
        return file.cases.filter { $0.kind == kind }.map { DenoiseChannelPin(c: $0, image: file.images[$0.image]) }
    }

    private func ints<I: FixedWidthInteger>(_: I.Type) -> [I] { image.x.map { I($0.text)! } }

    /// Runs the pinned call on the pinned image in its own type; the result widened to Double.
    func run() throws -> [Double] {
        let (r, w, ch, p) = (image.h, image.w, image.channels, c.params)
        func float<S: BinaryFloatingPoint & Sendable>(_ x: [S]) throws -> [Double] {
            switch c.kind {
            case "chambolle":
                return try Denoise.totalVariationChambolle(x, rows: r, columns: w, channels: ch, weight: p.weight!, eps: p.eps!,
                                                           maxNumIter: p.max_num_iter!).map { Double($0) }
            case "bregman":
                return try Denoise.totalVariationBregman(x, rows: r, columns: w, channels: ch, weight: p.weight!,
                                                         maxNumIter: p.max_num_iter!, eps: p.eps!, isotropic: p.isotropic!).map { Double($0) }
            default:
                return try Denoise.nonLocalMeans(x, rows: r, columns: w, channels: ch, patchSize: p.patch_size!,
                                                 patchDistance: p.patch_distance!, h: p.h!, fastMode: p.fast_mode!,
                                                 sigma: p.sigma!).map { Double($0) }
            }
        }
        func integer<I: FixedWidthInteger>(_ x: [I]) throws -> [Double] {
            switch c.kind {
            case "chambolle":
                return try Denoise.totalVariationChambolle(x, rows: r, columns: w, channels: ch, weight: p.weight!, eps: p.eps!,
                                                           maxNumIter: p.max_num_iter!)
            case "bregman":
                return try Denoise.totalVariationBregman(x, rows: r, columns: w, channels: ch, weight: p.weight!,
                                                         maxNumIter: p.max_num_iter!, eps: p.eps!, isotropic: p.isotropic!)
            default:
                return try Denoise.nonLocalMeans(x, rows: r, columns: w, channels: ch, patchSize: p.patch_size!,
                                                 patchDistance: p.patch_distance!, h: p.h!, fastMode: p.fast_mode!, sigma: p.sigma!,
                                                 preserveRange: p.preserve_range!)
            }
        }
        switch image.dtype {
        case "float32": return try float(image.x.map { Float($0.double) })
        case "float64": return try float(image.x.map(\.double))
        case "uint8": return try integer(ints(UInt8.self))
        case "uint16": return try integer(ints(UInt16.self))
        case "uint32": return try integer(ints(UInt32.self))
        case "uint64": return try integer(ints(UInt64.self))
        case "int8": return try integer(ints(Int8.self))
        case "int16": return try integer(ints(Int16.self))
        case "int32": return try integer(ints(Int32.self))
        case "int64": return try integer(ints(Int64.self))
        default: Issue.record("unknown dtype \(image.dtype)"); return []
        }
    }
}

private func check(_ got: [Double], _ pin: DenoiseChannelPin) {
    let rtol = pin.c.dtype == "float32" ? 1e-5 : 1e-7
    #expect(got.count == pin.c.y.count, "\(pin)")
    guard got.count == pin.c.y.count else { return }
    var worst: (rel: Double, at: Int)?
    for k in 0..<got.count {
        let diff = abs(got[k] - pin.c.y[k])
        guard !(diff <= rtol * abs(pin.c.y[k])) else { continue }
        let rel = pin.c.y[k] == 0 ? Double.infinity : diff / abs(pin.c.y[k])
        if worst == nil || !(rel <= worst!.rel) { worst = (rel, k) }
    }
    #expect(worst == nil, "\(pin): worst relative difference \(worst?.rel ?? 0) at \(worst?.at ?? 0)")
}

/// The fixture holds every case its generator writes, so no parameterised test below runs empty.
@Test func denoiseChannelPinCounts() {
    #expect(DenoiseChannelPinFile.shared.cases.count == 268)
    #expect(DenoiseChannelPin.all("chambolle").count == 52)
    #expect(DenoiseChannelPin.all("bregman").count == 48)
    #expect(DenoiseChannelPin.all("nlmeans").count == 168)
}

@Test(arguments: DenoiseChannelPin.all("chambolle"))
func chambolleChannelsAndIntegersMatchSciKitImage(_ pin: DenoiseChannelPin) throws { check(try pin.run(), pin) }

@Test(arguments: DenoiseChannelPin.all("bregman"))
func bregmanChannelsAndIntegersMatchSciKitImage(_ pin: DenoiseChannelPin) throws { check(try pin.run(), pin) }

@Test(arguments: DenoiseChannelPin.all("nlmeans"))
func nonLocalMeansChannelsAndIntegersMatchSciKitImage(_ pin: DenoiseChannelPin) throws { check(try pin.run(), pin) }

/// Records the largest differences per group (visible with `swift test --filter denoiseChannelAgreementReport`); never fails.
@Test func denoiseChannelAgreementReport() throws {
    var lines = [String]()
    for kind in ["chambolle", "bregman", "nlmeans"] {
        for dtype in ["float64", "float32"] {
            var maxRel = 0.0, count = 0
            for pin in DenoiseChannelPin.all(kind) where pin.c.dtype == dtype {
                let got = try pin.run()
                count += 1
                for k in 0..<got.count where pin.c.y[k] != 0 { maxRel = max(maxRel, abs(got[k] - pin.c.y[k]) / abs(pin.c.y[k])) }
            }
            lines.append("\(kind) \(dtype): \(count) pins, max relative \(maxRel)")
        }
    }
    print(lines.joined(separator: "\n"))
}
