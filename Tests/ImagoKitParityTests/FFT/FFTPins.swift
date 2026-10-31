// The FFT, spot and angle pins: the committed subset in Fixtures/fft-spot-angle-pins.json (NumPy 2.5.4 on planted lattices,
// frozen: the generator read macTEM's parity harnesses from a commit whose history was squashed, so it was retired), or,
// when IMAGOKIT_PINS_FULL names a folder holding the full case set, every case, read from the harness manifests there.
import Foundation
import ImagoKitTestSupport
import ImagoCore

struct CropPin: Decodable, Sendable { let row, col, height, width: Int }

struct SpotPinMask: Decodable, Sendable { let kx, ky, r, e: Double }

struct FFTPin: Decodable, Sendable, CustomStringConvertible {
    let name, kind, input, units: String
    let width, height: Int
    let scale_x, scale_y: Double
    let crop: CropPin?
    let power, axis_x, axis_y, axis_units, pmean, pcount, cropped_input: String?
    let bin_width: Double?
    let bin_width_given: Double?
    let ring_bin: Int?
    let peaks: [[Int]]?
    let ifft: String?
    let spots: [SpotPinMask]?
    let imag_max, real_max: Double?
    var description: String { name }
}

struct SpotPin: Decodable, Sendable, CustomStringConvertible {
    let name, input, units, kind, tier: String
    let width, height, click_row, click_col, snapped_row, snapped_col: Int
    let scale_x, scale_y, d0: Double
    let centroid_row, centroid_col, d: Double?
    let neighbour: Bool
    let error: String?
    var description: String { name }
}

struct AngleSpotPin: Decodable, Sendable {
    let planted_kx, planted_ky: Double
    let click_row, click_col, snapped_row, snapped_col: Int
    let centroid_row, centroid_col, qx, qy, d: Double
}

struct AnglePin: Decodable, Sendable, CustomStringConvertible {
    let name, input, units, tier: String
    let width, height: Int
    let scale_x, scale_y, theta0, theta_numpy, ratio0, ratio_numpy: Double
    let spots: [AngleSpotPin]
    var description: String { name }
}

enum FFTPins {
    static let fixtures = FixtureDirs.fixtures(of: "ImagoFFTTests")
    static let repository = FixtureDirs.repository
    static let full: URL? = ProcessInfo.processInfo.environment["IMAGOKIT_PINS_FULL"].map { URL(fileURLWithPath: $0) }
        .flatMap { FileManager.default.fileExists(atPath: $0.appendingPathComponent("fft/ref/manifest.json").path) ? $0 : nil }

    private struct Committed: Decodable { let fft: [FFTPin]; let spot: [SpotPin]; let angle: [AnglePin] }
    private struct Harness<C: Decodable>: Decodable { let cases: [C] }

    private static let committed: Committed = {
        let data = try! Data(contentsOf: fixtures.appendingPathComponent("fft-spot-angle-pins.json"))
        return try! JSONDecoder().decode(Committed.self, from: data)
    }()

    private static func harness<C: Decodable>(_ path: String) -> [C] {
        let data = try! Data(contentsOf: full!.appendingPathComponent(path))
        return try! JSONDecoder().decode(Harness<C>.self, from: data).cases
    }

    static let fft: [FFTPin] = full == nil ? committed.fft
        : (harness("fft/ref/manifest.json") as [FFTPin]).filter { ["spectrum", "profile", "ifft"].contains($0.kind) }
    static let spot: [SpotPin] = full == nil ? committed.spot : harness("spot/ref/manifest.json")
    static let angle: [AnglePin] = full == nil ? committed.angle : harness("angle/manifest.json")

    /// A file a pin names: absolute; or in the committed fixtures' section folder; or in the full set's section folders;
    /// or relative to the repository (the spot harness writes paths relative to it).
    static func url(_ name: String, section: String) -> URL {
        if name.hasPrefix("/") { return URL(fileURLWithPath: name) }
        var candidates = [fixtures.appendingPathComponent(section).appendingPathComponent(name)]
        if let full {
            candidates += [full.appendingPathComponent(section).appendingPathComponent(name),
                           full.appendingPathComponent(section).appendingPathComponent("ref").appendingPathComponent(name)]
        }
        candidates.append(repository.appendingPathComponent(name))
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) } ?? candidates[0]
    }

    static func floats(_ name: String, section: String) -> [Float] {
        let data = try! Data(contentsOf: url(name, section: section))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func doubles(_ name: String, section: String) -> [Double] {
        let data = try! Data(contentsOf: url(name, section: section))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
    }

    static func ints(_ name: String, section: String) -> [Int] {
        let data = try! Data(contentsOf: url(name, section: section))
        return data.withUnsafeBytes { $0.bindMemory(to: Int64.self).map(Int.init) }
    }

    static func image(_ pixels: [Float], width: Int, height: Int, scaleX: Double, scaleY: Double, units: String) -> Micrograph {
        try! Micrograph(width: width, height: height, pixels: pixels,
                        x: AxisCalibration(scale: scaleX, unit: units), y: AxisCalibration(scale: scaleY, unit: units))
    }

    /// max |got − want| / max |want|.
    static func relativeToMaximum(_ got: [Double], _ want: [Double]) -> Double {
        precondition(got.count == want.count)
        let top = want.map(abs).max() ?? 0
        let worst = zip(got, want).map { abs($0 - $1) }.max() ?? 0
        return top > 0 ? worst / top : worst
    }
}
