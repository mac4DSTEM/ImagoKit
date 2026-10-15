// The slice-2 pins (tools/slice-2/pins.py; registration-slice-2.md sections 2 to 4): the committed subset in
// Fixtures/slice-2-pins.json, or, when IMAGOKIT_PINS_FULL names a folder (tools/slice-2/full-parity.sh sets it to
// .build/pins-full), every case, read from alignment/slice-2-pins.json there.
import Foundation
import ImagoAlignment
import ImagoCore

struct OraclePin: Decodable, Sendable {
    let shift: [Double]
    let error, phasediff: Double
    let dtype: String
}

struct RegisterPin: Decodable, Sendable, CustomStringConvertible {
    let name, normalization, ref, mov: String
    let height, width, u: Int
    let sigma: Double?
    let truth: [Double]?
    let truth_band: Double?
    let truth_error64: Double?
    let f64, f32: OraclePin
    let gap_integer: Double
    let gap_refined: Double?
    let tie64, tie32, committed: Bool
    /// The measured noise of the correlation arrays (relative to their maxima): integer stage, refinement stage.
    let eps64, eps32: [Double]
    /// For a near-tie case, the bounding box [row min, row max, column min, column max] of the shifts indistinguishable
    /// from the oracle's at that noise.
    let box64, box32: [Double]?
    var description: String { name }
    var norm: Normalization { normalization == "phase" ? .phase : .unnormalized }
}

struct ShiftImagePin: Decodable, Sendable { let name, file: String; let height, width: Int }
struct ShiftComboPin: Decodable, Sendable, CustomStringConvertible {
    let mode, file: String
    let order: Int
    var description: String { "\(mode)-o\(order)" }
}
struct ShiftPins: Decodable, Sendable { let cval: Double; let shifts: [[Double]]; let images: [ShiftImagePin]; let combos: [ShiftComboPin] }

struct FourierImagePin: Decodable, Sendable, CustomStringConvertible {
    let name, file, out: String
    let height, width: Int
    let imag_max: [Double]
    var description: String { name }
}
struct FourierPins: Decodable, Sendable { let shifts: [[Double]]; let images: [FourierImagePin] }

struct DriftPin: Decodable, Sendable, CustomStringConvertible {
    let name, noisy, clean, sum_fourier, sum_bilinear, sum_count: String
    let size, frames, u: Int
    let committed: Bool
    let truth, sum_shifts: [[Double]]
    let truth_band: [String: [Double]]
    let oracle: [String: [OraclePair?]]
    let fourier_mean_vs_image, bilinear_correlation: Double
    let full_overlap_pixels: Int
    var description: String { name }
}

struct OraclePair: Decodable, Sendable {
    let shift: [Double]
    let error, phasediff: Double
    let pair: [Double]?
    let tie64, tie32: Bool
    let box64, box32: [Double]?
    let eps64, eps32: [Double]
}

enum Pins {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    static let full: URL? = ProcessInfo.processInfo.environment["IMAGOKIT_PINS_FULL"].map {
        URL(fileURLWithPath: $0).appendingPathComponent("alignment")
    }
    static var folder: URL { full ?? fixtures }

    private struct Manifest: Decodable {
        let register: [RegisterPin]
        let shift: ShiftPins
        let fourier: FourierPins
        let drift: [DriftPin]
    }

    private static let manifest: Manifest = {
        let data = try! Data(contentsOf: folder.appendingPathComponent("slice-2-pins.json"))
        return try! JSONDecoder().decode(Manifest.self, from: data)
    }()

    /// Every registration case (the full set when IMAGOKIT_PINS_FULL is set, else the committed ones).
    static let register: [RegisterPin] = manifest.register
    static let shift: ShiftPins = manifest.shift
    static let fourier: FourierPins = manifest.fourier
    static let drift: [DriftPin] = manifest.drift

    static func floats(_ name: String) -> [Float] {
        let data = try! Data(contentsOf: folder.appendingPathComponent(name))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func doubles(_ name: String) -> [Double] {
        let data = try! Data(contentsOf: folder.appendingPathComponent(name))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
    }

    static func int32s(_ name: String) -> [Int] {
        let data = try! Data(contentsOf: folder.appendingPathComponent(name))
        return data.withUnsafeBytes { $0.bindMemory(to: Int32.self).map(Int.init) }
    }

    static func image(_ pixels: [Float], width: Int, height: Int) -> Micrograph {
        try! Micrograph(width: width, height: height, pixels: pixels)
    }
}
