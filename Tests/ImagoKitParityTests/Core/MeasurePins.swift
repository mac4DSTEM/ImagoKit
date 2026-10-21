// The pins of tools/measure/pins.py: numpy.histogram, skimage.measure.profile_line and closed-form measurements.
import Foundation
import ImagoKitTestSupport

struct MeasurePins: Decodable, Sendable {
    struct Versions: Decodable, Sendable { let numpy, scipy, skimage, python: String }
    struct Histogram: Decodable, Sendable, CustomStringConvertible {
        let name: String
        let bins: Int
        let pixels: [UInt32]
        let range: [UInt32]?
        let counts: [Int]?
        let edges: [UInt32]?
        let error: String?
        let min, max: UInt32?
        let finite: Int?
        let mean, meanAbs: Double?
        var description: String { name }
        var floats: [Float] { pixels.map { Float(bitPattern: $0) } }
    }
    struct Image: Decodable, Sendable { let height, width: Int; let bits: [UInt32] }
    struct Profile: Decodable, Sendable, CustomStringConvertible {
        let src, dst: [Double]
        let lw, order: Int
        let mode, reduce: String
        let cval: Double
        let f32, f64: [Double]
        var description: String { "\(src)->\(dst) w\(lw) o\(order) \(mode) \(reduce)" }
    }
    struct Unreduced: Decodable, Sendable {
        let src, dst: [Double]
        let lw, order: Int
        let mode: String
        let f32, f64: [[Double]]
    }
    struct Measure: Decodable, Sendable {
        let a, b, c: [Double]
        let sx, sy, distancePx, distancePhys, angleDegrees: Double
    }
    let versions: Versions
    let seed: Int
    let histogram: [Histogram]
    let image: Image
    let profile: [Profile]
    let unreduced: [Unreduced]
    let measure: [Measure]

    static func load() throws -> MeasurePins {
        let url = FixtureDirs.fixtures(of: "ImagoCoreTests").appendingPathComponent("measure-pins.json")
        return try JSONDecoder().decode(MeasurePins.self, from: Data(contentsOf: url))
    }
}
