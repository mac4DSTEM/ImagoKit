// The pins of tools/resampling/pins.py (suite task 20, K4a): SciPy 1.18.1 ndimage.zoom and ndimage.rotate on float32 images.
import Foundation
import ImagoKitTestSupport

struct ResamplingPins: Decodable, Sendable {
    struct Case: Decodable, Sendable, CustomStringConvertible {
        let kind: String
        let image: String
        let height, width: Int
        let pixels: [Float]
        let order: Int
        let mode: String
        let cval: Double
        let zoomRows, zoomColumns: Double?
        let gridMode: Bool?
        let angle: Double?
        let reshape: Bool?
        let outHeight, outWidth: Int
        let result: [Float]
        var description: String {
            kind == "zoom" ? "zoom \(image) order \(order) \(mode) x(\(zoomRows!), \(zoomColumns!)) grid \(gridMode!)"
                : "rotate \(image) order \(order) \(mode) \(angle!) reshape \(reshape!)"
        }
    }
    struct Meta: Decodable, Sendable { let numpy, scipy: String }
    let meta: Meta
    let cases: [Case]

    static let all: ResamplingPins = {
        let url = FixtureDirs.fixtures(of: "ImagoAlignmentTests").appendingPathComponent("resampling-pins.json")
        return try! JSONDecoder().decode(ResamplingPins.self, from: Data(contentsOf: url))
    }()
}
