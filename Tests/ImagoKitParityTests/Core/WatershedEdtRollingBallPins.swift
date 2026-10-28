// The pins of tools/segmentation-k1/pins.py: SciPy 1.18.1 distance_transform_edt, scikit-image 0.26.0 watershed and rolling_ball.
import Foundation
import ImagoKitTestSupport

struct WatershedEdtRollingBallPins: Decodable, Sendable {
    struct EDT: Decodable, Sendable, CustomStringConvertible {
        let name: String
        let width, height: Int
        let mask: String
        let sampling: [Double]?
        let distance: [Double]
        let hasBackground: Bool
        var description: String { name }
    }
    struct Watershed: Decodable, Sendable, CustomStringConvertible {
        let name: String
        let width, height: Int
        let image: [Double]
        let markers: [Int]
        let mask: String?
        let connectivity: Int
        let labels: [Int]
        var description: String { name }
    }
    struct Kernel: Decodable, Sendable {
        let height, width: Int
        let values: [Double?]
    }
    struct RollingBall: Decodable, Sendable, CustomStringConvertible {
        let name: String
        let width, height: Int
        let dtype: String
        let image: [Double]
        let radius: Double?
        let kernel: Kernel?
        let background: [Double]
        var description: String { name }
    }
    struct Meta: Decodable, Sendable { let skimage, numpy, scipy: String }
    let meta: Meta
    let edt: [EDT]
    let watershed: [Watershed]
    let rollingBall: [RollingBall]

    static let all: WatershedEdtRollingBallPins = {
        let url = FixtureDirs.fixtures(of: "ImagoCoreTests").appendingPathComponent("watershed-edt-rollingball-pins.json")
        return try! JSONDecoder().decode(WatershedEdtRollingBallPins.self, from: Data(contentsOf: url))
    }()
}
