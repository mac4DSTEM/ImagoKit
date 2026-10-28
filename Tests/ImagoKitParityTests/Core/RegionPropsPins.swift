// The pins of tools/precipitation/pins.py: scikit-image 0.26.0 regionprops on needle masks and SciPy 1.18.1 ndimage
// measurements over a label image.
import Foundation
import ImagoKitTestSupport

struct RegionPropsPins: Decodable, Sendable {
    struct Props: Decodable, Sendable {
        let label, area: Int
        let bbox: [Int]
        let centroid: [Double]
        let orientation, axis_major_length, axis_minor_length, feret_diameter_max, perimeter, perimeter_crofton: Double
        let eccentricity, solidity, equivalent_diameter_area, extent: Double
        let euler_number, area_filled, area_convex: Int
    }
    struct Case: Decodable, Sendable, CustomStringConvertible {
        let name: String
        let width, height: Int
        let mask: String
        let labels1, labels2: [Int]
        let count1, count2: Int
        let props1, props2: [Props]
        var description: String { name }
    }
    struct IntensityProps: Decodable, Sendable {
        let label: Int
        let intensity_mean, intensity_min, intensity_max, intensity_std: Double
        let centroid_weighted: [Double]
    }
    struct Intensity: Decodable, Sendable {
        let width, height: Int
        let image: [Double]
        let labels: [Int]
        let count: Int
        let sum, mean, std, min, max: [Double]
        let center_of_mass: [[Double]]
        let min_position, max_position: [[Int]]
        let find_objects: [[Int]]
        let regionprops: [IntensityProps]
    }
    let skimage, numpy, scipy: String
    let cases: [Case]
    let intensity: Intensity

    static let all: RegionPropsPins = {
        let url = FixtureDirs.fixtures(of: "ImagoCoreTests").appendingPathComponent("regionprops-needle-pins.json")
        return try! JSONDecoder().decode(RegionPropsPins.self, from: Data(contentsOf: url))
    }()
}
