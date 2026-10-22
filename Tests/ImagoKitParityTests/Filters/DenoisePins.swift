// The pinned runs of tools/denoise/pins.py: skimage.restoration.denoise_tv_chambolle / denoise_tv_bregman /
// denoise_nl_means (scikit-image 0.26.0, NumPy 2.5.4, SciPy 1.18.1) on float32 and float64 images.
import Foundation
import ImagoKitTestSupport

struct DenoisePinFile: Decodable, Sendable {
    struct Image: Decodable, Sendable {
        let id: Int; let name: String; let h, w: Int; let dtype: String; let x: [Double]
    }
    struct Params: Decodable, Sendable {
        let weight, eps: Double?; let max_num_iter: Int?; let isotropic: Bool?
        let patch_size, patch_distance: Int?; let h: Double?; let fast_mode: Bool?; let sigma: Double?
    }
    struct Case: Decodable, Sendable, CustomStringConvertible {
        let kind: String; let image: Int; let dtype: String; let params: Params; let y: [Double]
        var description: String { "\(kind) image\(image) \(dtype) \(params)" }
    }
    let images: [Image]
    let cases: [Case]

    static let shared: DenoisePinFile = {
        let url = FixtureDirs.fixtures(of: "ImagoFiltersTests").appendingPathComponent("denoise-pins.json")
        return try! JSONDecoder().decode(DenoisePinFile.self, from: Data(contentsOf: url))
    }()
}

/// One pinned call with its image resolved.
struct DenoisePin: Sendable, CustomStringConvertible {
    let c: DenoisePinFile.Case
    let image: DenoisePinFile.Image
    var description: String { "\(c.kind) \(image.name) \(c.dtype) \(c.params)" }

    static func all(_ kind: String) -> [DenoisePin] {
        let file = DenoisePinFile.shared
        return file.cases.filter { $0.kind == kind }.map { DenoisePin(c: $0, image: file.images[$0.image]) }
    }
}
