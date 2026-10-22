// scikit-image 0.26.0 thresholds, SciPy 1.18.1 / scikit-image binary morphology, remove_small_*, clear_border against the pins of
// tools/precipitation/pins.py (suite task 32). Tolerance: the scikit-image 0.26.0 wheel ships no tests, so its numbers use
// assert_allclose dtype defaults (rtol 1e-7 Float64, 1e-5 Float32; integer images return exact integer thresholds); every mask is exact.
import Foundation
import ImagoKitTestSupport
import ImagoCore
import ImagoFilters
import Testing

struct ThresholdMorphologyPins: Decodable, Sendable {
    struct Hysteresis: Decodable, Sendable { let low, high: Double; let mask: String }
    struct Grey: Decodable, Sendable, CustomStringConvertible {
        let name, dtype: String
        let width, height: Int
        let values: [Double]
        let otsu, otsu_nbins64, li, yen, triangle, mean, isodata: Double
        let local_b15, local_b15_off, local_b15_mean: [Double]?
        let hysteresis: Hysteresis
        let otsu_mask: String
        var description: String { name }
        var pixels: ThresholdPixels {
            switch dtype {
            case "float64": .float64(values)
            case "float32": .float32(values.map { Float($0) })
            default: .integers(values.map { Int($0) })
            }
        }
        var isFloat32: Bool { dtype == "float32" }
    }
    struct NDImage: Decodable, Sendable {
        let erosion, erosion_border1, dilation, opening, closing, erosion_iter2, dilation_iter2: String
    }
    struct Mode: Decodable, Sendable { let erosion, dilation, opening, closing: String }
    struct Removal: Decodable, Sendable { let max_size, connectivity: Int; let mask: String }
    struct Border: Decodable, Sendable { let buffer0, buffer2: String }
    struct Binary: Decodable, Sendable, CustomStringConvertible {
        let name: String
        let width, height: Int
        let mask: String
        let ndimage_cross, ndimage_square: NDImage
        let fill_holes, fill_holes_square: String
        let skimage_default: Mode
        let remove_small_objects, remove_small_holes: [Removal]
        let clear_border: Border
        var description: String { name }
    }
    struct Structure: Decodable, Sendable { let rank, connectivity: Int; let structure: [Int]; let shape: [Int] }
    let skimage, numpy, scipy: String
    let grey: [Grey]
    let binary: [Binary]
    let structures: [String: Structure]

    static let all: ThresholdMorphologyPins = {
        let url = FixtureDirs.fixtures(of: "ImagoCoreTests").appendingPathComponent("threshold-morphology-pins.json")
        return try! JSONDecoder().decode(ThresholdMorphologyPins.self, from: Data(contentsOf: url))
    }()
}

func maskFrom(_ s: String) -> [Bool] { s.map { $0 == "1" } }

@Suite("Thresholds (scikit-image 0.26.0)") struct ThresholdPinTests {
    static let images = ThresholdMorphologyPins.all.grey

    @Test func versions() {
        let p = ThresholdMorphologyPins.all
        #expect(p.skimage == "0.26.0" && p.numpy == "2.5.4" && p.scipy == "1.18.1")
    }

    private func close(_ got: Double, _ expected: Double, _ g: ThresholdMorphologyPins.Grey, _ what: String) {
        if g.dtype.contains("int") { #expect(got == expected, "\(g.name) \(what): \(got) != \(expected)") }
        else {
            let rtol = g.isFloat32 ? 1e-5 : 1e-7
            #expect(abs(got - expected) <= rtol * abs(expected) + 1e-300, "\(g.name) \(what): \(got) vs \(expected)")
        }
    }

    @Test(arguments: images) func otsu(_ g: ThresholdMorphologyPins.Grey) throws {
        close(try Thresholding.otsu(g.pixels), g.otsu, g, "otsu")
        close(try Thresholding.otsu(g.pixels, bins: 64), g.otsu_nbins64, g, "otsu nbins 64")
    }

    @Test(arguments: images) func li(_ g: ThresholdMorphologyPins.Grey) throws { close(try Thresholding.li(g.pixels), g.li, g, "li") }
    @Test(arguments: images) func yen(_ g: ThresholdMorphologyPins.Grey) throws { close(try Thresholding.yen(g.pixels), g.yen, g, "yen") }
    @Test(arguments: images) func triangle(_ g: ThresholdMorphologyPins.Grey) throws {
        close(try Thresholding.triangle(g.pixels), g.triangle, g, "triangle")
    }
    @Test(arguments: images) func mean(_ g: ThresholdMorphologyPins.Grey) throws { close(try Thresholding.mean(g.pixels), g.mean, g, "mean") }
    @Test(arguments: images) func isodata(_ g: ThresholdMorphologyPins.Grey) throws {
        close(try Thresholding.isodata(g.pixels), g.isodata, g, "isodata")
    }

    @Test(arguments: images) func otsuMaskIsExact(_ g: ThresholdMorphologyPins.Grey) throws {
        let t = try Thresholding.otsu(g.pixels)
        #expect(g.values.map { $0 > t } == maskFrom(g.otsu_mask))
    }

    @Test(arguments: images) func hysteresisMaskIsExact(_ g: ThresholdMorphologyPins.Grey) throws {
        let got = try Thresholding.hysteresis(g.pixels, width: g.width, height: g.height, low: g.hysteresis.low, high: g.hysteresis.high)
        #expect(got == maskFrom(g.hysteresis.mask))
    }

    @Test(arguments: images.filter { $0.local_b15 != nil }) func localSurfaces(_ g: ThresholdMorphologyPins.Grey) throws {
        let cases: [(Thresholding.LocalMethod, Double, [Double])] = [
            (.gaussian(), 0, g.local_b15!), (.gaussian(), 0.05, g.local_b15_off!), (.mean, 0, g.local_b15_mean!)]
        for (method, offset, expected) in cases {
            let got = try Thresholding.local(g.pixels, width: g.width, height: g.height, blockSize: 15, method: method, offset: offset)
            #expect(got.count == expected.count)
            for (a, b) in zip(got, expected) { #expect(abs(a - b) <= 1e-7 * abs(b) + 1e-12, "\(g.name) \(method): \(a) vs \(b)") }
        }
    }

    @Test func refusals() throws {
        #expect(throws: ThresholdError.shape) { try Thresholding.otsu(.float64([])) }
        #expect(throws: ThresholdError.nonFinite) { try Thresholding.otsu(.float64([1, 2, .nan])) }
        #expect(throws: ThresholdError.parameter("nbins")) { try Thresholding.yen(.float64([1, 2, 3]), bins: 0) }
        #expect(throws: ThresholdError.parameter("block_size must be odd")) {
            try Thresholding.local(.float64([1, 2, 3, 4]), width: 2, height: 2, blockSize: 2)
        }
        #expect(try Thresholding.otsu(.integers([7, 7, 7])) == 7)
        // Review findings (2026-10-22): a range that overflows is refused (it trapped); bins is ignored for integers; one bin is refused
        // for a float Otsu (argmax of nothing); Li with only equal finite pixels and an inf is refused.
        #expect(throws: ThresholdError.parameter("the pixel range overflows")) { try Thresholding.otsu(.float64([-1e308, 1e308])) }
        #expect(throws: ThresholdError.parameter("the pixel range overflows")) { try Thresholding.yen(.float32([-3e38, 3e38])) }
        #expect(try Thresholding.otsu(.integers([1, 1, 9, 9]), bins: 0) == 1)
        #expect(throws: ThresholdError.parameter("nbins")) { try Thresholding.otsu(.float64([1, 2, 3]), bins: 1) }
        #expect(throws: ThresholdError.noThreshold) { try Thresholding.li(.float64([1, 1, .infinity])) }
    }

    @Test func hysteresisComparesInTheImagesOwnType() throws {
        // Float(0.1) is 0.10000000149...; NumPy casts a float64 `low` of 0.1 to float32 for a float32 image, so `image > low` is false.
        let pixels: [Float] = [Float(0.1), 0.9]
        let f32 = try Thresholding.hysteresis(.float32(pixels), width: 2, height: 1, low: 0.1, high: 0.5)
        #expect(f32 == [false, true])
        let f64 = try Thresholding.hysteresis(.float64(pixels.map(Double.init)), width: 2, height: 1, low: 0.1, high: 0.5)
        #expect(f64 == [true, true])
    }
}

@Suite("Binary morphology (SciPy 1.18.1, scikit-image 0.26.0)") struct BinaryMorphologyPinTests {
    static let cases = ThresholdMorphologyPins.all.binary

    @Test(arguments: cases) func ndimageCrossAndSquare(_ c: ThresholdMorphologyPins.Binary) throws {
        let m = maskFrom(c.mask)
        for (structure, pins) in [(StructuringElement.cross, c.ndimage_cross), (.square, c.ndimage_square)] {
            func run(_ f: ([Bool], Int, Int, StructuringElement, Int, Bool) throws -> [Bool], iterations: Int = 1, border: Bool = false) throws -> [Bool] {
                try f(m, c.width, c.height, structure, iterations, border)
            }
            #expect(try BinaryMorphology.erosion(m, width: c.width, height: c.height, structure: structure) == maskFrom(pins.erosion))
            #expect(try BinaryMorphology.erosion(m, width: c.width, height: c.height, structure: structure, borderValue: true) == maskFrom(pins.erosion_border1))
            #expect(try BinaryMorphology.dilation(m, width: c.width, height: c.height, structure: structure) == maskFrom(pins.dilation))
            #expect(try BinaryMorphology.opening(m, width: c.width, height: c.height, structure: structure) == maskFrom(pins.opening))
            #expect(try BinaryMorphology.closing(m, width: c.width, height: c.height, structure: structure) == maskFrom(pins.closing))
            #expect(try BinaryMorphology.erosion(m, width: c.width, height: c.height, structure: structure, iterations: 2) == maskFrom(pins.erosion_iter2))
            #expect(try BinaryMorphology.dilation(m, width: c.width, height: c.height, structure: structure, iterations: 2) == maskFrom(pins.dilation_iter2))
        }
    }

    @Test(arguments: cases) func skimageDefaults(_ c: ThresholdMorphologyPins.Binary) throws {
        let m = maskFrom(c.mask)
        #expect(try BinaryMorphology.skimageErosion(m, width: c.width, height: c.height) == maskFrom(c.skimage_default.erosion))
        #expect(try BinaryMorphology.skimageDilation(m, width: c.width, height: c.height) == maskFrom(c.skimage_default.dilation))
        #expect(try BinaryMorphology.skimageOpening(m, width: c.width, height: c.height) == maskFrom(c.skimage_default.opening))
        #expect(try BinaryMorphology.skimageClosing(m, width: c.width, height: c.height) == maskFrom(c.skimage_default.closing))
    }

    @Test(arguments: cases) func fillHoles(_ c: ThresholdMorphologyPins.Binary) throws {
        let m = maskFrom(c.mask)
        #expect(try BinaryMorphology.fillHoles(m, width: c.width, height: c.height) == maskFrom(c.fill_holes))
        #expect(try BinaryMorphology.fillHoles(m, width: c.width, height: c.height, structure: .square) == maskFrom(c.fill_holes_square))
    }

    @Test(arguments: cases) func removeSmall(_ c: ThresholdMorphologyPins.Binary) throws {
        let m = maskFrom(c.mask)
        func conn(_ k: Int) -> Segmentation.Connectivity { k == 1 ? .four : .eight }
        for r in c.remove_small_objects {
            #expect(try BinaryMorphology.removeSmallObjects(m, width: c.width, height: c.height, maxSize: r.max_size, connectivity: conn(r.connectivity))
                    == maskFrom(r.mask), "objects max_size \(r.max_size) connectivity \(r.connectivity)")
        }
        for r in c.remove_small_holes {
            #expect(try BinaryMorphology.removeSmallHoles(m, width: c.width, height: c.height, maxSize: r.max_size, connectivity: conn(r.connectivity))
                    == maskFrom(r.mask), "holes max_size \(r.max_size) connectivity \(r.connectivity)")
        }
    }

    @Test(arguments: cases) func clearBorder(_ c: ThresholdMorphologyPins.Binary) throws {
        let m = maskFrom(c.mask)
        #expect(try BinaryMorphology.clearBorder(m, width: c.width, height: c.height) == maskFrom(c.clear_border.buffer0))
        if c.width > 2 && c.height > 2 {
            #expect(try BinaryMorphology.clearBorder(m, width: c.width, height: c.height, bufferSize: 2) == maskFrom(c.clear_border.buffer2))
        }
    }

    @Test func generateBinaryStructure() throws {
        for (name, s) in ThresholdMorphologyPins.all.structures {
            let got = try StructuringElement.generateStructure(rank: s.rank, connectivity: s.connectivity)
            #expect(got.map { $0 ? 1 : 0 } == s.structure, "\(name)")
        }
        #expect(StructuringElement.cross.cells == (try StructuringElement.generateStructure(rank: 2, connectivity: 1)))
        #expect(StructuringElement.square.cells == (try StructuringElement.generateStructure(rank: 2, connectivity: 2)))
    }

    @Test func refusals() throws {
        #expect(throws: MorphologyError.structure) { try StructuringElement(rows: 2, columns: 3, cells: [Bool](repeating: true, count: 6)) }
        #expect(throws: MorphologyError.shape) { try BinaryMorphology.erosion([true, false], width: 3, height: 1) }
        #expect(throws: MorphologyError.buffer) { try BinaryMorphology.clearBorder([Bool](repeating: true, count: 9), width: 3, height: 3, bufferSize: 3) }
        // Until stable: iterations below 1 erode a 5x5 block to nothing with border_value 0.
        let block = [Bool](repeating: true, count: 25)
        #expect(try BinaryMorphology.erosion(block, width: 5, height: 5, iterations: 0) == [Bool](repeating: false, count: 25))
    }
}
