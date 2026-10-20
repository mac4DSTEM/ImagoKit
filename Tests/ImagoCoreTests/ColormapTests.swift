// Colormap tables and DisplayMapping.sampleRGB against tools/colormap/pins.py: an independent NumPy reading of the
// definition, itself asserted equal to the application's own table bytes when the pins were generated.
// Tables and sampler are integer-valued outputs of a Double recipe: tested exact (D12: indices and bytes are exact).
import Foundation
import ImagoCore
import Testing

private struct ColormapPins: Decodable, Sendable {
    let luts: [String: [UInt8]]
    let sampleGrid: [Double]
    let sample: [UInt8]
    let appChecked: Bool
    static func load() throws -> ColormapPins {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/colormap-pins.json")
        return try JSONDecoder().decode(ColormapPins.self, from: Data(contentsOf: url))
    }
}

@Suite("Colormap") struct ColormapTests {
    @Test func tablesEqualThePinsByteForByte() throws {
        let pins = try ColormapPins.load()
        #expect(pins.appChecked)
        for map in Colormap.allCases {
            for n in [256, 44, 2, 3] {
                #expect(map.lutRGBA(count: n) == pins.luts["\(map.rawValue)-\(n)"], "\(map.rawValue) \(n)")
            }
        }
    }

    @Test func endsAndMiddleArePlanted() {
        let gray = Colormap.gray.lutRGBA()
        #expect(Array(gray[0..<4]) == [0, 0, 0, 255])
        #expect(Array(gray[1020..<1024]) == [255, 255, 255, 255])
        #expect(Array(gray[512..<516]) == [128, 128, 128, 255])           // t = 128/255 = 0.50196 -> 127.99... rounds to 128
        #expect(Colormap.rdbu.isDiverging && !Colormap.viridis.isDiverging)
        #expect(Colormap.viridis.lutRGBA(count: 1).isEmpty)
        #expect(Colormap.allCases.count == 4)
    }

    @Test func samplerEqualsThePinsOnTheGrid() throws {
        let pins = try ColormapPins.load()
        let lut = Colormap.inferno.lutRGBA()
        for (i, v) in pins.sampleGrid.enumerated() {
            let c = try #require(DisplayMapping.sampleRGB(lut, v))
            #expect([c.0, c.1, c.2] == Array(pins.sample[3 * i..<3 * i + 3]), "v = \(v)")
        }
        #expect(DisplayMapping.sampleRGB([0, 0, 0, 255], 0.5) == nil)
        let nan = try #require(DisplayMapping.sampleRGB(lut, .nan))
        #expect([nan.0, nan.1, nan.2] == Array(lut[0..<3]))
    }

    @Test func mappingThroughAColormapIsTheTableMapping() throws {
        let direct = DisplayMapping(low: 0.1, high: 0.9, gamma: 1.4, colormap: .viridis)
        let viaLut = try #require(DisplayMapping(low: 0.1, high: 0.9, gamma: 1.4, lut: Colormap.viridis.lutRGBA()))
        for n in stride(from: Float(-0.1), through: 1.1, by: 0.01) {
            #expect(direct.rgb(n) == viaLut.rgb(n))
        }
        // window 0...1, gamma 1, grey: the value reaches its own texel (centre i/256 + 1/512)
        let grey = DisplayMapping(low: 0, high: 1, gamma: 1, colormap: .gray)
        #expect(grey.rgb(0) == (0, 0, 0) && grey.rgb(1) == (255, 255, 255))
    }
}
