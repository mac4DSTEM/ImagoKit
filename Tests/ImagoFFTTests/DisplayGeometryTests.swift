// SpectrumGeometry (regions, bins, decades, units, aspect) and SpectrumDisplay against tools/display/pins.py.
import Foundation
import ImagoCore
import ImagoFFT
import Testing

private struct RegionPin: Decodable, Sendable, Equatable {
    let row, col, height, width: Int
    var crop: PixelCrop { PixelCrop(row: row, col: col, height: height, width: width) }
}

private struct GeometryPins: Decodable, Sendable {
    struct Region: Decodable, Sendable {
        struct Move: Decodable, Sendable { let dx, dy: Double; let crop: RegionPin }
        let width, height: Int
        let p0, p1: [Double]
        let drawn, typed, centred: RegionPin?
        let moved: Move?
        let maximum: Int?
    }
    struct Bin: Decodable, Sendable {
        let width, height: Int
        let qx, qy: Double
        let row, col, kx, ky: Int
        let k: [Int]?
    }
    struct Decades: Decodable, Sendable { let lo, hi, plotHeight, minimumSpacing: Double; let marks: [Double] }
    struct Unit: Decodable, Sendable { let unit: String; let d: Double; let length, reciprocal: [PairValue]; let spectrum: [PairValue] }
    struct Aspect: Decodable, Sendable { let width, height: Int; let sx: Double; let ux: String; let sy: Double; let uy: String; let aspect: Double }
    struct Display: Decodable, Sendable {
        let width, height: Int
        let power, values: [UInt32]
        let window: [UInt32]?
        let sortedMedian, binWidth: Double?
    }
    let region: [Region]
    let bins: [Bin]
    let decades: [Decades]
    let units: [Unit]
    let aspects: [Aspect]
    let display: [Display]

    static func load() throws -> GeometryPins {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/display-fft-pins.json")
        return try JSONDecoder().decode(GeometryPins.self, from: Data(contentsOf: url))
    }
}

/// A JSON array element that is a number or a string (a (value, label) pair).
private enum PairValue: Decodable, Sendable, Equatable {
    case number(Double), text(String)
    init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let d = try? c.decode(Double.self) { self = .number(d) } else { self = .text(try c.decode(String.self)) }
    }
    var number: Double { if case .number(let d) = self { d } else { .nan } }
    var text: String { if case .text(let s) = self { s } else { "" } }
}

@Suite("SpectrumGeometry") struct SpectrumGeometryTests {
    @Test func regionRulesMatchThePinnedDefinition() throws {
        for r in try GeometryPins.load().region {
            #expect(FFTRegion.maximumSide(width: r.width, height: r.height) == r.maximum)
            let drawn = FFTRegion.drawn(from: PixelPoint(x: r.p0[0], y: r.p0[1]), to: PixelPoint(x: r.p1[0], y: r.p1[1]),
                                        width: r.width, height: r.height)
            #expect(drawn == r.drawn?.crop, "drawn \(r.width)x\(r.height) \(r.p0) \(r.p1)")
            if let c = drawn { #expect(FFTRegion.isValid(c, width: r.width, height: r.height)) }
            #expect(FFTRegion.centredSquare(width: r.width, height: r.height) == r.centred?.crop)
            if let t = r.typed, let m = r.moved {
                #expect(FFTRegion.moved(t.crop, dx: m.dx, dy: m.dy, width: r.width, height: r.height) == m.crop.crop)
            }
        }
    }

    @Test func typedRegionsAreEvenSquaresInsideTheImage() throws {
        #expect(FFTRegion.typed(left: -5, top: 900, size: 101, width: 200, height: 150) == PixelCrop(row: 50, col: 0, height: 100, width: 100))
        #expect(FFTRegion.typed(left: 0, top: 0, size: 10, width: 200, height: 150)?.width == 32)
        #expect(FFTRegion.typed(left: 0, top: 0, size: 10, width: 31, height: 150) == nil)
        #expect(FFTRegion.filterable(width: 64, height: 64) && !FFTRegion.filterable(width: 64, height: 66)
                && !FFTRegion.filterable(width: 65, height: 65))
        let crop = PixelCrop(row: 40, col: 60, height: 64, width: 64)
        #expect(FFTRegion.point(of: .topLeft, in: crop) == PixelPoint(x: 59.5, y: 39.5))
        #expect(FFTRegion.point(of: .bottomRight, in: crop) == PixelPoint(x: 123.5, y: 103.5))
        // Dragging the bottom-right corner back to the top-left corner's pixel edge shrinks to the minimum side.
        let small = FFTRegion.resized(crop, corner: .bottomRight, to: PixelPoint(x: 60, y: 41), width: 400, height: 400)
        #expect(small?.width == 32 && small?.height == 32)
    }

    @Test func binsMatchThePinnedDefinition() throws {
        for b in try GeometryPins.load().bins {
            let got = SpectrumBin.at(PixelPoint(x: b.qx, y: b.qy), width: b.width, height: b.height)
            #expect(got.row == b.row && got.col == b.col)
            let k = SpectrumBin.at(kx: b.kx, ky: b.ky, width: b.width, height: b.height)
            #expect(k.map { [$0.row, $0.col] } == b.k)
        }
        #expect(SpectrumBin.at(PixelPoint(x: .nan, y: 3), width: 10, height: 10) == (row: 3, col: 0))
        #expect(SpectrumBin.at(kx: Int.max, ky: 0, width: 10, height: 10) == nil)
    }

    @Test func decadeMarksMatchThePinnedDefinition() throws {
        for d in try GeometryPins.load().decades {
            #expect(RadialChartScale.decadeMarks(lowest: d.lo, highest: d.hi, plotHeight: d.plotHeight,
                                                 minimumSpacing: d.minimumSpacing) == d.marks, "\(d.lo) \(d.hi)")
        }
        #expect(RadialChartScale.decadeMarks(lowest: 1, highest: 1, plotHeight: 100, minimumSpacing: 10) == [1, 10])
    }

    @Test func unitsMatchThePinnedDefinition() throws {
        let pins = try GeometryPins.load()
        for u in pins.units {
            let l = FFTUnits.length(u.d, unit: u.unit)
            #expect(l.value == u.length[0].number && l.label == u.length[1].text, "length \(u.unit)")
            let r = FFTUnits.reciprocal(u.d, realUnit: u.unit)
            #expect(r.value == u.reciprocal[0].number && r.label == u.reciprocal[1].text, "reciprocal \(u.unit)")
            let s = FFTUnits.spectrumUnit(u.unit)
            #expect(s.factor == u.spectrum[0].number && s.label == u.spectrum[1].text, "spectrum \(u.unit)")
        }
        for a in pins.aspects {
            #expect(SpectrumShape.aspect(width: a.width, height: a.height, spacingX: a.sx, unitsX: a.ux, spacingY: a.sy, unitsY: a.uy)
                    == a.aspect, "\(a.ux) \(a.uy)")
        }
    }

    @Test func plantedUnits() {
        #expect(FFTUnits.length(0.5, unit: "µm").value == 500 && FFTUnits.length(0.5, unit: "µm").label == "nm")
        #expect(FFTUnits.reciprocal(2, realUnit: "px").label == FFTUnits.reciprocalPixelLabel)
        // An axis already in "1/nm" (a diffraction image) has spectrum frequencies in nm, shown as nm.
        #expect(FFTUnits.reciprocal(2, realUnit: "1/nm").label == "nm" && FFTUnits.reciprocal(2, realUnit: "1/nm").value == 2)
        #expect(FFTUnits.spectrumUnit("1/nm").factor == 1)
    }
}

@Suite("SpectrumDisplay") struct SpectrumDisplayTests {
    @Test func valuesAndWindowMatchTheDefinition() throws {
        var worstUlp = 0, exact = 0, total = 0
        for d in try GeometryPins.load().display {
            let power = d.power.map { Float(bitPattern: $0) }
            let got = SpectrumDisplay.make(power: power, width: d.width, height: d.height)
            #expect(got.values.count == d.values.count)
            for (g, w) in zip(got.values, d.values.map { Float(bitPattern: $0) }) {
                total += 1
                if g.isNaN || w.isNaN { #expect(g.isNaN && w.isNaN); exact += 1; continue }
                if g.bitPattern == w.bitPattern { exact += 1 } else {
                    worstUlp = max(worstUlp, abs(Int(g.bitPattern) - Int(w.bitPattern)))
                }
            }
            guard let w = d.window else { #expect(got.defaultWindow == nil); continue }
            let win = try #require(got.defaultWindow)
            #expect(win.high.bitPattern == w[1], "\(d.width)x\(d.height) white")
            // The median is the histogram rule's; allow the log's last-bit differences to move it by a bin edge.
            let bin = try #require(d.binWidth)
            #expect(abs(Double(win.low) - Double(Float(bitPattern: w[0]))) <= max(bin * 1e-3, 1e-6), "\(d.width)x\(d.height) black")
            if let m = d.sortedMedian { #expect(abs(Double(win.low) - m) <= bin * 1.0001 + 1e-6, "within one bin of the ceil(n/2)-th smallest value") }
        }
        // Measured: libm's log10 and NumPy's agree to the last bit of Float32.
        #expect(worstUlp <= 1, "worst \(worstUlp) ulp, exact \(exact)/\(total)")
    }

    @Test func theCentreStaysNaNAndTheWhiteIgnoresIt() {
        let w = 9, h = 9
        var power = [Float](repeating: 1, count: w * h)
        power[4 * w + 4] = 1e20                                     // the DC
        power[4 * w + 5] = 1e10                                     // 1 bin away: ignored for the white
        power[0] = 1e4                                              // far: sets the white
        let d = SpectrumDisplay.make(power: power, width: w, height: h)
        #expect(d.values[4 * w + 4].isNaN)
        let scale = Float(log10(Double(1e4) / 6561))
        #expect(d.defaultWindow?.high == scale)
        #expect(SpectrumDisplay.make(power: [1, 2], width: 3, height: 3).defaultWindow == nil)
    }

    @Test func aPowerSpectrumFeedsTheSameValues() throws {
        let w = 16, h = 16
        let pixels = (0..<(w * h)).map { Float(5 + cos(2 * Double.pi * 3 * Double($0 % w) / Double(w))) }
        let image = try Micrograph(width: w, height: h, pixels: pixels)
        let spectrum = try PowerSpectrum.compute(image)
        let d = SpectrumDisplay.make(spectrum)
        guard case .float32(let p) = spectrum.power else { Issue.record("float32 expected"); return }
        #expect(d.values.map(\.bitPattern) == SpectrumDisplay.make(power: p, width: w, height: h).values.map(\.bitPattern))
        // The planted peaks are the brightest bins: ±3 columns from the centre.
        let peak = d.values.enumerated().filter { !$0.element.isNaN }.max { $0.element < $1.element }!.offset
        #expect(peak % w == w / 2 + 3 || peak % w == w / 2 - 3)
    }
}
