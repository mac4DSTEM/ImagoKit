// ImageMeasurement, IntensityHistogram and LineProfile against the pins of tools/measure/pins.py (NumPy 2.5.4,
// SciPy 1.18.1, scikit-image 0.26.0) and planted closed forms.
import Foundation
import ImagoCore
import Testing

@Suite("Pins") struct PinVersionTests {
    @Test func versions() throws {
        let p = try MeasurePins.load()
        #expect(p.versions.numpy == "2.5.4" && p.versions.scipy == "1.18.1" && p.versions.skimage == "0.26.0")
    }
}

@Suite("ImageMeasurement") struct ImageMeasurementTests {
    private let nm = { (s: Double) in AxisCalibration(scale: s, unit: "nm") }

    @Test func distanceUsesEachAxisOwnScale() {
        let d = ImageMeasurement.distance(PixelPoint(x: 1, y: 2), PixelPoint(x: 4, y: 6), x: nm(2), y: nm(0.5))
        #expect(d.pixels == 5)
        #expect(d.physical == hypot(6, 2) && d.unit == "nm")
    }

    @Test func physicalNeedsTwoCalibratedAxesInOneUnit() {
        let a = PixelPoint(x: 0, y: 0), b = PixelPoint(x: 3, y: 4)
        #expect(ImageMeasurement.distance(a, b, x: nil, y: nil).physical == nil)
        #expect(ImageMeasurement.distance(a, b, x: nm(1), y: AxisCalibration(scale: 1, unit: "µm")).physical == nil)
        #expect(ImageMeasurement.distance(a, b, x: nm(1), y: AxisCalibration(scale: 0, unit: "nm")).physical == nil)
        #expect(ImageMeasurement.distance(a, b, x: nm(1), y: nm(1)).physical == 5)
    }

    @Test func plantedAngles() throws {
        let b = PixelPoint(x: 10, y: 10)
        func angle(_ degrees: Double, scale: Double = 1) throws -> ImageMeasurement.Angle {
            let t = degrees * .pi / 180
            let c = PixelPoint(x: 10 + 7 * cos(t), y: 10 + 7 * sin(t))
            return try #require(ImageMeasurement.angle(PixelPoint(x: 13, y: 10), vertex: b, c, x: nil, y: nil))
        }
        for d in [0.0, 30, 60, 90, 120, 179.5, 180] { #expect(abs(try angle(d).degrees - d) <= 1e-12 * max(d, 1)) }
        #expect(try angle(90).plane == .pixel)
        // Two long, almost parallel arms keep their small angle (Kahan's cross product): 1e-9 rad over 1e6 px.
        let far = ImageMeasurement.angle(PixelPoint(x: 1e6, y: 0), vertex: PixelPoint(x: 0, y: 0),
                                         PixelPoint(x: 1e6, y: 1e-3), x: nil, y: nil)
        #expect(abs(try #require(far).degrees - 1e-9 * 180 / .pi) <= 1e-12 * 1e-9 * 180 / .pi)
        #expect(ImageMeasurement.angle(b, vertex: b, PixelPoint(x: 1, y: 1), x: nil, y: nil) == nil)
    }

    @Test func angleInThePhysicalPlaneOfNonSquarePixels() throws {
        // (1, 0) and (0, 1) pixel arms are at 90 degrees in pixels but at atan2(2, 1) physically for (1, 2) vectors.
        let b = PixelPoint(x: 0, y: 0)
        let ax = AxisCalibration(scale: 1, unit: "nm"), ay = AxisCalibration(scale: 2, unit: "nm")
        let r = try #require(ImageMeasurement.angle(PixelPoint(x: 1, y: 0), vertex: b, PixelPoint(x: 1, y: 1), x: ax, y: ay))
        #expect(r.plane == .physical)
        #expect(abs(r.degrees - atan2(2, 1) * 180 / .pi) <= 1e-13 * r.degrees)
        let p = try #require(ImageMeasurement.angle(PixelPoint(x: 1, y: 0), vertex: b, PixelPoint(x: 1, y: 1), x: ax, y: nil))
        #expect(p.plane == .pixel && abs(p.degrees - 45) <= 1e-13 * 45)
    }

    @Test func randomTriplesMatchNumPy() throws {
        for m in try MeasurePins.load().measure {
            let a = PixelPoint(x: m.a[0], y: m.a[1]), b = PixelPoint(x: m.b[0], y: m.b[1]), c = PixelPoint(x: m.c[0], y: m.c[1])
            let ax = AxisCalibration(scale: m.sx, unit: "nm"), ay = AxisCalibration(scale: m.sy, unit: "nm")
            let d = ImageMeasurement.distance(a, b, x: ax, y: ay)
            #expect(abs(d.pixels - m.distancePx) <= 4e-16 * m.distancePx)
            #expect(abs(try #require(d.physical) - m.distancePhys) <= 4e-16 * m.distancePhys)
            let ang = try #require(ImageMeasurement.angle(a, vertex: b, c, x: ax, y: ay))
            #expect(abs(ang.degrees - m.angleDegrees) <= 1e-13 * max(m.angleDegrees, 1))
        }
    }
}

@Suite("IntensityHistogram") struct IntensityHistogramTests {
    @Test func matchesNumPyBinForBin() throws {
        for h in try MeasurePins.load().histogram {
            let range = h.range.map { (Float(bitPattern: $0[0]), Float(bitPattern: $0[1])) }
            if let error = h.error {
                #expect(throws: HistogramError.self, "\(h.name): numpy raised \(error)") {
                    try IntensityHistogram.histogram(h.floats, bins: h.bins, range: range)
                }
                continue
            }
            let got = try IntensityHistogram.histogram(h.floats, bins: h.bins, range: range)
            #expect(got.counts == h.counts, "\(h.name) counts")
            #expect(got.edges.map(\.bitPattern) == h.edges, "\(h.name) edges")
            guard h.finite != nil else { #expect(got.finiteCount == 0 && got.minimum.isNaN); continue }
            #expect(got.minimum.bitPattern == h.min && got.maximum.bitPattern == h.max, "\(h.name) extremes")
            #expect(got.finiteCount == h.finite)
            // The Double mean: blockwise sum against numpy's pairwise float64 mean.
            let scale = max(abs(try #require(h.mean)), try #require(h.meanAbs))
            #expect(abs(got.mean - h.mean!) <= 1e-9 * scale, "\(h.name) mean")
        }
    }

    @Test func numpyErrorsAreTheKitsErrors() throws {
        let p = try MeasurePins.load().histogram
        func error(_ name: String) throws -> HistogramError? {
            let h = try #require(p.first { $0.name == name })
            let range = h.range.map { (Float(bitPattern: $0[0]), Float(bitPattern: $0[1])) }
            do { _ = try IntensityHistogram.histogram(h.floats, bins: h.bins, range: range); return nil } catch {
                return error as? HistogramError
            }
        }
        #expect(try error("constant-2e24") == .tooManyBins)
        #expect(try error("one-ulp-apart") == .tooManyBins)
        #expect(try error("all-nan") == nil)
        #expect(try error("range-reversed") == .rangeOrder)
        #expect(try error("range-inf") == .nonFiniteRange)
        #expect(throws: HistogramError.invalidBins(0)) { try IntensityHistogram.histogram([1, 2], bins: 0) }
    }

    @Test func computeKeepsTheStatisticsWhereNumPyRaises() throws {
        let h = try #require(IntensityHistogram.compute([1, 1.0000001, .nan], bins: 96))
        #expect(h.counts.isEmpty && h.edges.isEmpty && h.finiteCount == 2)
        #expect(h.minimum == 1 && h.firstEdge == 1 && h.lastEdge == h.maximum)
        #expect(IntensityHistogram.compute([.nan, .infinity], bins: 5) == nil)
        let ok = try #require(IntensityHistogram.compute([0, 1, 2, 3], bins: 3))
        #expect(ok.counts == [1, 1, 2])
    }

    @Test func defaultBinCountIsNumPys() throws {
        #expect(try IntensityHistogram.histogram([0, 1, 2]).counts.count == 10)
    }
}

@Suite("LineProfile") struct LineProfileTests {
    private func image() throws -> Micrograph {
        let i = try MeasurePins.load().image
        return try Micrograph(width: i.width, height: i.height, pixels: i.bits.map { Float(bitPattern: $0) })
    }
    private func reduction(_ name: String) -> ProfileReduction {
        switch name {
        case "sum": .sum
        case "min": .min
        case "max": .max
        case "median": .median
        default: .mean
        }
    }
    private func run(_ p: MeasurePins.Profile, _ a: ProfileArithmetic) throws -> [Double] {
        try LineProfile.sample(try image(), from: PixelPoint(x: p.src[1], y: p.src[0]), to: PixelPoint(x: p.dst[1], y: p.dst[0]),
                               linewidth: p.lw, order: ProfileOrder(rawValue: p.order)!, mode: ProfileMode(rawValue: p.mode)!,
                               cval: p.cval, reduce: reduction(p.reduce), arithmetic: a).samples
    }

    @Test func float32MatchesSkimageOnAFloat32Image() throws {
        var worst = 0.0, exact = 0, total = 0
        for p in try MeasurePins.load().profile {
            let got = try run(p, .float32), want = p.f32
            #expect(got.count == want.count, "\(p)")
            for (g, w) in zip(got, want) {
                total += 1
                if g == w { exact += 1 }
                worst = max(worst, abs(g - w) / 128)        // the image's largest magnitude is below 128
            }
        }
        // Measured: every one of the pinned samples is bit for bit (widths 1 to 25, the five reductions, both orders).
        #expect(exact == total && worst == 0)
    }

    @Test func float64MatchesSkimageOnAFloat64Image() throws {
        var worst = 0.0
        for p in try MeasurePins.load().profile {
            let got = try run(p, .float64)
            #expect(got.count == p.f64.count, "\(p)")
            for (g, w) in zip(got, p.f64) { worst = max(worst, abs(g - w) / 128) }
        }
        // Measured 8.9e-16 of the image maximum (rounding of the interpolation weights and of the sum).
        #expect(worst <= 4e-15)
    }

    @Test func unreducedRowsAreSkimagesFlippedPerpendicular() throws {
        for u in try MeasurePins.load().unreduced {
            let got64 = try LineProfile.unreduced(try image(), from: PixelPoint(x: u.src[1], y: u.src[0]),
                                                  to: PixelPoint(x: u.dst[1], y: u.dst[0]), linewidth: u.lw,
                                                  order: ProfileOrder(rawValue: u.order)!, mode: ProfileMode(rawValue: u.mode)!,
                                                  arithmetic: .float64)
            #expect(got64.count == u.f64.count)
            for (r, w) in zip(got64, u.f64) { for (g, e) in zip(r, w) { #expect(abs(g - e) <= 1e-13 * 128) } }
            let got32 = try LineProfile.unreduced(try image(), from: PixelPoint(x: u.src[1], y: u.src[0]),
                                                  to: PixelPoint(x: u.dst[1], y: u.dst[0]), linewidth: u.lw,
                                                  order: ProfileOrder(rawValue: u.order)!, mode: ProfileMode(rawValue: u.mode)!)
            #expect(got32 == u.f32)
        }
    }

    /// The consumer's own Double-precision sampling (order 1, "reflect", mean over the width), transcribed as it stands:
    /// the fold of a coordinate into [−0.5, n − 0.5] and the bilinear read.
    private func pinnedConsumerProfile(_ m: Micrograph, from: PixelPoint, to: PixelPoint, width w: Int) -> [Double] {
        func fold(_ c: Double, _ n: Int) -> Double {
            if c >= -0.5 && c <= Double(n) - 0.5 { return c }
            let period = Double(2 * n)
            var q = c + 0.5
            q -= (q / period).rounded(.down) * period
            if q >= Double(n) { q = period - q }
            return q - 0.5
        }
        func bilinear(_ row: Double, _ col: Double) -> Double {
            let r = fold(row, m.height), c = fold(col, m.width)
            let rf = r.rounded(.down), cf = c.rounded(.down)
            let fr = r - rf, fc = c - cf
            let r0 = min(max(Int(rf), 0), m.height - 1), r1 = min(max(Int(rf) + 1, 0), m.height - 1)
            let c0 = min(max(Int(cf), 0), m.width - 1), c1 = min(max(Int(cf) + 1, 0), m.width - 1)
            let top = Double(m.pixels[r0 * m.width + c0]) * (1 - fc) + Double(m.pixels[r0 * m.width + c1]) * fc
            let bottom = Double(m.pixels[r1 * m.width + c0]) * (1 - fc) + Double(m.pixels[r1 * m.width + c1]) * fc
            return top * (1 - fr) + bottom * fr
        }
        let dRow = to.y - from.y, dCol = to.x - from.x
        let n = Int((hypot(dRow, dCol) + 1).rounded(.up))
        let theta = atan2(dRow, dCol)
        let colHalf = Double(w - 1) * sin(-theta) / 2, rowHalf = Double(w - 1) * cos(theta) / 2
        let rowStep = n > 1 ? dRow / Double(n - 1) : 0, colStep = n > 1 ? dCol / Double(n - 1) : 0
        return (0..<n).map { i in
            let last = n > 1 && i == n - 1
            let lineRow = last ? to.y : Double(i) * rowStep + from.y
            let lineCol = last ? to.x : Double(i) * colStep + from.x
            let rs = lineRow - rowHalf, re = lineRow + rowHalf, cs = lineCol - colHalf, ce = lineCol + colHalf
            let prs = w > 1 ? (re - rs) / Double(w - 1) : 0, pcs = w > 1 ? (ce - cs) / Double(w - 1) : 0
            var sum = 0.0
            for k in 0..<w {
                sum += bilinear(w > 1 && k == w - 1 ? re : Double(k) * prs + rs, w > 1 && k == w - 1 ? ce : Double(k) * pcs + cs)
            }
            return sum / Double(w)
        }
    }

    @Test func float64PathIsTheConsumersNamedVariant() throws {
        let m = try image()
        var worst = 0.0
        for (a, b, w) in [((2.0, 3.0), (12.0, 19.0), 1), ((-4.5, -3.25), (9.75, 30.5), 3), ((3.3, 2.2), (25.1, 40.7), 5),
                          ((14.0, 20.0), (2.0, 4.0), 4), ((0.4, 0.9), (0.4, 1.3), 2), ((-30.7, 12.3), (40.1, 11.9), 9)] {
            let from = PixelPoint(x: a.1, y: a.0), to = PixelPoint(x: b.1, y: b.0)
            let got = try LineProfile.sample(m, from: from, to: to, linewidth: w, arithmetic: .float64).samples
            let want = pinnedConsumerProfile(m, from: from, to: to, width: w)
            #expect(got.count == want.count)
            for (g, e) in zip(got, want) { worst = max(worst, abs(g - e) / 128) }
        }
        // Measured gap: rounding of the weights only (the fold and the reflect mapping agree).
        #expect(worst <= 4e-15)
    }

    @Test func positionsFollowTheAxes() throws {
        let m = try Micrograph(width: 8, height: 8, pixels: [Float](repeating: 1, count: 64),
                               x: AxisCalibration(scale: 0.5, unit: "nm"), y: AxisCalibration(scale: 0.5, unit: "nm"))
        let p = try LineProfile.sample(m, from: PixelPoint(x: 0, y: 0), to: PixelPoint(x: 4, y: 3))
        #expect(p.samples.count == 6 && p.isPhysical && p.unit == "nm")
        #expect(p.positions.first == 0 && p.positions.last == 2.5)
        let q = try LineProfile.sample(try image(), from: PixelPoint(x: 0, y: 0), to: PixelPoint(x: 4, y: 3))
        #expect(!q.isPhysical && q.unit == "px" && q.positions.last == 5)
    }

    @Test func refusals() throws {
        let m = try image()
        let o = PixelPoint(x: 0, y: 0)
        #expect(throws: LineProfileError.invalidWidth(0)) { try LineProfile.sample(m, from: o, to: o, linewidth: 0) }
        #expect(throws: LineProfileError.invalidSegment) { try LineProfile.sample(m, from: o, to: PixelPoint(x: .nan, y: 1)) }
        #expect(throws: LineProfileError.invalidSegment) { try LineProfile.sample(m, from: o, to: PixelPoint(x: 1e16, y: 0)) }
        #expect(throws: LineProfileError.invalidSegment) { try LineProfile.sample(m, from: o, to: PixelPoint(x: 3e7, y: 0)) }
    }
}
