// Spot spacings and the angle between spots against NumPy 2.5.4 and the planted lattices (registration sections 3, 5.2, 5.4).
import Foundation
import ImagoCore
import ImagoFFT
import Testing

private let centroidBand: [Precision: Double] = [.float32: 1e-6, .float64: 1e-10]
private let angleBand: [Precision: Double] = [.float32: 1e-6, .float64: 1e-9]

@Test(arguments: Pins.spot, Precision.allCases)
func spotMatchesNumPyAndThePlant(_ c: SpotPin, _ precision: Precision) throws {
    let image = Pins.image(Pins.floats(c.input, section: "spot"), width: c.width, height: c.height,
                           scaleX: c.scale_x, scaleY: c.scale_y, units: c.units)
    if c.kind == "refusal" {
        let expected: SpotMeasurementError = c.error == "nearEdge" ? .nearEdge : .nearDC
        #expect(throws: expected) {
            try SpotMeasurement.measure(image, clickRow: c.click_row, clickColumn: c.click_col, precision: precision)
        }
        return
    }
    let m = try SpotMeasurement.measure(image, clickRow: c.click_row, clickColumn: c.click_col, precision: precision)
    #expect(m.snappedRow == c.snapped_row && m.snappedCol == c.snapped_col)
    let dr = abs(m.centroidRow - c.centroid_row!), dc = abs(m.centroidCol - c.centroid_col!)
    #expect(dr <= centroidBand[precision]! && dc <= centroidBand[precision]!, "centroid Δ (\(dr), \(dc)) px")
    #expect(m.neighbour == c.neighbour)
    // Truth: the planted spacing.
    let relative = abs(m.d - c.d0) / c.d0
    let band = c.tier == "clean" ? 0.002 : 0.005
    #expect(relative < band, "|d − d₀|/d₀ = \(relative), band \(band)")
    #expect(m.units == "nm" && m.reciprocalUnits == "1/nm" && m.dNanometres == m.d)
}

@Test(arguments: Pins.angle, Precision.allCases)
func angleMatchesNumPyAndThePlant(_ c: AnglePin, _ precision: Precision) throws {
    let image = Pins.image(Pins.floats(c.input, section: "angle"), width: c.width, height: c.height,
                           scaleX: c.scale_x, scaleY: c.scale_y, units: c.units)
    var spots: [SpotMeasurement] = []
    for s in c.spots {
        let m = try SpotMeasurement.measure(image, clickRow: s.click_row, clickColumn: s.click_col, precision: precision)
        #expect(m.snappedRow == s.snapped_row && m.snappedCol == s.snapped_col)
        let dr = abs(m.centroidRow - s.centroid_row), dc = abs(m.centroidCol - s.centroid_col)
        #expect(dr <= centroidBand[precision]! && dc <= centroidBand[precision]!, "centroid Δ (\(dr), \(dc)) px")
        spots.append(m)
    }
    let pair = try LatticePair.measure(spots[0], spots[1])
    let fromNumPy = abs(pair.angle - c.theta_numpy)
    #expect(fromNumPy <= angleBand[precision]!, "angle \(pair.angle)° vs NumPy \(c.theta_numpy)°: \(fromNumPy)°")
    // Truth: the planted angle and spacing ratio (d1/d2 = |q2|/|q1|).
    let truthBand = c.tier == "clean" ? 0.1 : 0.3
    #expect(abs(pair.angle - c.theta0) <= truthBand, "angle \(pair.angle)° vs planted \(c.theta0)°")
    #expect(abs(pair.ratio / c.ratio0 - 1) <= 0.004, "ratio \(pair.ratio) vs planted \(c.ratio0)")
}

@Test func angleClosedForms() throws {
    #expect(try SpotMeasurement.angle(qa: (1, 0), qb: (0, 1)) == 90)
    #expect(try SpotMeasurement.angle(qa: (2, 0), qb: (3, 0)) == 0)
    #expect(try SpotMeasurement.angle(qa: (1, 0), qb: (-1, 0)) == 180)
    let e = 1e-8
    let small = try SpotMeasurement.angle(qa: (1, 0), qb: (cos(e), sin(e)))
    #expect(abs(small - e * 180 / .pi) <= 1e-15, "atan2 keeps a 1e-8 rad angle: \(small)°")
    #expect(throws: SpotMeasurementError.zeroFrequency) { try SpotMeasurement.angle(qa: (0, 0), qb: (0, 1)) }
}

@Test func angleRefusesMixedUnits() throws {
    let pixels = (0..<4096).map { Float(cos(2 * Double.pi * 9 * Double($0 % 64) / 64)) }
    let nm = try Micrograph(width: 64, height: 64, pixels: pixels,
                            x: AxisCalibration(scale: 0.1, unit: "nm"), y: AxisCalibration(scale: 0.1, unit: "nm"))
    let px = try Micrograph(width: 64, height: 64, pixels: pixels)
    let a = try SpotMeasurement.measure(nm, clickRow: 32, clickColumn: 41)
    let b = try SpotMeasurement.measure(px, clickRow: 32, clickColumn: 41)
    #expect(b.units == "px" && b.reciprocalUnits == "1/px" && b.dNanometres == nil)
    #expect(throws: SpotMeasurementError.mixedUnits("1/nm", "1/px")) { try SpotMeasurement.angle(a, b) }
}

@Test func hanningIsNumPys() {
    #expect(SpotMeasurement.hanning(1) == [1])
    #expect(SpotMeasurement.hanning(2) == [0, 0])
    let h5 = SpotMeasurement.hanning(5)
    #expect(h5[0] == 0 && h5[2] == 1 && h5[4] == 0 && abs(h5[1] - 0.5) < 1e-16)
}

@Test func spotRefusesClicksOutsideAndMixedUnits() throws {
    let image = try Micrograph(width: 32, height: 32, pixels: [Float](repeating: 1, count: 1024),
                               x: AxisCalibration(scale: 0.1, unit: "nm"), y: AxisCalibration(scale: 0.1, unit: "pm"))
    #expect(throws: SpotMeasurementError.outside(row: 32, col: 0)) {
        try SpotMeasurement.measure(image, clickRow: 32, clickColumn: 0)
    }
    #expect(throws: SpotMeasurementError.mixedUnits("nm", "pm")) {
        try SpotMeasurement.measure(image, clickRow: 3, clickColumn: 3)
    }
}
