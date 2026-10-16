import Foundation
import Testing
import ImagoFFT

private struct FFT1DPin: Decodable, Sendable {
    let n: Int
    let re, im, forwardRe, forwardIm, inverseRe, inverseIm: [Double]
}

private struct FFT1DPins: Decodable {
    let numpy, python: String
    let cases: [FFT1DPin]
}

private func fft1dPins() throws -> FFT1DPins {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/fft1d.json")
    return try JSONDecoder().decode(FFT1DPins.self, from: Data(contentsOf: url))
}

// Complex-bin error |got - oracle| <= tolerance * max(1, |oracle|).
// The absolute floor handles bins near zero. Relative error uses the complex
// magnitude, so cancellation in one component does not define the bin's scale.
private func expectFFT(_ re: [Double], _ im: [Double], _ wantRe: [Double], _ wantIm: [Double],
                       tolerance: Double) {
    #expect(re.count == wantRe.count && im.count == wantIm.count)
    for i in re.indices {
        let error = hypot(re[i] - wantRe[i], im[i] - wantIm[i])
        let scale = max(1, hypot(wantRe[i], wantIm[i]))
        #expect(error.isFinite && error <= tolerance * scale)
    }
}

@Test func fft1dFloatAgainstNumPy() throws {
    let pins = try fft1dPins()
    #expect(pins.numpy == "2.5.4" && pins.python == "3.14.7")
    #expect(pins.cases.map(\.n) == [1, 2, 4, 8, 64, 256, 1024])
    for pin in pins.cases {
        let fft = try #require(FFT1D(n: pin.n))
        #expect(fft.n == pin.n)
        for forward in [true, false] {
            var re = pin.re.map(Float.init), im = pin.im.map(Float.init)
            fft.transform(re: &re, im: &im, forward: forward)
            expectFFT(re.map(Double.init), im.map(Double.init),
                      forward ? pin.forwardRe : pin.inverseRe,
                      forward ? pin.forwardIm : pin.inverseIm, tolerance: 1e-5)
        }
        var re = pin.re.map(Float.init), im = pin.im.map(Float.init)
        fft.transform(re: &re, im: &im, forward: true)
        fft.transform(re: &re, im: &im, forward: false)
        expectFFT(re.map { Double($0) / Double(pin.n) }, im.map { Double($0) / Double(pin.n) },
                  pin.re, pin.im, tolerance: 1e-5)
    }
}

@Test func fft1dDoubleAgainstNumPy() throws {
    for pin in try fft1dPins().cases {
        let fft = try #require(FFT1DDouble(n: pin.n))
        #expect(fft.n == pin.n)
        for forward in [true, false] {
            var re = pin.re, im = pin.im
            fft.transform(re: &re, im: &im, forward: forward)
            expectFFT(re, im, forward ? pin.forwardRe : pin.inverseRe,
                      forward ? pin.forwardIm : pin.inverseIm, tolerance: 1e-12)
        }
        var re = pin.re, im = pin.im
        fft.transform(re: &re, im: &im, forward: true)
        fft.transform(re: &re, im: &im, forward: false)
        expectFFT(re.map { $0 / Double(pin.n) }, im.map { $0 / Double(pin.n) },
                  pin.re, pin.im, tolerance: 1e-12)
    }
}

@Test func fft1dLengthsAndPadding() {
    for n in [Int.min, -1, 0, 3, 63, 65, 255, 1023, Int.max] {
        #expect(FFT1D(n: n) == nil)
        #expect(FFT1DDouble(n: n) == nil)
    }
    for (input, expected) in [(-1, 1), (0, 1), (1, 1), (2, 2), (3, 4), (63, 64),
                              (64, 64), (65, 128), (1024, 1024), (1025, 2048),
                              (1 << (Int.bitWidth - 2), 1 << (Int.bitWidth - 2))] {
        #expect(FFT1D.nextPow2(input) == expected)
        #expect(FFT1DDouble.nextPow2(input) == expected)
    }
}

@Test func fft1dSharedSetupIsSendable() async throws {
    let pin = try #require(fft1dPins().cases.first { $0.n == 256 })
    let fft = try #require(FFT1D(n: pin.n))
    let double = try #require(FFT1DDouble(n: pin.n))
    await withTaskGroup(of: Void.self) { group in
        for _ in 0..<8 {
            group.addTask {
                var re = pin.re.map(Float.init), im = pin.im.map(Float.init)
                fft.transform(re: &re, im: &im, forward: true)
                expectFFT(re.map(Double.init), im.map(Double.init), pin.forwardRe, pin.forwardIm, tolerance: 1e-5)
                var dr = pin.re, di = pin.im
                double.transform(re: &dr, im: &di, forward: false)
                expectFFT(dr, di, pin.inverseRe, pin.inverseIm, tolerance: 1e-12)
            }
        }
    }
}
