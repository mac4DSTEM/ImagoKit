// The real-input transforms and the mixed-radix lengths against NumPy 2.5.4 (tools/fft-real/pins.py, seed 20261023):
// rfft2 / irfft2 at larger and odd shapes with `s=` truncation and padding, rfft / irfft with `n=` and `norm=`, and fft / ifft
// at lengths of small prime factors. Tolerance: numpy/fft/tests/test_pocketfft.py asserts atol 1e-6 (rtol 1e-7 default) on
// these compositions; Float64 is held to that, Float32 to 1e-5 of the largest magnitude (the upstream float32 tolerance).
import Foundation
import ImagoKitTestSupport
import Testing
@testable import ImagoCore
import ImagoFFT

struct RealPins: Decodable, Sendable {
    struct IRFFT2: Decodable, Sendable { let cols: Int; let z: [[Double]]; let out: [Double] }
    struct RFFT2: Decodable, Sendable, CustomStringConvertible {
        let h, w: Int; let norm: String; let x: [Double]; let X: [[Double]]
        let irfft: [String: IRFFT2]; let zdefault: IRFFT2
        var description: String { "\(h)x\(w) \(norm)" }
    }
    struct RFFT1: Decodable, Sendable, CustomStringConvertible {
        let n, length: Int; let norm: String; let x: [Double]; let X: [[Double]]
        var description: String { "n=\(n) length=\(length) \(norm)" }
    }
    struct IRFFT1: Decodable, Sendable, CustomStringConvertible {
        let m, n: Int; let explicit: Bool; let norm: String; let z: [[Double]]; let out: [Double]
        var description: String { "m=\(m) n=\(n) explicit=\(explicit) \(norm)" }
    }
    struct Line: Decodable, Sendable, CustomStringConvertible {
        let n: Int; let z, fft, ifft: [[Double]]
        var description: String { "n=\(n)" }
    }
    struct Plane: Decodable, Sendable, CustomStringConvertible {
        let h, w: Int; let z, fft2, ifft2: [[Double]]
        var description: String { "\(h)x\(w)" }
    }
    let numpy: String; let seed: Int
    let rfft2: [RFFT2]; let rfft1: [RFFT1]; let irfft1: [IRFFT1]; let lines: [Line]; let planes: [Plane]
    static let all: RealPins = {
        let url = FixtureDirs.fixtures(of: "ImagoFFTTests").appendingPathComponent("fft-real-pins.json")
        return try! JSONDecoder().decode(RealPins.self, from: Data(contentsOf: url))
    }()
}

private func norm(_ s: String) -> FFTNorm { s == "backward" ? .backward : (s == "forward" ? .forward : .ortho) }

private func rClose64(_ got: [Double], _ want: [Double], _ label: String) {
    #expect(got.count == want.count, "\(label) count")
    var worst = 0.0
    for (g, w) in zip(got, want) { worst = max(worst, abs(g - w) - (1e-6 + 1e-7 * abs(w))) }
    #expect(worst <= 0, "\(label): exceeds atol 1e-6 + rtol 1e-7 by \(worst)")
}

private func rClose32(_ got: [Float], _ want: [Double], _ label: String) {
    #expect(got.count == want.count, "\(label) count")
    let scale = max(1, want.map(abs).max() ?? 1)
    var worst = 0.0
    for (g, w) in zip(got, want) { worst = max(worst, abs(Double(g) - w)) }
    #expect(worst <= 1e-5 * scale, "\(label): \(worst) vs \(1e-5 * scale)")
}

@Test func realPinsAreNumPy254() {
    #expect(RealPins.all.numpy == "2.5.4" && RealPins.all.seed == 20261023)
}

@Test(arguments: RealPins.all.rfft2)
func rfft2AndIrfft2MatchNumPyAtLargerShapes(_ c: RealPins.RFFT2) throws {
    let r = try FFTSurface.rfft2(real: c.x, rows: c.h, columns: c.w, norm: norm(c.norm))
    #expect(r.columns == c.w / 2 + 1 && r.rows == c.h)
    rClose64(r.re, c.X[0], "rfft2 re \(c)"); rClose64(r.im, c.X[1], "rfft2 im \(c)")
    rClose64(try FFTSurface.irfft2(r, columns: c.w, norm: norm(c.norm)), c.x, "irfft2(rfft2) \(c)")
    let x32 = c.x.map(Float.init)
    let r32 = try FFTSurface.rfft2(real: x32, rows: c.h, columns: c.w, norm: norm(c.norm))
    rClose32(r32.re, c.X[0], "rfft2f re \(c)"); rClose32(r32.im, c.X[1], "rfft2f im \(c)")
    rClose32(try FFTSurface.irfft2(r32, columns: c.w, norm: norm(c.norm)), c.x, "irfft2f(rfft2f) \(c)")
    // irfft2 of arbitrary half spectra: s = (h, w) crops or zero pads the columns; the default width is 2 (cols - 1).
    for (key, p) in c.irfft {
        let z = ComplexImage<Double>(rows: c.h, columns: p.cols, re: p.z[0], im: p.z[1])
        rClose64(try FFTSurface.irfft2(z, columns: c.w, norm: norm(c.norm)), p.out, "irfft2 \(key) \(c)")
        let z32 = ComplexImage<Float>(rows: c.h, columns: p.cols, re: p.z[0].map(Float.init), im: p.z[1].map(Float.init))
        rClose32(try FFTSurface.irfft2(z32, columns: c.w, norm: norm(c.norm)), p.out, "irfft2f \(key) \(c)")
    }
    let d = c.zdefault
    let zd = ComplexImage<Double>(rows: c.h, columns: d.cols, re: d.z[0], im: d.z[1])
    rClose64(try FFTSurface.irfft2(zd, norm: norm(c.norm)), d.out, "irfft2 default s \(c)")
}

@Test(arguments: RealPins.all.lines)
func complexLinesOfSmallPrimeLengthsMatchNumPy(_ c: RealPins.Line) throws {
    let z = ComplexImage<Double>(rows: 1, columns: c.n, re: c.z[0], im: c.z[1])
    let f = try FFTSurface.fft2(z, axes: .axis1), i = try FFTSurface.ifft2(z, axes: .axis1)
    rClose64(f.re, c.fft[0], "fft re \(c)"); rClose64(f.im, c.fft[1], "fft im \(c)")
    rClose64(i.re, c.ifft[0], "ifft re \(c)"); rClose64(i.im, c.ifft[1], "ifft im \(c)")
    let z32 = ComplexImage<Float>(rows: 1, columns: c.n, re: c.z[0].map(Float.init), im: c.z[1].map(Float.init))
    let f32 = try FFTSurface.fft2(z32, axes: .axis1)
    rClose32(f32.re, c.fft[0], "fftf re \(c)"); rClose32(f32.im, c.fft[1], "fftf im \(c)")
}

@Test(arguments: RealPins.all.planes)
func complexPlanesOfSmallPrimeShapesMatchNumPy(_ c: RealPins.Plane) throws {
    let z = ComplexImage<Double>(rows: c.h, columns: c.w, re: c.z[0], im: c.z[1])
    let f = try FFTSurface.fft2(z), i = try FFTSurface.ifft2(z)
    rClose64(f.re, c.fft2[0], "fft2 re \(c)"); rClose64(f.im, c.fft2[1], "fft2 im \(c)")
    rClose64(i.re, c.ifft2[0], "ifft2 re \(c)"); rClose64(i.im, c.ifft2[1], "ifft2 im \(c)")
}

@Test(arguments: RealPins.all.rfft1)
func rfftMatchesNumPy(_ c: RealPins.RFFT1) throws {
    let nrm = norm(c.norm)
    let plan = try #require(FFT1DDouble(n: c.n))
    let r = plan.rfft(c.x, norm: nrm)
    #expect(r.re.count == c.n / 2 + 1 && r.im.count == r.re.count)
    rClose64(r.re, c.X[0], "rfft re \(c)"); rClose64(r.im, c.X[1], "rfft im \(c)")
    let s = try #require(FFT1DDouble.rfft(c.x, n: c.n, norm: nrm))
    #expect(s.re == r.re && s.im == r.im)
    let plan32 = try #require(FFT1D(n: c.n))
    let r32 = plan32.rfft(c.x.map(Float.init), norm: nrm)
    rClose32(r32.re, c.X[0], "rfftf re \(c)"); rClose32(r32.im, c.X[1], "rfftf im \(c)")
    let s32 = try #require(FFT1D.rfft(c.x.map(Float.init), n: c.n, norm: nrm))
    #expect(s32.re == r32.re && s32.im == r32.im)
    if c.length == c.n {
        // n defaults to the length of x; the round trip returns x.
        let d = try #require(FFT1DDouble.rfft(c.x, norm: nrm))
        #expect(d.re == r.re)
        rClose64(plan.irfft(re: r.re, im: r.im, norm: nrm), c.x, "irfft(rfft) \(c)")
        rClose32(plan32.irfft(re: r32.re, im: r32.im, norm: nrm), c.x, "irfftf(rfftf) \(c)")
    }
}

@Test(arguments: RealPins.all.irfft1)
func irfftMatchesNumPy(_ c: RealPins.IRFFT1) throws {
    let nrm = norm(c.norm)
    let plan = try #require(FFT1DDouble(n: c.n))
    rClose64(plan.irfft(re: c.z[0], im: c.z[1], norm: nrm), c.out, "irfft \(c)")
    let s = try #require(FFT1DDouble.irfft(re: c.z[0], im: c.z[1], n: c.explicit ? c.n : nil, norm: nrm))
    rClose64(s, c.out, "irfft static \(c)")
    let plan32 = try #require(FFT1D(n: c.n))
    rClose32(plan32.irfft(re: c.z[0].map(Float.init), im: c.z[1].map(Float.init), norm: nrm), c.out, "irfftf \(c)")
    let s32 = try #require(FFT1D.irfft(re: c.z[0].map(Float.init), im: c.z[1].map(Float.init), n: c.explicit ? c.n : nil, norm: nrm))
    rClose32(s32, c.out, "irfftf static \(c)")
}

@Test func rfftRefusesLengthsThatAreNotPowersOfTwo() {
    #expect(FFT1DDouble.rfft([1, 2, 3], norm: .backward) == nil)
    #expect(FFT1DDouble.rfft([1, 2, 3], n: 4) != nil)
    #expect(FFT1D.rfft([1, 2, 3, 4, 5, 6], n: 6) == nil)
    #expect(FFT1DDouble.irfft(re: [1, 2, 3, 4], im: [0, 0, 0, 0]) == nil)       // default n = 6
    #expect(FFT1DDouble.irfft(re: [], im: []) == nil)
    #expect(FFT1DDouble.irfft(re: [], im: [], n: 4)?.map { $0 } == [0, 0, 0, 0])
}
