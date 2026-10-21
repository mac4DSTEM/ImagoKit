// numpy.fft.fft2 / ifft2 (axes, norm), rfft2 / irfft2, fftshift / ifftshift against NumPy 2.5.4 (tools/fftsurface/pins.py).
// Tolerance: numpy/fft/tests/test_pocketfft.py asserts `assert_allclose(..., atol=1e-6)` (rtol 1e-7 default) on these
// compositions; Float64 is held to that, Float32 to 1e-5 of the largest magnitude (the upstream float32 tolerance).
import Foundation
import Testing
@testable import ImagoCore

struct SurfacePins: Decodable, Sendable {
    struct FFT: Decodable, Sendable, CustomStringConvertible {
        let h, w: Int; let axes, norm: String
        let re, im: [Double]; let forward, inverse: [[Double]]
        var description: String { "\(h)x\(w) \(axes) \(norm)" }
    }
    struct RFFT: Decodable, Sendable, CustomStringConvertible {
        let h, w: Int; let norm: String
        let x: [Double]; let X: [[Double]]; let zre, zim: [Double]; let zcols: Int
        let z_default, z_w, round_trip: [Double]
        var description: String { "\(h)x\(w) \(norm)" }
    }
    struct Shift: Decodable, Sendable, CustomStringConvertible {
        let h, w: Int; let axes: String; let fftshift, ifftshift: [Int]
        var description: String { "\(h)x\(w) \(axes)" }
    }
    let fft: [FFT]; let rfft: [RFFT]; let shift: [Shift]
    static let all: SurfacePins = {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/fft-surface-pins.json")
        return try! JSONDecoder().decode(SurfacePins.self, from: Data(contentsOf: url))
    }()
}

private func axes(_ s: String) -> FFTAxes { s == "both" ? .both : (s == "axis0" ? .axis0 : .axis1) }
private func norm(_ s: String) -> FFTNorm { s == "backward" ? .backward : (s == "forward" ? .forward : .ortho) }

private func close64(_ got: [Double], _ want: [Double], _ label: String) {
    #expect(got.count == want.count, "\(label) count")
    var worst = 0.0
    for (g, w) in zip(got, want) { worst = max(worst, abs(g - w) - (1e-6 + 1e-7 * abs(w))) }
    #expect(worst <= 0, "\(label): exceeds atol 1e-6 + rtol 1e-7 by \(worst)")
}

private func close32(_ got: [Float], _ want: [Double], _ label: String) {
    #expect(got.count == want.count, "\(label) count")
    let scale = max(1, want.map(abs).max() ?? 1)
    var worst = 0.0
    for (g, w) in zip(got, want) { worst = max(worst, abs(Double(g) - w)) }
    #expect(worst <= 1e-5 * scale, "\(label): \(worst) vs \(1e-5 * scale)")
}

@Test(arguments: SurfacePins.all.fft)
func fft2AxesAndNormMatchNumPy(_ c: SurfacePins.FFT) throws {
    let x = ComplexImage<Double>(rows: c.h, columns: c.w, re: c.re, im: c.im)
    let f = try FFTSurface.fft2(x, axes: axes(c.axes), norm: norm(c.norm))
    close64(f.re, c.forward[0], "fft2 re \(c)"); close64(f.im, c.forward[1], "fft2 im \(c)")
    let i = try FFTSurface.ifft2(x, axes: axes(c.axes), norm: norm(c.norm))
    close64(i.re, c.inverse[0], "ifft2 re \(c)"); close64(i.im, c.inverse[1], "ifft2 im \(c)")
    let x32 = ComplexImage<Float>(rows: c.h, columns: c.w, re: c.re.map(Float.init), im: c.im.map(Float.init))
    let f32 = try FFTSurface.fft2(x32, axes: axes(c.axes), norm: norm(c.norm))
    close32(f32.re, c.forward[0], "fft2f re \(c)"); close32(f32.im, c.forward[1], "fft2f im \(c)")
    let i32 = try FFTSurface.ifft2(x32, axes: axes(c.axes), norm: norm(c.norm))
    close32(i32.re, c.inverse[0], "ifft2f re \(c)"); close32(i32.im, c.inverse[1], "ifft2f im \(c)")
}

@Test(arguments: SurfacePins.all.rfft)
func rfft2AndIrfft2MatchNumPy(_ c: SurfacePins.RFFT) throws {
    let r = try FFTSurface.rfft2(real: c.x, rows: c.h, columns: c.w, norm: norm(c.norm))
    #expect(r.columns == c.w / 2 + 1)
    close64(r.re, c.X[0], "rfft2 re \(c)"); close64(r.im, c.X[1], "rfft2 im \(c)")
    close64(try FFTSurface.irfft2(r, columns: c.w, norm: norm(c.norm)), c.x, "irfft2(rfft2) \(c)")
    let z = ComplexImage<Double>(rows: c.h, columns: c.zcols, re: c.zre, im: c.zim)
    close64(try FFTSurface.irfft2(z, norm: norm(c.norm)), c.z_default, "irfft2 default s \(c)")
    close64(try FFTSurface.irfft2(z, columns: c.w, norm: norm(c.norm)), c.z_w, "irfft2 s=(h,w) \(c)")
    let r32 = try FFTSurface.rfft2(real: c.x.map(Float.init), rows: c.h, columns: c.w, norm: norm(c.norm))
    close32(r32.re, c.X[0], "rfft2f re \(c)"); close32(r32.im, c.X[1], "rfft2f im \(c)")
    close32(try FFTSurface.irfft2(r32, columns: c.w, norm: norm(c.norm)), c.x, "irfft2f \(c)")
}

@Test(arguments: SurfacePins.all.shift)
func fftshiftMatchesNumPyExactly(_ c: SurfacePins.Shift) throws {
    let a = Array(0..<(c.h * c.w))
    #expect(try FFTSurface.fftshift(a, rows: c.h, columns: c.w, axes: axes(c.axes)) == c.fftshift)
    #expect(try FFTSurface.ifftshift(a, rows: c.h, columns: c.w, axes: axes(c.axes)) == c.ifftshift)
}

@Test func surfaceRefusesBadShapes() {
    #expect(throws: FFTSurfaceError.shape) { try FFTSurface.fftshift([1, 2, 3], rows: 2, columns: 2) }
    #expect(throws: FFTSurfaceError.shape) { try FFTSurface.rfft2(real: [Double](), rows: 0, columns: 0) }
}
