// ImagoCore: the FFT engines against a direct Double DFT, round trips, the image and the units (registration 1.1, 3, 4).
import Foundation
import ImagoCore
import Testing

/// The direct O(N²) 2D DFT in Double, forward, unnormalised: the independent check of the FFT plans.
private func directDFT(_ re: [Double], _ im: [Double], nx: Int, ny: Int) -> ([Double], [Double]) {
    var outRe = [Double](repeating: 0, count: nx * ny), outIm = outRe
    for v in 0..<ny { for u in 0..<nx {
        var sr = 0.0, si = 0.0
        for y in 0..<ny { for x in 0..<nx {
            // The phase reduced modulo the length first, so it stays exact for every size here.
            let phase = -2 * Double.pi * (Double((u * x) % nx) / Double(nx) + Double((v * y) % ny) / Double(ny))
            let c = cos(phase), s = sin(phase)
            sr += re[y * nx + x] * c - im[y * nx + x] * s
            si += re[y * nx + x] * s + im[y * nx + x] * c
        } }
        outRe[v * nx + u] = sr; outIm[v * nx + u] = si
    } }
    return (outRe, outIm)
}

/// The largest |z − w| over two complex arrays in split form (one helper: the runner's compiler times out type-checking the
/// inline expression).
private func largestDistance(_ re: [Double], _ im: [Double], _ wantRe: [Double], _ wantIm: [Double]) -> Double {
    var worst = 0.0
    for i in re.indices {
        let dr: Double = re[i] - wantRe[i]
        let di: Double = im[i] - wantIm[i]
        worst = max(worst, (dr * dr + di * di).squareRoot())
    }
    return worst
}

private func signal(_ n: Int, seed: UInt64) -> [Double] {
    var state = seed
    return (0..<n).map { _ in
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Double(state >> 11) / Double(1 << 53) - 0.5
    }
}

/// Sizes that take every plan: 2D radix-2 (64 × 64, 16 × 32), the small-power-of-two exception (8 × 32, 4 × 8), radix-3
/// (27, 9), radix-5 (25), the vDSP DFT (48 = 3·16, 40 = 5·8, 30 = 15·2), Bluestein (7, 13, 50, 31), and length 1.
private let sizes: [(Int, Int)] = [(64, 64), (16, 32), (8, 32), (4, 8), (27, 9), (25, 5), (48, 40), (30, 7), (13, 50),
                                   (31, 1), (1, 1)]

@Test(arguments: sizes.indices)
func doubleFFTEqualsTheDirectDFT(_ k: Int) throws {
    let (nx, ny) = sizes[k]
    let fft = try #require(FFT2DDouble(nx: nx, ny: ny))
    let re0 = signal(nx * ny, seed: UInt64(k + 1)), im0 = signal(nx * ny, seed: UInt64(k + 101))
    var re = re0, im = im0
    fft.transform(re: &re, im: &im, forward: true)
    let (wantRe, wantIm) = directDFT(re0, im0, nx: nx, ny: ny)
    let top = zip(wantRe, wantIm).map { ($0 * $0 + $1 * $1).squareRoot() }.max()!
    let worst = largestDistance(re, im, wantRe, wantIm)
    #expect(worst <= 1e-12 * top, "\(nx) × \(ny): \(worst / top) of the maximum")
    // The inverse returns the input, scaled by 1/(nx·ny).
    fft.transform(re: &re, im: &im, forward: false, scaleInverse: true)
    let back = (0..<(nx * ny)).map { max(abs(re[$0] - re0[$0]), abs(im[$0] - im0[$0])) }.max()!
    #expect(back <= 1e-13, "\(nx) × \(ny) round trip: \(back)")
}

@Test(arguments: sizes.indices)
func floatFFTIsWithinItsBandOfTheDirectDFT(_ k: Int) throws {
    let (nx, ny) = sizes[k]
    let fft = try #require(FFT2D(nx: nx, ny: ny))
    let re0 = signal(nx * ny, seed: UInt64(k + 1)), im0 = signal(nx * ny, seed: UInt64(k + 101))
    var re = re0.map { Float($0) }, im = im0.map { Float($0) }
    fft.transform(re: &re, im: &im, forward: true)
    let (wantRe, wantIm) = directDFT(re.indices.map { Double(Float(re0[$0])) }, im.indices.map { Double(Float(im0[$0])) },
                                     nx: nx, ny: ny)
    let top = zip(wantRe, wantIm).map { ($0 * $0 + $1 * $1).squareRoot() }.max()!
    let worst = (0..<(nx * ny)).map { max(abs(Double(re[$0]) - wantRe[$0]), abs(Double(im[$0]) - wantIm[$0])) }.max()!
    #expect(worst <= 1e-5 * top, "\(nx) × \(ny): \(worst / top) of the maximum")
}

@Test func fftfreqAndNextPow2() {
    #expect((0..<5).map { FFT2D.fftfreq($0, 5) } == [0, 0.2, 0.4, -0.4, -0.2])
    #expect((0..<4).map { FFT2D.fftfreq($0, 4) } == [0, 0.25, -0.5, -0.25])
    #expect(FFT2D.nextPow2(1) == 1 && FFT2D.nextPow2(17) == 32 && FFT2D.nextPow2(32) == 32)
    #expect(FFT2D(nx: 0, ny: 4) == nil && FFT2DDouble(nx: 4, ny: 0) == nil)
}

@Test func micrographRefusesWrongSizes() {
    #expect(throws: MicrographError.emptySize(width: 0, height: 3)) { try Micrograph(width: 0, height: 3, pixels: []) }
    #expect(throws: MicrographError.pixelCount(expected: 6, actual: 5)) {
        try Micrograph(width: 2, height: 3, pixels: [0, 0, 0, 0, 0])
    }
    #expect(throws: MicrographError.pixelCount(expected: Int.max, actual: 1)) {
        try Micrograph(width: Int.max, height: 2, pixels: [0])
    }
}

@Test func cropKeepsThePhysicalPosition() throws {
    let image = try Micrograph(width: 4, height: 3, pixels: (0..<12).map(Float.init),
                               x: AxisCalibration(scale: 0.5, unit: "nm", origin: 1), y: AxisCalibration(scale: 2, unit: "nm"))
    let c = try #require(image.cropped(PixelCrop(row: 1, col: 2, height: 2, width: 2)))
    #expect(c.pixels == [6, 7, 10, 11])
    #expect(c.x?.value(at: 0) == image.x?.value(at: 2) && c.y?.value(at: 1) == image.y?.value(at: 2))
    #expect(image.cropped(PixelCrop(row: 2, col: 0, height: 2, width: 1)) == nil)
    #expect(image.cropped(PixelCrop(row: 0, col: 0, height: 0, width: 1)) == nil)
    #expect(image.cropped(PixelCrop(row: -1, col: 0, height: 1, width: 1)) == nil)
}

@Test func lengthUnits() {
    #expect(LengthUnit.nanometres(per: "nm") == 1 && LengthUnit.nanometres(per: " NM ") == 1)
    #expect(LengthUnit.nanometres(per: "Å") == 0.1 && LengthUnit.nanometres(per: "A\u{030A}") == 0.1)   // decomposed Å
    #expect(LengthUnit.nanometres(per: "\u{00B5}m") == 1e3 && LengthUnit.nanometres(per: "\u{03BC}m") == 1e3)
    #expect(LengthUnit.nanometres(per: "pm") == 1e-3 && LengthUnit.nanometres(per: "m") == 1e9)
    #expect(LengthUnit.nanometres(per: "px") == nil && LengthUnit.nanometres(per: "1/nm") == nil)
    #expect(LengthUnit.frequencySampling(AxisCalibration(scale: 0.2, unit: "nm")) == (0.2, "1/nm"))
    #expect(LengthUnit.frequencySampling(AxisCalibration(scale: 0.2, unit: "1/nm")) == (0.2, "nm"))
    #expect(LengthUnit.frequencySampling(AxisCalibration(scale: 1, unit: "")) == (1, "1/px"))
    #expect(LengthUnit.frequencySampling(nil) == (1, "1/px"))
    #expect(LengthUnit.realSampling(AxisCalibration(scale: -1, unit: "nm")) == (1, "px"))
}

/// At roundoff on one axis (added after the slice-1a mutants, amendment 6): a prime Bluestein length and radix-5 lengths,
/// where the registered 1e-12 is too loose to see a chirp without its mod-2N reduction (3e-13 at 1009) or a radix-5
/// constant off by 7e-15. Measured unmutated: 1.9e-15 (1009), 2.5e-16 (5), 6.2e-16 (125).
@Test(arguments: [(1009, 1e-14), (5, 2e-15), (125, 2e-15)])
func doubleFFTIsAtRoundoffOnOneAxis(_ n: Int, _ band: Double) throws {
    let fft = try #require(FFT2DDouble(nx: n, ny: 1))
    let re0 = signal(n, seed: UInt64(n)), im0 = signal(n, seed: UInt64(n + 7))
    var re = re0, im = im0
    fft.transform(re: &re, im: &im, forward: true)
    let (wantRe, wantIm) = directDFT(re0, im0, nx: n, ny: 1)
    let top = zip(wantRe, wantIm).map { ($0 * $0 + $1 * $1).squareRoot() }.max()!
    let worst = largestDistance(re, im, wantRe, wantIm)
    #expect(worst <= band * top, "\(n): \(worst / top) of the maximum, band \(band)")
}

@Test func micrographRefusesASizeThatWrapsToItsPixelCount() {
    // 2³² × 2³² wraps to 0 in Int; without the overflow guard it would match an empty pixel array.
    #expect(throws: MicrographError.pixelCount(expected: Int.max, actual: 0)) {
        try Micrograph(width: 1 << 32, height: 1 << 32, pixels: [])
    }
}
