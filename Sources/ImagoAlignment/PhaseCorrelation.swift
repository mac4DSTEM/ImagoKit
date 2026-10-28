//
//  PhaseCorrelation.swift (ImagoAlignment)
//  The translation that registers one image onto another: scikit-image 0.26.0's `phase_cross_correlation` without masks and
//  without `disambiguate`, on the Float32 FFT (`FFT2D`) or the Float64 reference path.
//
//  THE STEPS, as scikit-image takes them: P = F_R · conj(F_M); with `.phase`, P /= max(|P|, 100·eps); CC = ifft2(P); the first
//  maximum of |CC| in row-major order; a peak index above trunc(n/2) has n subtracted. With an upsample factor u > 1 the
//  peak is refined on the 1/u grid in a region of ceil(1.5·u) samples per axis around it, by the matrix-multiply DFT of
//  Guizar-Sicairos, Thurman and Fienup (Opt. Lett. 33, 156, 2008): two small complex matrix products, no zero padding. All
//  of it runs in the chosen precision's scalar, as scikit-image computes in its input's dtype (the kernels are evaluated
//  in Double and rounded to the scalar).
//

import Foundation
import ImagoCore

public nonisolated enum PhaseCorrelation {
    /// The largest upsample factor accepted: the refinement region holds ceil(1.5·u)² samples per image.
    public static let maximumUpsampleFactor = 1024

    /// The translation that registers `moving` onto `reference` (see `ImageRegistration.shift`). The images must have one
    /// size and hold finite pixels. Calibrations are ignored: the shift is in pixels.
    public static func register(reference: Micrograph, moving: Micrograph, upsampleFactor: Int = 1,
                                normalization: Normalization = .phase,
                                precision: Precision = .float32) throws -> ImageRegistration {
        try checkUpsample(upsampleFactor)
        try checkPair(reference, moving)
        switch precision {
        case .float32:
            guard let fft = FFT2D(nx: reference.width, ny: reference.height) else {
                throw AlignmentError.fftSetup(width: reference.width, height: reference.height)
            }
            return register(fft, reference.pixels, moving.pixels, reference.width, reference.height,
                            upsampleFactor, normalization, precision)
        case .float64:
            guard let fft = FFT2DDouble(nx: reference.width, ny: reference.height) else {
                throw AlignmentError.fftSetup(width: reference.width, height: reference.height)
            }
            return register(fft, reference.pixels.map(Double.init), moving.pixels.map(Double.init), reference.width,
                            reference.height, upsampleFactor, normalization, precision)
        }
    }

    /// The refinement region's samples per axis, ceil(1.5·u).
    package static func refinementRegion(for u: Int) -> Int { Int((Double(u) * 1.5).rounded(.up)) }

    /// The region's centre sample, trunc(S/2): the sample that is the integer-stage peak's own position.
    package static func refinementCentre(for u: Int) -> Int { refinementRegion(for: u) / 2 }

    static func checkUpsample(_ u: Int) throws {
        guard u >= 1, u <= maximumUpsampleFactor else { throw AlignmentError.upsampleFactor(u) }
    }

    static func checkPair(_ reference: Micrograph, _ moving: Micrograph) throws {
        guard reference.width == moving.width, reference.height == moving.height else {
            throw AlignmentError.sizeMismatch(index: 1, width: moving.width, height: moving.height,
                                              expectedWidth: reference.width, expectedHeight: reference.height)
        }
        guard reference.pixels.allSatisfy(\.isFinite) else { throw AlignmentError.nonFinitePixel(index: 0) }
        guard moving.pixels.allSatisfy(\.isFinite) else { throw AlignmentError.nonFinitePixel(index: 1) }
    }

    private static func register<T: LineTransformPlan>(_ fft: T, _ reference: [T.Scalar], _ moving: [T.Scalar], _ width: Int,
                                                      _ height: Int, _ u: Int, _ normalization: Normalization,
                                                      _ precision: Precision) -> ImageRegistration {
        let a = Spectrum.of(reference, fft)
        let b = Spectrum.of(moving, fft)
        return correlate(fft, a, b, width, height, u, normalization, precision)
    }

    /// The registration of spectrum `b` (moving) onto spectrum `a` (reference), both of the engine `fft`. The spectra are the
    /// half spectra of real images (`height × (width/2 + 1)`), so P is Hermitian and CC = ifft2(P) is real: one
    /// complex-to-real inverse replaces the complex one (the imaginary part of scikit-image's CC is rounding noise, at 1e-16
    /// of the peak; the phase difference of a registered pair, atan2(noise, positive), is 0 there and here).
    package static func correlate<T: LineTransformPlan>(_ fft: T, _ a: Spectrum<T.Scalar>, _ b: Spectrum<T.Scalar>,
                                                      _ width: Int, _ height: Int, _ u: Int,
                                                      _ normalization: Normalization,
                                                      _ precision: Precision) -> ImageRegistration {
        typealias S = T.Scalar
        let count = width * height
        let half = width / 2 + 1
        let halfCount = height * half
        var pr = [S](repeating: 0, count: halfCount)
        var pi = [S](repeating: 0, count: halfCount)
        for i in 0..<halfCount {                                // P = F_R · conj(F_M)
            let ar = a.re[i], ai = a.im[i], br = b.re[i], bi = b.im[i]
            pr[i] = ar * br + ai * bi
            pi[i] = ai * br - ar * bi
        }
        if normalization == .phase {
            let floor = Double(S.ulpOfOne) * 100
            for i in 0..<halfCount {
                let magnitude = hypot(Double(pr[i]), Double(pi[i]))
                let divisor = S(max(magnitude, floor))
                pr[i] /= divisor
                pi[i] /= divisor
            }
        }
        var cr = fft.realInverse(re: pr, im: pi, inputColumns: half)
        let inverseScale = 1 / S(count)
        for i in 0..<count { cr[i] *= inverseScale }
        let peak = firstMaximum(cr)
        let peakRow = peak / width, peakColumn = peak % width
        var rowShift: S = wrapped(peakRow, height)
        var columnShift: S = wrapped(peakColumn, width)
        let amplitudeA: Double, amplitudeB: Double
        let ccMax: (re: S, im: S)
        if u == 1 {
            let size = Double(count)
            amplitudeA = a.power / size
            amplitudeB = b.power / size
            ccMax = (cr[peak], 0)
        } else {
            let factor = S(u)
            rowShift = (rowShift * factor).rounded(.toNearestOrEven) / factor
            columnShift = (columnShift * factor).rounded(.toNearestOrEven) / factor
            let region = refinementRegion(for: u)
            let dftShift = S(refinementCentre(for: u))
            let rowOffset = dftShift - rowShift * factor
            let columnOffset = dftShift - columnShift * factor
            // The matrix-multiply DFT sums over every column of P: complete the half spectrum by P[r, c] = conj P[−r, −c].
            let (fullRe, fullIm) = completed(pr, pi, width, height)
            let (rr, ri) = upsampledDFT(fullRe, fullIm, width, height, region, u, Double(rowOffset), Double(columnOffset))
            let refined = firstMaximum(rr, ri)
            let maxRow = S(refined / region) - dftShift, maxColumn = S(refined % region) - dftShift
            rowShift += maxRow / factor
            columnShift += maxColumn / factor
            amplitudeA = a.power
            amplitudeB = b.power
            ccMax = (rr[refined], ri[refined])
        }
        if height == 1 { rowShift = 0 }
        if width == 1 { columnShift = 0 }
        let re = Double(ccMax.re), im = Double(ccMax.im)
        let amplitude = amplitudeA * amplitudeB
        let ratio = (re * re + im * im) / amplitude
        return ImageRegistration(shift: PixelShift(row: Double(rowShift), column: Double(columnShift)),
                                 error: abs(1 - ratio).squareRoot(), phaseDifference: atan2(im, re),
                                 upsampleFactor: u, normalization: normalization, precision: precision)
    }

    /// The first index of the largest magnitude of real data (NumPy's `argmax(abs(z))` for a real `z`).
    private static func firstMaximum<S: BinaryFloatingPoint>(_ re: [S]) -> Int {
        var best = 0
        var bestValue = -1.0
        for i in 0..<re.count {
            let r = Double(re[i])
            let m = r * r
            if m > bestValue {
                bestValue = m
                best = i
            }
        }
        return best
    }

    /// The full `height × width` spectrum of a real image from its half (`height × (width/2 + 1)`):
    /// F[r, c] = conj F[(−r) mod h, w − c] for the columns the half does not hold.
    package static func completed<S: BinaryFloatingPoint>(_ re: [S], _ im: [S], _ width: Int, _ height: Int) -> ([S], [S]) {
        let m = width / 2 + 1
        var fullRe = [S](repeating: 0, count: height * width), fullIm = fullRe
        for r in 0..<height {
            let rowFrom = r * m, rowTo = r * width
            for c in 0..<m { fullRe[rowTo + c] = re[rowFrom + c]; fullIm[rowTo + c] = im[rowFrom + c] }
            let mirror = ((height - r) % height) * m
            if m < width { for c in m..<width {
                fullRe[rowTo + c] = re[mirror + width - c]
                fullIm[rowTo + c] = -im[mirror + width - c]
            } }
        }
        return (fullRe, fullIm)
    }

    /// NumPy's `argmax(abs(z))` over row-major data: the first index of the largest magnitude.
    private static func firstMaximum<S: BinaryFloatingPoint>(_ re: [S], _ im: [S]) -> Int {
        var best = 0
        var bestValue = -1.0
        for i in 0..<re.count {
            let r = Double(re[i]), q = Double(im[i])
            let m = r * r + q * q
            if m > bestValue {
                bestValue = m
                best = i
            }
        }
        return best
    }

    /// A peak index above trunc(n/2) has n subtracted, so the shift lies in (−n/2, n/2].
    private static func wrapped<S: BinaryFloatingPoint>(_ index: Int, _ n: Int) -> S {
        index > n / 2 ? S(index - n) : S(index)
    }

    /// The matrix-multiply DFT: `conj(upsampledDFT(conj(P), region, u, offsets))`, the `region × region` samples (rows, then
    /// columns) of the cross-correlation around the shift whose offsets are given. Axis by axis from the last, with the
    /// kernel `exp(−2πi (k − offset) · fftfreq(n, u))`.
    private static func upsampledDFT<S: BinaryFloatingPoint>(_ pr: [S], _ pi: [S], _ width: Int, _ height: Int,
                                                              _ region: Int, _ u: Int, _ rowOffset: Double,
                                                              _ columnOffset: Double) -> ([S], [S]) {
        let (kcRe, kcIm): ([S], [S]) = kernel(region, width, u, columnOffset)
        let (krRe, krIm): ([S], [S]) = kernel(region, height, u, rowOffset)
        // T[k, i] = Σ_j Kc[k, j] · conj(P)[i, j]
        var tRe = [S](repeating: 0, count: region * height)
        var tIm = [S](repeating: 0, count: region * height)
        pr.withUnsafeBufferPointer { dr in
            pi.withUnsafeBufferPointer { di in
                kcRe.withUnsafeBufferPointer { kr in
                    kcIm.withUnsafeBufferPointer { ki in
                        for k in 0..<region {
                            for i in 0..<height {
                                var sr: S = 0, si: S = 0
                                let kBase = k * width, dBase = i * width
                                for j in 0..<width {
                                    let xr = kr[kBase + j], xi = ki[kBase + j]
                                    let yr = dr[dBase + j], yi = -di[dBase + j]
                                    sr += xr * yr - xi * yi
                                    si += xr * yi + xi * yr
                                }
                                tRe[k * height + i] = sr
                                tIm[k * height + i] = si
                            }
                        }
                    }
                }
            }
        }
        // out[l, k] = Σ_i Kr[l, i] · T[k, i], then conjugated
        var outRe = [S](repeating: 0, count: region * region)
        var outIm = [S](repeating: 0, count: region * region)
        for l in 0..<region {
            for k in 0..<region {
                var sr: S = 0, si: S = 0
                for i in 0..<height {
                    let xr = krRe[l * height + i], xi = krIm[l * height + i]
                    let yr = tRe[k * height + i], yi = tIm[k * height + i]
                    sr += xr * yr - xi * yi
                    si += xr * yi + xi * yr
                }
                outRe[l * region + k] = sr
                outIm[l * region + k] = -si
            }
        }
        return (outRe, outIm)
    }

    /// The `region × n` kernel `exp(−2πi (k − offset) · fftfreq(n, u))`, evaluated in Double and rounded to the scalar.
    private static func kernel<S: BinaryFloatingPoint>(_ region: Int, _ n: Int, _ u: Int, _ offset: Double) -> ([S], [S]) {
        let step = 1.0 / (Double(n) * Double(u))
        var re = [S](repeating: 0, count: region * n)
        var im = [S](repeating: 0, count: region * n)
        for j in 0..<n {
            let signed = j <= (n - 1) / 2 ? j : j - n
            let frequency = Double(signed) * step
            for k in 0..<region {
                let angle = -(2.0 * Double.pi) * ((Double(k) - offset) * frequency)
                re[k * n + j] = S(cos(angle))
                im[k * n + j] = S(sin(angle))
            }
        }
        return (re, im)
    }
}

/// The forward transform of a real image: its half spectrum (`height × (width/2 + 1)`, split complex) and `Σ|F|²` of the
/// full spectrum.
package nonisolated struct Spectrum<S: BinaryFloatingPoint & Sendable>: Sendable {
    package let re: [S]
    package let im: [S]
    package let power: Double

    /// One real-input transform. The power counts the columns the half omits through their mirrors: columns 1 … w − m
    /// of the half appear twice in the full spectrum (columns 0 and, for even w, w/2 once).
    package static func of<T: LineTransformPlan>(_ pixels: [S], _ fft: T) -> Spectrum<S> where T.Scalar == S {
        let h = fft.ny, w = fft.nx, m = w / 2 + 1
        let half = fft.realForward(pixels)
        var power = 0.0
        for r in 0..<h {
            for c in 0..<m {
                let rr = Double(half.re[r * m + c]), qq = Double(half.im[r * m + c])
                let weight = (c >= 1 && c <= w - m) ? 2.0 : 1.0
                power += weight * (rr * rr + qq * qq)
            }
        }
        return Spectrum(re: half.re, im: half.im, power: power)
    }
}
