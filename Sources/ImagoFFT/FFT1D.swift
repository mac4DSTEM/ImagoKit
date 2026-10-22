import Accelerate
import ImagoCore

/// An in-place, radix-2 complex Fourier transform in Float precision.
///
/// Both directions are unnormalised: forward uses exp(-2πikj/n), inverse uses
/// exp(+2πikj/n). A forward/inverse round trip multiplies the input by `n`.
/// Divide inverse output by `n` for NumPy's default `ifft` convention.
/// No padding, windowing or frequency centring is applied.
///
/// The vDSP setup is immutable and shareable across threads. Each call must
/// own its real and imaginary buffers; the plan stores no mutable scratch space.
public nonisolated final class FFT1D: @unchecked Sendable {
    /// Number of complex samples, a positive power of two.
    public let n: Int
    private let log2n: vDSP_Length
    private let setup: FFTSetup
    /// For `rfft` / `irfft`: the twiddles exp(−2πik/n), k = 0 … n/2.
    private let twiddleRe: [Float]
    private let twiddleIm: [Float]

    /// Creates a plan, or returns `nil` for a nonpositive length, a length that
    /// is not a power of two, or a vDSP setup allocation failure.
    /// A length of one is supported as the identity transform.
    public init?(n: Int) {
        guard n > 0, n & (n - 1) == 0 else { return nil }
        self.n = n
        self.log2n = vDSP_Length(n.trailingZeroBitCount)
        guard let setup = vDSP_create_fftsetup(max(1, log2n), FFTRadix(kFFTRadix2)) else { return nil }
        self.setup = setup
        var wr = [Float](repeating: 1, count: n / 2 + 1), wi = [Float](repeating: 0, count: n / 2 + 1)
        for k in 0...(n / 2) {
            let angle = -2 * Double.pi * Double(k) / Double(n)
            wr[k] = Float(cos(angle)); wi[k] = Float(sin(angle))
        }
        if n >= 2 { wr[n / 2] = -1; wi[n / 2] = 0 }
        twiddleRe = wr
        twiddleIm = wi
    }

    deinit { vDSP_destroy_fftsetup(setup) }

    /// Transforms split-complex samples in place, with no scaling in either direction.
    ///
    /// - Parameters:
    ///   - re: Real components, exactly `n` elements.
    ///   - im: Imaginary components, exactly `n` elements.
    ///   - forward: `true` for the forward transform, `false` for the inverse.
    /// - Precondition: Both arrays contain exactly `n` elements.
    public func transform(re: inout [Float], im: inout [Float], forward: Bool) {
        precondition(re.count == n && im.count == n, "FFT buffers must match the plan length")
        guard n > 1 else { return }
        re.withUnsafeMutableBufferPointer { rp in
            im.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                vDSP_fft_zip(setup, &split, 1, log2n,
                                  FFTDirection(forward ? kFFTDirection_Forward : kFFTDirection_Inverse))
            }
        }
    }

    // MARK: real-input transforms (numpy.fft.rfft / irfft)

    /// `numpy.fft.rfft(x, n: self.n, norm:)`: the `n / 2 + 1` bins of the transform of a real signal, as (re, im). `x` is cut
    /// or padded with zeros to `n`, as NumPy's `n=` does. Computed with one complex transform of length `n / 2` on the even
    /// and odd samples plus a split step, half the work of the complex transform.
    public func rfft(_ x: [Float], norm: FFTNorm = .backward) -> (re: [Float], im: [Float]) {
        let m = n / 2
        let scale = Self.forwardScale(norm, n)
        guard n > 1 else { return ([(x.first ?? 0) * scale], [0]) }
        var zr = [Float](repeating: 0, count: m), zi = zr
        for j in 0..<m {
            let even = 2 * j, odd = even + 1
            if even < x.count { zr[j] = x[even] }
            if odd < x.count { zi[j] = x[odd] }
        }
        halfTransform(re: &zr, im: &zi, forward: true)
        var re = [Float](repeating: 0, count: m + 1), im = re
        for k in 0...m {
            let a = k % m, b = (m - k) % m                            // Z[k] and conj Z[m − k] (indices mod m)
            let zre = zr[a], zim = zi[a], cre = zr[b], cim = -zi[b]
            let er = (zre + cre) * 0.5, ei = (zim + cim) * 0.5        // E = DFT of the even samples
            let or = (zim - cim) * 0.5, oi = (cre - zre) * 0.5        // O = (Z − conj Z′) / 2i, the odd samples
            let wr = twiddleRe[k], wi = twiddleIm[k]
            re[k] = (er + wr * or - wi * oi) * scale
            im[k] = (ei + wr * oi + wi * or) * scale
        }
        return (re, im)
    }

    /// `numpy.fft.irfft(X, n: self.n, norm:)`: the `n` real samples whose transform has the bins (`re`, `im`). The bins are cut
    /// or padded with zeros to `n / 2 + 1`; the imaginary parts of bin 0 and of the Nyquist bin `n / 2` do not count.
    public func irfft(re: [Float], im: [Float], norm: FFTNorm = .backward) -> [Float] {
        let m = n / 2
        let scale = Self.inverseScale(norm, n)
        guard n > 1 else { return [(re.first ?? 0) * scale] }
        precondition(re.count == im.count, "irfft: real and imaginary parts must have one length")
        func bin(_ k: Int) -> (Float, Float) {
            guard k < re.count else { return (0, 0) }
            return (re[k], k == 0 || k == m ? 0 : im[k])
        }
        var zr = [Float](repeating: 0, count: m), zi = zr
        for k in 0..<m {
            let (xr, xi) = bin(k), (yr, yi) = bin(m - k)              // X[k] and X[m − k]
            let er = (xr + yr) * 0.5, ei = (xi - yi) * 0.5            // E = (X[k] + conj X[m − k]) / 2
            let dr = (xr - yr) * 0.5, di = (xi + yi) * 0.5            // D = (X[k] − conj X[m − k]) / 2 = w O
            let wr = twiddleRe[k], wi = -twiddleIm[k]                  // conj w
            let orr = dr * wr - di * wi, oi = dr * wi + di * wr        // O = D · conj w
            zr[k] = er - oi
            zi[k] = ei + orr
        }
        halfTransform(re: &zr, im: &zi, forward: false)
        var out = [Float](repeating: 0, count: n)
        for j in 0..<m { out[2 * j] = zr[j] * scale; out[2 * j + 1] = zi[j] * scale }
        return out
    }

    /// `numpy.fft.rfft(x, n, norm)` with a plan built for `n` (default `x.count`); nil when `n` is not a positive power of two.
    public static func rfft(_ x: [Float], n: Int? = nil, norm: FFTNorm = .backward) -> (re: [Float], im: [Float])? {
        guard let plan = FFT1D(n: n ?? x.count) else { return nil }
        return plan.rfft(x, norm: norm)
    }

    /// `numpy.fft.irfft(X, n, norm)` with a plan built for `n` (default `2 · (bins − 1)`); nil when `n` is not a positive power
    /// of two or the two parts differ in length.
    public static func irfft(re: [Float], im: [Float], n: Int? = nil, norm: FFTNorm = .backward) -> [Float]? {
        guard re.count == im.count else { return nil }
        guard let plan = FFT1D(n: n ?? 2 * (re.count - 1)) else { return nil }
        return plan.irfft(re: re, im: im, norm: norm)
    }

    private static func forwardScale(_ norm: FFTNorm, _ n: Int) -> Float {
        switch norm {
        case .backward: return 1
        case .forward: return 1 / Float(n)
        case .ortho: return 1 / Float(n).squareRoot()
        }
    }

    /// The factor on the unnormalised length-`n / 2` inverse that gives NumPy's irfft: n · (1/n, 1, 1/√n) / (n / 2).
    private static func inverseScale(_ norm: FFTNorm, _ n: Int) -> Float {
        switch norm {
        case .backward: return n > 1 ? 2 / Float(n) : 1
        case .forward: return n > 1 ? 2 : 1
        case .ortho: return n > 1 ? 2 / Float(n).squareRoot() : 1
        }
    }

    /// The complex transform of length `n / 2` of the packed even/odd samples, in place.
    private func halfTransform(re: inout [Float], im: inout [Float], forward: Bool) {
        let m = n / 2
        guard m > 1 else { return }
        re.withUnsafeMutableBufferPointer { rp in
            im.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                vDSP_fft_zip(setup, &split, 1, log2n - 1,
                              FFTDirection(forward ? kFFTDirection_Forward : kFFTDirection_Inverse))
            }
        }
    }

    /// Smallest power of two at least `v`; returns one when `v <= 1`.
    ///
    /// - Precondition: `v` is at most the largest positive power of two
    ///   representable by `Int` (1 << (Int.bitWidth - 2)).
    public static func nextPow2(_ v: Int) -> Int {
        precondition(v <= 1 << (Int.bitWidth - 2), "Next power of two would overflow Int")
        guard v > 1 else { return 1 }
        return 1 << (Int.bitWidth - (v - 1).leadingZeroBitCount)
    }
}
