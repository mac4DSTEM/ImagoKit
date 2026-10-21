import Accelerate
import ImagoCore

/// An in-place, radix-2 complex Fourier transform in Double precision.
///
/// Both directions are unnormalised: forward uses exp(-2πikj/n), inverse uses
/// exp(+2πikj/n). A forward/inverse round trip multiplies the input by `n`.
/// Divide inverse output by `n` for NumPy's default `ifft` convention.
/// No padding, windowing or frequency centring is applied.
///
/// The vDSP setup is immutable and shareable across threads. Each call must
/// own its real and imaginary buffers; the plan stores no mutable scratch space.
public nonisolated final class FFT1DDouble: @unchecked Sendable {
    /// Number of complex samples, a positive power of two.
    public let n: Int
    private let log2n: vDSP_Length
    private let setup: FFTSetupD
    /// `vDSP_DFT_zop` is 5 to 7 times faster than the radix-2 Double call from 128 points up; `setup` is the fallback.
    private let pair: DFTPairD?

    /// Creates a plan, or returns `nil` for a nonpositive length, a length that
    /// is not a power of two, or a vDSP setup allocation failure.
    /// A length of one is supported as the identity transform.
    public init?(n: Int) {
        guard n > 0, n & (n - 1) == 0 else { return nil }
        self.n = n
        self.log2n = vDSP_Length(n.trailingZeroBitCount)
        guard let setup = vDSP_create_fftsetupD(max(1, log2n), FFTRadix(kFFTRadix2)) else { return nil }
        self.setup = setup
        self.pair = DFTPairD(length: n)
    }

    deinit { vDSP_destroy_fftsetupD(setup) }

    /// Transforms split-complex samples in place, with no scaling in either direction.
    ///
    /// - Parameters:
    ///   - re: Real components, exactly `n` elements.
    ///   - im: Imaginary components, exactly `n` elements.
    ///   - forward: `true` for the forward transform, `false` for the inverse.
    /// - Precondition: Both arrays contain exactly `n` elements.
    public func transform(re: inout [Double], im: inout [Double], forward: Bool) {
        precondition(re.count == n && im.count == n, "FFT buffers must match the plan length")
        guard n > 1 else { return }
        if let pair {
            var outRe = [Double](repeating: 0, count: n), outIm = [Double](repeating: 0, count: n)
            re.withUnsafeBufferPointer { r in im.withUnsafeBufferPointer { i in
                outRe.withUnsafeMutableBufferPointer { or in outIm.withUnsafeMutableBufferPointer { oi in
                    pair.execute(forward: forward, r.baseAddress!, i.baseAddress!, or.baseAddress!, oi.baseAddress!)
                } }
            } }
            swap(&re, &outRe)
            swap(&im, &outIm)
            return
        }
        re.withUnsafeMutableBufferPointer { rp in
            im.withUnsafeMutableBufferPointer { ip in
                var split = DSPDoubleSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                vDSP_fft_zipD(setup, &split, 1, log2n,
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
