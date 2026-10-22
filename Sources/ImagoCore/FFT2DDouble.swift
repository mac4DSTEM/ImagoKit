//
//  FFT2DDouble.swift (ImagoCore)
//  The reference path (registration-slice-1.md 1.1, amendment 2): FFT2D.swift translated line by line to Double — the vDSP
//  `…D` calls, DSPDoubleSplitComplex, Double twiddles — with the same plans and the same small-power-of-two exception.
//  A change to FFT2D.swift is made here too, in the same commit. One exception, on purpose: the radix-2 Double calls
//  (`vDSP_fft_zipD`, `vDSP_fft2d_zipD`) are 5 to 7 times slower than `vDSP_DFT_zop` from 128² up, so this file runs
//  every power-of-two axis, and Bluestein's padded transforms, through `DFTPairD` and keeps the radix-2 call only as the
//  fallback when vDSP has no DFT plan (agreement with the radix-2 path 1.6e-15, machine precision).
//

import Foundation
import Accelerate

nonisolated private struct ExactRadixStageD: Sendable {
    package let groupSize: Int
    package let butterflyStride: Int
    package let twiddleReal: [Double]
    package let twiddleImaginaryForward: [Double]
}

nonisolated private struct ExactRadixPlanD: Sendable {
    package let length: Int
    package let radix: Int
    package let reversedIndices: [Int]
    package let stages: [ExactRadixStageD]

    package init(length: Int, radix: Int, exponent: Int) {
        self.length = length
        self.radix = radix

        var reversed = [Int](repeating: 0, count: length)
        for index in 0..<length {
            var source = index
            var destination = 0
            for _ in 0..<exponent {
                destination = destination * radix + source % radix
                source /= radix
            }
            reversed[index] = destination
        }
        reversedIndices = reversed

        var builtStages: [ExactRadixStageD] = []
        var groupSize = radix
        for _ in 0..<exponent {
            let stride = groupSize / radix
            var twiddleReal = [Double](repeating: 1, count: radix * stride)
            var twiddleImaginary = [Double](repeating: 0, count: radix * stride)
            for branch in 1..<radix {
                for offset in 0..<stride {
                    let angle = -2 * Double.pi * Double(branch * offset) / Double(groupSize)
                    let index = branch * stride + offset
                    twiddleReal[index] = cos(angle)
                    twiddleImaginary[index] = sin(angle)
                }
            }
            builtStages.append(ExactRadixStageD(
                groupSize: groupSize,
                butterflyStride: stride,
                twiddleReal: twiddleReal,
                twiddleImaginaryForward: twiddleImaginary
            ))
            groupSize *= radix
        }
        stages = builtStages
    }
}

/// Bluestein / chirp-z plan for one axis of arbitrary length N.
///
/// Identity: nk = (n² + k² − (k−n)²)/2, so
///   X_k = w_k · Σ_n (x_n · w_n) · conj(w_{k−n}),  w_m = exp(−iπ m²/N),
/// a linear convolution of the chirped input with conj(w), which a length-M
/// circular convolution (M ≥ 2N−1, power of two) computes exactly through
/// vDSP's radix-2 FFT. The inverse transform uses conj(w) throughout and is
/// left unnormalized, matching vDSP's convention that `FFT2D.transform`
/// already relies on. The padding is internal to evaluating the exact
/// N-point DFT — the data itself is never padded.
///
/// The chirp phase is built from n² reduced modulo 2N, so its accuracy does
/// not degrade with N (a Float angle of 2π·n·k/N, as the old scalar loop
/// used, loses ~1e-4 rad by n·k ≈ 60 000). The reduction is what matters:
/// A Float chirp with the same reduction was measured at the same accuracy;
/// the Double is belt-and-braces.
///
/// Immutable after construction (an FFTSetup is documented shareable across
/// threads), so one plan serves every concurrent worker.
nonisolated private final class BluesteinPlanD: @unchecked Sendable {
    package let length: Int
    package let paddedLength: Int
    package let paddedLog2: vDSP_Length
    package let setup: FFTSetupD
    /// The padded transform on `vDSP_DFT_zop`, when vDSP has a plan for `paddedLength`; `setup` serves the filter spectra.
    package let padded: DFTPairD?
    /// w_n = exp(−iπ n²/N) for n < N (the FORWARD chirp; inverse = conj).
    package let chirpRe: [Double]
    package let chirpIm: [Double]
    /// FFT_M of the symmetric conj-chirp filter, one per direction.
    package let filterForwardRe: [Double]
    package let filterForwardIm: [Double]
    package let filterInverseRe: [Double]
    package let filterInverseIm: [Double]

    package init?(length: Int) {
        guard length > 0 else { return nil }
        let padded = FFT2DDouble.nextPow2(2 * length - 1)
        let log2Padded = vDSP_Length(log2(Double(padded)))
        guard let setup = vDSP_create_fftsetupD(log2Padded, FFTRadix(kFFTRadix2)) else {
            return nil
        }
        self.length = length
        self.paddedLength = padded
        self.paddedLog2 = log2Padded
        self.setup = setup
        self.padded = DFTPairD(length: padded)

        var wRe = [Double](repeating: 0, count: length)
        var wIm = [Double](repeating: 0, count: length)
        for n in 0..<length {
            // exp(−iπ n²/N) has period 2N in n², so reduce first — exact in
            // integer arithmetic, and it keeps the Double angle small.
            let reduced = (n * n) % (2 * length)
            let angle = -Double.pi * Double(reduced) / Double(length)
            wRe[n] = Double(cos(angle))
            wIm[n] = Double(sin(angle))
        }
        chirpRe = wRe
        chirpIm = wIm

        // Filter for a direction whose chirp is c: b_m = conj(c_m) for
        // |m| < N, placed at index m (m ≥ 0) and M + m (m < 0); symmetric in
        // m because m² is. Forward: c = w → b = conj(w). Inverse: c = conj(w)
        // → b = w.
        func filter(conjugateChirp: Bool) -> ([Double], [Double]) {
            var bRe = [Double](repeating: 0, count: padded)
            var bIm = [Double](repeating: 0, count: padded)
            let sign: Double = conjugateChirp ? -1 : 1
            for m in 0..<length {
                bRe[m] = wRe[m]
                bIm[m] = sign * wIm[m]
                if m > 0 {
                    bRe[padded - m] = wRe[m]
                    bIm[padded - m] = sign * wIm[m]
                }
            }
            bRe.withUnsafeMutableBufferPointer { rp in
                bIm.withUnsafeMutableBufferPointer { ip in
                    var split = DSPDoubleSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    vDSP_fft_zipD(setup, &split, 1, log2Padded,
                                 FFTDirection(kFFTDirection_Forward))
                }
            }
            return (bRe, bIm)
        }
        (filterForwardRe, filterForwardIm) = filter(conjugateChirp: true)
        (filterInverseRe, filterInverseIm) = filter(conjugateChirp: false)
    }

    deinit {
        vDSP_destroy_fftsetupD(setup)
    }

    /// Exact length-N DFT of one line. `scratchRe/Im` must hold
    /// `2 * paddedLength` doubles each (the second half is the out-of-place target of the padded DFT); they are caller-owned so concurrent
    /// workers never share them.
    package func execute(
        forward: Bool,
        _ inputRe: UnsafePointer<Double>, _ inputIm: UnsafePointer<Double>,
        _ outputRe: UnsafeMutablePointer<Double>, _ outputIm: UnsafeMutablePointer<Double>,
        scratchRe: UnsafeMutablePointer<Double>, scratchIm: UnsafeMutablePointer<Double>
    ) {
        let n = vDSP_Length(length)
        let m = vDSP_Length(paddedLength)
        // vDSP_zvmul's conjugate flag applies to its FIRST operand: 1 = plain
        // product, −1 = conj(A)·B. The forward chirp is w, the inverse conj(w).
        let conjugate: Int32 = forward ? 1 : -1
        chirpRe.withUnsafeBufferPointer { cr in
            chirpIm.withUnsafeBufferPointer { ci in
                var chirp = DSPDoubleSplitComplex(
                    realp: UnsafeMutablePointer(mutating: cr.baseAddress!),
                    imagp: UnsafeMutablePointer(mutating: ci.baseAddress!)
                )
                var input = DSPDoubleSplitComplex(
                    realp: UnsafeMutablePointer(mutating: inputRe),
                    imagp: UnsafeMutablePointer(mutating: inputIm)
                )
                var work = DSPDoubleSplitComplex(realp: scratchRe, imagp: scratchIm)

                // 1. a_n = x_n · c_n, zero-padded to M.
                vDSP_zvmulD(&chirp, 1, &input, 1, &work, 1, n, conjugate)
                if paddedLength > length {
                    (scratchRe + length).update(repeating: 0, count: paddedLength - length)
                    (scratchIm + length).update(repeating: 0, count: paddedLength - length)
                }
                // 2. A = FFT_M(a).
                let alt = scratchRe + paddedLength, altIm = scratchIm + paddedLength
                if let padded {
                    padded.execute(forward: true, scratchRe, scratchIm, alt, altIm)
                    work = DSPDoubleSplitComplex(realp: alt, imagp: altIm)
                } else {
                    vDSP_fft_zipD(setup, &work, 1, paddedLog2, FFTDirection(kFFTDirection_Forward))
                }
                // 3. A ·= B (the direction's precomputed filter spectrum).
                let filterRe = forward ? filterForwardRe : filterInverseRe
                let filterIm = forward ? filterForwardIm : filterInverseIm
                filterRe.withUnsafeBufferPointer { fr in
                    filterIm.withUnsafeBufferPointer { fi in
                        var filter = DSPDoubleSplitComplex(
                            realp: UnsafeMutablePointer(mutating: fr.baseAddress!),
                            imagp: UnsafeMutablePointer(mutating: fi.baseAddress!)
                        )
                        vDSP_zvmulD(&work, 1, &filter, 1, &work, 1, m, 1)
                    }
                }
                // 4. a = IFFT_M(A) / M (vDSP's inverse is unnormalized).
                if let padded {
                    // `work` is the second half; the inverse lands back in the first.
                    padded.execute(forward: false, alt, altIm, scratchRe, scratchIm)
                    work = DSPDoubleSplitComplex(realp: scratchRe, imagp: scratchIm)
                } else {
                    vDSP_fft_zipD(setup, &work, 1, paddedLog2, FFTDirection(kFFTDirection_Inverse))
                }
                var scale = 1 / Double(paddedLength)
                vDSP_vsmulD(scratchRe, 1, &scale, scratchRe, 1, m)
                vDSP_vsmulD(scratchIm, 1, &scale, scratchIm, 1, m)
                // 5. X_k = c_k · a_k for k < N.
                var output = DSPDoubleSplitComplex(realp: outputRe, imagp: outputIm)
                vDSP_zvmulD(&chirp, 1, &work, 1, &output, 1, n, conjugate)
            }
        }
    }
}

public nonisolated final class FFT2DDouble: FFT2DTransform, @unchecked Sendable {
    public typealias Scalar = Double

    public let nx: Int
    public let ny: Int
    private let log2x: vDSP_Length
    private let log2y: vDSP_Length
    private let radix2Setup: FFTSetupD?
    /// The power-of-two 2D path on `vDSP_DFT_zop` (rows, transpose, columns, transpose); nil when `radix2Setup` serves.
    private let dft2DX: DFTPairD?
    private let dft2DY: DFTPairD?
    /// The power-of-two axes of the axis-wise path on `vDSP_DFT_zop`, ahead of `axisXSetup` / `axisYSetup`.
    private let axisPairX: DFTPairD?
    private let axisPairY: DFTPairD?
    private let axisXSetup: FFTSetupD?
    private let axisXLog: vDSP_Length
    private let axisXPlan: ExactRadixPlanD?
    private let axisYSetup: FFTSetupD?
    private let axisYLog: vDSP_Length
    private let axisYPlan: ExactRadixPlanD?
    private let forwardX: vDSP_DFT_SetupD?
    private let inverseX: vDSP_DFT_SetupD?
    private let forwardY: vDSP_DFT_SetupD?
    private let inverseY: vDSP_DFT_SetupD?
    /// Exact arbitrary-length plans, built only for an axis vDSP cannot
    /// serve natively (no radix-2 setup, no exact radix-3/5 plan, no DFT
    /// setup) — e.g. 250, 254, 300.
    private let bluesteinX: BluesteinPlanD?
    private let bluesteinY: BluesteinPlanD?
    /// Lengths of prime factors 2, 3, 5, 7 that no vDSP plan serves (e.g. 500, 504, 360 = 2³·3²·5): the mixed-radix Stockham
    /// plan, ahead of Bluestein. nil on an axis another engine serves.
    private let mixedX: MixedRadixPlanD?
    private let mixedY: MixedRadixPlanD?

    /// Fails only when a required Accelerate FFT setup cannot be allocated.
    public init?(nx: Int, ny: Int) {
        guard nx > 0, ny > 0 else { return nil }
        self.nx = nx
        self.ny = ny
        // The one-call vDSP 2D path is taken only for power-of-two sizes of
        // at least 16 on BOTH axes. MEASURED 2026-09-15:
        // `vDSP_fft2d_zip` returns garbage — forward→inverse off by O(1),
        // the DC bin zero for a non-zero-mean field — for nx = 8 or 4 with
        // ny ≥ 8 (8×8, 8×16, 8×32, 8×64, 4×8), while 16×8, 64×8, 8×4, 4×4
        // and everything ≥ 16×16 round-trip to 1e-7. The axis-wise path
        // below round-trips every one of those shapes to 1e-7, so small
        // power-of-two sizes go there.
        let isRadix2 = nx & (nx - 1) == 0 && ny & (ny - 1) == 0 && nx >= 16 && ny >= 16
        log2x = isRadix2 ? vDSP_Length(log2(Double(nx))) : 0
        log2y = isRadix2 ? vDSP_Length(log2(Double(ny))) : 0
        if isRadix2 {
            let pairX = DFTPairD(length: nx), pairY = DFTPairD(length: ny)
            if let pairX, let pairY {
                dft2DX = pairX; dft2DY = pairY
                radix2Setup = nil
            } else {
                guard let setup = vDSP_create_fftsetupD(
                    max(log2x, log2y), FFTRadix(kFFTRadix2)
                ) else { return nil }
                radix2Setup = setup
                dft2DX = nil; dft2DY = nil
            }
            axisPairX = nil; axisPairY = nil
            axisXSetup = nil; axisXLog = 0; axisXPlan = nil
            axisYSetup = nil; axisYLog = 0; axisYPlan = nil
            forwardX = nil; inverseX = nil; forwardY = nil; inverseY = nil
            bluesteinX = nil; bluesteinY = nil
            mixedX = nil; mixedY = nil
        } else {
            radix2Setup = nil
            dft2DX = nil; dft2DY = nil
            let xRadix2Log = Self.exactPowerExponent(nx, radix: 2)
            let yRadix2Log = Self.exactPowerExponent(ny, radix: 2)
            let xPlan = Self.exactRadixPlan(length: nx)
            let yPlan = Self.exactRadixPlan(length: ny)
            if let xRadix2Log {
                guard let setup = vDSP_create_fftsetupD(
                    xRadix2Log, FFTRadix(kFFTRadix2)
                ) else {
                    return nil
                }
                axisXSetup = setup
                axisXLog = xRadix2Log
            } else {
                axisXSetup = nil
                axisXLog = 0
            }
            axisXPlan = xPlan
            axisPairX = xRadix2Log != nil ? DFTPairD(length: nx) : nil
            axisPairY = yRadix2Log != nil ? DFTPairD(length: ny) : nil
            if let yRadix2Log {
                guard let setup = vDSP_create_fftsetupD(
                    yRadix2Log, FFTRadix(kFFTRadix2)
                ) else {
                    if let axisXSetup { vDSP_destroy_fftsetupD(axisXSetup) }
                    return nil
                }
                axisYSetup = setup
                axisYLog = yRadix2Log
            } else {
                axisYSetup = nil
                axisYLog = 0
            }
            axisYPlan = yPlan
            let fx = xRadix2Log == nil && xPlan == nil
                ? vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(nx), .FORWARD) : nil
            let ix = xRadix2Log == nil && xPlan == nil
                ? vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(nx), .INVERSE) : nil
            let fy = yRadix2Log == nil && yPlan == nil
                ? vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(ny), .FORWARD) : nil
            let iy = yRadix2Log == nil && yPlan == nil
                ? vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(ny), .INVERSE) : nil
            forwardX = fx; inverseX = ix; forwardY = fy; inverseY = iy
            let mx = xRadix2Log == nil && xPlan == nil && fx == nil ? MixedRadixPlanD(length: nx) : nil
            let my = yRadix2Log == nil && yPlan == nil && fy == nil ? MixedRadixPlanD(length: ny) : nil
            mixedX = mx; mixedY = my
            let needsBluesteinX = xRadix2Log == nil && xPlan == nil && fx == nil && mx == nil
            let needsBluesteinY = yRadix2Log == nil && yPlan == nil && fy == nil && my == nil
            let bx = needsBluesteinX ? BluesteinPlanD(length: nx) : nil
            let by = needsBluesteinY ? BluesteinPlanD(length: ny) : nil
            if (needsBluesteinX && bx == nil) || (needsBluesteinY && by == nil) {
                if let axisXSetup { vDSP_destroy_fftsetupD(axisXSetup) }
                if let axisYSetup { vDSP_destroy_fftsetupD(axisYSetup) }
                if let fx { vDSP_DFT_DestroySetupD(fx) }
                if let ix { vDSP_DFT_DestroySetupD(ix) }
                if let fy { vDSP_DFT_DestroySetupD(fy) }
                if let iy { vDSP_DFT_DestroySetupD(iy) }
                return nil
            }
            bluesteinX = bx
            bluesteinY = by
        }
    }

    deinit {
        if let radix2Setup { vDSP_destroy_fftsetupD(radix2Setup) }
        if let axisXSetup { vDSP_destroy_fftsetupD(axisXSetup) }
        if let axisYSetup { vDSP_destroy_fftsetupD(axisYSetup) }
        if let forwardX { vDSP_DFT_DestroySetupD(forwardX) }
        if let inverseX { vDSP_DFT_DestroySetupD(inverseX) }
        if let forwardY { vDSP_DFT_DestroySetupD(forwardY) }
        if let inverseY { vDSP_DFT_DestroySetupD(inverseY) }
    }

    /// In-place complex 2D FFT over row-major [ny * nx] split-complex data.
    /// The inverse is unnormalized, matching vDSP — `scaleInverse` divides by
    /// nx*ny so forward→inverse round-trips to the input.
    public func transform(re: inout [Double], im: inout [Double],
                   forward: Bool, scaleInverse: Bool = true) {
        precondition(re.count == nx * ny && im.count == nx * ny)
        if let dft2DX, let dft2DY {
            transformPowerOfTwo(re: &re, im: &im, forward: forward, pairX: dft2DX, pairY: dft2DY)
        } else if let radix2Setup {
            re.withUnsafeMutableBufferPointer { rp in
                im.withUnsafeMutableBufferPointer { ip in
                    var split = DSPDoubleSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    vDSP_fft2d_zipD(radix2Setup, &split, 1, 0, log2x, log2y,
                                   FFTDirection(forward ? kFFTDirection_Forward
                                                        : kFFTDirection_Inverse))
                }
            }
        } else {
            let xSetup = forward ? forwardX : inverseX
            let ySetup = forward ? forwardY : inverseY
            // Per-call Bluestein scratch (padded length of the larger axis):
            // caller-owned like the data, so concurrent workers sharing this
            // plan never share a buffer.
            let scratchCount = 2 * max(bluesteinX?.paddedLength ?? 0,
                                       bluesteinY?.paddedLength ?? 0)
            var scratchRe = [Double](repeating: 0, count: scratchCount)
            var scratchIm = [Double](repeating: 0, count: scratchCount)
            var horizontalRe = [Double](repeating: 0, count: nx * ny)
            var horizontalIm = [Double](repeating: 0, count: nx * ny)
            scratchRe.withUnsafeMutableBufferPointer { sr in
            scratchIm.withUnsafeMutableBufferPointer { si in
            let scratch = (sr.baseAddress, si.baseAddress)
            re.withUnsafeBufferPointer { inputRe in
                im.withUnsafeBufferPointer { inputIm in
                    horizontalRe.withUnsafeMutableBufferPointer { outputRe in
                        horizontalIm.withUnsafeMutableBufferPointer { outputIm in
                            if let mixedX {
                                mixedX.transformLines(count: ny, forward: forward, inputStride: nx,
                                                      inputRe.baseAddress!, inputIm.baseAddress!,
                                                      outputRe.baseAddress!, outputIm.baseAddress!)
                            } else { for y in 0..<ny {
                                executeAxis(
                                    pair: axisPairX, fftSetup: axisXSetup, fftLog: axisXLog,
                                    exactPlan: axisXPlan,
                                    dftSetup: xSetup, bluestein: bluesteinX,
                                    length: nx, forward: forward,
                                    inputRe.baseAddress! + y * nx,
                                    inputIm.baseAddress! + y * nx,
                                    outputRe.baseAddress! + y * nx,
                                    outputIm.baseAddress! + y * nx,
                                    scratch: scratch
                                )
                            } }
                        }
                    }
                }
            }
            if let mixedY {
                // Two neighbouring columns per pass, read and written in place (stride nx).
                horizontalRe.withUnsafeMutableBufferPointer { hr in horizontalIm.withUnsafeMutableBufferPointer { hi in
                    mixedY.transformColumns(count: nx, forward: forward, stride: nx, hr.baseAddress!, hi.baseAddress!)
                } }
                swap(&re, &horizontalRe)
                swap(&im, &horizontalIm)
            } else {
            // Columns: transpose so each is a contiguous line, transform the lines, transpose back (strided column
            // gathers were most of the time of the vDSP and Bluestein lengths).
            var tRe = [Double](repeating: 0, count: nx * ny), tIm = tRe
            var uRe = tRe, uIm = tRe
            vDSP_mtransD(horizontalRe, 1, &tRe, 1, vDSP_Length(nx), vDSP_Length(ny))
            vDSP_mtransD(horizontalIm, 1, &tIm, 1, vDSP_Length(nx), vDSP_Length(ny))
            tRe.withUnsafeBufferPointer { inputRe in tIm.withUnsafeBufferPointer { inputIm in
                uRe.withUnsafeMutableBufferPointer { outputRe in uIm.withUnsafeMutableBufferPointer { outputIm in
                    for x in 0..<nx {
                        executeAxis(
                            pair: axisPairY, fftSetup: axisYSetup, fftLog: axisYLog,
                            exactPlan: axisYPlan,
                            dftSetup: ySetup, bluestein: bluesteinY,
                            length: ny, forward: forward,
                            inputRe.baseAddress! + x * ny, inputIm.baseAddress! + x * ny,
                            outputRe.baseAddress! + x * ny, outputIm.baseAddress! + x * ny,
                            scratch: scratch
                        )
                    }
                } }
            } }
            vDSP_mtransD(uRe, 1, &re, 1, vDSP_Length(ny), vDSP_Length(nx))
            vDSP_mtransD(uIm, 1, &im, 1, vDSP_Length(ny), vDSP_Length(nx))
            }
            }
            }
        }
        if !forward && scaleInverse {
            var s = 1 / Double(nx * ny)
            vDSP_vsmulD(re, 1, &s, &re, 1, vDSP_Length(re.count))
            vDSP_vsmulD(im, 1, &s, &im, 1, vDSP_Length(im.count))
        }
    }

    /// `count` independent lines of the x axis (contiguous, length `nx`) or of the y axis (length `ny`), each transformed out of
    /// place with this plan's one-axis engine (the unnormalised inverse). The building block of the real-input 2D transforms
    /// (`RealFFT2D.swift`); `input` and `output` must not overlap.
    package func transformLines(alongX: Bool, count: Int, forward: Bool, inputStride: Int? = nil,
                                _ inputRe: UnsafePointer<Double>, _ inputIm: UnsafePointer<Double>,
                                _ outputRe: UnsafeMutablePointer<Double>, _ outputIm: UnsafeMutablePointer<Double>) {
        let length = alongX ? nx : ny
        let inStride = inputStride ?? length
        if length == 1 {
            for i in 0..<count { outputRe[i] = inputRe[i * inStride]; outputIm[i] = inputIm[i * inStride] }
            return
        }
        if dft2DX != nil || radix2Setup != nil {
            if let pair = alongX ? dft2DX : dft2DY {
                for i in 0..<count {
                    pair.execute(forward: forward, inputRe + i * inStride, inputIm + i * inStride,
                                 outputRe + i * length, outputIm + i * length)
                }
            } else if let radix2Setup {
                let log2n = alongX ? log2x : log2y
                for i in 0..<count {
                    (outputRe + i * length).update(from: inputRe + i * inStride, count: length)
                    (outputIm + i * length).update(from: inputIm + i * inStride, count: length)
                    var split = DSPDoubleSplitComplex(realp: outputRe + i * length, imagp: outputIm + i * length)
                    vDSP_fft_zipD(radix2Setup, &split, 1, log2n,
                                  FFTDirection(forward ? kFFTDirection_Forward : kFFTDirection_Inverse))
                }
            }
            return
        }
        if let mixed = alongX ? mixedX : mixedY {
            mixed.transformLines(count: count, forward: forward, inputStride: inStride, inputRe, inputIm, outputRe, outputIm)
            return
        }
        let bluestein = alongX ? bluesteinX : bluesteinY
        var scratchRe = [Double](repeating: 0, count: 2 * (bluestein?.paddedLength ?? 0))
        var scratchIm = scratchRe
        scratchRe.withUnsafeMutableBufferPointer { sr in scratchIm.withUnsafeMutableBufferPointer { si in
            let scratch = (sr.baseAddress, si.baseAddress)
            for i in 0..<count {
                executeAxis(pair: alongX ? axisPairX : axisPairY, fftSetup: alongX ? axisXSetup : axisYSetup,
                            fftLog: alongX ? axisXLog : axisYLog, exactPlan: alongX ? axisXPlan : axisYPlan,
                            dftSetup: alongX ? (forward ? forwardX : inverseX) : (forward ? forwardY : inverseY),
                            bluestein: bluestein, length: length, forward: forward,
                            inputRe + i * inStride, inputIm + i * inStride, outputRe + i * length, outputIm + i * length,
                            scratch: scratch)
            }
        } }
    }

    /// Rows, transpose, columns, transpose, each row a `vDSP_DFT_zop` call (the unnormalised inverse is scaled by the caller).
    private func transformPowerOfTwo(re: inout [Double], im: inout [Double], forward: Bool,
                                     pairX: DFTPairD, pairY: DFTPairD) {
        let count = nx * ny
        var tRe = [Double](repeating: 0, count: count), tIm = [Double](repeating: 0, count: count)
        var uRe = [Double](repeating: 0, count: count), uIm = [Double](repeating: 0, count: count)
        re.withUnsafeMutableBufferPointer { r in im.withUnsafeMutableBufferPointer { i in
        tRe.withUnsafeMutableBufferPointer { tr in tIm.withUnsafeMutableBufferPointer { ti in
        uRe.withUnsafeMutableBufferPointer { ur in uIm.withUnsafeMutableBufferPointer { ui in
            let (r, i, tr, ti, ur, ui) = (r.baseAddress!, i.baseAddress!, tr.baseAddress!, ti.baseAddress!,
                                          ur.baseAddress!, ui.baseAddress!)
            for y in 0..<ny { pairX.execute(forward: forward, r + y * nx, i + y * nx, tr + y * nx, ti + y * nx) }
            vDSP_mtransD(tr, 1, ur, 1, vDSP_Length(nx), vDSP_Length(ny))
            vDSP_mtransD(ti, 1, ui, 1, vDSP_Length(nx), vDSP_Length(ny))
            for x in 0..<nx { pairY.execute(forward: forward, ur + x * ny, ui + x * ny, tr + x * ny, ti + x * ny) }
            vDSP_mtransD(tr, 1, r, 1, vDSP_Length(ny), vDSP_Length(nx))
            vDSP_mtransD(ti, 1, i, 1, vDSP_Length(ny), vDSP_Length(nx))
        } } } } } }
    }

    private func executeAxis(
        pair: DFTPairD?, fftSetup: FFTSetupD?, fftLog: vDSP_Length,
        exactPlan: ExactRadixPlanD?,
        dftSetup: vDSP_DFT_SetupD?, bluestein: BluesteinPlanD?,
        length: Int, forward: Bool,
        _ inputRe: UnsafePointer<Double>, _ inputIm: UnsafePointer<Double>,
        _ outputRe: UnsafeMutablePointer<Double>, _ outputIm: UnsafeMutablePointer<Double>,
        scratch: (UnsafeMutablePointer<Double>?, UnsafeMutablePointer<Double>?)
    ) {
        if let pair {
            pair.execute(forward: forward, inputRe, inputIm, outputRe, outputIm)
            return
        }
        if let fftSetup {
            outputRe.update(from: inputRe, count: length)
            outputIm.update(from: inputIm, count: length)
            var split = DSPDoubleSplitComplex(realp: outputRe, imagp: outputIm)
            vDSP_fft_zipD(
                fftSetup, &split, 1, fftLog,
                FFTDirection(forward ? kFFTDirection_Forward : kFFTDirection_Inverse)
            )
            return
        }
        if let exactPlan {
            executeExactRadix(
                plan: exactPlan, forward: forward,
                inputRe, inputIm, outputRe, outputIm
            )
            return
        }
        if let dftSetup {
            vDSP_DFT_ExecuteD(dftSetup, inputRe, inputIm, outputRe, outputIm)
            return
        }
        // `init` guarantees a plan exists for every axis no other path
        // serves, so this is unreachable rather than a silent fallback.
        guard let bluestein, let scratchRe = scratch.0, let scratchIm = scratch.1 else {
            preconditionFailure("FFT2DDouble: no transform plan for length \(length)")
        }
        bluestein.execute(
            forward: forward, inputRe, inputIm, outputRe, outputIm,
            scratchRe: scratchRe, scratchIm: scratchIm
        )
    }

    private func executeExactRadix(
        plan: ExactRadixPlanD, forward: Bool,
        _ inputRe: UnsafePointer<Double>, _ inputIm: UnsafePointer<Double>,
        _ outputRe: UnsafeMutablePointer<Double>, _ outputIm: UnsafeMutablePointer<Double>
    ) {
        for source in 0..<plan.length {
            let destination = plan.reversedIndices[source]
            outputRe[destination] = inputRe[source]
            outputIm[destination] = inputIm[source]
        }
        for stage in plan.stages {
            var groupStart = 0
            while groupStart < plan.length {
                for offset in 0..<stage.butterflyStride {
                    if plan.radix == 3 {
                        radix3Butterfly(
                            groupStart: groupStart, offset: offset, stage: stage,
                            forward: forward, outputRe, outputIm
                        )
                    } else {
                        radix5Butterfly(
                            groupStart: groupStart, offset: offset, stage: stage,
                            forward: forward, outputRe, outputIm
                        )
                    }
                }
                groupStart += stage.groupSize
            }
        }
    }

    @inline(__always)
    private func twiddledValue(
        branch: Int, groupStart: Int, offset: Int,
        stage: ExactRadixStageD, forward: Bool,
        _ real: UnsafeMutablePointer<Double>, _ imaginary: UnsafeMutablePointer<Double>
    ) -> (Double, Double) {
        let dataIndex = groupStart + offset + branch * stage.butterflyStride
        let twiddleIndex = branch * stage.butterflyStride + offset
        let wr = stage.twiddleReal[twiddleIndex]
        let storedWI = stage.twiddleImaginaryForward[twiddleIndex]
        let wi = forward ? storedWI : -storedWI
        let xr = real[dataIndex], xi = imaginary[dataIndex]
        return (xr * wr - xi * wi, xr * wi + xi * wr)
    }

    @inline(__always)
    private func radix3Butterfly(
        groupStart: Int, offset: Int, stage: ExactRadixStageD, forward: Bool,
        _ real: UnsafeMutablePointer<Double>, _ imaginary: UnsafeMutablePointer<Double>
    ) {
        let index0 = groupStart + offset
        let index1 = index0 + stage.butterflyStride
        let index2 = index1 + stage.butterflyStride
        let x0r = real[index0], x0i = imaginary[index0]
        let (x1r, x1i) = twiddledValue(
            branch: 1, groupStart: groupStart, offset: offset,
            stage: stage, forward: forward, real, imaginary
        )
        let (x2r, x2i) = twiddledValue(
            branch: 2, groupStart: groupStart, offset: offset,
            stage: stage, forward: forward, real, imaginary
        )
        let sumR = x1r + x2r, sumI = x1i + x2i
        let differenceR = x1r - x2r, differenceI = x1i - x2i
        let baseR = x0r - 0.5 * sumR, baseI = x0i - 0.5 * sumI
        let direction: Double = forward ? -1 : 1
        let uR = direction * 0.8660254037844386 * differenceR
        let uI = direction * 0.8660254037844386 * differenceI
        real[index0] = x0r + sumR
        imaginary[index0] = x0i + sumI
        real[index1] = baseR - uI
        imaginary[index1] = baseI + uR
        real[index2] = baseR + uI
        imaginary[index2] = baseI - uR
    }

    @inline(__always)
    private func radix5Butterfly(
        groupStart: Int, offset: Int, stage: ExactRadixStageD, forward: Bool,
        _ real: UnsafeMutablePointer<Double>, _ imaginary: UnsafeMutablePointer<Double>
    ) {
        let stride = stage.butterflyStride
        let index0 = groupStart + offset
        let index1 = index0 + stride, index2 = index1 + stride
        let index3 = index2 + stride, index4 = index3 + stride
        let x0r = real[index0], x0i = imaginary[index0]
        let (x1r, x1i) = twiddledValue(
            branch: 1, groupStart: groupStart, offset: offset,
            stage: stage, forward: forward, real, imaginary
        )
        let (x2r, x2i) = twiddledValue(
            branch: 2, groupStart: groupStart, offset: offset,
            stage: stage, forward: forward, real, imaginary
        )
        let (x3r, x3i) = twiddledValue(
            branch: 3, groupStart: groupStart, offset: offset,
            stage: stage, forward: forward, real, imaginary
        )
        let (x4r, x4i) = twiddledValue(
            branch: 4, groupStart: groupStart, offset: offset,
            stage: stage, forward: forward, real, imaginary
        )

        let t1r = x1r + x4r, t1i = x1i + x4i
        let t2r = x2r + x3r, t2i = x2i + x3i
        let t3r = x1r - x4r, t3i = x1i - x4i
        let t4r = x2r - x3r, t4i = x2i - x3i
        let c1: Double = 0.30901699437494745
        let c2: Double = -0.8090169943749473
        let s1: Double = 0.9510565162951535
        let s2: Double = 0.5877852522924731
        let direction: Double = forward ? -1 : 1

        let base1r = x0r + c1 * t1r + c2 * t2r
        let base1i = x0i + c1 * t1i + c2 * t2i
        let u1r = direction * (s1 * t3r + s2 * t4r)
        let u1i = direction * (s1 * t3i + s2 * t4i)
        let base2r = x0r + c2 * t1r + c1 * t2r
        let base2i = x0i + c2 * t1i + c1 * t2i
        let u2r = direction * (s2 * t3r - s1 * t4r)
        let u2i = direction * (s2 * t3i - s1 * t4i)

        real[index0] = x0r + t1r + t2r
        imaginary[index0] = x0i + t1i + t2i
        real[index1] = base1r - u1i
        imaginary[index1] = base1i + u1r
        real[index4] = base1r + u1i
        imaginary[index4] = base1i - u1r
        real[index2] = base2r - u2i
        imaginary[index2] = base2i + u2r
        real[index3] = base2r + u2i
        imaginary[index3] = base2i - u2r
    }

    private static func exactRadixPlan(length: Int) -> ExactRadixPlanD? {
        for radix in [3, 5] {
            if let exponent = exactPowerExponent(length, radix: radix) {
                return ExactRadixPlanD(
                    length: length, radix: radix, exponent: Int(exponent)
                )
            }
        }
        return nil
    }

    private static func exactPowerExponent(
        _ value: Int, radix: Int
    ) -> vDSP_Length? {
        guard value > 1 else { return nil }
        var remainder = value
        var exponent: vDSP_Length = 0
        while remainder.isMultiple(of: radix) {
            remainder /= radix
            exponent += 1
        }
        return remainder == 1 ? exponent : nil
    }

    public static func nextPow2(_ v: Int) -> Int {
        var p = 1
        while p < v { p <<= 1 }
        return p
    }

    /// Frequency of bin k for an N-point FFT, in cycles per pixel (np.fftfreq).
    public static func fftfreq(_ k: Int, _ n: Int) -> Double {
        let kk = k <= (n - 1) / 2 ? k : k - n
        return Double(kk) / Double(n)
    }
}
