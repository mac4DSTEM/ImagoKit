//
//  RealFFT2D.swift (ImagoCore)
//  The real-input 2D transform behind `FFTSurface.rfft2` / `irfft2` and the real-image callers (butterworth, registration):
//  half the work of a complex transform of real data, which is what NumPy's pocketfft does (r2c).
//
//  FORWARD (`rows × columns` real -> `rows × (columns/2 + 1)` half spectrum): two real rows a, b are packed into one complex
//  row z = a + i·b and transformed together; with Z = FFT(z), A_k = (Z_k + conj Z_{n−k})/2 and
//  B_k = (Z_k − conj Z_{n−k})/(2i). Then the (columns/2 + 1) columns are transformed as complex lines of length `rows`.
//  INVERSE (the same steps backwards): complex lines down the columns first, then two Hermitian rows A, B are packed as
//  Z = A + i·B (completed by Z_{n−k} = conj A_k + i·conj B_k) and one complex inverse yields a in its real part and b in its
//  imaginary part. The imaginary parts of the zero-frequency and (even n) Nyquist bins do not count, as in NumPy's c2r.
//
//  Both directions are unnormalised; the callers scale. The line transforms are the plan's own one-axis engines (vDSP DFT,
//  radix-2, the exact radix plans, Bluestein), so every length the complex transform supports works here, and Float64 is the
//  reference path.
//

import Foundation
import Accelerate

/// A plan that transforms batches of lines along one axis, out of place; `FFT2D` and `FFT2DDouble`.
package protocol LineTransformPlan: FFT2DTransform {
    /// `inputStride` is the distance between consecutive input lines (nil: the line length); outputs are contiguous.
    func transformLines(alongX: Bool, count: Int, forward: Bool, inputStride: Int?,
                        _ inputRe: UnsafePointer<Scalar>, _ inputIm: UnsafePointer<Scalar>,
                        _ outputRe: UnsafeMutablePointer<Scalar>, _ outputIm: UnsafeMutablePointer<Scalar>)
    /// The unnormalised half spectrum (`ny × (nx/2 + 1)`, split complex) of a real `ny × nx` image.
    func realForward(_ real: [Scalar]) -> (re: [Scalar], im: [Scalar])
    /// The unnormalised real image (`ny × nx`) of a half spectrum of `inputColumns` columns per row.
    func realInverse(re: [Scalar], im: [Scalar], inputColumns: Int) -> [Scalar]
    /// `dst` (`columns × rows`) = transpose of `src` (`rows × columns`).
    static func transpose(_ src: UnsafePointer<Scalar>, _ dst: UnsafeMutablePointer<Scalar>, rows: Int, columns: Int)
}

extension FFT2DDouble: LineTransformPlan {
    package static func transpose(_ src: UnsafePointer<Double>, _ dst: UnsafeMutablePointer<Double>, rows: Int, columns: Int) {
        vDSP_mtransD(src, 1, dst, 1, vDSP_Length(columns), vDSP_Length(rows))
    }

    /// The unnormalised half spectrum (`ny × (nx/2 + 1)`, split complex) of a real `ny × nx` image.
    package func realForward(_ real: [Double]) -> (re: [Double], im: [Double]) { RealFFT2D.forward(self, real) }

    /// The unnormalised real image (`ny × nx`) of a half spectrum of `inputColumns` columns per row.
    package func realInverse(re: [Double], im: [Double], inputColumns: Int) -> [Double] {
        RealFFT2D.inverse(self, re: re, im: im, inputColumns: inputColumns)
    }
}

extension FFT2D: LineTransformPlan {
    package static func transpose(_ src: UnsafePointer<Float>, _ dst: UnsafeMutablePointer<Float>, rows: Int, columns: Int) {
        vDSP_mtrans(src, 1, dst, 1, vDSP_Length(columns), vDSP_Length(rows))
    }

    /// The unnormalised half spectrum (`ny × (nx/2 + 1)`, split complex) of a real `ny × nx` image.
    package func realForward(_ real: [Float]) -> (re: [Float], im: [Float]) { RealFFT2D.forward(self, real) }

    /// The unnormalised real image (`ny × nx`) of a half spectrum of `inputColumns` columns per row.
    package func realInverse(re: [Float], im: [Float], inputColumns: Int) -> [Float] {
        RealFFT2D.inverse(self, re: re, im: im, inputColumns: inputColumns)
    }
}

/// A scratch buffer of `count` scalars (zeroed only when asked: the transforms overwrite it), freed by `deinit`.
private final class Scratch<S: BinaryFloatingPoint> {
    let pointer: UnsafeMutablePointer<S>
    init(_ count: Int, zero: Bool = false) {
        pointer = .allocate(capacity: max(count, 1))
        if zero { pointer.initialize(repeating: 0, count: max(count, 1)) } else { pointer.initialize(to: 0) }
    }
    deinit { pointer.deallocate() }
}

nonisolated package enum RealFFT2D {
    /// Columns handled per sweep of the split and pack loops (their other side is a transposed, strided array).
    private static let block = 8

    static func forward<P: LineTransformPlan>(_ plan: P, _ real: [P.Scalar]) -> (re: [P.Scalar], im: [P.Scalar]) {
        typealias S = P.Scalar
        let h = plan.ny, w = plan.nx, m = w / 2 + 1, pairs = (h + 1) / 2
        precondition(real.count == h * w)
        let tr = Scratch<S>(pairs * w), ti = Scratch<S>(pairs * w)
        let br = Scratch<S>(m * h), bi = Scratch<S>(m * h)
        let cr = Scratch<S>(m * h), ci = Scratch<S>(m * h)
        var outRe = [S](repeating: 0, count: h * m), outIm = outRe
        // Rows 2p and 2p + 1 are one complex row: the real input array itself serves as the real and the imaginary part.
        real.withUnsafeBufferPointer { xb in
            let x = xb.baseAddress!
            if h >= 2 {
                plan.transformLines(alongX: true, count: h / 2, forward: true, inputStride: 2 * w, x, x + w, tr.pointer, ti.pointer)
            }
            if h % 2 == 1 {
                let zero = Scratch<S>(w, zero: true)
                plan.transformLines(alongX: true, count: 1, forward: true, inputStride: w, x + (h - 1) * w, zero.pointer,
                                    tr.pointer + (pairs - 1) * w, ti.pointer + (pairs - 1) * w)
            }
        }
        // Split each pair into the two half spectra, written transposed (`m × h`): bin k of rows 2p and 2p + 1 is
        // b[k * h + 2p] and b[k * h + 2p + 1].
        let half = S(0.5)
        var k0 = 0
        while k0 < m {
            let k1 = min(k0 + block, m)
            for p in 0..<pairs {
                let row = tr.pointer + p * w, rowI = ti.pointer + p * w
                let hasB = 2 * p + 1 < h
                for k in k0..<k1 {
                    let j = k == 0 ? 0 : w - k
                    let zre = row[k], zim = rowI[k], cre = row[j], cim = -rowI[j]
                    let at = k * h + 2 * p
                    br.pointer[at] = (zre + cre) * half
                    bi.pointer[at] = (zim + cim) * half
                    if hasB {
                        br.pointer[at + 1] = (zim - cim) * half
                        bi.pointer[at + 1] = (cre - zre) * half
                    }
                }
            }
            k0 = k1
        }
        // Complex lines down the m columns, then back to `h × m`.
        plan.transformLines(alongX: false, count: m, forward: true, inputStride: nil, br.pointer, bi.pointer, cr.pointer, ci.pointer)
        outRe.withUnsafeMutableBufferPointer { o in P.transpose(cr.pointer, o.baseAddress!, rows: m, columns: h) }
        outIm.withUnsafeMutableBufferPointer { o in P.transpose(ci.pointer, o.baseAddress!, rows: m, columns: h) }
        return (outRe, outIm)
    }

    static func inverse<P: LineTransformPlan>(_ plan: P, re: [P.Scalar], im: [P.Scalar], inputColumns mi: Int) -> [P.Scalar] {
        typealias S = P.Scalar
        let h = plan.ny, n = plan.nx, pairs = (h + 1) / 2
        let usable = min(mi, n / 2 + 1)
        precondition(re.count == h * mi && im.count == h * mi)
        // Complex inverse lines down the columns (transposed to `mi × h` for the contiguous line transform). Only the
        // `usable` columns count; the rest are dropped before the transform.
        let tr = Scratch<S>(usable * h), ti = Scratch<S>(usable * h), ur = Scratch<S>(usable * h), ui = Scratch<S>(usable * h)
        if usable == mi {
            re.withUnsafeBufferPointer { P.transpose($0.baseAddress!, tr.pointer, rows: h, columns: mi) }
            im.withUnsafeBufferPointer { P.transpose($0.baseAddress!, ti.pointer, rows: h, columns: mi) }
        } else {
            for c in 0..<usable { for r in 0..<h { tr.pointer[c * h + r] = re[r * mi + c]; ti.pointer[c * h + r] = im[r * mi + c] } }
        }
        plan.transformLines(alongX: false, count: usable, forward: false, inputStride: nil, tr.pointer, ti.pointer, ur.pointer, ui.pointer)
        // Pack two Hermitian rows into one complex row; bin k of row r is u[k * h + r].
        let zr = Scratch<S>(pairs * n, zero: true), zi = Scratch<S>(pairs * n, zero: true)
        let nyquist = n % 2 == 0 ? n / 2 : -1
        var k0 = 0
        while k0 < usable {
            let k1 = min(k0 + block, usable)
            for p in 0..<pairs {
                let hasB = 2 * p + 1 < h
                let zRow = zr.pointer + p * n, zIRow = zi.pointer + p * n
                for k in k0..<k1 {
                    let at = k * h + 2 * p
                    let drop = k == 0 || k == nyquist
                    let ar = ur.pointer[at], ai = drop ? 0 : ui.pointer[at]
                    let br: S = hasB ? ur.pointer[at + 1] : 0, bi: S = hasB && !drop ? ui.pointer[at + 1] : 0
                    zRow[k] = ar - bi
                    zIRow[k] = ai + br
                    if k > 0, n - k != k {
                        zRow[n - k] = ar + bi
                        zIRow[n - k] = br - ai
                    }
                }
            }
            k0 = k1
        }
        let vr = Scratch<S>(pairs * n), vi = Scratch<S>(pairs * n)
        plan.transformLines(alongX: true, count: pairs, forward: false, inputStride: nil, zr.pointer, zi.pointer, vr.pointer, vi.pointer)
        var out = [S](repeating: 0, count: h * n)
        out.withUnsafeMutableBufferPointer { o in
            for p in 0..<pairs {
                (o.baseAddress! + 2 * p * n).update(from: vr.pointer + p * n, count: n)
                if 2 * p + 1 < h { (o.baseAddress! + (2 * p + 1) * n).update(from: vi.pointer + p * n, count: n) }
            }
        }
        return out
    }
}
