//
//  MixedRadixPlanD.swift (ImagoCore)
//  The exact Float64 DFT of a length whose prime factors are 2, 3, 5 and 7, with no padding and no Bluestein: a Stockham
//  autosort FFT in radix-2, 3, 4, 5 and 7 passes (the pass structure of pocketfft's `cfftp`, NumPy's engine), with the
//  twiddles of every pass precomputed in Double from exact integer phases.
//
//  TWO LINES AT A TIME. Every pass works on `SIMD2<Double>`: lane 0 is one line, lane 1 the next, so a pair of lines costs
//  one pass of 128-bit arithmetic (what pocketfft does across lines with its vector type). A column pass of a row-major
//  image reads the two lanes as two neighbouring columns straight from memory; a row pass interleaves two rows on the way
//  in and out. An odd last line runs with a zero partner.
//
//  The plan is immutable after construction and shareable across threads; the work area is the caller's.
//  Used by `FFT2DDouble` for the lengths vDSP_DFT_zop, the radix-2 plans and the pure radix-3/5 plans do not serve, ahead of
//  Bluestein (design rule: the vDSP path for 2^n × {1, 3, 5, 15} stays, and so does Bluestein for every other length).
//

import Foundation

package typealias Lane2 = SIMD2<Double>

nonisolated package final class MixedRadixPlanD: @unchecked Sendable {
    package let length: Int
    private let factors: [Int]
    /// Per pass, the twiddles exp(+2πi · j · l1 · i / N) at (j − 1) · (ido − 1) + (i − 1).
    private let twiddleRe: [[Double]]
    private let twiddleIm: [[Double]]

    /// The pass radices of `n` (2 first, then 4s, then 3, 5, 7), or nil when `n` < 2 or has another prime factor.
    package static func factorization(_ n: Int) -> [Int]? {
        guard n > 1 else { return nil }
        var rest = n
        var fours: [Int] = []
        while rest % 4 == 0 { fours.append(4); rest /= 4 }
        var out: [Int] = []
        if rest % 2 == 0 { out.append(2); rest /= 2 }
        out += fours
        for p in [3, 5, 7] { while rest % p == 0 { out.append(p); rest /= p } }
        return rest == 1 ? out : nil
    }

    package static func supports(_ n: Int) -> Bool { factorization(n) != nil }

    package init?(length: Int) {
        guard let factors = Self.factorization(length) else { return nil }
        self.length = length
        self.factors = factors
        var re: [[Double]] = [], im: [[Double]] = []
        var l1 = 1
        for ip in factors {
            let ido = length / (l1 * ip)
            var wr = [Double](repeating: 1, count: max((ip - 1) * (ido - 1), 0)), wi = [Double](repeating: 0, count: wr.count)
            if ido > 1 {
                for j in 1..<ip { for i in 1..<ido {
                    let phase = (j * l1 * i) % length                       // exact integer phase, reduced mod N
                    let angle = 2 * Double.pi * Double(phase) / Double(length)
                    wr[(j - 1) * (ido - 1) + i - 1] = cos(angle)
                    wi[(j - 1) * (ido - 1) + i - 1] = sin(angle)
                } }
            }
            re.append(wr); im.append(wi)
            l1 *= ip
        }
        twiddleRe = re
        twiddleIm = im
    }

    /// Lanes the work area must hold: two buffers each for the real and the imaginary parts.
    package var workCount: Int { 4 * length }

    // MARK: drivers

    /// `count` lines of this length. Line i starts at `inputRe/Im + i · inputStride` and its result at `outputRe/Im + i · length`.
    /// The unnormalised transform in either direction. Input and output may not overlap.
    package func transformLines(count: Int, forward: Bool, inputStride: Int,
                                _ inputRe: UnsafePointer<Double>, _ inputIm: UnsafePointer<Double>,
                                _ outputRe: UnsafeMutablePointer<Double>, _ outputIm: UnsafeMutablePointer<Double>) {
        let n = length
        let work = UnsafeMutablePointer<Lane2>.allocate(capacity: workCount)
        defer { work.deallocate() }
        var line = 0
        while line < count {
            let two = line + 1 < count
            let r0 = inputRe + line * inputStride, i0 = inputIm + line * inputStride
            let r1 = inputRe + (two ? line + 1 : line) * inputStride, i1 = inputIm + (two ? line + 1 : line) * inputStride
            for e in 0..<n {
                work[e] = Lane2(r0[e], two ? r1[e] : 0)
                work[n + e] = Lane2(i0[e], two ? i1[e] : 0)
            }
            let (rr, ri) = run(forward: forward, work: work)
            let o0 = outputRe + line * n, q0 = outputIm + line * n
            for e in 0..<n { o0[e] = rr[e][0]; q0[e] = ri[e][0] }
            if two {
                let o1 = outputRe + (line + 1) * n, q1 = outputIm + (line + 1) * n
                for e in 0..<n { o1[e] = rr[e][1]; q1[e] = ri[e][1] }
            }
            line += 2
        }
    }

    /// `count` columns of a row-major array with `length` rows and row stride `stride`, transformed in place; two neighbouring
    /// columns share a pass.
    package func transformColumns(count: Int, forward: Bool, stride: Int,
                                  _ re: UnsafeMutablePointer<Double>, _ im: UnsafeMutablePointer<Double>) {
        let n = length
        let work = UnsafeMutablePointer<Lane2>.allocate(capacity: workCount)
        defer { work.deallocate() }
        var x = 0
        while x < count {
            let two = x + 1 < count
            if two {
                for e in 0..<n {
                    work[e] = UnsafeRawPointer(re + e * stride + x).loadUnaligned(as: Lane2.self)
                    work[n + e] = UnsafeRawPointer(im + e * stride + x).loadUnaligned(as: Lane2.self)
                }
            } else {
                for e in 0..<n { work[e] = Lane2(re[e * stride + x], 0); work[n + e] = Lane2(im[e * stride + x], 0) }
            }
            let (rr, ri) = run(forward: forward, work: work)
            if two {
                for e in 0..<n {
                    UnsafeMutableRawPointer(re + e * stride + x).storeBytes(of: rr[e], as: Lane2.self)
                    UnsafeMutableRawPointer(im + e * stride + x).storeBytes(of: ri[e], as: Lane2.self)
                }
            } else {
                for e in 0..<n { re[e * stride + x] = rr[e][0]; im[e * stride + x] = ri[e][0] }
            }
            x += 2
        }
    }

    // MARK: passes

    /// The Stockham passes on a work area whose first two buffers (real, imaginary) hold the input; returns the buffers that
    /// hold the result.
    private func run(forward: Bool, work: UnsafeMutablePointer<Lane2>) -> (UnsafeMutablePointer<Lane2>, UnsafeMutablePointer<Lane2>) {
        let n = length
        var fromRe = work, fromIm = work + n, toRe = work + 2 * n, toIm = work + 3 * n
        var l1 = 1
        for (pass, ip) in factors.enumerated() {
            let ido = n / (l1 * ip)
            twiddleRe[pass].withUnsafeBufferPointer { wr in twiddleIm[pass].withUnsafeBufferPointer { wi in
                let w = (wr.baseAddress, wi.baseAddress)
                switch ip {
                case 2: pass2(ido, l1, fromRe, fromIm, toRe, toIm, w, forward)
                case 3: pass3(ido, l1, fromRe, fromIm, toRe, toIm, w, forward)
                case 4: pass4(ido, l1, fromRe, fromIm, toRe, toIm, w, forward)
                case 5: pass5(ido, l1, fromRe, fromIm, toRe, toIm, w, forward)
                default: pass7(ido, l1, fromRe, fromIm, toRe, toIm, w, forward)
                }
            } }
            swap(&fromRe, &toRe)
            swap(&fromIm, &toIm)
            l1 *= ip
        }
        return (fromRe, fromIm)
    }

    private typealias Ptr = UnsafeMutablePointer<Lane2>
    private typealias Tw = (UnsafePointer<Double>?, UnsafePointer<Double>?)

    /// u · w for the stored w = exp(+iθ); the forward transform multiplies by conj(w).
    @inline(__always)
    private func twiddled(_ r: Lane2, _ q: Lane2, _ w: Tw, _ index: Int, _ sign: Double) -> (Lane2, Lane2) {
        let wr = w.0![index], wi = w.1![index] * sign
        return (r * wr - q * wi, r * wi + q * wr)
    }

    private func pass2(_ ido: Int, _ l1: Int, _ cr: Ptr, _ ci: Ptr, _ hr: Ptr, _ hi: Ptr, _ w: Tw, _ forward: Bool) {
        let sign: Double = forward ? -1 : 1
        for k in 0..<l1 {
            for i in 0..<ido {
                let a = i + ido * 2 * k, b = a + ido
                let t0r = cr[a], t0i = ci[a], t1r = cr[b], t1i = ci[b]
                let o = i + ido * k
                hr[o] = t0r + t1r; hi[o] = t0i + t1i
                let ur = t0r - t1r, ui = t0i - t1i
                let p = o + ido * l1
                if i == 0 { hr[p] = ur; hi[p] = ui } else {
                    (hr[p], hi[p]) = twiddled(ur, ui, w, i - 1, sign)
                }
            }
        }
    }

    private func pass3(_ ido: Int, _ l1: Int, _ cr: Ptr, _ ci: Ptr, _ hr: Ptr, _ hi: Ptr, _ w: Tw, _ forward: Bool) {
        let sign: Double = forward ? -1 : 1
        let c = Lane2(repeating: 0.8660254037844386 * sign)                 // σ · √3/2
        let half = Lane2(repeating: 0.5)
        for k in 0..<l1 {
            for i in 0..<ido {
                let a = i + ido * 3 * k
                let t0r = cr[a], t0i = ci[a]
                let t1r = cr[a + ido], t1i = ci[a + ido], t2r = cr[a + 2 * ido], t2i = ci[a + 2 * ido]
                let sr = t1r + t2r, si = t1i + t2i, dr = t1r - t2r, di = t1i - t2i
                let mr = t0r - half * sr, mi = t0i - half * si
                let o = i + ido * k
                hr[o] = t0r + sr; hi[o] = t0i + si
                // u1 = m + iσ(√3/2) d, u2 = m − iσ(√3/2) d
                let u1r = mr - c * di, u1i = mi + c * dr, u2r = mr + c * di, u2i = mi - c * dr
                let p1 = o + ido * l1, p2 = p1 + ido * l1
                if i == 0 { hr[p1] = u1r; hi[p1] = u1i; hr[p2] = u2r; hi[p2] = u2i } else {
                    (hr[p1], hi[p1]) = twiddled(u1r, u1i, w, i - 1, sign)
                    (hr[p2], hi[p2]) = twiddled(u2r, u2i, w, (ido - 1) + i - 1, sign)
                }
            }
        }
    }

    private func pass4(_ ido: Int, _ l1: Int, _ cr: Ptr, _ ci: Ptr, _ hr: Ptr, _ hi: Ptr, _ w: Tw, _ forward: Bool) {
        let sign: Double = forward ? -1 : 1
        for k in 0..<l1 {
            for i in 0..<ido {
                let a = i + ido * 4 * k
                let t0r = cr[a], t0i = ci[a], t1r = cr[a + ido], t1i = ci[a + ido]
                let t2r = cr[a + 2 * ido], t2i = ci[a + 2 * ido], t3r = cr[a + 3 * ido], t3i = ci[a + 3 * ido]
                let a0r = t0r + t2r, a0i = t0i + t2i, a1r = t0r - t2r, a1i = t0i - t2i
                let b0r = t1r + t3r, b0i = t1i + t3i, b1r = t1r - t3r, b1i = t1i - t3i
                let o = i + ido * k
                hr[o] = a0r + b0r; hi[o] = a0i + b0i
                // u1 = a1 + iσ b1, u3 = a1 − iσ b1, u2 = a0 − b0
                let u1r = a1r - sign * b1i, u1i = a1i + sign * b1r
                let u3r = a1r + sign * b1i, u3i = a1i - sign * b1r
                let u2r = a0r - b0r, u2i = a0i - b0i
                let p1 = o + ido * l1, p2 = p1 + ido * l1, p3 = p2 + ido * l1
                if i == 0 {
                    hr[p1] = u1r; hi[p1] = u1i; hr[p2] = u2r; hi[p2] = u2i; hr[p3] = u3r; hi[p3] = u3i
                } else {
                    (hr[p1], hi[p1]) = twiddled(u1r, u1i, w, i - 1, sign)
                    (hr[p2], hi[p2]) = twiddled(u2r, u2i, w, (ido - 1) + i - 1, sign)
                    (hr[p3], hi[p3]) = twiddled(u3r, u3i, w, 2 * (ido - 1) + i - 1, sign)
                }
            }
        }
    }

    private func pass5(_ ido: Int, _ l1: Int, _ cr: Ptr, _ ci: Ptr, _ hr: Ptr, _ hi: Ptr, _ w: Tw, _ forward: Bool) {
        let sign: Double = forward ? -1 : 1
        let c1 = Lane2(repeating: 0.30901699437494745), c2 = Lane2(repeating: -0.8090169943749473)
        let s1 = Lane2(repeating: 0.9510565162951535 * sign), s2 = Lane2(repeating: 0.5877852522924731 * sign)
        for k in 0..<l1 {
            for i in 0..<ido {
                let a = i + ido * 5 * k
                let t0r = cr[a], t0i = ci[a]
                let t1r = cr[a + ido], t1i = ci[a + ido], t2r = cr[a + 2 * ido], t2i = ci[a + 2 * ido]
                let t3r = cr[a + 3 * ido], t3i = ci[a + 3 * ido], t4r = cr[a + 4 * ido], t4i = ci[a + 4 * ido]
                let p1r = t1r + t4r, p1i = t1i + t4i, m1r = t1r - t4r, m1i = t1i - t4i
                let p2r = t2r + t3r, p2i = t2i + t3i, m2r = t2r - t3r, m2i = t2i - t3i
                let o = i + ido * k
                hr[o] = t0r + p1r + p2r; hi[o] = t0i + p1i + p2i
                let a1r = t0r + c1 * p1r + c2 * p2r, a1i = t0i + c1 * p1i + c2 * p2i
                let b1r = s1 * m1r + s2 * m2r, b1i = s1 * m1i + s2 * m2i
                let a2r = t0r + c2 * p1r + c1 * p2r, a2i = t0i + c2 * p1i + c1 * p2i
                let b2r = s2 * m1r - s1 * m2r, b2i = s2 * m1i - s1 * m2i
                // u1 = A1 + iB1, u4 = A1 − iB1, u2 = A2 + iB2, u3 = A2 − iB2
                let p1 = o + ido * l1, p2 = p1 + ido * l1, p3 = p2 + ido * l1, p4 = p3 + ido * l1
                let u1r = a1r - b1i, u1i = a1i + b1r, u4r = a1r + b1i, u4i = a1i - b1r
                let u2r = a2r - b2i, u2i = a2i + b2r, u3r = a2r + b2i, u3i = a2i - b2r
                if i == 0 {
                    hr[p1] = u1r; hi[p1] = u1i; hr[p2] = u2r; hi[p2] = u2i
                    hr[p3] = u3r; hi[p3] = u3i; hr[p4] = u4r; hi[p4] = u4i
                } else {
                    let d = ido - 1, j = i - 1
                    (hr[p1], hi[p1]) = twiddled(u1r, u1i, w, j, sign)
                    (hr[p2], hi[p2]) = twiddled(u2r, u2i, w, d + j, sign)
                    (hr[p3], hi[p3]) = twiddled(u3r, u3i, w, 2 * d + j, sign)
                    (hr[p4], hi[p4]) = twiddled(u4r, u4i, w, 3 * d + j, sign)
                }
            }
        }
    }

    private func pass7(_ ido: Int, _ l1: Int, _ cr: Ptr, _ ci: Ptr, _ hr: Ptr, _ hi: Ptr, _ w: Tw, _ forward: Bool) {
        let sign: Double = forward ? -1 : 1
        let c1 = Lane2(repeating: 0.6234898018587336), c2 = Lane2(repeating: -0.2225209339563144)
        let c3 = Lane2(repeating: -0.9009688679024191)
        let s1 = Lane2(repeating: 0.7818314824680298 * sign), s2 = Lane2(repeating: 0.9749279121818236 * sign)
        let s3 = Lane2(repeating: 0.4338837391175581 * sign)
        for k in 0..<l1 {
            for i in 0..<ido {
                let a = i + ido * 7 * k
                let t0r = cr[a], t0i = ci[a]
                let t1r = cr[a + ido], t1i = ci[a + ido], t6r = cr[a + 6 * ido], t6i = ci[a + 6 * ido]
                let t2r = cr[a + 2 * ido], t2i = ci[a + 2 * ido], t5r = cr[a + 5 * ido], t5i = ci[a + 5 * ido]
                let t3r = cr[a + 3 * ido], t3i = ci[a + 3 * ido], t4r = cr[a + 4 * ido], t4i = ci[a + 4 * ido]
                let p1r = t1r + t6r, p1i = t1i + t6i, m1r = t1r - t6r, m1i = t1i - t6i
                let p2r = t2r + t5r, p2i = t2i + t5i, m2r = t2r - t5r, m2i = t2i - t5i
                let p3r = t3r + t4r, p3i = t3i + t4i, m3r = t3r - t4r, m3i = t3i - t4i
                let o = i + ido * k
                hr[o] = t0r + p1r + p2r + p3r; hi[o] = t0i + p1i + p2i + p3i
                let a1r = t0r + c1 * p1r + c2 * p2r + c3 * p3r, a1i = t0i + c1 * p1i + c2 * p2i + c3 * p3i
                let b1r = s1 * m1r + s2 * m2r + s3 * m3r, b1i = s1 * m1i + s2 * m2i + s3 * m3i
                let a2r = t0r + c2 * p1r + c3 * p2r + c1 * p3r, a2i = t0i + c2 * p1i + c3 * p2i + c1 * p3i
                let b2r = s2 * m1r - s3 * m2r - s1 * m3r, b2i = s2 * m1i - s3 * m2i - s1 * m3i
                let a3r = t0r + c3 * p1r + c1 * p2r + c2 * p3r, a3i = t0i + c3 * p1i + c1 * p2i + c2 * p3i
                let b3r = s3 * m1r - s1 * m2r + s2 * m3r, b3i = s3 * m1i - s1 * m2i + s2 * m3i
                let u1r = a1r - b1i, u1i = a1i + b1r, u6r = a1r + b1i, u6i = a1i - b1r
                let u2r = a2r - b2i, u2i = a2i + b2r, u5r = a2r + b2i, u5i = a2i - b2r
                let u3r = a3r - b3i, u3i = a3i + b3r, u4r = a3r + b3i, u4i = a3i - b3r
                let step = ido * l1
                let p1 = o + step, p2 = p1 + step, p3 = p2 + step, p4 = p3 + step, p5 = p4 + step, p6 = p5 + step
                if i == 0 {
                    hr[p1] = u1r; hi[p1] = u1i; hr[p2] = u2r; hi[p2] = u2i; hr[p3] = u3r; hi[p3] = u3i
                    hr[p4] = u4r; hi[p4] = u4i; hr[p5] = u5r; hi[p5] = u5i; hr[p6] = u6r; hi[p6] = u6i
                } else {
                    let d = ido - 1, j = i - 1
                    (hr[p1], hi[p1]) = twiddled(u1r, u1i, w, j, sign)
                    (hr[p2], hi[p2]) = twiddled(u2r, u2i, w, d + j, sign)
                    (hr[p3], hi[p3]) = twiddled(u3r, u3i, w, 2 * d + j, sign)
                    (hr[p4], hi[p4]) = twiddled(u4r, u4i, w, 3 * d + j, sign)
                    (hr[p5], hi[p5]) = twiddled(u5r, u5i, w, 4 * d + j, sign)
                    (hr[p6], hi[p6]) = twiddled(u6r, u6i, w, 5 * d + j, sign)
                }
            }
        }
    }
}
