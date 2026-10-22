//
//  Denoise.swift (ImagoFilters)
//  scikit-image 0.26.0's `restoration.denoise_tv_chambolle`, `denoise_tv_bregman` and `denoise_nl_means` for one 2D
//  single-channel image, Float32 and Float64 (scikit-image keeps a float32 image float32).
//
//  Chambolle is the NumPy code of `_denoise_tv_chambolle_nd` in the image's own precision, every sum a NumPy pairwise sum,
//  so the stop test sees the same energies. Bregman is the Gauss-Seidel loop of the compiled `_denoise_tv_bregman`
//  (reflected one-pixel border, dx/dy/bx/by along rows and columns as that kernel has them). Non-local means follows the
//  compiled `_nl_means_denoising` kernels, which are not the textbook algorithm in four places the references pins:
//    - the patch weight is not exp(-d) but a Schraudolph-style bit-pattern approximation of it (an Int32 built from
//      -d / ln 2 * 2^20, offset by 1072632447, written into the high word of a double). It is piecewise linear, weighs
//      a zero distance 0.9711, and is what the pins hold;
//    - `fast_mode=False` (Gaussian-weighted s x s patches, search window clipped to the image) runs in the image's own
//      precision with fused multiply-adds and gives up on a patch pair once the running distance passes 5 at the start of
//      a patch row; `fast_mode=True` (integral images) always computes in Float64 and rounds once at the end, and its patch
//      is the s - 1 rows and columns from (pixel - s/2 + 1) to (pixel + s/2), still normalised by s^2;
//    - borders are NumPy's iterative `reflect` pad of s/2 + patch_distance + 1 pixels, also on axes of length 1.
//  Each is recreated from the compiled kernels' machine code and checked against the compiled package (docs/parity.md).
//

import Foundation

/// Why a denoising call was refused.
public nonisolated enum DenoiseError: Error, Equatable, Sendable {
    /// `rows` or `columns` below 1, or the sample count is not `rows * columns`.
    case shape
    /// A sample is NaN or infinite.
    case nonFinite
    /// A parameter is out of range; the text names it.
    case parameter(String)
    /// The image is smaller than the method can handle; the text says by how much.
    case tooSmall(String)
}

public nonisolated enum Denoise {
    // MARK: - Total variation, Chambolle

    /// `skimage.restoration.denoise_tv_chambolle(image, weight=0.1, eps=2e-4, max_num_iter=200)` for a 2D image (row-major).
    /// `weight`, `eps` and `maxNumIter` must be positive. The result has the image's precision.
    @_specialize(where S == Float)
    @_specialize(where S == Double)
    public static func totalVariationChambolle<S: BinaryFloatingPoint & Sendable>(
        _ image: [S], rows: Int, columns: Int, weight: Double = 0.1, eps: Double = 2.0e-4, maxNumIter: Int = 200
    ) throws -> [S] {
        try checkImage(image, rows: rows, columns: columns)
        try requirePositive(weight, "weight")
        try requirePositive(eps, "eps")
        guard maxNumIter >= 1 else { throw DenoiseError.parameter("maxNumIter must be at least 1") }
        let n = rows * columns
        var p0 = [S](repeating: 0, count: n), p1 = p0, g0 = p0, g1 = p0, d = p0, norm = p0, tmp = p0
        var out = image
        let weightS = S(weight), tau = S(0.25), scale = S(0.25 / weight), epsS = S(eps), size = S(n)
        var ePrevious: S = 0, eInit: S = 0
        var i = 0
        while i < maxNumIter {
            if i > 0 {
                for k in 0..<n { d[k] = -(p0[k] + p1[k]) }
                if rows > 1 {
                    for r in 1..<rows { for c in 0..<columns { d[r * columns + c] += p0[(r - 1) * columns + c] } }
                }
                if columns > 1 {
                    for r in 0..<rows { for c in 1..<columns { d[r * columns + c] += p1[r * columns + c - 1] } }
                }
                for k in 0..<n { out[k] = image[k] + d[k] }
            }
            for k in 0..<n { tmp[k] = d[k] * d[k] }
            var e = pairwiseSum(tmp)
            if rows > 1 {
                for r in 0..<(rows - 1) { for c in 0..<columns { g0[r * columns + c] = out[(r + 1) * columns + c] - out[r * columns + c] } }
            }
            if columns > 1 {
                for r in 0..<rows { for c in 0..<(columns - 1) { g1[r * columns + c] = out[r * columns + c + 1] - out[r * columns + c] } }
            }
            for k in 0..<n { norm[k] = (g0[k] * g0[k] + g1[k] * g1[k]).squareRoot() }
            e += weightS * pairwiseSum(norm)
            for k in 0..<n {
                let scaled = norm[k] * scale
                norm[k] = scaled + 1
                p0[k] = (p0[k] - tau * g0[k]) / norm[k]
                p1[k] = (p1[k] - tau * g1[k]) / norm[k]
            }
            e /= size
            if i == 0 {
                eInit = e
                ePrevious = e
            } else if abs(ePrevious - e) < epsS * eInit {
                break
            } else {
                ePrevious = e
            }
            i += 1
        }
        return out
    }

    // MARK: - Total variation, split Bregman

    /// `skimage.restoration.denoise_tv_bregman(image, weight=5.0, max_num_iter=100, eps=1e-3, isotropic=True)` for a 2D
    /// image (row-major). `weight`, `eps` and `maxNumIter` must be positive and the image at least 2 x 2: the compiled
    /// kernel reflects the border from the second row and column and reads outside a 1-pixel axis.
    @_specialize(where S == Float)
    @_specialize(where S == Double)
    public static func totalVariationBregman<S: BinaryFloatingPoint & Sendable>(
        _ image: [S], rows: Int, columns: Int, weight: Double = 5.0, maxNumIter: Int = 100, eps: Double = 1.0e-3,
        isotropic: Bool = true
    ) throws -> [S] {
        try checkImage(image, rows: rows, columns: columns)
        try requirePositive(weight, "weight")
        try requirePositive(eps, "eps")
        guard maxNumIter >= 1 else { throw DenoiseError.parameter("maxNumIter must be at least 1") }
        guard rows >= 2, columns >= 2 else { throw DenoiseError.tooSmall("total variation (Bregman) needs at least 2 rows and 2 columns") }
        let nc = columns + 2, size = (rows + 2) * nc
        var out = [S](repeating: 0, count: size), dx = out, dy = out, bx = out, by = out
        for r in 0..<rows { for c in 0..<columns { out[(r + 1) * nc + c + 1] = image[r * columns + c] } }
        for c in 0..<columns {
            out[c + 1] = image[columns + c]
            out[(rows + 1) * nc + c + 1] = image[(rows - 1) * columns + c]
        }
        for r in 0..<rows {
            out[(r + 1) * nc] = image[r * columns + 1]
            out[(r + 1) * nc + columns + 1] = image[r * columns + columns - 1]
        }
        let w = S(weight), lam = S(2) * w, norm = w + S(4) * lam, one = S(1), total = S(rows * columns)
        let inverseLambda = one / lam
        var rmse: S = .infinity
        var iteration = 0
        while iteration < maxNumIter && Double(rmse) > eps {
            rmse = 0
            for r in 1...rows {
                for c in 1...columns {
                    let i = r * nc + c
                    let uprev = out[i]
                    let ux = out[i + nc] - uprev
                    let uy = out[i + 1] - uprev
                    let sum1 = out[i + nc] + out[i - nc] + out[i + 1] + out[i - 1]
                    let sum2 = sum1 + dx[i - nc] - dx[i] + dy[i - 1] - dy[i]
                    let sum3 = sum2 - bx[i - nc] + bx[i] - by[i - 1] + by[i]
                    let unew = (lam * sum3 + w * image[(r - 1) * columns + c - 1]) / norm
                    out[i] = unew
                    let diff = unew - uprev
                    rmse += diff * diff
                    let bxx = bx[i], byy = by[i]
                    var dxx: S, dyy: S
                    if isotropic {
                        let s = ((ux + bxx) * (ux + bxx) + (uy + byy) * (uy + byy)).squareRoot()
                        dxx = s * lam * (ux + bxx) / (s * lam + one)
                        dyy = s * lam * (uy + byy) / (s * lam + one)
                    } else {
                        var s = ux + bxx
                        dxx = s > inverseLambda ? s - inverseLambda : (s < -inverseLambda ? s + inverseLambda : 0)
                        s = uy + byy
                        dyy = s > inverseLambda ? s - inverseLambda : (s < -inverseLambda ? s + inverseLambda : 0)
                    }
                    dx[i] = dxx
                    dy[i] = dyy
                    bx[i] += ux - dxx
                    by[i] += uy - dyy
                }
            }
            rmse = (rmse / total).squareRoot()
            iteration += 1
        }
        var result = [S](repeating: 0, count: rows * columns)
        for r in 0..<rows { for c in 0..<columns { result[r * columns + c] = out[(r + 1) * nc + c + 1] } }
        return result
    }

    // MARK: - Non-local means

    /// `skimage.restoration.denoise_nl_means(image, patch_size=7, patch_distance=11, h=0.1, fast_mode=True, sigma=0.0)` for
    /// a 2D image (row-major). `patchSize` (at least 2; an even size is raised by one, as scikit-image does) and
    /// `patchDistance` are in pixels and at most 1024, `h` is positive and `sigma` is not negative.
    /// `fastMode` computes in Float64 (rounded once at the end), the exact mode in the image's own precision.
    @_specialize(where S == Float)
    @_specialize(where S == Double)
    public static func nonLocalMeans<S: BinaryFloatingPoint & Sendable>(
        _ image: [S], rows: Int, columns: Int, patchSize: Int = 7, patchDistance: Int = 11, h: Double = 0.1,
        fastMode: Bool = true, sigma: Double = 0.0
    ) throws -> [S] {
        try checkImage(image, rows: rows, columns: columns)
        guard patchSize >= 2, patchSize <= 1024 else { throw DenoiseError.parameter("patchSize must be between 2 and 1024") }
        guard patchDistance >= 1, patchDistance <= 1024 else { throw DenoiseError.parameter("patchDistance must be between 1 and 1024") }
        try requirePositive(h, "h")
        guard sigma.isFinite, sigma >= 0 else { throw DenoiseError.parameter("sigma must be finite and not negative") }
        let s = patchSize % 2 == 0 ? patchSize + 1 : patchSize
        let variance = sigma * sigma
        if fastMode {
            let wide = image.map { Double($0) }
            return nonLocalMeansFast(wide, rows: rows, columns: columns, s: s, d: patchDistance, h: h, variance: variance).map { S($0) }
        }
        return nonLocalMeansExact(image, rows: rows, columns: columns, s: s, d: patchDistance, h: h, variance: variance)
    }

    // MARK: - Non-local means, fast mode (integral images, Float64)

    private static func nonLocalMeansFast(_ image: [Double], rows: Int, columns: Int, s: Int, d: Int, h: Double, variance: Double) -> [Double] {
        let off = s / 2, pad = off + d + 1
        let nr = rows + 2 * pad, nc = columns + 2 * pad
        let padded = reflectPad(image, rows: rows, columns: columns, pad: pad)
        var result = [Double](repeating: 0, count: nr * nc), weights = result, integral = result
        let h2s2 = ((h * 1.0 * h) * Double(s)) * Double(s)
        let twoVar = variance + variance
        for tRow in -d...d {
            let rowStart = max(off, off - tRow), rowEnd = min(nr - off, nr - off - tRow)
            let rLo = max(1, -tRow), rHi = min(nr, nr - tRow)
            for tCol in 0...d {
                let colStart = max(off, off - tCol), colEnd = min(nc - off, nc - off - tCol)
                let cLo = max(1, -tCol), cHi = min(nc, nc - tCol)
                let alpha = tCol == 0 ? 0.5 : 1.0
                if rLo < rHi && cLo < cHi {
                    for r in rLo..<rHi {
                        for c in cLo..<cHi {
                            let diff = padded[r * nc + c] - padded[(r + tRow) * nc + c + tCol]
                            var t = 0.0.addingProduct(diff, diff)
                            t = t.addingProduct(-1.0, twoVar)
                            integral[r * nc + c] = ((t + integral[(r - 1) * nc + c]) + integral[r * nc + c - 1]) - integral[(r - 1) * nc + c - 1]
                        }
                    }
                }
                guard rowStart < rowEnd, colStart < colEnd else { continue }
                for r in rowStart..<rowEnd {
                    for c in colStart..<colEnd {
                        var distance = ((integral[(r + off) * nc + c + off] + integral[(r - off) * nc + c - off])
                                        - integral[(r - off) * nc + c + off]) - integral[(r + off) * nc + c - off]
                        if distance < 0 { distance = 0 }
                        distance /= h2s2
                        if distance > 5.0 { continue }
                        let weight = alpha * schraudolphExp(distance)
                        let here = r * nc + c, there = (r + tRow) * nc + c + tCol
                        weights[here] += weight
                        weights[there] += weight
                        result[here] = result[here].addingProduct(weight, padded[there])
                        result[there] = result[there].addingProduct(weight, padded[here])
                    }
                }
            }
        }
        var out = [Double](repeating: 0, count: rows * columns)
        for r in 0..<rows {
            for c in 0..<columns {
                let k = (pad + r) * nc + pad + c
                out[r * columns + c] = result[k] / weights[k]
            }
        }
        return out
    }

    // MARK: - Non-local means, exact mode (Gaussian-weighted patches, image precision)

    @_specialize(where S == Float)
    @_specialize(where S == Double)
    private static func nonLocalMeansExact<S: BinaryFloatingPoint>(
        _ image: [S], rows: Int, columns: Int, s: Int, d: Int, h: Double, variance: Double
    ) -> [S] {
        let off = s / 2, pad = off + d + 1
        let nc = columns + 2 * pad
        let padded = reflectPad(image, rows: rows, columns: columns, pad: pad)
        // the Gaussian patch weights, built the way the kernel's NumPy lines build them, in the image's precision
        let a = (Double(s) - 1.0) / 4.0
        let twoASquared = S(2.0 * a * a)
        var w = [S](repeating: 0, count: s * s)
        for i in 0..<s {
            for j in 0..<s {
                let xr = S(i - off), xc = S(j - off)
                w[i * s + j] = expOf(-(xr * xr + xc * xc) / twoASquared)
            }
        }
        let normaliser = S(1.0) / (((S(1) * pairwiseSum(w)) * S(h)) * S(h))
        for k in 0..<w.count { w[k] *= normaliser }
        let minusTwoVar = S(-(2.0 * variance))
        let five = S(5)
        var out = [S](repeating: 0, count: rows * columns)
        padded.withUnsafeBufferPointer { P in
            w.withUnsafeBufferPointer { W in
                for row in 0..<rows {
                    let i0 = row - min(d, row), i1 = row + min(d + 1, rows - row)
                    for col in 0..<columns {
                        let j0 = col - min(d, col), j1 = col + min(d + 1, columns - col)
                        var weightSum: S = 0, newValue: S = 0
                        let centre = (pad + row - off) * nc + pad + col - off
                        for i in i0..<i1 {
                            for j in j0..<j1 {
                                let other = (pad + i - off) * nc + pad + j - off
                                var distance: S = 0
                                var complete = true
                                for pr in 0..<s {
                                    if distance > five { complete = false; break }
                                    let c1 = centre + pr * nc, c2 = other + pr * nc
                                    for pc in 0..<s {
                                        let t = P[c1 + pc] - P[c2 + pc]
                                        let sq = minusTwoVar.addingProduct(t, t)
                                        distance = distance.addingProduct(W[pr * s + pc], sq)
                                    }
                                }
                                var weight: S = 0
                                if complete {
                                    weight = S(schraudolphExp(distance > 0 ? Double(distance) : 0))
                                }
                                weightSum += weight
                                newValue = newValue.addingProduct(weight, P[(pad + i) * nc + pad + j])
                            }
                        }
                        out[row * columns + col] = newValue / weightSum
                    }
                }
            }
        }
        return out
    }

    // MARK: - Shared helpers

    private static func checkImage<S: BinaryFloatingPoint>(_ image: [S], rows: Int, columns: Int) throws {
        guard rows >= 1, columns >= 1 else { throw DenoiseError.shape }
        let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
        guard !overflow, image.count == count else { throw DenoiseError.shape }
        for v in image where !v.isFinite { throw DenoiseError.nonFinite }
    }

    private static func requirePositive(_ value: Double, _ name: String) throws {
        guard value.isFinite, value > 0 else { throw DenoiseError.parameter("\(name) must be finite and positive") }
    }

    /// exp in the image's precision (`expf` for Float, `exp` otherwise).
    @inline(__always)
    private static func expOf<S: BinaryFloatingPoint>(_ x: S) -> S {
        if S.self == Float.self { return S(Foundation.expf(Float(x))) }
        return S(Foundation.exp(Double(x)))
    }

    /// The kernel's weight for a non-negative distance: `Int32(-d * 2^20 / ln 2)` (saturating, truncating) plus 1072632447
    /// (wrapping), as the high 32 bits of a double. `d` of zero or below gives 0.9711.
    @inline(__always)
    private static func schraudolphExp(_ distance: Double) -> Double {
        let scale = Double(bitPattern: 0xc1371547652b82fe)  // -2^20 / ln 2
        let x = distance > 0 ? distance * scale : 0
        let truncated: Int32 = x <= -2147483648.0 ? Int32.min : (x >= 2147483647.0 ? Int32.max : Int32(x))
        let high = UInt32(bitPattern: truncated &+ 0x3fef127f)
        return Double(bitPattern: UInt64(high) << 32)
    }

    /// NumPy's `ndarray.sum()` of a contiguous float array: the pairwise sum with eight accumulators per block of up to 128.
    private static func pairwiseSum<S: BinaryFloatingPoint>(_ a: [S]) -> S {
        a.withUnsafeBufferPointer { pairwise($0, 0, $0.count) }
    }

    private static func pairwise<S: BinaryFloatingPoint>(_ a: UnsafeBufferPointer<S>, _ lo: Int, _ n: Int) -> S {
        if n < 8 {
            var res: S = 0
            for i in 0..<n { res += a[lo + i] }
            return res
        } else if n <= 128 {
            var r0 = a[lo], r1 = a[lo + 1], r2 = a[lo + 2], r3 = a[lo + 3]
            var r4 = a[lo + 4], r5 = a[lo + 5], r6 = a[lo + 6], r7 = a[lo + 7]
            var i = 8
            while i < n - (n % 8) {
                r0 += a[lo + i]; r1 += a[lo + i + 1]; r2 += a[lo + i + 2]; r3 += a[lo + i + 3]
                r4 += a[lo + i + 4]; r5 += a[lo + i + 5]; r6 += a[lo + i + 6]; r7 += a[lo + i + 7]
                i += 8
            }
            var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < n { res += a[lo + i]; i += 1 }
            return res
        }
        var half = n / 2
        half -= half % 8
        return pairwise(a, lo, half) + pairwise(a, lo + half, n - half)
    }

    /// `np.pad(image, pad, mode='reflect')` of a 2D array: NumPy fills wide pads by repeated reflection (and an axis of
    /// length 1 by its single value), which is not the periodic triangle wave, so its index bookkeeping is replayed.
    private static func reflectPad<S>(_ image: [S], rows: Int, columns: Int, pad: Int) -> [S] {
        let ri = reflectIndices(rows, pad), ci = reflectIndices(columns, pad)
        let nc = columns + 2 * pad
        var out = [S]()
        out.reserveCapacity(ri.count * nc)
        for r in ri { for c in ci { out.append(image[r * columns + c]) } }
        return out
    }

    private static func reflectIndices(_ n: Int, _ pad: Int) -> [Int] {
        let total = n + 2 * pad
        if n == 1 { return [Int](repeating: 0, count: total) }
        var a = [Int](repeating: -1, count: total)
        for i in 0..<n { a[pad + i] = i }
        var left = pad, right = pad
        while left > 0 || right > 0 {
            let current = total - right - left
            let old = ((current - 1) / (n - 1)) * (n - 1)
            if left > 0 {
                let chunk = min(old, left)
                let stop = left, start = left + chunk
                let destination = left - chunk
                var k = 0
                var source = start
                while source > stop { a[destination + k] = a[source]; k += 1; source -= 1 }
                left -= chunk
            }
            if right > 0 {
                let chunk = min(old, right)
                let destination = total - right
                var source = total - right - 2
                for k in 0..<chunk { a[destination + k] = a[source]; source -= 1 }
                right -= chunk
            }
        }
        return a
    }
}
