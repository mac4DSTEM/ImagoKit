// SPDX-License-Identifier: GPL-3.0-only
//
//  DistanceTransform.swift (ImagoCore)
//  SciPy 1.18.1's `ndimage.distance_transform_edt` on a 2D mask: for every foreground pixel the Euclidean distance to the nearest
//  background pixel, 0 on the background, with optional per-axis `sampling` (the pixel pitch along rows and along columns).
//
//  METHOD. The exact separable transform of Felzenszwalb and Huttenlocher: the squared distance to the nearest background pixel is
//  the minimum over background pixels of s_r^2 (r - r')^2 + s_c^2 (c - c')^2, found by a lower-envelope pass along each axis.
//  It is exact, as SciPy's Maurer-type algorithm is, so the two agree to rounding in the square root (the pins hold 1e-12).
//  `euclideanWithIndices` is SciPy's feature transform itself (`_ComputeFT`/`_VoronoiFT`), ported term by term with the fused
//  multiply-adds of its arm64 build, so the indices, ties included, are SciPy's exactly.
//
//  DEVIATION. An image with no background pixel is refused (`MorphologyError.noBackground`): SciPy then returns distances to a
//  point outside the image that its implementation picks (row -1, column 0 for a 4 x 6 image of ones), which is no distance.
//

import Foundation

public nonisolated enum DistanceTransform {
    /// `distance_transform_edt(mask, sampling=, return_indices=True)`: the distances and, per pixel, the row and column of the
    /// background pixel they are measured to (SciPy's feature transform; a background pixel names itself).
    public struct NearestBackground: Sendable, Equatable {
        /// `distances[row * columns + column]`, as `euclidean(foreground:rows:columns:sampling:)` (to 1e-12).
        public let distances: [Double]
        /// The row of the nearest background pixel, `indices[0]` in SciPy.
        public let nearestRows: [Int]
        /// The column of the nearest background pixel, `indices[1]` in SciPy.
        public let nearestColumns: [Int]
    }

    /// `distance_transform_edt(mask, sampling=)`: `mask[row * width + column]` true is foreground. `sampling` is (rows, columns);
    /// both must be finite and positive.
    public static func euclidean(foreground mask: [Bool], rows height: Int, columns width: Int,
                                 sampling: (rows: Double, columns: Double) = (1, 1)) throws -> [Double] {
        let n = try validate(mask, height, width, sampling)
        // Pass 1, down each column: squared distance (in sampled units) to the nearest background pixel of that column.
        var g = [Double](repeating: .infinity, count: n)
        var f = [Double](repeating: 0, count: max(width, height))
        var out = [Double](repeating: 0, count: max(width, height))
        for c in 0..<width {
            for r in 0..<height { f[r] = mask[r * width + c] ? .infinity : 0 }
            envelope(f, count: height, step: sampling.rows, into: &out)
            for r in 0..<height { g[r * width + c] = out[r] }
        }
        // Pass 2, along each row, over the squared distances of pass 1.
        var result = [Double](repeating: 0, count: n)
        for r in 0..<height {
            for c in 0..<width { f[c] = g[r * width + c] }
            envelope(f, count: width, step: sampling.columns, into: &out)
            for c in 0..<width { result[r * width + c] = out[c].squareRoot() }
        }
        return result
    }

    /// `distance_transform_edt(mask, sampling=, return_indices=True)`: SciPy's feature transform (`euclidean_feature_transform`,
    /// the Voronoi sweep of Maurer, Qi and Raghavan), down each column and then along each row, with its tie rule: of two equally
    /// near background pixels the sweep keeps the one it met first. The distances are then computed from the indices as SciPy
    /// does: sqrt(((row' - row) s_r)^2 + ((column' - column) s_c)^2). Same refusals as `euclidean(foreground:rows:columns:sampling:)`.
    public static func euclideanWithIndices(foreground mask: [Bool], rows height: Int, columns width: Int,
                                            sampling: (rows: Double, columns: Double) = (1, 1)) throws -> NearestBackground {
        let n = try validate(mask, height, width, sampling)
        let s = (sampling.rows, sampling.columns)
        var ftRow = [Int](repeating: 0, count: n), ftColumn = [Int](repeating: 0, count: n)
        var lineRow = [Int](repeating: 0, count: max(width, height)), lineColumn = lineRow
        // Pass 1 (SciPy's d = 0), down each column: a background pixel names itself, a foreground one gets row -1 (column 0, from
        // SciPy's zeroed output); a column without background keeps that.
        for c in 0..<width {
            for r in 0..<height {
                let isForeground = mask[r * width + c]
                lineRow[r] = isForeground ? -1 : r
                lineColumn[r] = isForeground ? 0 : c
            }
            voronoi(&lineRow, &lineColumn, count: height, axis: 0, coordinates: (0, c), sampling: s)
            for r in 0..<height { ftRow[r * width + c] = lineRow[r]; ftColumn[r * width + c] = lineColumn[r] }
        }
        // Pass 2 (d = 1), along each row over the features of pass 1.
        for r in 0..<height {
            for c in 0..<width { lineRow[c] = ftRow[r * width + c]; lineColumn[c] = ftColumn[r * width + c] }
            voronoi(&lineRow, &lineColumn, count: width, axis: 1, coordinates: (r, 0), sampling: s)
            for c in 0..<width { ftRow[r * width + c] = lineRow[c]; ftColumn[r * width + c] = lineColumn[c] }
        }
        var distances = [Double](repeating: 0, count: n)
        for r in 0..<height {
            for c in 0..<width {
                let i = r * width + c
                let dr = Double(ftRow[i] - r) * s.0, dc = Double(ftColumn[i] - c) * s.1
                distances[i] = (dr * dr + dc * dc).squareRoot()
            }
        }
        return NearestBackground(distances: distances, nearestRows: ftRow, nearestColumns: ftColumn)
    }

    private static func validate(_ mask: [Bool], _ height: Int, _ width: Int, _ sampling: (rows: Double, columns: Double)) throws -> Int {
        let (n, overflow) = width.multipliedReportingOverflow(by: height)
        guard width > 0, height > 0, !overflow, mask.count == n else { throw MorphologyError.shape }
        guard sampling.rows.isFinite, sampling.rows > 0, sampling.columns.isFinite, sampling.columns > 0,
              (sampling.rows * sampling.rows).isNormal, (sampling.columns * sampling.columns).isNormal,
              (sampling.rows * Double(height)).isFinite, (sampling.columns * Double(width)).isFinite,
              (sampling.rows * Double(height) * sampling.rows * Double(height)).isFinite,
              (sampling.columns * Double(width) * sampling.columns * Double(width)).isFinite else {
            throw MorphologyError.parameter("sampling must be finite and positive, with squares that neither underflow nor overflow")
        }
        guard mask.contains(false) else { throw MorphologyError.noBackground }
        return n
    }

    /// SciPy's `_VoronoiFT` on one line of a 2D feature array: `row[i]`, `column[i]` is the feature at position i along `axis`
    /// (row -1: none yet); `coordinates` holds the line's fixed coordinate on the other axis. The arithmetic is SciPy's, term by term,
    /// with the fused multiply-adds of the reference macOS arm64 build (they decide exact ties under a non-dyadic `sampling`; a SciPy
    /// build without them can break such a tie the other way).
    private static func voronoi(_ row: inout [Int], _ column: inout [Int], count len: Int, axis d: Int, coordinates: (Int, Int),
                                sampling: (Double, Double)) {
        let f0 = Array(row[0..<len]), f1 = Array(column[0..<len])
        @inline(__always) func f(_ i: Int, _ jj: Int) -> Int { jj == 0 ? f0[i] : f1[i] }
        let coor = [coordinates.0, coordinates.1], samp = [sampling.0, sampling.1]
        let o = 1 - d
        var g = [Int](repeating: 0, count: len)
        var l = -1
        for ii in 0..<len where f0[ii] >= 0 {
            let fd = Double(f(ii, d))
            var wR = 0.0
            let tw = Double(f(ii, o) - coor[o]) * samp[o]
            wR += tw * tw
            while l >= 1 {
                let idx1 = g[l], idx2 = g[l - 1]
                let fl = Double(f(idx1, d))
                let a = (fl - Double(f(idx2, d))) * samp[d]
                let b = (fd - fl) * samp[d]
                let c = a + b
                let cc = Double(coor[o])
                let tu = (Double(f(idx2, o)) - cc) * samp[o], tv = (Double(f(idx1, o)) - cc) * samp[o]
                var uR = 0.0, vR = 0.0
                uR += tu * tu
                vR += tv * tv
                // c vR - b uR - a wR - a b c, contracted as the arm64 build of SciPy compiles it.
                if (-(b * uR)).addingProduct(c, vR).addingProduct(-a, wR).addingProduct(-(a * b), c) <= 0 { break }
                l -= 1
            }
            l += 1
            g[l] = ii
        }
        let maxl = l
        guard maxl >= 0 else { return }
        // The squared distance, row term first, the column term fused into it as SciPy's arm64 build does.
        @inline(__always) func delta(_ k: Int, _ ii: Int) -> Double {
            let t0 = Double(d == 0 ? f(k, 0) - ii : f(k, 0) - coor[0]) * samp[0]
            let t1 = Double(d == 1 ? f(k, 1) - ii : f(k, 1) - coor[1]) * samp[1]
            return (t0 * t0).addingProduct(t1, t1)
        }
        l = 0
        for ii in 0..<len {
            var delta1 = delta(g[l], ii)
            while l < maxl {
                let delta2 = delta(g[l + 1], ii)
                if delta1 <= delta2 { break }
                delta1 = delta2
                l += 1
            }
            row[ii] = f0[g[l]]
            column[ii] = f1[g[l]]
        }
    }

    /// The lower envelope of the parabolas `f[q] + (step * (x - q))^2` over the finite `f[q]`, evaluated at x = 0..<count.
    private static func envelope(_ f: [Double], count: Int, step: Double, into out: inout [Double]) {
        var v = [Int](), z = [Double]()
        v.reserveCapacity(count); z.reserveCapacity(count + 1)
        let s2 = step * step
        for q in 0..<count where f[q].isFinite {
            if v.isEmpty { v.append(q); z.append(-.infinity); z.append(.infinity); continue }
            var s: Double
            while true {
                let p = v[v.count - 1]
                s = ((f[q] / s2 + Double(q * q)) - (f[p] / s2 + Double(p * p))) / Double(2 * q - 2 * p)
                if s <= z[z.count - 2], v.count > 1 { v.removeLast(); z.removeLast(); z[z.count - 1] = .infinity; continue }
                break
            }
            z[z.count - 1] = s
            v.append(q); z.append(.infinity)
        }
        guard !v.isEmpty else { for x in 0..<count { out[x] = .infinity }; return }
        var k = 0
        for x in 0..<count {
            while z[k + 1] < Double(x) { k += 1 }
            let d = Double(x - v[k]) * step
            out[x] = d * d + f[v[k]]
        }
    }
}
