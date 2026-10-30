// SPDX-License-Identifier: GPL-3.0-only
//
//  DistanceTransform.swift (ImagoCore)
//  SciPy 1.18.1's `ndimage.distance_transform_edt` on a 2D mask: for every foreground pixel the Euclidean distance to the nearest
//  background pixel, 0 on the background, with optional per-axis `sampling` (the pixel pitch along rows and along columns).
//
//  METHOD. The exact separable transform of Felzenszwalb and Huttenlocher: the squared distance to the nearest background pixel is
//  the minimum over background pixels of s_r^2 (r - r')^2 + s_c^2 (c - c')^2, found by a lower-envelope pass along each axis.
//  It is exact, as SciPy's Maurer-type algorithm is, so the two agree to rounding in the square root (the pins hold 1e-12).
//
//  DEVIATION. An image with no background pixel is refused (`MorphologyError.noBackground`): SciPy then returns distances to a
//  point outside the image that its implementation picks (row -1, column 0 for a 4 x 6 image of ones), which is no distance.
//

import Foundation

public nonisolated enum DistanceTransform {
    /// `distance_transform_edt(mask, sampling=)`: `mask[row * width + column]` true is foreground. `sampling` is (rows, columns);
    /// both must be finite and positive.
    public static func euclidean(foreground mask: [Bool], rows height: Int, columns width: Int,
                                 sampling: (rows: Double, columns: Double) = (1, 1)) throws -> [Double] {
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
