//
//  SeparableFilters.swift (ImagoFilters)
//  SciPy 1.18.1's `ndimage.gaussian_filter` (order 0) and `ndimage.uniform_filter` on a 2D row-major array, in the five
//  boundary extensions the HRTEM path uses. Each axis is one 1D correlation done in Double (the reference path, Accelerate
//  `vDSP_convD` / `vDSP_vsmaD`); the intermediate and final arrays are rounded to the caller's precision after every axis, the way
//  SciPy rounds to its input dtype. Axis 0 (down the rows) first, then axis 1, as SciPy.
//
//  The Gaussian kernel is SciPy's `_gaussian_kernel1d`: radius `int(truncate · σ + 0.5)`, weights exp(−x²/2σ²) normalised to a
//  sum of one; σ ≤ 1e-15 skips the axis. The uniform filter's window for size s is [i − s/2, i + s − s/2 − 1] (origin 0).
//

import Foundation
import Accelerate

/// How a filter reads beyond the edge (SciPy's `mode=`; `grid-constant` is `.constant` and `grid-wrap` is `.wrap` here).
public nonisolated enum BoundaryMode: String, Sendable, Equatable, Hashable, CaseIterable {
    /// `d c b a | a b c d | d c b a` (SciPy's default for `gaussian_filter`).
    case reflect
    /// `k k k k | a b c d | k k k k`, k = `cval`.
    case constant
    /// `a a a a | a b c d | d d d d` (scikit-image's default for `gaussian`).
    case nearest
    /// `d c b | a b c d | c b a`.
    case mirror
    /// `a b c d | a b c d | a b c d`.
    case wrap
}

public nonisolated enum FilterError: Error, Equatable, Sendable {
    /// Counts do not match `rows * columns`, or a length is below 1.
    case shape
    /// A sigma is negative or not finite, `truncate` is not positive, or a size is below 1.
    case parameter(String)
    /// High sigma smaller than low sigma on an axis (`difference_of_gaussians`).
    case sigmaOrder
    /// `cutoffFrequencyRatio` outside [0, 0.5] or `npad` below 0.
    case cutoff
    /// An Accelerate setup could not be allocated.
    case setup
}

public nonisolated enum SeparableFilters {
    /// `ndimage.gaussian_filter(a, sigma=(row, column), truncate=, mode=, cval=)` (order 0).
    @_specialize(where S == Float)
    @_specialize(where S == Double)
    public static func gaussianFilter<S: BinaryFloatingPoint & Sendable>(_ a: [S], rows: Int, columns: Int,
                                                              sigma: (row: Double, column: Double), truncate: Double = 4,
                                                              mode: BoundaryMode = .reflect, cval: Double = 0) throws -> [S] {
        guard rows > 0, columns > 0, a.count == rows * columns else { throw FilterError.shape }
        guard sigma.row.isFinite, sigma.column.isFinite, sigma.row >= 0, sigma.column >= 0 else {
            throw FilterError.parameter("sigma")
        }
        guard truncate.isFinite, truncate > 0 else { throw FilterError.parameter("truncate") }
        guard max(sigma.row, sigma.column) * truncate < Double(1 << 24) else { throw FilterError.parameter("sigma · truncate") }
        var data = a.map(Double.init)
        if sigma.row > 1e-15 { data = alongAxis0(data, rows, columns, kernel: gaussianKernel(sigma.row, truncate), mode, cval, S.self) }
        if sigma.column > 1e-15 { data = alongAxis1(data, rows, columns, kernel: gaussianKernel(sigma.column, truncate), mode, cval, S.self) }
        return data.map { S($0) }
    }

    /// `ndimage.uniform_filter(a, size=(row, column), mode=, cval=)`.
    @_specialize(where S == Float)
    @_specialize(where S == Double)
    public static func uniformFilter<S: BinaryFloatingPoint & Sendable>(_ a: [S], rows: Int, columns: Int, size: (row: Int, column: Int),
                                                             mode: BoundaryMode = .reflect, cval: Double = 0) throws -> [S] {
        guard rows > 0, columns > 0, a.count == rows * columns else { throw FilterError.shape }
        guard size.row >= 1, size.column >= 1, max(size.row, size.column) < (1 << 24) else { throw FilterError.parameter("size") }
        var data = a.map(Double.init)
        if size.row > 1 { data = boxAxis0(data, rows, columns, size.row, mode, cval, S.self) }
        if size.column > 1 { data = boxAxis1(data, rows, columns, size.column, mode, cval, S.self) }
        return data.map { S($0) }
    }

    /// SciPy's `_gaussian_kernel1d(sigma, 0, int(truncate·sigma + 0.5))`.
    static func gaussianKernel(_ sigma: Double, _ truncate: Double = 4) -> [Double] {
        let radius = Int(truncate * sigma + 0.5)
        let s2 = sigma * sigma
        var k = (-radius...radius).map { exp(-0.5 / s2 * Double($0 * $0)) }
        let sum = k.reduce(0, +)
        for i in k.indices { k[i] /= sum }
        return k
    }

    // MARK: index mapping

    /// The index in 0..<n a read at `i` takes, or nil for the constant mode's `cval`.
    @inline(__always)
    static func map(_ i: Int, _ n: Int, _ mode: BoundaryMode) -> Int? {
        if i >= 0 && i < n { return i }
        switch mode {
        case .constant: return nil
        case .nearest: return i < 0 ? 0 : n - 1
        case .wrap: return ((i % n) + n) % n
        case .reflect:
            let j = ((i % (2 * n)) + 2 * n) % (2 * n)
            return j < n ? j : 2 * n - 1 - j
        case .mirror:
            if n == 1 { return 0 }
            let p = 2 * n - 2
            let j = ((i % p) + p) % p
            return j < n ? j : p - j
        }
    }

    /// A line extended by `before` on the left and `after` on the right.
    private static func extended(_ line: UnsafeBufferPointer<Double>, _ before: Int, _ after: Int, _ mode: BoundaryMode,
                                 _ cval: Double, into ext: inout [Double]) {
        let n = line.count
        ext.withUnsafeMutableBufferPointer { e in
            for i in 0..<n { e[before + i] = line[i] }
            for i in stride(from: -before, to: 0, by: 1) { e[i + before] = map(i, n, mode).map { line[$0] } ?? cval }
            for i in n..<(n + after) { e[i + before] = map(i, n, mode).map { line[$0] } ?? cval }
        }
    }

    // MARK: Gaussian (correlation)

    private static func alongAxis1<S: BinaryFloatingPoint & Sendable>(_ a: [Double], _ h: Int, _ w: Int, kernel: [Double], _ mode: BoundaryMode,
                                                           _ cval: Double, _: S.Type) -> [Double] {
        let r = kernel.count / 2
        var out = [Double](repeating: 0, count: h * w)
        var ext = [Double](repeating: 0, count: w + 2 * r)
        a.withUnsafeBufferPointer { src in
            for row in 0..<h {
                extended(UnsafeBufferPointer(rebasing: src[row * w ..< (row + 1) * w]), r, r, mode, cval, into: &ext)
                out.withUnsafeMutableBufferPointer { o in
                    vDSP_convD(ext, 1, kernel, 1, o.baseAddress! + row * w, 1, vDSP_Length(w), vDSP_Length(kernel.count))
                }
            }
        }
        return rounded(out, S.self)
    }

    private static func alongAxis0<S: BinaryFloatingPoint & Sendable>(_ a: [Double], _ h: Int, _ w: Int, kernel: [Double], _ mode: BoundaryMode,
                                                           _ cval: Double, _: S.Type) -> [Double] {
        let r = kernel.count / 2
        var out = [Double](repeating: 0, count: h * w)
        a.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { o in
                for row in 0..<h {
                    var constantWeight = 0.0
                    for p in 0..<kernel.count {
                        if let j = map(row + p - r, h, mode) {
                            var wt = kernel[p]
                            vDSP_vsmaD(src.baseAddress! + j * w, 1, &wt, o.baseAddress! + row * w, 1, o.baseAddress! + row * w, 1, vDSP_Length(w))
                        } else {
                            constantWeight += kernel[p]
                        }
                    }
                    if constantWeight != 0 && cval != 0 {
                        var add = constantWeight * cval
                        vDSP_vsaddD(o.baseAddress! + row * w, 1, &add, o.baseAddress! + row * w, 1, vDSP_Length(w))
                    }
                }
            }
        }
        return rounded(out, S.self)
    }

    // MARK: uniform (running sum)

    private static func boxAxis1<S: BinaryFloatingPoint & Sendable>(_ a: [Double], _ h: Int, _ w: Int, _ size: Int, _ mode: BoundaryMode,
                                                         _ cval: Double, _: S.Type) -> [Double] {
        let before = size / 2, after = size - before - 1
        var out = [Double](repeating: 0, count: h * w)
        var ext = [Double](repeating: 0, count: w + before + after)
        let inv = 1.0 / Double(size)
        a.withUnsafeBufferPointer { src in
            for row in 0..<h {
                extended(UnsafeBufferPointer(rebasing: src[row * w ..< (row + 1) * w]), before, after, mode, cval, into: &ext)
                var acc = 0.0
                for p in 0..<size { acc += ext[p] }
                out[row * w] = acc * inv
                for c in 1..<max(w, 1) {
                    acc += ext[c + size - 1] - ext[c - 1]
                    out[row * w + c] = acc * inv
                }
            }
        }
        return rounded(out, S.self)
    }

    private static func boxAxis0<S: BinaryFloatingPoint & Sendable>(_ a: [Double], _ h: Int, _ w: Int, _ size: Int, _ mode: BoundaryMode,
                                                         _ cval: Double, _: S.Type) -> [Double] {
        let before = size / 2
        var out = [Double](repeating: 0, count: h * w)
        var acc = [Double](repeating: 0, count: w)
        let constant = UnsafeMutablePointer<Double>.allocate(capacity: w)
        constant.initialize(repeating: cval, count: w)
        defer { constant.deallocate() }
        let inv = 1.0 / Double(size)
        a.withUnsafeBufferPointer { src in
            func rowPointer(_ i: Int) -> UnsafePointer<Double> {
                if let j = map(i, h, mode) { return src.baseAddress! + j * w }
                return UnsafePointer(constant)
            }
            for p in 0..<size {
                acc.withUnsafeMutableBufferPointer { ac in
                    vDSP_vaddD(ac.baseAddress!, 1, rowPointer(p - before), 1, ac.baseAddress!, 1, vDSP_Length(w))
                }
            }
            for row in 0..<h {
                if row > 0 {
                    acc.withUnsafeMutableBufferPointer { ac in
                        vDSP_vaddD(ac.baseAddress!, 1, rowPointer(row + size - 1 - before), 1, ac.baseAddress!, 1, vDSP_Length(w))
                        vDSP_vsubD(rowPointer(row - 1 - before), 1, ac.baseAddress!, 1, ac.baseAddress!, 1, vDSP_Length(w))
                    }
                }
                for c in 0..<w { out[row * w + c] = acc[c] * inv }
            }
        }
        return rounded(out, S.self)
    }

    private static func rounded<S: BinaryFloatingPoint & Sendable>(_ a: [Double], _: S.Type) -> [Double] {
        S.self == Double.self ? a : a.map { Double(S($0)) }
    }
}
