// SPDX-License-Identifier: GPL-3.0-only
//
//  GeometricPhaseAnalysis.swift (ImagoFilters)
//  Geometric phase analysis (GPA) of a lattice image after Hytch, Snoeck and Kilaas, Ultramicroscopy 74 (1998) 131.
//
//  Conventions (rows and columns, never x and y): every vector is (row, column); a reciprocal vector g is in cycles per pixel,
//  g = (g_row, g_col); a lattice image is I(r) = sum over +-g of a cos(2 pi g . (r - u(r))) with u the displacement (pixels),
//  so the demodulated Bragg-filtered image is H_g = (a/2) exp(i P_g) with P_g = -2 pi g . u, grad P_g = -2 pi G D, G the 2x2
//  matrix whose rows are g1, g2, and D_ij = du_i / dr_j. D is returned; strain is its symmetric part, rotation
//  (D_rc - D_cr) / 2 its antisymmetric part (small-deformation form, no second order terms).
//
//  Steps per reflection: FFT (FFTSurface, Float64), Gaussian mask centred on +g (isotropic, sigma in cycles per pixel),
//  inverse FFT, multiplication by exp(-2 pi i g . r) (the perfect carrier), phase gradient without unwrapping,
//  Im(conj(H) grad H) / |H|^2, with grad by central differences (one-sided on the frame edge, as numpy.gradient).
//  The mask distance is measured on the signed frequency axis (no periodic aliasing). Everything is Float64 on the CPU,
//  the reference path; there is no Metal path yet.
//

import Foundation
import ImagoCore

public nonisolated enum GeometricPhaseAnalysisError: Error, Equatable, Sendable {
    /// The pixel count is not rows x columns, or a side is below 8.
    case invalidSize
    /// A pixel is NaN or infinite.
    case nonFinitePixel
    /// g1 and g2 are (nearly) collinear: |sin(angle between them)| below 0.02.
    case collinearReflections
    /// A reflection has a component with |g| >= 0.5 cycles per pixel, is not finite, or is zero.
    case reflectionOutsideNyquist
    /// The mask sigma is not a positive finite number.
    case nonPositiveSigma
    /// The reference window is empty or leaves the image.
    case invalidReferenceWindow
    /// The FFT could not be set up.
    case fftSetup
    /// `findReflections` found no two non-collinear peaks.
    case reflectionsNotFound
}

public nonisolated enum GeometricPhaseAnalysis {
    /// A rows x columns window of the image whose mean distortion is taken as zero (the usual GPA reference).
    public nonisolated struct ReferenceWindow: Sendable, Equatable {
        public var rows: Range<Int>
        public var columns: Range<Int>
        public init(rows: Range<Int>, columns: Range<Int>) { self.rows = rows; self.columns = columns }
    }

    /// Full-size, row-major, `rows * columns` arrays.
    ///
    /// `validation` is "none": the result agrees with planted fields (tools/gpa/planted.py) and has not been checked on a
    /// dataset with truth, so an app must show it as unvalidated until one passes.
    public nonisolated struct Result: Sendable {
        /// The badge every result carries until a dataset with truth passes.
        public static let validationNone = "none"
        public let validation: String = "none"
        public var rows: Int
        public var columns: Int
        /// The reflections used, (g_row, g_col) in cycles per pixel.
        public var g1: (row: Double, column: Double)
        public var g2: (row: Double, column: Double)
        /// The Gaussian mask sigma used, cycles per pixel.
        public var sigma: Double
        /// D_ij = du_i / dr_j after the reference subtraction: dRR = du_row/d_row, dRC = du_row/d_col, dCR, dCC.
        public var dRR: [Double], dRC: [Double], dCR: [Double], dCC: [Double]
        /// Symmetric part of D.
        public var strainRR: [Double], strainCC: [Double], strainRC: [Double]
        /// (D_rc - D_cr) / 2, radians; positive when u_row grows along the columns faster than u_col grows along the rows.
        public var rotation: [Double]
        /// Wrapped phase of the demodulated filtered image, P_g = -2 pi g . u + const, in (-pi, pi]. For display only; D never unwraps it.
        public var phase1: [Double], phase2: [Double]
        /// |H_g| of the filtered, demodulated image (half the cosine amplitude of the lattice fringe).
        public var amplitude1: [Double], amplitude2: [Double]
        /// The mean D removed by the reference window (zeros when none).
        public var referenceOffset: (dRR: Double, dRC: Double, dCR: Double, dCC: Double)
    }

    /// Default mask sigma, in units of the shorter of |g1| and |g2| (the reciprocal-lattice spacing): 0.2, which puts the nearest
    /// other reflection (at least one spacing away) 5 sigma from the mask centre. A smaller value gives smoother strain with
    /// coarser resolution, about 1 / (2 pi sigma) pixels.
    public static let defaultSigmaInLatticeUnits = 0.2

    /// GPA of a real lattice image.
    /// - Parameters:
    ///   - image: row-major `rows * columns`, finite.
    ///   - g1, g2: reflections (g_row, g_col) in cycles per pixel, non-collinear, each component inside (-0.5, 0.5).
    ///   - sigmaInLatticeUnits: mask sigma as a multiple of min(|g1|, |g2|); default `defaultSigmaInLatticeUnits`.
    ///   - reference: optional window whose mean D is subtracted from D before strain and rotation are formed.
    /// Pixels where the filtered amplitude is below 1e-10 of its maximum have no phase; D there is NaN.
    public static func analyze(_ image: [Double], rows: Int, columns: Int,
                               g1: (row: Double, column: Double), g2: (row: Double, column: Double),
                               sigmaInLatticeUnits: Double = defaultSigmaInLatticeUnits,
                               reference: ReferenceWindow? = nil) throws -> Result {
        typealias E = GeometricPhaseAnalysisError
        guard rows >= 8, columns >= 8, !rows.multipliedReportingOverflow(by: columns).overflow, image.count == rows * columns else { throw E.invalidSize }
        guard image.allSatisfy({ $0.isFinite }) else { throw E.nonFinitePixel }
        for g in [g1, g2] {
            guard g.row.isFinite, g.column.isFinite, abs(g.row) < 0.5, abs(g.column) < 0.5, g.row != 0 || g.column != 0
            else { throw E.reflectionOutsideNyquist }
        }
        let n1 = (g1.row * g1.row + g1.column * g1.column).squareRoot()
        let n2 = (g2.row * g2.row + g2.column * g2.column).squareRoot()
        let cross = g1.row * g2.column - g1.column * g2.row
        guard abs(cross) >= 0.02 * n1 * n2 else { throw E.collinearReflections }
        guard sigmaInLatticeUnits.isFinite, sigmaInLatticeUnits > 0 else { throw E.nonPositiveSigma }
        if let w = reference {
            guard !w.rows.isEmpty, !w.columns.isEmpty, w.rows.lowerBound >= 0, w.columns.lowerBound >= 0,
                  w.rows.upperBound <= rows, w.columns.upperBound <= columns else { throw E.invalidReferenceWindow }
        }
        let sigma = sigmaInLatticeUnits * min(n1, n2)
        let count = rows * columns

        let spectrum: ComplexImage<Double>
        do { spectrum = try FFTSurface.fft2(ComplexImage(rows: rows, columns: columns, real: image)) } catch { throw E.fftSetup }

        // signed frequencies as numpy.fft.fftfreq
        let fr = (0..<rows).map { Double($0 < (rows + 1) / 2 ? $0 : $0 - rows) / Double(rows) }
        let fc = (0..<columns).map { Double($0 < (columns + 1) / 2 ? $0 : $0 - columns) / Double(columns) }

        var phases: [[Double]] = [], amps: [[Double]] = [], grads: [(r: [Double], c: [Double])] = []
        for g in [g1, g2] {
            var mre = [Double](repeating: 0, count: count), mim = mre
            let inv2s2 = 1 / (2 * sigma * sigma)
            for i in 0..<rows {
                let dr = fr[i] - g.row
                for j in 0..<columns {
                    let dc = fc[j] - g.column
                    let m = exp(-(dr * dr + dc * dc) * inv2s2)
                    mre[i * columns + j] = spectrum.re[i * columns + j] * m
                    mim[i * columns + j] = spectrum.im[i * columns + j] * m
                }
            }
            let back: ComplexImage<Double>
            do { back = try FFTSurface.ifft2(ComplexImage(rows: rows, columns: columns, re: mre, im: mim)) } catch { throw E.fftSetup }
            var hr = back.re, hi = back.im
            for i in 0..<rows {
                for j in 0..<columns {
                    let a = -2 * Double.pi * (g.row * Double(i) + g.column * Double(j))
                    let c = cos(a), s = sin(a), k = i * columns + j
                    let (r, m) = (hr[k], hi[k])
                    hr[k] = r * c - m * s; hi[k] = r * s + m * c
                }
            }
            let amp = (0..<count).map { (hr[$0] * hr[$0] + hi[$0] * hi[$0]).squareRoot() }
            phases.append((0..<count).map { atan2(hi[$0], hr[$0]) })
            let top = amp.max() ?? 0
            let floor = 1e-20 * top * top
            let drr = gradient(hr, rows: rows, columns: columns, axis: 0), dri = gradient(hi, rows: rows, columns: columns, axis: 0)
            let dcr = gradient(hr, rows: rows, columns: columns, axis: 1), dci = gradient(hi, rows: rows, columns: columns, axis: 1)
            var gr = [Double](repeating: 0, count: count), gc = gr
            for k in 0..<count {
                let den = hr[k] * hr[k] + hi[k] * hi[k]
                if den > floor, den > 0 {
                    gr[k] = (hr[k] * dri[k] - hi[k] * drr[k]) / den
                    gc[k] = (hr[k] * dci[k] - hi[k] * dcr[k]) / den
                } else { gr[k] = .nan; gc[k] = .nan }
            }
            amps.append(amp)
            grads.append((gr, gc))
        }

        // D = -(1 / 2 pi) G^-1 grad P with G = [[g1r, g1c], [g2r, g2c]] and G^-1 = (1 / cross) [[g2c, -g1c], [-g2r, g1r]]
        let k = -1 / (2 * Double.pi * cross)
        var dRR = [Double](repeating: 0, count: count), dRC = dRR, dCR = dRR, dCC = dRR
        for p in 0..<count {
            let a1r = grads[0].r[p], a1c = grads[0].c[p], a2r = grads[1].r[p], a2c = grads[1].c[p]
            dRR[p] = k * (g2.column * a1r - g1.column * a2r)
            dRC[p] = k * (g2.column * a1c - g1.column * a2c)
            dCR[p] = k * (-g2.row * a1r + g1.row * a2r)
            dCC[p] = k * (-g2.row * a1c + g1.row * a2c)
        }
        var off = (dRR: 0.0, dRC: 0.0, dCR: 0.0, dCC: 0.0)
        if let w = reference {
            // The mean over the window's pixels that have a phase (a vacuum corner holds NaN); none finite is a refusal.
            func mean(_ a: [Double]) -> Double? {
                var s = 0.0, k = 0
                for i in w.rows { for j in w.columns where a[i * columns + j].isFinite { s += a[i * columns + j]; k += 1 } }
                return k > 0 ? s / Double(k) : nil
            }
            guard let mrr = mean(dRR), let mrc = mean(dRC), let mcr = mean(dCR), let mcc = mean(dCC) else { throw E.invalidReferenceWindow }
            off = (mrr, mrc, mcr, mcc)
            for p in 0..<count { dRR[p] -= off.dRR; dRC[p] -= off.dRC; dCR[p] -= off.dCR; dCC[p] -= off.dCC }
        }
        return Result(rows: rows, columns: columns, g1: g1, g2: g2, sigma: sigma, dRR: dRR, dRC: dRC, dCR: dCR, dCC: dCC,
                      strainRR: dRR, strainCC: dCC, strainRC: (0..<count).map { 0.5 * (dRC[$0] + dCR[$0]) },
                      rotation: (0..<count).map { 0.5 * (dRC[$0] - dCR[$0]) },
                      phase1: phases[0], phase2: phases[1], amplitude1: amps[0], amplitude2: amps[1], referenceOffset: off)
    }

    /// `numpy.gradient` along one axis, unit spacing: central differences inside, one-sided on the two edges.
    private static func gradient(_ a: [Double], rows: Int, columns: Int, axis: Int) -> [Double] {
        var out = [Double](repeating: 0, count: a.count)
        for i in 0..<rows {
            for j in 0..<columns {
                let k = i * columns + j
                if axis == 0 {
                    out[k] = i == 0 ? a[k + columns] - a[k] : (i == rows - 1 ? a[k] - a[k - columns] : 0.5 * (a[k + columns] - a[k - columns]))
                } else {
                    out[k] = j == 0 ? a[k + 1] - a[k] : (j == columns - 1 ? a[k] - a[k - 1] : 0.5 * (a[k + 1] - a[k - 1]))
                }
            }
        }
        return out
    }

    // MARK: finding the reflections

    /// The two strongest non-collinear reflections of a lattice image, (g_row, g_col) in cycles per pixel, each in the half
    /// plane g_col > 0 (or g_col = 0 and g_row > 0), the stronger first. The spectrum is taken after the mean is removed and a
    /// Hann window applied; each peak is refined by a parabola through the log magnitudes of the three bins on each axis.
    /// Peaks within `minRadius` cycles per pixel of the origin are skipped; the second peak must be at least 3 bins from the
    /// first with |sin angle| >= 0.2. Unvalidated beyond the planted images.
    public static func findReflections(_ image: [Double], rows: Int, columns: Int, minRadius: Double = 0.02) throws
        -> (g1: (row: Double, column: Double), g2: (row: Double, column: Double)) {
        typealias E = GeometricPhaseAnalysisError
        guard rows >= 8, columns >= 8, !rows.multipliedReportingOverflow(by: columns).overflow, image.count == rows * columns else { throw E.invalidSize }
        guard image.allSatisfy({ $0.isFinite }) else { throw E.nonFinitePixel }
        let mean = image.reduce(0, +) / Double(image.count)
        var buf = image
        for i in 0..<rows {
            let wr = 0.5 - 0.5 * cos(2 * Double.pi * (Double(i) + 0.5) / Double(rows))
            for j in 0..<columns {
                let wc = 0.5 - 0.5 * cos(2 * Double.pi * (Double(j) + 0.5) / Double(columns))
                buf[i * columns + j] = (image[i * columns + j] - mean) * wr * wc
            }
        }
        let f: ComplexImage<Double>
        do { f = try FFTSurface.fft2(ComplexImage(rows: rows, columns: columns, real: buf)) } catch { throw E.fftSetup }
        let mag = (0..<buf.count).map { (f.re[$0] * f.re[$0] + f.im[$0] * f.im[$0]).squareRoot() }
        func signed(_ k: Int, _ n: Int) -> Int { k < (n + 1) / 2 ? k : k - n }
        var cands: [(i: Int, j: Int, m: Double)] = []
        for i in 0..<rows {
            for j in 0..<columns {
                let gr = Double(signed(i, rows)) / Double(rows), gc = Double(signed(j, columns)) / Double(columns)
                guard gc > 0 || (gc == 0 && gr > 0), (gr * gr + gc * gc).squareRoot() >= minRadius else { continue }
                let m = mag[i * columns + j]
                guard m > 0 else { continue }
                var isMax = true
                for di in -1...1 {
                    for dj in -1...1 where di != 0 || dj != 0 {
                        if mag[((i + di + rows) % rows) * columns + (j + dj + columns) % columns] > m { isMax = false }
                    }
                }
                if isMax { cands.append((i, j, m)) }
            }
        }
        cands.sort { $0.m > $1.m }
        func parabola(_ a: Double, _ b: Double, _ c: Double) -> Double {
            let la = log(max(a, 1e-300)), lb = log(max(b, 1e-300)), lc = log(max(c, 1e-300))
            let d = la - 2 * lb + lc
            return d == 0 ? 0 : 0.5 * (la - lc) / d
        }
        func refine(_ c: (i: Int, j: Int, m: Double)) -> (row: Double, column: Double) {
            func at(_ i: Int, _ j: Int) -> Double { mag[((i + rows) % rows) * columns + (j + columns) % columns] }
            let di = parabola(at(c.i - 1, c.j), at(c.i, c.j), at(c.i + 1, c.j))
            let dj = parabola(at(c.i, c.j - 1), at(c.i, c.j), at(c.i, c.j + 1))
            return ((Double(signed(c.i, rows)) + di) / Double(rows), (Double(signed(c.j, columns)) + dj) / Double(columns))
        }
        guard let first = cands.first else { throw E.reflectionsNotFound }
        let a = refine(first)
        let na = (a.row * a.row + a.column * a.column).squareRoot()
        for c in cands.dropFirst() {
            let b = refine(c)
            let nb = (b.row * b.row + b.column * b.column).squareRoot()
            let sinA = abs(a.row * b.column - a.column * b.row) / (na * nb)
            let dr = (a.row - b.row) * Double(rows), dc = (a.column - b.column) * Double(columns)
            if sinA >= 0.2, (dr * dr + dc * dc).squareRoot() >= 3 { return (a, b) }
        }
        throw E.reflectionsNotFound
    }
}
