//
//  PlantedLattice.swift (ImagoFiltersTests)
//  The planted-truth lattice images of tools/gpa/planted.py, rebuilt from its docstring formulas: I = sum over g1, g2 of
//  cos(2 pi g . (r - u)) (the +g and -g terms of 0.5 cos each), u the displacement in pixels, D_ij = du_i / dr_j analytic.
//  Cases: uniform (constant D), bump (constant D plus Gaussian displacement bumps), perfect (u = 0), bump_oblique (bump with
//  a non-orthogonal pair).
//

import Foundation

struct PlantedLattice {
    let name: String
    let n: Int
    let g1: (row: Double, column: Double)
    let g2: (row: Double, column: Double)
    var image: [Double]
    /// D_ij = du_i / dr_j, row-major `n * n` each.
    var dRR: [Double], dRC: [Double], dCR: [Double], dCC: [Double]

    /// The planted.py mask sigma (cycles per pixel) and interior margin (pixels) at this size.
    var maskSigma: Double { 3.2 / Double(n) }
    var margin: Int { Int((3 / (2 * Double.pi * maskSigma)).rounded(.up)) }

    init(_ name: String, n: Int = 128) {
        self.name = name; self.n = n
        let k2: (Double, Double) = name == "bump_oblique" ? (14, 8) : (16, 0)
        g1 = (0, 16 / Double(n)); g2 = (k2.0 / Double(n), k2.1 / Double(n))
        let kind = name == "bump_oblique" ? "bump" : name
        let count = n * n
        var u0 = [Double](repeating: 0, count: count), u1 = u0
        dRR = u0; dRC = u0; dCR = u0; dCC = u0
        if kind != "perfect" {
            let lin = [[0.010, 0.004], [-0.002, -0.006]]
            let c = Double(n - 1) / 2
            for i in 0..<n {
                for j in 0..<n {
                    let k = i * n + j
                    u0[k] = lin[0][0] * (Double(i) - c) + lin[0][1] * (Double(j) - c)
                    u1[k] = lin[1][0] * (Double(i) - c) + lin[1][1] * (Double(j) - c)
                    dRR[k] = lin[0][0]; dRC[k] = lin[0][1]; dCR[k] = lin[1][0]; dCC[k] = lin[1][1]
                }
            }
            if kind == "bump" {
                let bumps: [(amp: Double, ri: Double, rj: Double, s: Double)] = [(2.0, 0.4, 0.6, 40), (-1.5, 0.55, 0.45, 50)]
                for (a, b) in bumps.enumerated() {
                    let ri = b.ri * Double(n), rj = b.rj * Double(n)
                    for i in 0..<n {
                        for j in 0..<n {
                            let k = i * n + j
                            let di = Double(i) - ri, dj = Double(j) - rj
                            let g = b.amp * exp(-(di * di + dj * dj) / (2 * b.s * b.s))
                            if a == 0 { u0[k] += g; dRR[k] += g * (-di / (b.s * b.s)); dRC[k] += g * (-dj / (b.s * b.s)) }
                            else { u1[k] += g; dCR[k] += g * (-di / (b.s * b.s)); dCC[k] += g * (-dj / (b.s * b.s)) }
                        }
                    }
                }
            }
        }
        image = PlantedLattice.render(n: n, gs: [g1, g2], u0: u0, u1: u1)
    }

    /// A lattice with a given displacement field (for sign-convention tests).
    static func render(n: Int, gs: [(row: Double, column: Double)], u0: [Double], u1: [Double]) -> [Double] {
        var img = [Double](repeating: 0, count: n * n)
        for g in gs {
            for i in 0..<n {
                for j in 0..<n {
                    let k = i * n + j
                    img[k] += cos(2 * Double.pi * (g.row * (Double(i) - u0[k]) + g.column * (Double(j) - u1[k])))
                }
            }
        }
        return img
    }

    /// Worst |a - b| inside the interior (margin pixels from each frame edge).
    func interiorMax(_ a: [Double], _ b: [Double]) -> Double {
        var m = 0.0
        for i in margin..<(n - margin) { for j in margin..<(n - margin) { m = max(m, abs(a[i * n + j] - b[i * n + j])) } }
        return m
    }
}
