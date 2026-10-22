//
//  GeometricPhaseAnalysisTests.swift (ImagoFiltersTests)
//  GPA against the planted truth of tools/gpa/planted.py, size 128, mask sigma 3.2 / 128 (the default rule, 0.2 of the lattice
//  spacing 16 / 128), compared inside the interior (margin 20 px).
//
//  Tolerance: 2 x the selfcheck max_abs_error_interior of a plain numpy GPA at size 128 (python3 planted.py --size 128), floor 1e-6.
//  numpy selfcheck: uniform 4.887e-4, bump 1.560e-3, bump_oblique 1.485e-3, perfect 2.5e-11.
//  Tolerances used: uniform 9.8e-4, bump 3.12e-3, bump_oblique 2.97e-3, perfect 1e-6.
//  Measured Swift errors are recorded in docs/notes-k5-gpa.md. validation stays "none".
//

import Foundation
import Testing
@testable import ImagoFilters

@Suite struct GeometricPhaseAnalysisTests {
    typealias GPA = GeometricPhaseAnalysis

    private func check(_ name: String, tol: Double) throws {
        let p = PlantedLattice(name)
        let r = try GPA.analyze(p.image, rows: p.n, columns: p.n, g1: p.g1, g2: p.g2)
        let errs = [p.interiorMax(r.dRR, p.dRR), p.interiorMax(r.dRC, p.dRC), p.interiorMax(r.dCR, p.dCR), p.interiorMax(r.dCC, p.dCC)]
        let worst = errs.max()!
        print("GPA planted \(name): max interior |D - truth| = \(worst)")
        #expect(worst <= tol, "\(name): \(worst) > \(tol)")
        // strain and rotation are the symmetric and antisymmetric parts of the truth
        let sRC = zip(p.dRC, p.dCR).map { 0.5 * ($0 + $1) }, rot = zip(p.dRC, p.dCR).map { 0.5 * ($0 - $1) }
        #expect(p.interiorMax(r.strainRR, p.dRR) <= tol)
        #expect(p.interiorMax(r.strainCC, p.dCC) <= tol)
        #expect(p.interiorMax(r.strainRC, sRC) <= tol)
        #expect(p.interiorMax(r.rotation, rot) <= tol)
        #expect(r.validation == "none")
        #expect(GPA.Result.validationNone == "none")
    }

    @Test func uniform() throws { try check("uniform", tol: 9.8e-4) }
    @Test func bump() throws { try check("bump", tol: 3.12e-3) }
    @Test func bumpOblique() throws { try check("bump_oblique", tol: 2.97e-3) }
    @Test func perfectGivesZeroStrain() throws { try check("perfect", tol: 1e-6) }

    @Test func outputsAreFullSize() throws {
        let p = PlantedLattice("uniform")
        let r = try GPA.analyze(p.image, rows: p.n, columns: p.n, g1: p.g1, g2: p.g2)
        for a in [r.dRR, r.dRC, r.dCR, r.dCC, r.strainRR, r.strainCC, r.strainRC, r.rotation, r.phase1, r.phase2, r.amplitude1, r.amplitude2] {
            #expect(a.count == p.n * p.n)
        }
        #expect(abs(r.sigma - 3.2 / 128) < 1e-15)
        // amplitude of a unit cosine fringe is 0.5 in the middle of the frame
        #expect(abs(r.amplitude1[64 * 128 + 64] - 0.5) < 1e-3)
    }

    /// Sign convention: a lattice displaced by u = +s along the columns (u_col = s * (col - c), D_cc = +s, a stretch) gives
    /// positive e_cc; u_row growing along columns gives positive D_rc and positive rotation.
    @Test func conventions() throws {
        let n = 128, c = 63.5
        let g1 = (row: 0.0, column: 16.0 / 128), g2 = (row: 16.0 / 128, column: 0.0)
        func run(_ u0: (Int, Int) -> Double, _ u1: (Int, Int) -> Double) throws -> GPA.Result {
            var a = [Double](repeating: 0, count: n * n), b = a
            for i in 0..<n { for j in 0..<n { a[i * n + j] = u0(i, j); b[i * n + j] = u1(i, j) } }
            return try GPA.analyze(PlantedLattice.render(n: n, gs: [g1, g2], u0: a, u1: b), rows: n, columns: n, g1: g1, g2: g2)
        }
        let mid = 64 * n + 64
        let stretchCol = try run({ _, _ in 0 }, { _, j in 0.01 * (Double(j) - c) })
        #expect(abs(stretchCol.dCC[mid] - 0.01) < 2e-3 && stretchCol.strainCC[mid] > 0 && abs(stretchCol.strainRR[mid]) < 2e-3)
        let stretchRow = try run({ i, _ in 0.01 * (Double(i) - c) }, { _, _ in 0 })
        #expect(abs(stretchRow.dRR[mid] - 0.01) < 2e-3 && stretchRow.strainRR[mid] > 0 && abs(stretchRow.strainCC[mid]) < 2e-3)
        let shear = try run({ _, j in 0.01 * (Double(j) - c) }, { _, _ in 0 })   // du_row / dcol = +0.01
        #expect(abs(shear.dRC[mid] - 0.01) < 2e-3 && abs(shear.dCR[mid]) < 2e-3)
        #expect(abs(shear.rotation[mid] - 0.005) < 2e-3 && abs(shear.strainRC[mid] - 0.005) < 2e-3)
        // a rigid rotation: D_rc = +w, D_cr = -w gives rotation w and no strain
        let rot = try run({ _, j in 0.01 * (Double(j) - c) }, { i, _ in -0.01 * (Double(i) - c) })
        #expect(abs(rot.rotation[mid] - 0.01) < 2e-3 && abs(rot.strainRC[mid]) < 2e-3)
    }

    @Test func anOverflowingSizeIsRefusedNotTrapped() throws {
        #expect(throws: GeometricPhaseAnalysisError.invalidSize) {
            try GPA.analyze([1, 2], rows: Int.max, columns: 3, g1: (0, 0.25), g2: (0.25, 0))
        }
        #expect(throws: GeometricPhaseAnalysisError.invalidSize) { try GPA.findReflections([1, 2], rows: Int.max, columns: 3) }
    }

    @Test func referenceRegionZeroesStrainOverIt() throws {
        let p = PlantedLattice("bump")
        let w = GPA.ReferenceWindow(rows: 40..<60, columns: 25..<45)
        let r = try GPA.analyze(p.image, rows: p.n, columns: p.n, g1: p.g1, g2: p.g2, reference: w)
        for a in [r.dRR, r.dRC, r.dCR, r.dCC] {
            var s = 0.0
            for i in w.rows { for j in w.columns { s += a[i * p.n + j] } }
            #expect(abs(s / Double(w.rows.count * w.columns.count)) < 1e-14)
        }
        let plain = try GPA.analyze(p.image, rows: p.n, columns: p.n, g1: p.g1, g2: p.g2)
        // the offset is the plain mean and the shift is the same everywhere
        let k = 50 * p.n + 70
        #expect(abs((plain.dRR[k] - r.dRR[k]) - r.referenceOffset.dRR) < 1e-14)
        #expect(abs(r.referenceOffset.dRR - plain.dRR[40 * p.n + 25]) < 0.02)
    }

    @Test func refusals() throws {
        let p = PlantedLattice("perfect")
        let n = p.n
        func fail(_ expected: GeometricPhaseAnalysisError, _ body: () throws -> Void) {
            do { try body(); Issue.record("no error, expected \(expected)") } catch { #expect(error as? GeometricPhaseAnalysisError == expected) }
        }
        fail(.invalidSize) { _ = try GPA.analyze(p.image, rows: n, columns: n - 1, g1: p.g1, g2: p.g2) }
        fail(.invalidSize) { _ = try GPA.analyze([Double](repeating: 0, count: 16), rows: 4, columns: 4, g1: p.g1, g2: p.g2) }
        var bad = p.image; bad[100] = .nan
        fail(.nonFinitePixel) { _ = try GPA.analyze(bad, rows: n, columns: n, g1: p.g1, g2: p.g2) }
        bad[100] = .infinity
        fail(.nonFinitePixel) { _ = try GPA.analyze(bad, rows: n, columns: n, g1: p.g1, g2: p.g2) }
        fail(.collinearReflections) { _ = try GPA.analyze(p.image, rows: n, columns: n, g1: p.g1, g2: (p.g1.row * 2, p.g1.column * 2)) }
        fail(.collinearReflections) { _ = try GPA.analyze(p.image, rows: n, columns: n, g1: p.g1, g2: (-p.g1.row, -p.g1.column)) }
        fail(.reflectionOutsideNyquist) { _ = try GPA.analyze(p.image, rows: n, columns: n, g1: (0, 0.5), g2: p.g2) }
        fail(.reflectionOutsideNyquist) { _ = try GPA.analyze(p.image, rows: n, columns: n, g1: p.g1, g2: (0.7, 0.1)) }
        fail(.reflectionOutsideNyquist) { _ = try GPA.analyze(p.image, rows: n, columns: n, g1: (0, 0), g2: p.g2) }
        fail(.nonPositiveSigma) { _ = try GPA.analyze(p.image, rows: n, columns: n, g1: p.g1, g2: p.g2, sigmaInLatticeUnits: 0) }
        fail(.nonPositiveSigma) { _ = try GPA.analyze(p.image, rows: n, columns: n, g1: p.g1, g2: p.g2, sigmaInLatticeUnits: -1) }
        fail(.nonPositiveSigma) { _ = try GPA.analyze(p.image, rows: n, columns: n, g1: p.g1, g2: p.g2, sigmaInLatticeUnits: .nan) }
        fail(.invalidReferenceWindow) {
            _ = try GPA.analyze(p.image, rows: n, columns: n, g1: p.g1, g2: p.g2, reference: .init(rows: 0..<0, columns: 0..<5))
        }
        fail(.invalidReferenceWindow) {
            _ = try GPA.analyze(p.image, rows: n, columns: n, g1: p.g1, g2: p.g2, reference: .init(rows: 0..<5, columns: 120..<129))
        }
    }

    @Test func findsTheReflectionsOfPlantedImages() throws {
        for name in ["perfect", "uniform", "bump", "bump_oblique"] {
            let p = PlantedLattice(name)
            let (a, b) = try GPA.findReflections(p.image, rows: p.n, columns: p.n)
            // the half-plane representative of each planted reflection (g_col > 0, or g_col = 0 and g_row > 0)
            func half(_ g: (row: Double, column: Double)) -> (row: Double, column: Double) {
                g.column > 0 || (g.column == 0 && g.row > 0) ? g : (-g.row, -g.column)
            }
            let truth = [half(p.g1), half(p.g2)]
            // either order; the strain of the planted fields moves a peak by up to 0.2 bins
            let tol = (name == "perfect" ? 1e-3 : 0.3) / Double(p.n)
            func close(_ x: (row: Double, column: Double), _ y: (row: Double, column: Double)) -> Bool {
                abs(x.row - y.row) < tol && abs(x.column - y.column) < tol
            }
            #expect((close(a, truth[0]) && close(b, truth[1])) || (close(a, truth[1]) && close(b, truth[0])), "\(name): \(a) \(b)")
            // and GPA with the found reflections still works on the unstrained image
            if name == "perfect" {
                let r = try GPA.analyze(p.image, rows: p.n, columns: p.n, g1: a, g2: b)
                #expect(p.interiorMax(r.strainRR, p.dRR) < 1e-3)
            }
        }
        do {
            let flat = [Double](repeating: 1, count: 64 * 64)
            #expect(throws: GeometricPhaseAnalysisError.reflectionsNotFound) { try GPA.findReflections(flat, rows: 64, columns: 64) }
        }
    }
}
