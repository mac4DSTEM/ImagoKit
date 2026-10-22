// Closed-form and refusal checks for Denoise (no fixtures); the pinned comparisons with scikit-image are in ImagoKitParityTests.
import Foundation
import ImagoFilters
import Testing

@Suite struct DenoiseClosedFormTests {
    @Test func constantImageIsUnchanged() throws {
        let c = [Double](repeating: 0.37, count: 9 * 8)
        #expect(try Denoise.totalVariationChambolle(c, rows: 9, columns: 8) == c)
        let b = try Denoise.totalVariationBregman(c, rows: 9, columns: 8)
        #expect(b.allSatisfy { abs($0 - 0.37) < 1e-12 })
        for fast in [true, false] {
            let n = try Denoise.nonLocalMeans(c, rows: 9, columns: 8, patchSize: 5, patchDistance: 3, fastMode: fast)
            #expect(n.allSatisfy { abs($0 - 0.37) < 1e-12 })
        }
        let f = [Float](repeating: 0.25, count: 6 * 7)
        #expect(try Denoise.totalVariationChambolle(f, rows: 6, columns: 7) == f)
        #expect(try Denoise.nonLocalMeans(f, rows: 6, columns: 7, patchSize: 3, patchDistance: 2, fastMode: false).allSatisfy { abs($0 - 0.25) < 1e-6 })
    }

    @Test func floatImagesStayFloat() throws {
        let x: [Float] = (0..<20).map { Float($0 % 5) * 0.1 }
        let a: [Float] = try Denoise.totalVariationChambolle(x, rows: 4, columns: 5)
        let b: [Float] = try Denoise.totalVariationBregman(x, rows: 4, columns: 5)
        let c: [Float] = try Denoise.nonLocalMeans(x, rows: 4, columns: 5, patchSize: 3, patchDistance: 2)
        #expect(a.count == 20 && b.count == 20 && c.count == 20)
    }

    @Test func singleSampleSurvivesChambolleAndNonLocalMeans() throws {
        #expect(try Denoise.totalVariationChambolle([0.5], rows: 1, columns: 1) == [0.5])
        let n = try Denoise.nonLocalMeans([0.5], rows: 1, columns: 1, patchSize: 3, patchDistance: 2, fastMode: false)
        #expect(abs(n[0] - 0.5) < 1e-12)
    }

    @Test func refusals() {
        let ok = [Double](repeating: 0.5, count: 12)
        #expect(throws: DenoiseError.shape) { try Denoise.totalVariationChambolle(ok, rows: 3, columns: 5) }
        #expect(throws: DenoiseError.shape) { try Denoise.totalVariationChambolle(ok, rows: 0, columns: 12) }
        #expect(throws: DenoiseError.shape) { try Denoise.totalVariationBregman(ok, rows: -3, columns: -4) }
        #expect(throws: DenoiseError.shape) { try Denoise.nonLocalMeans(ok, rows: 3, columns: 5) }
        var nan = ok; nan[4] = .nan
        var inf = ok; inf[7] = .infinity
        #expect(throws: DenoiseError.nonFinite) { try Denoise.totalVariationChambolle(nan, rows: 3, columns: 4) }
        #expect(throws: DenoiseError.nonFinite) { try Denoise.totalVariationBregman(inf, rows: 3, columns: 4) }
        #expect(throws: DenoiseError.nonFinite) { try Denoise.nonLocalMeans(nan, rows: 3, columns: 4) }
        #expect(throws: DenoiseError.nonFinite) { try Denoise.nonLocalMeans([Float.nan, 1, 2, 3], rows: 2, columns: 2) }

        func parameter<T>(_ body: () throws -> T) -> Bool {
            do { _ = try body(); return false } catch DenoiseError.parameter { return true } catch { return false }
        }
        #expect(parameter { try Denoise.totalVariationChambolle(ok, rows: 3, columns: 4, weight: 0) })
        #expect(parameter { try Denoise.totalVariationChambolle(ok, rows: 3, columns: 4, weight: -1) })
        #expect(parameter { try Denoise.totalVariationChambolle(ok, rows: 3, columns: 4, eps: 0) })
        #expect(parameter { try Denoise.totalVariationChambolle(ok, rows: 3, columns: 4, maxNumIter: 0) })
        #expect(parameter { try Denoise.totalVariationChambolle(ok, rows: 3, columns: 4, weight: .nan) })
        #expect(parameter { try Denoise.totalVariationBregman(ok, rows: 3, columns: 4, weight: 0) })
        #expect(parameter { try Denoise.totalVariationBregman(ok, rows: 3, columns: 4, maxNumIter: -2) })
        #expect(parameter { try Denoise.totalVariationBregman(ok, rows: 3, columns: 4, eps: -1e-3) })
        #expect(parameter { try Denoise.nonLocalMeans(ok, rows: 3, columns: 4, patchSize: 1) })
        #expect(parameter { try Denoise.nonLocalMeans(ok, rows: 3, columns: 4, patchDistance: 0) })
        #expect(parameter { try Denoise.nonLocalMeans(ok, rows: 3, columns: 4, h: 0) })
        #expect(parameter { try Denoise.nonLocalMeans(ok, rows: 3, columns: 4, h: .infinity) })
        #expect(parameter { try Denoise.nonLocalMeans(ok, rows: 3, columns: 4, sigma: -0.1) })
        #expect(parameter { try Denoise.nonLocalMeans(ok, rows: 3, columns: 4, patchSize: Int.max) })
        #expect(parameter { try Denoise.nonLocalMeans(ok, rows: 3, columns: 4, patchDistance: 1 << 40) })
    }

    @Test func bregmanNeedsTwoByTwo() {
        #expect(throws: DenoiseError.self) { try Denoise.totalVariationBregman([1.0, 2.0, 3.0], rows: 1, columns: 3) }
        #expect(throws: DenoiseError.self) { try Denoise.totalVariationBregman([1.0, 2.0, 3.0], rows: 3, columns: 1) }
        #expect(throws: DenoiseError.self) { try Denoise.totalVariationBregman([1.0], rows: 1, columns: 1) }
    }

    @Test func messagesNameNoApplication() {
        let samples: [DenoiseError] = [.shape, .nonFinite, .parameter("weight must be finite and positive"), .tooSmall("total variation (Bregman) needs at least 2 rows and 2 columns")]
        for e in samples {
            let text = "\(e)".lowercased()
            #expect(!text.contains("mac4dstem") && !text.contains("mactem"))
        }
    }
}
