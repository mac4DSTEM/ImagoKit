// ImagoKit#92: the 2D FFT's rows and columns run in bands on every core above `FFT2D.parallelElements`. The banded transform
// must be the serial one bit for bit (each line goes through the same code), and the bands must cover every line once.
import Foundation
import Synchronization
import Testing
import ImagoCore

@Suite struct FFTParallelLinesTests {

    /// Red before: `forEachBand` and `parallelElements` did not exist (every line ran on the caller's thread).
    @Test func testTheBandsCoverEveryLineOnce() {
        let count = 1000, length = 2000
        #expect(count * length >= FFT2D.parallelElements)
        let visits = Mutex([Int](repeating: 0, count: count))
        FFT2D.forEachBand(count: count, length: length, scratchCount: 16) { lines, scratch in
            #expect(scratch.0 != nil && scratch.1 != nil)
            visits.withLock { for i in lines { $0[i] += 1 } }
        }
        #expect(visits.withLock { $0 } == [Int](repeating: 1, count: count))
        FFT2D.forEachBand(count: 3, length: 4, scratchCount: 0) { lines, scratch in
            #expect(lines == 0..<3)
            #expect(scratch.0 == nil && scratch.1 == nil)
        }
    }

    /// A prime × prime grid above the threshold (both axes Bluestein): the banded transform equals the same lines transformed
    /// one at a time on the caller's thread, bit for bit, forward and inverse.
    @Test func testTheBandedTransformIsTheSerialOne() throws {
        let nx = 1031, ny = 1033
        #expect(nx * ny >= FFT2D.parallelElements)
        let plan = try #require(FFT2D(nx: nx, ny: ny))
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(seed >> 40) / Float(1 << 24) - 0.5 }
        let re0 = (0..<(nx * ny)).map { _ in next() }, im0 = (0..<(nx * ny)).map { _ in next() }
        for forward in [true, false] {
            var re = re0, im = im0
            plan.transform(re: &re, im: &im, forward: forward, scaleInverse: false)
            let (sRe, sIm) = serial(plan, re0, im0, forward: forward)
            #expect(re.map(\.bitPattern) == sRe.map(\.bitPattern))
            #expect(im.map(\.bitPattern) == sIm.map(\.bitPattern))
        }
    }

    /// Rows one line per call, transpose, columns one line per call, transpose back: each call is below the threshold.
    private func serial(_ plan: FFT2D, _ re: [Float], _ im: [Float], forward: Bool) -> ([Float], [Float]) {
        let nx = plan.nx, ny = plan.ny
        var rowsRe = [Float](repeating: 0, count: nx * ny), rowsIm = rowsRe
        for y in 0..<ny {
            re.withUnsafeBufferPointer { ir in im.withUnsafeBufferPointer { ii in
                rowsRe.withUnsafeMutableBufferPointer { orr in rowsIm.withUnsafeMutableBufferPointer { oi in
                    plan.transformLines(alongX: true, count: 1, forward: forward,
                                        ir.baseAddress! + y * nx, ii.baseAddress! + y * nx,
                                        orr.baseAddress! + y * nx, oi.baseAddress! + y * nx)
                } }
            } }
        }
        let tRe = transpose(rowsRe, rows: ny, columns: nx), tIm = transpose(rowsIm, rows: ny, columns: nx)
        var colsRe = [Float](repeating: 0, count: nx * ny), colsIm = colsRe
        for x in 0..<nx {
            tRe.withUnsafeBufferPointer { ir in tIm.withUnsafeBufferPointer { ii in
                colsRe.withUnsafeMutableBufferPointer { orr in colsIm.withUnsafeMutableBufferPointer { oi in
                    plan.transformLines(alongX: false, count: 1, forward: forward,
                                        ir.baseAddress! + x * ny, ii.baseAddress! + x * ny,
                                        orr.baseAddress! + x * ny, oi.baseAddress! + x * ny)
                } }
            } }
        }
        return (transpose(colsRe, rows: nx, columns: ny), transpose(colsIm, rows: nx, columns: ny))
    }

    private func transpose(_ a: [Float], rows: Int, columns: Int) -> [Float] {
        var out = [Float](repeating: 0, count: a.count)
        for r in 0..<rows { for c in 0..<columns { out[c * rows + r] = a[r * columns + c] } }
        return out
    }
}
