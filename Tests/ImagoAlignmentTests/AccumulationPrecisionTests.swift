// ImagoKit#79: the reference sums a float32 stack in float32. NumPy 2.5.4's `np.sum(stack, axis=0)` and `np.mean` on a float32
// (N, H, W) array return float32 equal bit for bit to adding the frames one after another in float32 (checked in the pinned venv),
// so `AlignedSum` at `.float32` accumulates in Float, as NumPy does, and is held to that here. Refuted: no wider accumulation.
import Foundation
import ImagoAlignment
import ImagoCore
import Testing

@Suite("Accumulation precision")
struct AccumulationPrecision {
    @Test func aFloat32StackIsSummedInFloat32AsNumPyDoes() throws {
        let w = 16, h = 12, n = 50
        var state: UInt64 = 79
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 24) * 1e4 + 0.1
        }
        let frames = try (0..<n).map { _ in try Micrograph(width: w, height: h, pixels: (0..<(w * h)).map { _ in next() }) }
        let stack = try AlignedSum.sum(frames, shifts: Array(repeating: PixelShift(row: 0, column: 0), count: n), method: .bilinear,
                                       precision: .float32)
        var float32 = [Float](repeating: 0, count: w * h), float64 = [Double](repeating: 0, count: w * h)
        for f in frames { for i in 0..<(w * h) { float32[i] += f.pixels[i]; float64[i] += Double(f.pixels[i]) } }
        #expect(stack.sum.pixels == float32, "frame by frame in Float, NumPy's np.sum(axis=0) on float32")
        #expect(stack.mean.pixels == float32.map { $0 / Float(n) }, "np.mean(axis=0) on float32")
        #expect(stack.sum.pixels != float64.map { Float($0) }, "a wider accumulator would differ: the check can tell them apart")
    }
}
