import ImagoCore

/// A random-looking, structure-rich image for the closed forms (a small LCG, deterministic).
package func noise(_ width: Int, _ height: Int, seed: UInt64) -> Micrograph {
    var state = seed
    let pixels: [Float] = (0..<(width * height)).map { _ in
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(Double(state >> 33) / Double(1 << 31)) * 100
    }
    return try! Micrograph(width: width, height: height, pixels: pixels)
}
