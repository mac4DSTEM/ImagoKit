//
//  Thumbnail.swift (ImagoCore)
//  The box-average downsample of display-normalised pixels into a small copy: aspect kept, never upscaled. No colormap.
//

import Foundation

/// A small display copy of an image: at most `maxSide` pixels on the longer side, the pixel aspect kept, Float32 values as
/// `DisplayNormalization.normalized` leaves them (finite range onto [0, 1], a masked pixel at `invalidValue`).
public nonisolated struct Thumbnail: Sendable, Equatable {
    /// The longest side the app's thumbnails have; the default of `make`.
    public static let defaultMaxSide = 96

    public let width: Int
    public let height: Int
    /// Row-major, `width` fastest.
    public let normalized: [Float]

    /// nil when the geometry is not one: a side under 1 px, or a value count that is not `width × height`.
    public init?(width: Int, height: Int, normalized: [Float]) {
        guard width >= 1, height >= 1, normalized.count == width * height else { return nil }
        self.width = width
        self.height = height
        self.normalized = normalized
    }

    /// As `init?(width:height:normalized:)`, and nil as well when a side is over `maxSide` (the consumer's cap, for example
    /// `defaultMaxSide`): a reader that cannot make a thumbnail of that size answers nil, it never makes a wrong one.
    public init?(width: Int, height: Int, normalized: [Float], maxSide: Int) {
        guard width <= maxSide, height <= maxSide else { return nil }
        self.init(width: width, height: height, normalized: normalized)
    }

    /// The thumbnail of a `width × height` image of display-normalised pixels (values in [0, 1] or `invalidValue`). nil
    /// when a side or `maxSide` is under 1 or the count is not `width × height`. An image whose longer side is at most
    /// `maxSide` is returned as it is (no upscaling). Otherwise the longer side becomes `maxSide` and the shorter keeps the
    /// aspect (rounded, at least 1). Output pixel (i, j) is the mean of the valid pixels in source columns
    /// [i·W/w, (i+1)·W/w) and rows [j·H/h, (j+1)·H/h) (integer division), so every source pixel is in exactly one box;
    /// the invalid sentinel and non-finite values are left out, and a box with no valid pixel is the sentinel. Sums are in
    /// Double (exact for the sizes thumbnails come from), the mean rounded once to Float32.
    public static func make(normalized: [Float], width: Int, height: Int, maxSide: Int = defaultMaxSide) -> Thumbnail? {
        guard width >= 1, height >= 1, maxSide >= 1, normalized.count == width * height else { return nil }
        let long = max(width, height)
        if long <= maxSide { return Thumbnail(width: width, height: height, normalized: normalized) }

        func scaled(_ side: Int) -> Int {
            side == long ? maxSide : max(1, Int((Double(side) * Double(maxSide) / Double(long)).rounded()))
        }
        let w = scaled(width), h = scaled(height)
        let invalid = DisplayNormalization.invalidValue

        // Source column / row -> output column / row: the inverse of the floor boxes above.
        var colBox = [Int](repeating: 0, count: width)
        var i = 0
        for x in 0..<width {
            while (i + 1) * width / w <= x { i += 1 }
            colBox[x] = i
        }
        var rowBox = [Int](repeating: 0, count: height)
        var j = 0
        for y in 0..<height {
            while (j + 1) * height / h <= y { j += 1 }
            rowBox[y] = j
        }

        var sums = [Double](repeating: 0, count: w * h)
        var counts = [Int32](repeating: 0, count: w * h)
        normalized.withUnsafeBufferPointer { src in
            for y in 0..<height {
                let rowBase = rowBox[y] * w
                let srcBase = y * width
                for x in 0..<width {
                    let v = src[srcBase + x]
                    if v.isFinite && v != invalid {
                        let k = rowBase + colBox[x]
                        sums[k] += Double(v)
                        counts[k] += 1
                    }
                }
            }
        }
        var out = [Float](repeating: invalid, count: w * h)
        for k in 0..<(w * h) where counts[k] > 0 { out[k] = Float(sums[k] / Double(counts[k])) }
        return Thumbnail(width: w, height: h, normalized: out)
    }
}
