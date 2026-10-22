//
//  BinaryMorphology.swift (ImagoCore)
//  SciPy 1.18.1's `ndimage` binary morphology on a 2D mask (`binary_erosion`, `binary_dilation`, `binary_opening`,
//  `binary_closing`, `binary_fill_holes`, `generate_binary_structure`) and scikit-image 0.26.0's `morphology.binary_*` defaults,
//  `remove_small_objects`, `remove_small_holes` and `segmentation.clear_border`. Masks are exact: every result is boolean.
//
//  SEMANTICS. A structuring element is an odd-sized grid of booleans centred at its middle cell (SciPy's `origin` 0). Erosion keeps
//  a pixel when the mask is true at every cell the structure marks around it; dilation sets a pixel when the mask is true at any
//  cell of the structure's reflection (the same cells for the symmetric cross and square). Pixels beyond the image read as
//  `borderValue`: SciPy's default is false, so erosion eats the image border; `skimage.morphology.binary_erosion` defaults to
//  `mode='ignore'`, which is a true border (the border does not erode), and `binary_dilation` a false one. Both families are here.
//  `iterations` below 1 repeats until the mask stops changing (SciPy's rule). Opening and closing pass one `borderValue` to
//  both their steps, as SciPy does.
//
//  DEVIATIONS. An even structure dimension is refused (SciPy shifts its origin; nothing in the suite needs it). 2D only, apart
//  from `generateStructure`, which makes any rank.
//

import Foundation

public nonisolated enum MorphologyError: Error, Equatable, Sendable {
    /// Counts that do not match `width * height`, or a non-positive dimension.
    case shape
    /// A structuring element with an even dimension, a count that is not rows * columns, or no true cell.
    case structure
    /// `connectivity` or `rank` outside what the call takes.
    case parameter(String)
    /// `bufferSize` is not smaller than both image dimensions (`clear_border`).
    case buffer
}

/// A 2D structuring element: `cells[row * columns + column]`, odd `rows` and `columns`, centred.
public nonisolated struct StructuringElement: Sendable, Equatable {
    public let rows: Int
    public let columns: Int
    public let cells: [Bool]

    public init(rows: Int, columns: Int, cells: [Bool]) throws {
        guard rows > 0, columns > 0, rows % 2 == 1, columns % 2 == 1, cells.count == rows * columns, cells.contains(true) else {
            throw MorphologyError.structure
        }
        self.rows = rows; self.columns = columns; self.cells = cells
    }

    /// `generate_binary_structure(2, 1)`: the cross.
    public static let cross = try! StructuringElement(rows: 3, columns: 3, cells: [false, true, false, true, true, true, false, true, false])
    /// `generate_binary_structure(2, 2)`: the 3 x 3 square.
    public static let square = try! StructuringElement(rows: 3, columns: 3, cells: [Bool](repeating: true, count: 9))

    /// `ndimage.generate_binary_structure(2, connectivity)`: connectivity 1 is the cross, 2 the square (clamped to 1...2).
    public static func connectivity(_ connectivity: Int) -> StructuringElement { connectivity <= 1 ? .cross : .square }

    /// `ndimage.generate_binary_structure(rank, connectivity)`: a 3^rank grid, true where the sum of the absolute offsets from the
    /// centre is at most `connectivity` (clamped to 1...rank), row-major.
    public static func generateStructure(rank: Int, connectivity: Int) throws -> [Bool] {
        guard rank >= 1, rank <= 10 else { throw MorphologyError.parameter("rank") }
        let k = max(1, min(rank, connectivity))
        var size = 1
        for _ in 0..<rank { size *= 3 }
        return (0..<size).map { index in
            var rest = index, sum = 0
            for _ in 0..<rank { sum += abs(rest % 3 - 1); rest /= 3 }
            return sum <= k
        }
    }
}

/// `skimage.morphology.binary_*`'s `mode`.
public nonisolated enum MorphologyMode: Sendable, Equatable {
    /// The image border does not erode and does not dilate in (scikit-image's default).
    case ignore
    /// Pixels beyond the image are false for both.
    case min
    /// Pixels beyond the image are true for both.
    case max
}

public nonisolated enum BinaryMorphology {
    // MARK: ndimage

    /// `ndimage.binary_erosion(mask, structure, iterations, border_value=)`.
    public static func erosion(_ mask: [Bool], width: Int, height: Int, structure: StructuringElement = .cross,
                               iterations: Int = 1, borderValue: Bool = false) throws -> [Bool] {
        try check(mask, width, height)
        return repeated(mask, iterations) { erodeOnce($0, width, height, structure, borderValue) }
    }

    /// `ndimage.binary_dilation(mask, structure, iterations, border_value=)`.
    public static func dilation(_ mask: [Bool], width: Int, height: Int, structure: StructuringElement = .cross,
                                iterations: Int = 1, borderValue: Bool = false) throws -> [Bool] {
        try check(mask, width, height)
        return repeated(mask, iterations) { dilateOnce($0, width, height, structure, borderValue) }
    }

    /// `ndimage.binary_opening`: erosion, then dilation, both with `borderValue`.
    public static func opening(_ mask: [Bool], width: Int, height: Int, structure: StructuringElement = .cross,
                               iterations: Int = 1, borderValue: Bool = false) throws -> [Bool] {
        let eroded = try erosion(mask, width: width, height: height, structure: structure, iterations: iterations, borderValue: borderValue)
        return try dilation(eroded, width: width, height: height, structure: structure, iterations: iterations, borderValue: borderValue)
    }

    /// `ndimage.binary_closing`: dilation, then erosion, both with `borderValue`.
    public static func closing(_ mask: [Bool], width: Int, height: Int, structure: StructuringElement = .cross,
                               iterations: Int = 1, borderValue: Bool = false) throws -> [Bool] {
        let dilated = try dilation(mask, width: width, height: height, structure: structure, iterations: iterations, borderValue: borderValue)
        return try erosion(dilated, width: width, height: height, structure: structure, iterations: iterations, borderValue: borderValue)
    }

    /// `ndimage.binary_fill_holes(mask, structure)`: background not connected, through `structure`, to the image border is filled.
    public static func fillHoles(_ mask: [Bool], width: Int, height: Int, structure: StructuringElement = .cross) throws -> [Bool] {
        try check(mask, width, height)
        let offsets = reflectedOffsets(structure)
        var reached = [Bool](repeating: false, count: mask.count)
        var stack: [Int] = []
        for r in 0..<height {
            for c in 0..<width where !mask[r * width + c] {
                // Seeded when a cell of the (reflected) structure around it falls outside the image: that cell reads as true.
                let outside = offsets.contains { r + $0.row < 0 || r + $0.row >= height || c + $0.col < 0 || c + $0.col >= width }
                if outside { reached[r * width + c] = true; stack.append(r * width + c) }
            }
        }
        while let i = stack.popLast() {
            let r = i / width, c = i % width
            for o in offsets {
                // Pixel p is reached when any pixel at p + offset is; so from i go to i - offset.
                let rr = r - o.row, cc = c - o.col
                guard rr >= 0, rr < height, cc >= 0, cc < width else { continue }
                let j = rr * width + cc
                if !mask[j], !reached[j] { reached[j] = true; stack.append(j) }
            }
        }
        return (0..<mask.count).map { !reached[$0] }
    }

    // MARK: scikit-image defaults

    /// `skimage.morphology.binary_erosion(mask, footprint, mode=)`.
    public static func skimageErosion(_ mask: [Bool], width: Int, height: Int, footprint: StructuringElement = .cross,
                                      mode: MorphologyMode = .ignore) throws -> [Bool] {
        try erosion(mask, width: width, height: height, structure: footprint, borderValue: mode != .min)
    }

    /// `skimage.morphology.binary_dilation(mask, footprint, mode=)`.
    public static func skimageDilation(_ mask: [Bool], width: Int, height: Int, footprint: StructuringElement = .cross,
                                       mode: MorphologyMode = .ignore) throws -> [Bool] {
        try dilation(mask, width: width, height: height, structure: footprint, borderValue: mode == .max)
    }

    /// `skimage.morphology.binary_opening`: its erosion, then its dilation, in `mode`.
    public static func skimageOpening(_ mask: [Bool], width: Int, height: Int, footprint: StructuringElement = .cross,
                                      mode: MorphologyMode = .ignore) throws -> [Bool] {
        let e = try skimageErosion(mask, width: width, height: height, footprint: footprint, mode: mode)
        return try skimageDilation(e, width: width, height: height, footprint: footprint, mode: mode)
    }

    /// `skimage.morphology.binary_closing`: its dilation, then its erosion, in `mode`.
    public static func skimageClosing(_ mask: [Bool], width: Int, height: Int, footprint: StructuringElement = .cross,
                                      mode: MorphologyMode = .ignore) throws -> [Bool] {
        let d = try skimageDilation(mask, width: width, height: height, footprint: footprint, mode: mode)
        return try skimageErosion(d, width: width, height: height, footprint: footprint, mode: mode)
    }

    // MARK: Small objects, holes, border

    /// `morphology.remove_small_objects(mask, max_size=, connectivity=)`: connected components of `connectivity` whose area is at most
    /// `maxSize` are cleared (the pre-0.26 `min_size=M` is `maxSize = M - 1`).
    public static func removeSmallObjects(_ mask: [Bool], width: Int, height: Int, maxSize: Int,
                                          connectivity: Segmentation.Connectivity = .four) throws -> [Bool] {
        try check(mask, width, height)
        guard maxSize > 0 else { return mask }
        let labels = try Segmentation.label(mask, width: width, height: height, connectivity: connectivity)
        var sizes = [Int](repeating: 0, count: labels.count + 1)
        for l in labels.labels where l > 0 { sizes[Int(l)] += 1 }
        return (0..<mask.count).map { i in
            let l = Int(labels.labels[i])
            return l > 0 && sizes[l] > maxSize
        }
    }

    /// `morphology.remove_small_holes(mask, max_size=, connectivity=)`: background components of at most `maxSize` pixels are filled.
    public static func removeSmallHoles(_ mask: [Bool], width: Int, height: Int, maxSize: Int,
                                        connectivity: Segmentation.Connectivity = .four) throws -> [Bool] {
        try check(mask, width, height)
        let inverse = mask.map { !$0 }
        let kept = try removeSmallObjects(inverse, width: width, height: height, maxSize: maxSize, connectivity: connectivity)
        return kept.map { !$0 }
    }

    /// `segmentation.clear_border(mask, buffer_size=)`: objects (8-connected, scikit-image's default) touching the border ring of
    /// width `bufferSize + 1` are cleared.
    public static func clearBorder(_ mask: [Bool], width: Int, height: Int, bufferSize: Int = 0) throws -> [Bool] {
        try check(mask, width, height)
        guard bufferSize >= 0, bufferSize < width, bufferSize < height else { throw MorphologyError.buffer }
        let labels = try Segmentation.label(mask, width: width, height: height, connectivity: .eight)
        let ring = bufferSize + 1
        var touching = Set<Int32>()
        for r in 0..<height {
            for c in 0..<width where r < ring || r >= height - ring || c < ring || c >= width - ring {
                let l = labels.labels[r * width + c]
                if l > 0 { touching.insert(l) }
            }
        }
        return (0..<mask.count).map { mask[$0] && !touching.contains(labels.labels[$0]) }
    }

    // MARK: Internals

    private static func check(_ mask: [Bool], _ width: Int, _ height: Int) throws {
        let (product, overflow) = width.multipliedReportingOverflow(by: height)
        guard width > 0, height > 0, !overflow, mask.count == product else { throw MorphologyError.shape }
    }

    /// Runs `step` `iterations` times, or until the mask stops changing when `iterations < 1`.
    private static func repeated(_ mask: [Bool], _ iterations: Int, _ step: ([Bool]) -> [Bool]) -> [Bool] {
        var current = mask
        if iterations >= 1 {
            for _ in 0..<iterations { current = step(current) }
            return current
        }
        while true {
            let next = step(current)
            if next == current { return current }
            current = next
        }
    }

    private struct Offset { let row: Int; let col: Int }

    /// The structure's cell offsets from its centre.
    private static func offsets(_ s: StructuringElement) -> [Offset] {
        var out: [Offset] = []
        for r in 0..<s.rows {
            for c in 0..<s.columns where s.cells[r * s.columns + c] { out.append(Offset(row: r - s.rows / 2, col: c - s.columns / 2)) }
        }
        return out
    }

    /// The offsets of the reflected structure (dilation reads the mask at pixel + reflected offset).
    private static func reflectedOffsets(_ s: StructuringElement) -> [Offset] {
        offsets(s).map { Offset(row: -$0.row, col: -$0.col) }
    }

    private static func erodeOnce(_ m: [Bool], _ w: Int, _ h: Int, _ s: StructuringElement, _ border: Bool) -> [Bool] {
        let o = offsets(s)
        var out = [Bool](repeating: false, count: m.count)
        for r in 0..<h {
            for c in 0..<w {
                var all = true
                for d in o {
                    let rr = r + d.row, cc = c + d.col
                    let v = rr >= 0 && rr < h && cc >= 0 && cc < w ? m[rr * w + cc] : border
                    if !v { all = false; break }
                }
                out[r * w + c] = all
            }
        }
        return out
    }

    private static func dilateOnce(_ m: [Bool], _ w: Int, _ h: Int, _ s: StructuringElement, _ border: Bool) -> [Bool] {
        let o = reflectedOffsets(s)
        var out = [Bool](repeating: false, count: m.count)
        for r in 0..<h {
            for c in 0..<w {
                var any = false
                for d in o {
                    let rr = r + d.row, cc = c + d.col
                    let v = rr >= 0 && rr < h && cc >= 0 && cc < w ? m[rr * w + cc] : border
                    if v { any = true; break }
                }
                out[r * w + c] = any
            }
        }
        return out
    }
}
