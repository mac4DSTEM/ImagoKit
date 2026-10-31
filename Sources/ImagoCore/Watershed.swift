// SPDX-License-Identifier: GPL-3.0-only
//
//  Watershed.swift (ImagoCore)
//  scikit-image 0.26.0's `segmentation.watershed(image, markers, mask=, connectivity=, compactness=, watershed_line=)` on a 2D
//  image: marker-controlled flooding
//  by priority queue. Labels are exact (the pins compare every label of every case).
//
//  SEMANTICS (scikit-image's `_watershed_cy.watershed_raveled`). The image and markers are padded by one pixel with a false mask, so
//  the flood never leaves the image. Marker pixels (marker != 0 inside the mask) enter a priority queue in raster order with age 0.
//  The queue pops the smallest (value, age); each neighbour not yet labelled and inside the mask takes the popped pixel's label AT ONCE
//  (not when it is popped) and enters the queue with the larger of its own image value and the popped pixel's value, and the next age
//  (1, 2, ...): a pixel is never flooded earlier than the level of the pixel that reached it (measured against the pins: without the
//  larger-of rule an image of integer levels differs in a third of its pixels). Neighbours are visited in
//  scikit-image's order: sorted by squared distance from the centre, ties in raster order: connectivity 4 is up, left, right, down;
//  connectivity 8 adds up-left, up-right, down-left, down-right. The queue is a binary heap with sift-up on push and sift-down on
//  pop, `smaller` being (value, then age); equal (value, age) pairs arise only among the markers, whose order the heap decides.
//
//  COMPACTNESS AND WATERSHED LINE. With `compactness` > 0 or `watershedLine`, a pixel is labelled when it leaves the queue (it can be
//  queued once per neighbour, a copy whose label is already set is skipped), and each queued copy remembers its marker (the source).
//  Compact: the queued value is image + compactness * (Euclidean distance to the source), fused (the arm64 build contracts it), then
//  raised to the popped pixel's value; no line is drawn even if asked (scikit-image's `if compact or not _diff_neighbors`). Line: a
//  pixel whose mask neighbours carry a nonzero label other than its source's leaves the mask with label 0, and still passes the
//  flood on to its neighbours. Adjacent markers can touch without a line between them, as upstream documents.
//
//  DEVIATIONS. A marker array of the wrong size or a connectivity other than 4 and 8 is refused. NaN image values are refused, and
//  so is a negative or non-finite `compactness` (scikit-image treats every value that is not > 0 as 0).
//

import Foundation

extension Segmentation {
    /// `watershed(image, markers, mask:, connectivity:, compactness:, watershed_line:)` with `connectivity` 4 (scikit-image 1) or
    /// 8 (scikit-image 2). Returns the label map: every pixel reached takes the label of the marker that flooded it; pixels outside
    /// `mask`, pixels no marker reaches and, with `watershedLine` (and no compactness), the line pixels are 0. Marker labels are kept
    /// as given (they need not be consecutive). `compactness` > 0 adds compactness x the Euclidean distance (in pixels) to the
    /// marker to each queued value (compact watershed; the line is then not drawn); a negative or non-finite one is refused.
    public static func watershed(image: [Double], markers: [Int32], rows height: Int, columns width: Int, mask: [Bool]? = nil,
                                 connectivity: Connectivity = .four, compactness: Double = 0,
                                 watershedLine: Bool = false) throws -> [Int32] {
        let (n, overflow) = width.multipliedReportingOverflow(by: height)
        guard width > 0, height > 0, !overflow, image.count == n, markers.count == n, mask.map({ $0.count == n }) ?? true else {
            throw MorphologyError.shape
        }
        guard !image.contains(where: \.isNaN) else { throw MorphologyError.parameter("image contains NaN") }
        guard compactness.isFinite, compactness >= 0 else { throw MorphologyError.parameter("compactness must be finite and not negative") }
        if compactness > 0 || watershedLine {
            return labelOnPop(image: image, markers: markers, width: width, height: height, mask: mask, connectivity: connectivity,
                              compactness: compactness, line: watershedLine)
        }
        let pw = width + 2, ph = height + 2
        var padImage = [Double](repeating: 0, count: pw * ph)
        var padMask = [Bool](repeating: false, count: pw * ph)
        var output = [Int32](repeating: 0, count: pw * ph)
        for r in 0..<height {
            for c in 0..<width {
                let i = r * width + c, p = (r + 1) * pw + c + 1
                padImage[p] = image[i]
                let inside = mask?[i] ?? true
                padMask[p] = inside
                if inside { output[p] = markers[i] }
            }
        }
        var offsets = [-pw, -1, 1, pw]
        if connectivity == .eight { offsets += [-pw - 1, -pw + 1, pw - 1, pw + 1] }
        var heap = FloodHeap()
        for p in 0..<pw * ph where output[p] != 0 { heap.push(FloodItem(value: padImage[p], age: 0, index: p)) }
        var age = 0
        while let item = heap.pop() {
            let label = output[item.index]
            for o in offsets {
                let q = item.index + o
                guard padMask[q], output[q] == 0 else { continue }
                age += 1
                output[q] = label
                heap.push(FloodItem(value: max(padImage[q], item.value), age: age, index: q))
            }
        }
        var result = [Int32](repeating: 0, count: n)
        for r in 0..<height { for c in 0..<width { result[r * width + c] = output[(r + 1) * pw + c + 1] } }
        return result
    }

    /// The compact and watershed-line flood: a pixel is labelled when it leaves the queue, not when it enters it, so it can be
    /// queued more than once (each queued copy carries the marker, the source, it came from).
    private static func labelOnPop(image: [Double], markers: [Int32], width: Int, height: Int, mask: [Bool]?,
                                   connectivity: Connectivity, compactness: Double, line: Bool) -> [Int32] {
        let n = width * height
        let pw = width + 2, ph = height + 2
        var padImage = [Double](repeating: 0, count: pw * ph)
        var padMask = [Bool](repeating: false, count: pw * ph)
        var output = [Int32](repeating: 0, count: pw * ph)
        for r in 0..<height {
            for c in 0..<width {
                let i = r * width + c, p = (r + 1) * pw + c + 1
                padImage[p] = image[i]
                let inside = mask?[i] ?? true
                padMask[p] = inside
                if inside { output[p] = markers[i] }
            }
        }
        var offsets = [-pw, -1, 1, pw]
        if connectivity == .eight { offsets += [-pw - 1, -pw + 1, pw - 1, pw + 1] }
        let compact = compactness > 0
        var heap = FloodHeap()
        for p in 0..<pw * ph where output[p] != 0 { heap.push(FloodItem(value: padImage[p], age: 0, index: p, source: p)) }
        var age = 0
        while let item = heap.pop() {
            if output[item.index] != 0, item.index != item.source { continue }
            // With compactness the line is never drawn (scikit-image's `if compact or not _diff_neighbors(...)`). A line pixel
            // leaves the mask unlabelled but still passes the flood on to its neighbours.
            let sourceLabel = output[item.source]
            if compact || !differentNeighbours(output, offsets, &padMask, item.index, sourceLabel) { output[item.index] = sourceLabel }
            for o in offsets {
                let q = item.index + o
                guard padMask[q], output[q] == 0 else { continue }
                age += 1
                var v = padImage[q]
                if compact {
                    let dr = Double(q / pw - item.source / pw), dc = Double(q % pw - item.source % pw)
                    let dist = (dr * dr + dc * dc).squareRoot()
                    // value + compactness * distance, fused as the arm64 build of scikit-image computes it (it decides exact ties).
                    v = v.addingProduct(compactness, dist)
                }
                if v < item.value { v = item.value }
                heap.push(FloodItem(value: v, age: age, index: q, source: item.source))
            }
        }
        var result = [Int32](repeating: 0, count: n)
        for r in 0..<height { for c in 0..<width { result[r * width + c] = output[(r + 1) * pw + c + 1] } }
        return result
    }

    /// scikit-image's `_diff_neighbors`: true (and the pixel leaves the mask, a line pixel) when the pixel is already outside the
    /// mask or a neighbour inside the mask carries a nonzero label other than `label`.
    private static func differentNeighbours(_ output: [Int32], _ offsets: [Int], _ mask: inout [Bool], _ index: Int,
                                            _ label: Int32) -> Bool {
        guard mask[index] else { return true }
        for o in offsets {
            let q = index + o
            guard mask[q] else { continue }
            let neighbour = output[q]
            if neighbour != 0, neighbour != label { mask[index] = false; return true }
        }
        return false
    }

    /// Marker image from peak coordinates, as the usual `markers[tuple(coords.T)] = arange(1, n + 1)`: the k-th coordinate (1-based)
    /// carries label k. Coordinates outside the image are refused; a repeated coordinate keeps the later label.
    public static func markers(peakRows: [Int], peakColumns: [Int], rows height: Int, columns width: Int) throws -> [Int32] {
        guard width > 0, height > 0, peakRows.count == peakColumns.count else { throw MorphologyError.shape }
        var out = [Int32](repeating: 0, count: width * height)
        for (k, rc) in zip(peakRows, peakColumns).enumerated() {
            guard rc.0 >= 0, rc.0 < height, rc.1 >= 0, rc.1 < width else { throw MorphologyError.parameter("peak outside the image") }
            out[rc.0 * width + rc.1] = Int32(k + 1)
        }
        return out
    }
}

private struct FloodItem {
    var value: Double
    var age: Int
    var index: Int
    var source: Int = 0
    @inline(__always) func smaller(than other: FloodItem) -> Bool { value != other.value ? value < other.value : age < other.age }
}

/// scikit-image's binary heap: sift-up on push, sift-down on pop (children compared with the parent in the order left, right).
private struct FloodHeap {
    private var data: [FloodItem] = []

    mutating func push(_ item: FloodItem) {
        data.append(item)
        var child = data.count - 1
        while child > 0 {
            let parent = (child - 1) / 2
            if data[child].smaller(than: data[parent]) { data.swapAt(child, parent); child = parent } else { break }
        }
    }

    mutating func pop() -> FloodItem? {
        guard !data.isEmpty else { return nil }
        let top = data[0]
        let last = data.removeLast()
        if !data.isEmpty {
            data[0] = last
            var i = 0
            while true {
                var smallest = i
                let l = 2 * i + 1, r = l + 1
                if l < data.count, data[l].smaller(than: data[smallest]) { smallest = l }
                if r < data.count, data[r].smaller(than: data[smallest]) { smallest = r }
                if smallest == i { break }
                data.swapAt(i, smallest); i = smallest
            }
        }
        return top
    }
}
