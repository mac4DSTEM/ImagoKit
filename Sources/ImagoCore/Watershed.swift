//
//  Watershed.swift (ImagoCore)
//  scikit-image 0.26.0's `segmentation.watershed(image, markers, mask=, connectivity=)` on a 2D image: marker-controlled flooding
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
//  DEVIATIONS. `compactness` and `watershed_line` are not ported (the study's use is plain flooding from peak markers); a marker
//  array of the wrong size or a connectivity other than 4 and 8 is refused. NaN image values are refused.
//

import Foundation

extension Segmentation {
    /// `watershed(image, markers, mask:, connectivity:)` with `connectivity` 4 (scikit-image 1) or 8 (scikit-image 2).
    /// Returns the label map: every pixel reached takes the label of the marker that flooded it; pixels outside `mask` and pixels no
    /// marker reaches are 0. Marker labels are kept as given (they need not be consecutive).
    public static func watershed(image: [Double], markers: [Int32], rows height: Int, columns width: Int, mask: [Bool]? = nil,
                                 connectivity: Connectivity = .four) throws -> [Int32] {
        let (n, overflow) = width.multipliedReportingOverflow(by: height)
        guard width > 0, height > 0, !overflow, image.count == n, markers.count == n, mask.map({ $0.count == n }) ?? true else {
            throw MorphologyError.shape
        }
        guard !image.contains(where: \.isNaN) else { throw MorphologyError.parameter("image contains NaN") }
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
