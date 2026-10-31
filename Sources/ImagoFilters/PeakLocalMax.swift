// SPDX-License-Identifier: GPL-3.0-only
//
//  PeakLocalMax.swift (ImagoFilters)
//  scikit-image 0.26.0's `feature.peak_local_max` for one 2D image: the pixels equal to the maximum of their
//  (2 · min_distance + 1)² window or of `footprint=` (edge-replicated, SciPy's `mode='nearest'`) and above the threshold, the
//  border cut off, ordered by decreasing intensity (stable, row-major among equals), thinned so that kept peaks lie at least
//  `minDistance` apart in the p-norm (greedy in that order: a kept peak removes every later one closer than `minDistance`),
//  cut to `numPeaks`. A constant image has no peak. Indices are exact against scikit-image. ±infinite pixels are ordinary extremes
//  (as scikit-image); a NaN pixel is refused (scikit-image answers by its comparisons with NaN, which depend on the threshold).
//  `labels=` runs the same per label inside its bounding box (see `labelledPeaks`), as scikit-image does.
//

import Foundation

public nonisolated enum PeakBorder: Sendable, Equatable {
    /// scikit-image's `exclude_border=True`: `minDistance` pixels on every side.
    case minDistance
    /// `exclude_border=(row, column)`: peaks within that many pixels of the border are dropped (0 keeps all).
    case width(row: Int, column: Int)
}

/// `peak_local_max(footprint=)`: the neighbourhood a candidate must be the maximum of, `rows × columns` row-major, its origin
/// at (rows / 2, columns / 2) as SciPy's `maximum_filter` places it. An all-false footprint of more than one cell is refused,
/// as SciPy refuses it.
public nonisolated struct PeakFootprint: Sendable, Equatable {
    public let rows: Int
    public let columns: Int
    public let mask: [Bool]

    public init(rows: Int, columns: Int, mask: [Bool]) throws {
        guard rows > 0, columns > 0, rows <= Int.max / columns, mask.count == rows * columns else { throw FilterError.shape }
        guard mask.count == 1 || mask.contains(true) else { throw FilterError.parameter("all-false footprint") }
        self.rows = rows; self.columns = columns; self.mask = mask
    }
}

public nonisolated struct PeakCoordinate: Sendable, Equatable, Hashable {
    public let row: Int
    public let column: Int

    public init(row: Int, column: Int) { self.row = row; self.column = column }
}

/// A centrosymmetric image (a power spectrum): peaks come in mirror pairs through the centre and the centre itself is the transform's
/// bright core. A kit extension to `peak_local_max`, not a scikit-image option.
public nonisolated struct SpectrumSymmetry: Sendable, Equatable {
    /// The centre pixel (a spectrum of n bins is centred at n / 2).
    public let centreRow: Int
    public let centreColumn: Int
    /// Peaks within this many pixels of the centre (Euclidean, inclusive) are dropped; 0 drops only the centre itself.
    public let excludedRadius: Int
    /// Keep one of each mirror pair: the peak in the half-plane `row > centreRow`, or on `row == centreRow` with `column > centreColumn`.
    public let oneOfEachMirrorPair: Bool

    public init(centreRow: Int, centreColumn: Int, excludedRadius: Int = 0, oneOfEachMirrorPair: Bool = true) {
        self.centreRow = centreRow; self.centreColumn = centreColumn
        self.excludedRadius = excludedRadius; self.oneOfEachMirrorPair = oneOfEachMirrorPair
    }
}

public nonisolated enum PeakLocalMax {
    /// `peak_local_max(image, min_distance, threshold_abs, threshold_rel, exclude_border, num_peaks, footprint, labels,
    /// num_peaks_per_label, p_norm)`. The threshold is `thresholdAbs ?? min(image)`, raised to `thresholdRel · max(image)`
    /// when that is larger, compared in the image's precision with `>`. `footprint` replaces the (2 · minDistance + 1)²
    /// square (minDistance still spaces the peaks and sets the default border); `labels` (one Int per pixel, 0 and below
    /// background) finds peaks per label, at most `numPeaksPerLabel` each. A `symmetry` with `labels` is refused.
    @_specialize(where S == Float)
    @_specialize(where S == Double)
    public static func find<S: BinaryFloatingPoint & Sendable>(
        _ a: [S], rows: Int, columns: Int, minDistance: Int = 1, thresholdAbs: Double? = nil, thresholdRel: Double? = nil,
        excludeBorder: PeakBorder = .minDistance, numPeaks: Int? = nil, footprint: PeakFootprint? = nil,
        labels: [Int]? = nil, numPeaksPerLabel: Int? = nil, pNorm: Double = .infinity,
        symmetry: SpectrumSymmetry? = nil
    ) throws -> [PeakCoordinate] {
        guard rows > 0, columns > 0 else { throw FilterError.shape }
        guard rows <= Int.max / columns, a.count == rows * columns else { throw FilterError.shape }
        guard !a.contains(where: { $0.isNaN }) else { throw FilterError.parameter("NaN pixel") }   // ±infinity is an ordinary extreme, as in scikit-image
        guard minDistance >= 0 else { throw FilterError.parameter("minDistance") }
        guard pNorm >= 1 else { throw FilterError.parameter("pNorm") }
        if let numPeaks, numPeaks < 0 { throw FilterError.parameter("numPeaks") }
        if let symmetry, symmetry.excludedRadius < 0 { throw FilterError.parameter("excludedRadius") }
        if let numPeaksPerLabel, numPeaksPerLabel < 0 { throw FilterError.parameter("numPeaksPerLabel") }
        if let labels {
            guard labels.count == a.count else { throw FilterError.shape }
            guard symmetry == nil else { throw FilterError.parameter("symmetry with labels") }
        }
        let (bRow, bCol): (Int, Int)
        switch excludeBorder {
        case .minDistance: (bRow, bCol) = (minDistance, minDistance)
        case .width(let r, let c):
            guard r >= 0, c >= 0 else { throw FilterError.parameter("excludeBorder") }
            (bRow, bCol) = (r, c)
        }
        var threshold: S = thresholdAbs.map { S($0) } ?? a.min()!
        if let rel = thresholdRel { threshold = max(threshold, S(rel) * a.max()!) }

        func inBorder(_ r: Int, _ c: Int) -> Bool { r < bRow || r >= rows - bRow || c < bCol || c >= columns - bCol }
        if let labels {
            return labelledPeaks(a, rows: rows, columns: columns, labels: labels, inBorder: inBorder, minDistance: minDistance,
                                 footprint: footprint, threshold: threshold, numPeaks: numPeaks,
                                 numPeaksPerLabel: numPeaksPerLabel, pNorm: pNorm)
        }
        // Candidates: image == maximum_filter(image, footprint, 'nearest') and image > threshold.
        var out = peakMask(a, rows, columns, minDistance: minDistance, footprint: footprint, threshold: threshold, labelMask: nil)
        for r in 0..<rows { for c in 0..<columns where inBorder(r, c) { out[r * columns + c] = false } }
        // With a symmetry the cut to `numPeaks` waits until the centre and the mirror half are dropped.
        var coords = highIntensityPeaks(a, out, columns: columns, minDistance: minDistance, pNorm: pNorm,
                                        maxOut: symmetry == nil ? numPeaks : nil)
        if let symmetry {
            coords = coords.filter { p in
                let dy = p.row - symmetry.centreRow, dx = p.column - symmetry.centreColumn
                if dx * dx + dy * dy <= symmetry.excludedRadius * symmetry.excludedRadius { return false }
                return !symmetry.oneOfEachMirrorPair || dy > 0 || (dy == 0 && dx > 0)
            }
        }
        if let numPeaks, coords.count > numPeaks { coords = Array(coords.prefix(numPeaks)) }
        return coords
    }

    /// `_get_peak_mask`: the pixels equal to the maximum of their footprint (the trivial image, all equal, has none, except
    /// with a label mask the pixels the mask's binary opening removes), above the threshold.
    private static func peakMask<S: BinaryFloatingPoint>(_ a: [S], _ rows: Int, _ columns: Int, minDistance: Int,
                                                         footprint: PeakFootprint?, threshold: S, labelMask: [Bool]?) -> [Bool] {
        let size = footprint.map { $0.rows * $0.columns } ?? (minDistance == 0 ? 1 : 2)
        var out = [Bool](repeating: false, count: rows * columns)
        if size == 1 || rows * columns == 1 {
            for i in out.indices { out[i] = a[i] > threshold }
            return out
        }
        let peak: [S]
        if let footprint {
            peak = footprintMax(a, rows, columns, footprint)
        } else {
            peak = windowMax(windowMax(a, rows, columns, minDistance, alongRows: false), rows, columns, minDistance, alongRows: true)
        }
        var trivial = true
        for i in out.indices {
            out[i] = a[i] == peak[i]
            if !out[i] && (labelMask?[i] ?? true) { trivial = false }
        }
        if trivial {
            out = [Bool](repeating: false, count: out.count)
            if let labelMask { out = isolatedPixels(labelMask, rows, columns) }
        }
        for i in out.indices where out[i] && !(a[i] > threshold) { out[i] = false }
        return out
    }

    /// `_get_high_intensity_peaks`: highest first, equal intensities in row-major order (a stable sort of −intensity), spaced
    /// by `ensure_spacing` when `minDistance > 1` (stopping at `maxOut`).
    private static func highIntensityPeaks<S: BinaryFloatingPoint>(_ a: [S], _ mask: [Bool], columns: Int, minDistance: Int,
                                                                   pNorm: Double, maxOut: Int?) -> [PeakCoordinate] {
        var index = mask.indices.filter { mask[$0] }
        index.sort { a[$0] > a[$1] || (a[$0] == a[$1] && $0 < $1) }
        let coords = index.map { PeakCoordinate(row: $0 / columns, column: $0 % columns) }
        return minDistance > 1 ? ensureSpacing(coords, spacing: Double(minDistance), pNorm: pNorm, maxOut: maxOut) : coords
    }

    /// The `labels=` branch: each label in the bounding box of what the border cut leaves of it, the pixels outside the label
    /// (by the uncut labels) set to the type's lowest finite value, peaks found there and cut to `numPeaksPerLabel`, in label
    /// order; when more than `numPeaks` remain, they are sorted and spaced again over the whole image and cut.
    private static func labelledPeaks<S: BinaryFloatingPoint>(
        _ a: [S], rows: Int, columns: Int, labels: [Int], inBorder: (Int, Int) -> Bool, minDistance: Int,
        footprint: PeakFootprint?, threshold: S, numPeaks: Int?, numPeaksPerLabel: Int?, pNorm: Double
    ) -> [PeakCoordinate] {
        // `find_objects` of the border-cut labels: one bounding box per positive label, in label order.
        var boxes: [Int: (r0: Int, r1: Int, c0: Int, c1: Int)] = [:]
        for r in 0..<rows { for c in 0..<columns where !inBorder(r, c) {
            let l = labels[r * columns + c]
            guard l > 0 else { continue }
            if let b = boxes[l] { boxes[l] = (min(b.r0, r), max(b.r1, r), min(b.c0, c), max(b.c1, c)) } else { boxes[l] = (r, r, c, c) }
        } }
        let background = -S.greatestFiniteMagnitude
        var all: [PeakCoordinate] = []
        for l in boxes.keys.sorted() {
            let b = boxes[l]!
            let h = b.r1 - b.r0 + 1, w = b.c1 - b.c0 + 1
            var object = [S](repeating: background, count: h * w)
            var labelMask = [Bool](repeating: false, count: h * w)
            for r in 0..<h { for c in 0..<w {
                let i = (r + b.r0) * columns + c + b.c0
                if labels[i] == l { labelMask[r * w + c] = true; object[r * w + c] = a[i] }
            } }
            let mask = peakMask(object, h, w, minDistance: minDistance, footprint: footprint, threshold: threshold, labelMask: labelMask)
            var coords = highIntensityPeaks(object, mask, columns: w, minDistance: minDistance, pNorm: pNorm, maxOut: numPeaksPerLabel)
            if let numPeaksPerLabel, coords.count > numPeaksPerLabel { coords = Array(coords.prefix(numPeaksPerLabel)) }
            all += coords.map { PeakCoordinate(row: $0.row + b.r0, column: $0.column + b.c0) }
        }
        if let numPeaks, all.count > numPeaks {
            var out = [Bool](repeating: false, count: rows * columns)
            for p in all { out[p.row * columns + p.column] = true }
            all = highIntensityPeaks(a, out, columns: columns, minDistance: minDistance, pNorm: pNorm, maxOut: numPeaks)
            if all.count > numPeaks { all = Array(all.prefix(numPeaks)) }
        }
        return all
    }

    /// SciPy's `maximum_filter(footprint=, mode='nearest')`: the footprint's origin at (rows / 2, columns / 2).
    private static func footprintMax<S: BinaryFloatingPoint>(_ a: [S], _ rows: Int, _ columns: Int, _ f: PeakFootprint) -> [S] {
        // A full rectangle is separable (SciPy takes that path too): offsets −size/2 … size − size/2 − 1 on each axis.
        if !f.mask.contains(false) {
            func pass(_ x: [S], alongRows: Bool) -> [S] {
                let n = alongRows ? rows : columns, s = alongRows ? f.rows : f.columns
                let lo = -(s / 2), hi = s - s / 2 - 1
                var out = x
                for line in 0..<(alongRows ? columns : rows) {
                    for i in 0..<n {
                        var m = -S.infinity
                        for k in max(i + lo, -1)...min(i + hi, n) {   // beyond the edge the edge sample repeats
                            let j = min(max(k, 0), n - 1)
                            let v = alongRows ? x[j * columns + line] : x[line * columns + j]
                            if v > m { m = v }
                        }
                        if alongRows { out[i * columns + line] = m } else { out[line * columns + i] = m }
                    }
                }
                return out
            }
            return pass(pass(a, alongRows: false), alongRows: true)
        }
        let offsets = (0..<(f.rows * f.columns)).filter { f.mask[$0] }.map { ($0 / f.columns - f.rows / 2, $0 % f.columns - f.columns / 2) }
        var out = [S](repeating: -S.infinity, count: a.count)
        for r in 0..<rows { for c in 0..<columns {
            var m = -S.infinity
            for (dr, dc) in offsets {
                let v = a[min(max(r + dr, 0), rows - 1) * columns + min(max(c + dc, 0), columns - 1)]
                if v > m { m = v }
            }
            out[r * columns + c] = m
        } }
        return out
    }

    /// `mask XOR binary_opening(mask)` with SciPy's default cross and a background border: the mask pixels the opening removes.
    private static func isolatedPixels(_ m: [Bool], _ rows: Int, _ columns: Int) -> [Bool] {
        func at(_ v: [Bool], _ r: Int, _ c: Int) -> Bool { r >= 0 && r < rows && c >= 0 && c < columns && v[r * columns + c] }
        var eroded = [Bool](repeating: false, count: m.count)
        for r in 0..<rows { for c in 0..<columns {
            eroded[r * columns + c] = at(m, r, c) && at(m, r - 1, c) && at(m, r + 1, c) && at(m, r, c - 1) && at(m, r, c + 1)
        } }
        var out = [Bool](repeating: false, count: m.count)
        for r in 0..<rows { for c in 0..<columns {
            let opened = at(eroded, r, c) || at(eroded, r - 1, c) || at(eroded, r + 1, c) || at(eroded, r, c - 1) || at(eroded, r, c + 1)
            out[r * columns + c] = m[r * columns + c] != opened
        } }
        return out
    }

    /// The running maximum over `2 · d + 1` samples along one axis, the edge replicated.
    private static func windowMax<S: BinaryFloatingPoint>(_ a: [S], _ rows: Int, _ columns: Int, _ d: Int, alongRows: Bool) -> [S] {
        var out = a
        let n = alongRows ? rows : columns
        let d = min(d, n)
        let lines = alongRows ? columns : rows
        for line in 0..<lines {
            for i in 0..<n {
                var m = -S.infinity
                for k in (i - d)...(i + d) {
                    let j = min(max(k, 0), n - 1)
                    let v = alongRows ? a[j * columns + line] : a[line * columns + j]
                    if v > m { m = v }
                }
                if alongRows { out[i * columns + line] = m } else { out[line * columns + i] = m }
            }
        }
        return out
    }

    /// `skimage._shared.coord.ensure_spacing`: greedy in the given order, a kept point removes later points at p-distance < spacing.
    static func ensureSpacing(_ coords: [PeakCoordinate], spacing: Double, pNorm: Double, maxOut: Int?) -> [PeakCoordinate] {
        func distance(_ p: PeakCoordinate, _ q: PeakCoordinate) -> Double {
            let dr = Double(abs(p.row - q.row)), dc = Double(abs(p.column - q.column))
            if pNorm == .infinity { return max(dr, dc) }
            if pNorm == 1 { return dr + dc }
            if pNorm == 2 { return (dr * dr + dc * dc).squareRoot() }
            return pow(pow(dr, pNorm) + pow(dc, pNorm), 1 / pNorm)
        }
        let cell = max(1, Int(spacing.rounded(.up)))
        var grid: [Int: [PeakCoordinate]] = [:]
        func key(_ r: Int, _ c: Int) -> Int { (r / cell) &* 1_000_003 &+ (c / cell) }
        var kept: [PeakCoordinate] = []
        for p in coords {
            var ok = true
            search: for dr in -1...1 { for dc in -1...1 {
                let r = p.row / cell + dr, c = p.column / cell + dc
                guard let bucket = grid[r &* 1_000_003 &+ c] else { continue }
                for q in bucket where distance(p, q) < spacing { ok = false; break search }
            } }
            if ok {
                kept.append(p)
                grid[key(p.row, p.column), default: []].append(p)
                if let maxOut, kept.count >= maxOut { break }
            }
        }
        return kept
    }
}
