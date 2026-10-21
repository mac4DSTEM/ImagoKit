//
//  PeakLocalMax.swift (ImagoFilters)
//  scikit-image 0.26.0's `feature.peak_local_max` for one 2D image without `footprint=` or `labels=`: the pixels equal to the
//  maximum of their (2 · min_distance + 1)² window (edge-replicated, SciPy's `mode='nearest'`) and above the threshold, the
//  border cut off, ordered by decreasing intensity (stable, row-major among equals), thinned so that kept peaks lie at least
//  `minDistance` apart in the p-norm (greedy in that order: a kept peak removes every later one closer than `minDistance`),
//  cut to `numPeaks`. A constant image has no peak. Indices are exact against scikit-image. ±infinite pixels are ordinary extremes
//  (as scikit-image); a NaN pixel is refused (scikit-image answers by its comparisons with NaN, which depend on the threshold).
//

import Foundation

public nonisolated enum PeakBorder: Sendable, Equatable {
    /// scikit-image's `exclude_border=True`: `minDistance` pixels on every side.
    case minDistance
    /// `exclude_border=(row, column)`: peaks within that many pixels of the border are dropped (0 keeps all).
    case width(row: Int, column: Int)
}

public nonisolated struct PeakCoordinate: Sendable, Equatable, Hashable {
    public let row: Int
    public let column: Int

    public init(row: Int, column: Int) { self.row = row; self.column = column }
}

public nonisolated enum PeakLocalMax {
    /// `peak_local_max(image, min_distance, threshold_abs, threshold_rel, exclude_border, num_peaks, p_norm=)`.
    /// The threshold is `thresholdAbs ?? min(image)`, raised to `thresholdRel · max(image)` when that is larger, compared in the
    /// image's precision with `>`.
    @_specialize(where S == Float)
    @_specialize(where S == Double)
    public static func find<S: BinaryFloatingPoint & Sendable>(
        _ a: [S], rows: Int, columns: Int, minDistance: Int = 1, thresholdAbs: Double? = nil, thresholdRel: Double? = nil,
        excludeBorder: PeakBorder = .minDistance, numPeaks: Int? = nil, pNorm: Double = .infinity
    ) throws -> [PeakCoordinate] {
        guard rows > 0, columns > 0 else { throw FilterError.shape }
        guard rows <= Int.max / columns, a.count == rows * columns else { throw FilterError.shape }
        guard !a.contains(where: { $0.isNaN }) else { throw FilterError.parameter("NaN pixel") }   // ±infinity is an ordinary extreme, as in scikit-image
        guard minDistance >= 0 else { throw FilterError.parameter("minDistance") }
        guard pNorm >= 1 else { throw FilterError.parameter("pNorm") }
        if let numPeaks, numPeaks < 0 { throw FilterError.parameter("numPeaks") }
        let (bRow, bCol): (Int, Int)
        switch excludeBorder {
        case .minDistance: (bRow, bCol) = (minDistance, minDistance)
        case .width(let r, let c):
            guard r >= 0, c >= 0 else { throw FilterError.parameter("excludeBorder") }
            (bRow, bCol) = (r, c)
        }
        var threshold: S = thresholdAbs.map { S($0) } ?? a.min()!
        if let rel = thresholdRel { threshold = max(threshold, S(rel) * a.max()!) }

        // Candidates: image == maximum_filter(image, window, 'nearest') and image > threshold.
        var out = [Bool](repeating: false, count: rows * columns)
        if minDistance == 0 || rows * columns == 1 {
            for i in 0..<out.count { out[i] = a[i] > threshold }
        } else {
            let alongColumns = windowMax(a, rows, columns, minDistance, alongRows: false)
            let peak = windowMax(alongColumns, rows, columns, minDistance, alongRows: true)
            var trivial = true
            for i in 0..<out.count { out[i] = a[i] == peak[i]; if !out[i] { trivial = false } }
            if trivial { out = [Bool](repeating: false, count: out.count) }
            for i in 0..<out.count where out[i] && !(a[i] > threshold) { out[i] = false }
        }
        for r in 0..<rows { for c in 0..<columns where r < bRow || r >= rows - bRow || c < bCol || c >= columns - bCol {
            out[r * columns + c] = false
        } }
        // Highest first; equal intensities keep row-major order (a stable sort of −intensity).
        var index = (0..<out.count).filter { out[$0] }
        index.sort { a[$0] > a[$1] || (a[$0] == a[$1] && $0 < $1) }
        var coords = index.map { PeakCoordinate(row: $0 / columns, column: $0 % columns) }
        if minDistance > 1 { coords = ensureSpacing(coords, spacing: Double(minDistance), pNorm: pNorm, maxOut: numPeaks) }
        if let numPeaks, coords.count > numPeaks { coords = Array(coords.prefix(numPeaks)) }
        return coords
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
