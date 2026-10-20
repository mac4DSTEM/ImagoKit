//
//  Colormap.swift (ImagoCore)
//  The four colour tables of the display, as 256-entry RGBA tables for `DisplayMapping`.
//
//  DEFINITION: each map is a short list of anchors (position t, red, green, blue in 0...1) joined by straight lines.
//  Entry i of an N-entry table sits at t = i / (N − 1); below the first anchor or above the last it holds that anchor's
//  colour; between two anchors each channel is `a + (b − a) · (t − t_a) / (t_b − t_a)` in Double. A byte is
//  `(channel · 255).rounded()` (half away from zero), clamped to 0...255; alpha is 255.
//
//  THESE ARE NOT matplotlib's tables: `viridis` and `inferno` are five and six anchors read off the matplotlib maps, so a
//  colour can differ from matplotlib's by several percent of full scale. The tables are pinned to the arithmetic above
//  (an independent NumPy reading and the application's own table bytes), not to matplotlib, which is not in the pinned
//  reference environment.
//

import Foundation

/// A display colour table: viridis, inferno, gray, and the diverging red-blue map.
public nonisolated enum Colormap: String, Sendable, CaseIterable, Equatable, Hashable {
    case viridis, inferno, gray, rdbu

    /// True for the map whose middle is the neutral value (`rdbu`).
    public var isDiverging: Bool { self == .rdbu }

    /// The anchors: (t, r, g, b), t ascending from 0 to 1, channels in 0...1.
    public var anchors: [(t: Double, r: Double, g: Double, b: Double)] {
        switch self {
        case .gray:
            return [(0, 0, 0, 0), (1, 1, 1, 1)]
        case .viridis:
            return [(0.00, 0.267, 0.005, 0.329), (0.25, 0.254, 0.265, 0.530), (0.50, 0.128, 0.567, 0.551),
                    (0.75, 0.369, 0.789, 0.383), (1.00, 0.993, 0.906, 0.144)]
        case .inferno:
            return [(0.00, 0.001, 0.000, 0.014), (0.25, 0.258, 0.039, 0.406), (0.50, 0.578, 0.148, 0.404),
                    (0.75, 0.865, 0.316, 0.226), (0.90, 0.988, 0.645, 0.040), (1.00, 0.988, 0.998, 0.645)]
        case .rdbu:
            return [(0.00, 0.129, 0.400, 0.674), (0.50, 0.969, 0.969, 0.969), (1.00, 0.698, 0.094, 0.168)]
        }
    }

    /// The colour at position `t` (0...1) in Double: the anchors joined by straight lines, held beyond the ends.
    public func color(at t: Double) -> (r: Double, g: Double, b: Double) {
        guard !t.isNaN else { let f = anchors[0]; return (f.r, f.g, f.b) }      // a NaN reads the first colour
        let stops = anchors
        let first = stops[0], last = stops[stops.count - 1]
        if t <= first.t { return (first.r, first.g, first.b) }
        if t >= last.t { return (last.r, last.g, last.b) }
        for index in 1..<stops.count {
            let start = stops[index - 1], end = stops[index]
            if t <= end.t {
                let f = (t - start.t) / (end.t - start.t)
                return (start.r + (end.r - start.r) * f, start.g + (end.g - start.g) * f, start.b + (end.b - start.b) * f)
            }
        }
        return (1, 1, 1)
    }

    /// RGBA bytes, `count * 4` of them (`count` ≥ 2; 256 is the display's table). A `count` below 2 returns an empty table.
    public func lutRGBA(count: Int = 256) -> [UInt8] {
        guard count >= 2 else { return [] }
        var out = [UInt8](repeating: 255, count: count * 4)
        @inline(__always) func byte(_ v: Double) -> UInt8 { UInt8(min(max((v * 255).rounded(), 0), 255)) }
        for index in 0..<count {
            let c = color(at: Double(index) / Double(count - 1))
            out[index * 4] = byte(c.r)
            out[index * 4 + 1] = byte(c.g)
            out[index * 4 + 2] = byte(c.b)
        }
        return out
    }
}

extension DisplayMapping {
    /// The mapping through one of the display's own colour tables (256 entries).
    public init(low: Float, high: Float, gamma: Double, colormap: Colormap) {
        self.init(low: low, high: high, gamma: gamma, lut: colormap.lutRGBA(count: 256))!
    }

    /// The colour a linearly filtered, clamp-to-edge 256-texel table returns for a normalised value `v` (texel centres at
    /// (i + ½) / 256), as 8-bit RGB; `lut` is 1024 RGBA bytes, nil if it is not. A NaN `v` reads the first texel.
    public static func sampleRGB(_ lut: [UInt8], _ v: Double) -> (UInt8, UInt8, UInt8)? {
        guard lut.count == 1024 else { return nil }
        let raw = v * 256 - 0.5
        let p = raw.isNaN ? 0 : min(max(raw, 0), 255)
        let i0 = Int(p.rounded(.down))
        let i1 = min(i0 + 1, 255)
        let t = p - Double(i0)
        @inline(__always) func channel(_ c: Int) -> UInt8 {
            let a = Double(lut[i0 * 4 + c]), b = Double(lut[i1 * 4 + c])
            return UInt8(min(max((a + (b - a) * t).rounded(), 0), 255))
        }
        return (channel(0), channel(1), channel(2))
    }
}
