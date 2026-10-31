// Closed-form checks for the multichannel (`channels`) and integer-image paths of Denoise (no fixtures); the pinned
// comparisons with scikit-image are in ImagoKitParityTests (DenoiseChannelTests).
import Foundation
import ImagoFilters
import Testing

@Suite struct DenoiseChannelClosedFormTests {
    /// A channel-last image of `channels` planes, each a different ramp with a deterministic ripple.
    private func interleaved(rows: Int, columns: Int, channels: Int) -> [Double] {
        var x = [Double](repeating: 0, count: rows * columns * channels)
        for r in 0..<rows {
            for c in 0..<columns {
                for k in 0..<channels {
                    let ripple = sin(Double(r * 7 + c * 3 + k * 11)) * 0.1
                    x[(r * columns + c) * channels + k] = 0.2 + 0.05 * Double(k) + 0.03 * Double(r) + 0.02 * Double(c) + ripple
                }
            }
        }
        return x
    }

    private func plane(_ x: [Double], _ k: Int, _ channels: Int) -> [Double] {
        stride(from: k, to: x.count, by: channels).map { x[$0] }
    }

    @Test func totalVariationRunsEachChannelAlone() throws {
        let (rows, columns, channels) = (7, 6, 3)
        let x = interleaved(rows: rows, columns: columns, channels: channels)
        let a = try Denoise.totalVariationChambolle(x, rows: rows, columns: columns, channels: channels, weight: 0.3)
        let b = try Denoise.totalVariationBregman(x, rows: rows, columns: columns, channels: channels, weight: 2)
        for k in 0..<channels {
            let p = plane(x, k, channels)
            #expect(plane(a, k, channels) == (try Denoise.totalVariationChambolle(p, rows: rows, columns: columns, weight: 0.3)))
            // Bregman: scikit-image reuses one output buffer across the channels and its kernel seeds dx, dy, bx and by from
            // it, so only the first channel is the single-channel result; the later ones start from the previous channel.
            let single = try Denoise.totalVariationBregman(p, rows: rows, columns: columns, weight: 2)
            if k == 0 { #expect(plane(b, k, channels) == single) } else { #expect(plane(b, k, channels) != single) }
        }
    }

    @Test func oneChannelIsTheSingleChannelCall() throws {
        let x = interleaved(rows: 6, columns: 5, channels: 1)
        for fast in [true, false] {
            #expect(try Denoise.nonLocalMeans(x, rows: 6, columns: 5, channels: 1, patchSize: 3, patchDistance: 2, fastMode: fast)
                    == (try Denoise.nonLocalMeans(x, rows: 6, columns: 5, patchSize: 3, patchDistance: 2, fastMode: fast)))
        }
    }

    /// Non-local means weighs patches over all channels together and divides the distance by the channel count, so two
    /// copies of one plane give that plane's single-channel result in both channels (to rounding of the doubled sums).
    @Test func nonLocalMeansSharesWeightsAcrossChannels() throws {
        let p = interleaved(rows: 8, columns: 7, channels: 1)
        var twice = [Double]()
        for v in p { twice.append(v); twice.append(v) }
        for fast in [true, false] {
            let one = try Denoise.nonLocalMeans(p, rows: 8, columns: 7, patchSize: 5, patchDistance: 3, h: 0.1, fastMode: fast, sigma: 0.02)
            let two = try Denoise.nonLocalMeans(twice, rows: 8, columns: 7, channels: 2, patchSize: 5, patchDistance: 3, h: 0.1,
                                                fastMode: fast, sigma: 0.02)
            for k in 0..<2 {
                let got = plane(two, k, 2)
                #expect(zip(got, one).allSatisfy { abs($0 - $1) <= 1e-12 * abs($1) }, "fast \(fast) channel \(k)")
            }
        }
    }

    @Test func integerImagesAreScaledAsImgAsFloat() throws {
        let u: [UInt8] = (0..<30).map { UInt8(($0 * 37) % 256) }
        let uf = u.map { Double($0) * (1.0 / 255.0) }
        #expect(try Denoise.totalVariationChambolle(u, rows: 5, columns: 6) == (try Denoise.totalVariationChambolle(uf, rows: 5, columns: 6)))
        #expect(try Denoise.totalVariationBregman(u, rows: 5, columns: 6) == (try Denoise.totalVariationBregman(uf, rows: 5, columns: 6)))
        #expect(try Denoise.nonLocalMeans(u, rows: 5, columns: 6, patchSize: 3, patchDistance: 2)
                == (try Denoise.nonLocalMeans(uf, rows: 5, columns: 6, patchSize: 3, patchDistance: 2)))
        // signed: x / max, the most negative value clipped to -1
        let s: [Int8] = [-128, -127, -1, 0, 1, 127]
        let sf = s.map { max(Double($0) * (1.0 / 127.0), -1) }
        #expect(sf[0] == -1 && sf[1] == -1 && sf[5] == 1)
        #expect(try Denoise.totalVariationChambolle(s, rows: 2, columns: 3) == (try Denoise.totalVariationChambolle(sf, rows: 2, columns: 3)))
        // preserve_range keeps the integer values
        let w: [UInt16] = (0..<20).map { UInt16($0 * 1000) }
        #expect(try Denoise.nonLocalMeans(w, rows: 4, columns: 5, patchSize: 3, patchDistance: 2, h: 500, preserveRange: true)
                == (try Denoise.nonLocalMeans(w.map { Double($0) }, rows: 4, columns: 5, patchSize: 3, patchDistance: 2, h: 500)))
        // 64-bit extremes stay finite
        let big: [UInt64] = [0, .max, 1 << 63, 12345]
        #expect(try Denoise.totalVariationChambolle(big, rows: 2, columns: 2).allSatisfy(\.isFinite))
    }

    /// The integer overloads are disfavoured: an integer array literal still means a Float64 image, as before they existed.
    @Test func integerLiteralsStayFloat64() throws {
        #expect(try Denoise.totalVariationChambolle([0, 1, 1, 0], rows: 2, columns: 2)
                == (try Denoise.totalVariationChambolle([0.0, 1, 1, 0], rows: 2, columns: 2)))
        #expect(try Denoise.totalVariationBregman([0, 1, 1, 0], rows: 2, columns: 2)
                == (try Denoise.totalVariationBregman([0.0, 1, 1, 0], rows: 2, columns: 2)))
        let n: [Double] = try Denoise.nonLocalMeans([0, 1, 2, 3, 4, 5], rows: 2, columns: 3, patchSize: 3, patchDistance: 1, h: 2)
        #expect(n == (try Denoise.nonLocalMeans([0.0, 1, 2, 3, 4, 5], rows: 2, columns: 3, patchSize: 3, patchDistance: 1, h: 2)))
        #expect(n.contains { $0 > 0.5 })
    }

    @Test func channelRefusals() {
        let x = [Double](repeating: 0.5, count: 24)
        #expect(throws: DenoiseError.shape) { try Denoise.totalVariationChambolle(x, rows: 2, columns: 4, channels: 0) }
        #expect(throws: DenoiseError.shape) { try Denoise.totalVariationChambolle(x, rows: 2, columns: 4, channels: 2) }
        #expect(throws: DenoiseError.shape) { try Denoise.totalVariationBregman(x, rows: 2, columns: 4, channels: -3) }
        #expect(throws: DenoiseError.shape) { try Denoise.nonLocalMeans(x, rows: 2, columns: 4, channels: 4) }
        #expect(throws: DenoiseError.shape) { try Denoise.nonLocalMeans(x, rows: 2, columns: 4, channels: Int.max) }
        #expect(throws: DenoiseError.self) { try Denoise.totalVariationBregman(x, rows: 1, columns: 8, channels: 3) }
        var nan = x; nan[5] = .nan
        #expect(throws: DenoiseError.nonFinite) { try Denoise.nonLocalMeans(nan, rows: 2, columns: 4, channels: 3) }
        #expect(throws: DenoiseError.shape) { try Denoise.totalVariationChambolle([UInt8](repeating: 1, count: 5), rows: 2, columns: 3) }
    }
}
