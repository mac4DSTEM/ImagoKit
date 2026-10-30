// SPDX-License-Identifier: GPL-3.0-only
//
//  MaskedInverseFFT.swift (ImagoFFT)
//  An image rebuilt from chosen spots of its spectrum (Bragg filtering).
//
//  THE MASK: each spot is a centre (kx, ky) in frequency-index units from the spectrum's centre, a radius r and a
//  raised-cosine edge e, both in index units: m = 1 for d ≤ r, ½(1 + cos(π(d − r)/e)) for r < d < r + e, else 0, d the index
//  distance to the centre. Each spot's Friedel partner (−kx, −ky) is added; the union of spots is their maximum. The image
//  is the real part of `ifft2(ifftshift(fftshift(fft2(x)) · M))`, and the largest |imaginary part| is reported, not hidden.
//  The mask does not wrap: on an even size the Nyquist row and column have no partner inside the centred frame, so a spot
//  whose disc reaches them (or crosses the spectrum's edge) leaves a mask that is not symmetric, and the result is not real
//  (`imaginaryMax` says how far). Index units, not physical: r and e are stated on the product.
//

import Foundation
import ImagoCore

/// Why a masked inverse cannot be computed.
public nonisolated enum MaskedInverseFFTError: Error, Equatable, Sendable, LocalizedError {
    case fftSetup(width: Int, height: Int)

    public var errorDescription: String? {
        switch self {
        case .fftSetup(let w, let h): "The Fourier transform of a \(w) × \(h) image could not be set up."
        }
    }
}

public nonisolated struct MaskedInverseFFT: Sendable {
    public struct Spot: Sendable, Equatable, Hashable {
        /// The centre in frequency-index units from the spectrum's centre (column, row).
        public let kx: Double
        public let ky: Double
        /// The radius and the raised-cosine edge, in index units.
        public let radius: Double
        public let edge: Double

        public init(kx: Double, ky: Double, radius: Double, edge: Double) {
            self.kx = kx
            self.ky = ky
            self.radius = radius
            self.edge = edge
        }

        /// The mask's value at index distance `d` from this spot's centre.
        public func weight(atDistance d: Double) -> Double {
            if d <= radius { return 1 }
            if edge > 0, d < radius + edge { return 0.5 * (1 + cos(.pi * (d - radius) / edge)) }
            return 0
        }
    }

    public let width: Int
    public let height: Int
    /// The rebuilt image, row-major, x fastest, row 0 first, in the precision it was computed in.
    public let image: SpectrumValues
    /// The largest |imaginary part| left by the inverse transform: rounding for a mask symmetric under (k → −k); large when a
    /// spot reaches the Nyquist row or column of an even size or crosses the spectrum's edge (the mask does not wrap).
    public let imaginaryMax: Double
    public let spots: [Spot]
    public let precision: Precision

    /// `includeFriedelPartners: false` builds the mask from the given spots only (the result is then complex in general).
    public static func compute(_ image: Micrograph, spots: [Spot], includeFriedelPartners: Bool = true,
                               precision: Precision = .float32) throws -> MaskedInverseFFT {
        let w = image.width, h = image.height
        let all = includeFriedelPartners
            ? spots + spots.map { Spot(kx: -$0.kx, ky: -$0.ky, radius: $0.radius, edge: $0.edge) } : spots
        switch precision {
        case .float32:
            guard let fft = FFT2D(nx: w, ny: h) else { throw MaskedInverseFFTError.fftSetup(width: w, height: h) }
            var re = image.pixels
            var im = [Float](repeating: 0, count: w * h)
            fft.transform(re: &re, im: &im, forward: true)
            for u in 0..<h {
                let j = Double((u + h / 2) % h - h / 2)          // the centred index of unshifted row u (fftshift)
                for v in 0..<w {
                    let i = Double((v + w / 2) % w - w / 2)
                    let m = mask(i, j, all)
                    re[u * w + v] *= Float(m)
                    im[u * w + v] *= Float(m)
                }
            }
            fft.transform(re: &re, im: &im, forward: false)
            let imaginaryMax = im.map { Double(abs($0)) }.max() ?? 0
            return MaskedInverseFFT(width: w, height: h, image: .float32(re), imaginaryMax: imaginaryMax, spots: spots,
                                    precision: precision)
        case .float64:
            guard let fft = FFT2DDouble(nx: w, ny: h) else { throw MaskedInverseFFTError.fftSetup(width: w, height: h) }
            var re = image.pixels.map(Double.init)
            var im = [Double](repeating: 0, count: w * h)
            fft.transform(re: &re, im: &im, forward: true)
            for u in 0..<h {
                let j = Double((u + h / 2) % h - h / 2)
                for v in 0..<w {
                    let i = Double((v + w / 2) % w - w / 2)
                    let m = mask(i, j, all)
                    re[u * w + v] *= m
                    im[u * w + v] *= m
                }
            }
            fft.transform(re: &re, im: &im, forward: false)
            let imaginaryMax = im.map { abs($0) }.max() ?? 0
            return MaskedInverseFFT(width: w, height: h, image: .float64(re), imaginaryMax: imaginaryMax, spots: spots,
                                    precision: precision)
        }
    }

    /// The union (maximum) of the spots' weights at centred index (i, j).
    private static func mask(_ i: Double, _ j: Double, _ spots: [Spot]) -> Double {
        var m = 0.0
        for s in spots {
            let d = ((i - s.kx) * (i - s.kx) + (j - s.ky) * (j - s.ky)).squareRoot()
            m = max(m, s.weight(atDistance: d))
        }
        return m
    }
}

extension SpotMeasurement {
    /// This spot as a mask spot: its centroid minus the spectrum's centre (w/2, h/2), with `radius` and `edge` in bins.
    public func maskSpot(radius: Double, edge: Double) -> MaskedInverseFFT.Spot {
        MaskedInverseFFT.Spot(kx: centroidCol - Double(width / 2), ky: centroidRow - Double(height / 2),
                              radius: radius, edge: edge)
    }
}
