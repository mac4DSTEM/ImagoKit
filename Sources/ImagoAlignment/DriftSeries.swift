//
//  DriftSeries.swift (ImagoAlignment)
//  The drift of a stack of frames: every frame registered against the first frame, the mean of the unshifted frames, or its
//  predecessor (registration-slice-2.md section 1).
//
//  `.first` and `.previous` give frame 0 the shift (0, 0) (not computed: frame 0 is the origin). `.previous` cumulates: the
//  shift of frame k is s_k = s_(k−1) + r_k, r_k frame k registered onto frame k − 1, added row and column in that order from
//  s_0 = 0, so the series is exactly the cumulated pairwise shifts; each result's `error` and `phaseDifference` are the
//  pair's. `.mean` registers every frame, frame 0 included, onto the mean image (a drift smears the mean, which biases a
//  large drift's reference toward the average position: the shifts are relative to that average).
//

import Foundation
import ImagoCore

public nonisolated enum DriftSeries {
    /// One `ImageRegistration` per frame, in the frames' order. The frames must have one size and hold finite pixels.
    public static func estimate(_ frames: [Micrograph], reference: DriftReference = .first, upsampleFactor: Int = 1,
                                normalization: Normalization = .phase,
                                precision: Precision = .float32) throws -> [ImageRegistration] {
        try PhaseCorrelation.checkUpsample(upsampleFactor)
        guard let first = frames.first else { throw AlignmentError.emptyStack }
        let w = first.width, h = first.height
        for (k, frame) in frames.enumerated() {
            guard frame.width == w, frame.height == h else {
                throw AlignmentError.sizeMismatch(index: k, width: frame.width, height: frame.height,
                                                  expectedWidth: w, expectedHeight: h)
            }
            guard frame.pixels.allSatisfy(\.isFinite) else { throw AlignmentError.nonFinitePixel(index: k) }
        }
        switch precision {
        case .float32:
            guard let fft = FFT2D(nx: w, ny: h) else { throw AlignmentError.fftSetup(width: w, height: h) }
            return estimate(fft, frames.map(\.pixels), w, h, reference, upsampleFactor, normalization, precision)
        case .float64:
            guard let fft = FFT2DDouble(nx: w, ny: h) else { throw AlignmentError.fftSetup(width: w, height: h) }
            return estimate(fft, frames.map { $0.pixels.map(Double.init) }, w, h, reference, upsampleFactor,
                            normalization, precision)
        }
    }

    private static func estimate<T: FFT2DTransform>(_ fft: T, _ frames: [[T.Scalar]], _ w: Int, _ h: Int,
                                                     _ reference: DriftReference, _ u: Int, _ normalization: Normalization,
                                                     _ precision: Precision) -> [ImageRegistration] {
        let spectra = frames.map { Spectrum.of($0, fft) }
        let identity = ImageRegistration(shift: .zero, error: 0, phaseDifference: 0, upsampleFactor: u,
                                         normalization: normalization, precision: precision)
        var out = [ImageRegistration]()
        out.reserveCapacity(frames.count)
        switch reference {
        case .first:
            out.append(identity)
            for k in 1..<max(frames.count, 1) {
                out.append(PhaseCorrelation.correlate(fft, spectra[0], spectra[k], w, h, u, normalization, precision))
            }
        case .mean:
            let mean = Spectrum.of(meanImage(frames), fft)
            for k in 0..<frames.count {
                out.append(PhaseCorrelation.correlate(fft, mean, spectra[k], w, h, u, normalization, precision))
            }
        case .previous:
            let pairs = (1..<max(frames.count, 1)).map {
                PhaseCorrelation.correlate(fft, spectra[$0 - 1], spectra[$0], w, h, u, normalization, precision)
            }
            out.append(identity)
            out.append(contentsOf: cumulated(pairs, identity))
        }
        return out
    }

    /// The cumulated pairwise registrations of a `.previous` series: s_k = s_(k−1) + r_k from s_0 = 0, row then column; each
    /// keeps its pair's error and phase difference. `pairs[k − 1]` is frame k against frame k − 1.
    package static func cumulated(_ pairs: [ImageRegistration], _ origin: ImageRegistration) -> [ImageRegistration] {
        var out = [ImageRegistration]()
        var row = origin.shift.row, column = origin.shift.column
        for p in pairs {
            row += p.shift.row
            column += p.shift.column
            out.append(ImageRegistration(shift: PixelShift(row: row, column: column), error: p.error,
                                         phaseDifference: p.phaseDifference, upsampleFactor: p.upsampleFactor,
                                         normalization: p.normalization, precision: p.precision))
        }
        return out
    }

    /// The mean of the frames, accumulated in Double and rounded once to the scalar.
    private static func meanImage<S: BinaryFloatingPoint>(_ frames: [[S]]) -> [S] {
        var sum = [Double](repeating: 0, count: frames[0].count)
        for f in frames {
            for i in 0..<sum.count { sum[i] += Double(f[i]) }
        }
        let n = Double(frames.count)
        return sum.map { S($0 / n) }
    }
}
