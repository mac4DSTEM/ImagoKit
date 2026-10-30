// SPDX-License-Identifier: GPL-3.0-only
//
//  AlignedSum.swift (ImagoAlignment)
//  A stack summed after each frame is shifted by its shift: the Fourier shift (exact for
//  a periodic image; edges wrap) or the bilinear one (zero outside the footprint; the overlap counted).
//

import Foundation
import ImagoCore

/// The sum of a shifted stack, the number of frames behind each pixel, and the mean.
public nonisolated struct AlignedStack: Sendable, Equatable {
    /// Σ over the frames of the shifted frame, in the precision it was computed in and rounded to Float32 once.
    public let sum: Micrograph
    /// `sum / counts` (0 where the count is 0).
    public let mean: Micrograph
    /// The frames that cover each pixel, row-major: the frame count everywhere for `.fourier`; for `.bilinear` the frames
    /// whose source coordinate for that pixel lies in [0, n − 1] on both axes.
    public let counts: [Int]
    public let frameCount: Int
    public let method: AlignmentMethod
    public let precision: Precision
    /// For `.fourier`: the largest |imaginary part| any frame's inverse transform left (see `ImageShift.fourierShift`);
    /// 0 for `.bilinear`.
    public let largestImaginaryPart: Double
}

public nonisolated enum AlignedSum {
    /// Each frame shifted by its shift (the shift to apply, as a registration reports it) and summed. The frames must have
    /// one size, hold finite pixels for `.fourier`, and come with one finite shift each. The calibration is the first
    /// frame's.
    public static func sum(_ frames: [Micrograph], shifts: [PixelShift], method: AlignmentMethod = .fourier,
                           precision: Precision = .float32) throws -> AlignedStack {
        guard let first = frames.first else { throw AlignmentError.emptyStack }
        guard shifts.count == frames.count else {
            throw AlignmentError.shiftCount(expected: frames.count, actual: shifts.count)
        }
        let w = first.width, h = first.height
        for (k, frame) in frames.enumerated() {
            guard frame.width == w, frame.height == h else {
                throw AlignmentError.sizeMismatch(index: k, width: frame.width, height: frame.height,
                                                  expectedWidth: w, expectedHeight: h)
            }
            try ImageShift.checkShift(shifts[k], index: k)
            if method == .fourier, !frame.pixels.allSatisfy(\.isFinite) { throw AlignmentError.nonFinitePixel(index: k) }
        }
        let sums: [Float], means: [Float], counts: [Int], imaginary: Double
        switch precision {
        case .float32:
            var engine: FFT2D?
            if method == .fourier {
                guard let fft = FFT2D(nx: w, ny: h) else { throw AlignmentError.fftSetup(width: w, height: h) }
                engine = fft
            }
            (sums, means, counts, imaginary) = accumulate(frames.map(\.pixels), shifts, w, h, method) { pixels, shift in
                ImageShift.fourierShifted(pixels, w, h, shift, engine!)
            }
        case .float64:
            var engine: FFT2DDouble?
            if method == .fourier {
                guard let fft = FFT2DDouble(nx: w, ny: h) else { throw AlignmentError.fftSetup(width: w, height: h) }
                engine = fft
            }
            (sums, means, counts, imaginary) = accumulate(frames.map { $0.pixels.map(Double.init) }, shifts, w, h,
                                                          method) { pixels, shift in
                ImageShift.fourierShifted(pixels, w, h, shift, engine!)
            }
        }
        return AlignedStack(sum: try Micrograph(width: w, height: h, pixels: sums, x: first.x, y: first.y),
                            mean: try Micrograph(width: w, height: h, pixels: means, x: first.x, y: first.y),
                            counts: counts, frameCount: frames.count, method: method, precision: precision,
                            largestImaginaryPart: imaginary)
    }

    private static func accumulate<S: BinaryFloatingPoint & Sendable>(
        _ frames: [[S]], _ shifts: [PixelShift], _ w: Int, _ h: Int, _ method: AlignmentMethod,
        fourier: ([S], PixelShift) -> ([S], Double)
    ) -> ([Float], [Float], [Int], Double) {
        var total = [S](repeating: 0, count: w * h)
        var counts = [Int](repeating: 0, count: w * h)
        var imaginary = 0.0
        switch method {
        case .fourier:
            for (k, frame) in frames.enumerated() {
                let (shifted, imag) = fourier(frame, shifts[k])
                imaginary = max(imaginary, imag)
                for i in 0..<total.count { total[i] += shifted[i] }
            }
            counts = [Int](repeating: frames.count, count: w * h)
        case .bilinear:
            for (k, frame) in frames.enumerated() {
                let rows = ImageShift.axisSamples(h, shifts[k].row, .linear, .constant)
                let columns = ImageShift.axisSamples(w, shifts[k].column, .linear, .constant)
                let shifted = ImageShift.shifted(frame, w, h, shifts[k], .linear, .constant, 0)
                for r in 0..<h where !rows[r].constant {
                    for c in 0..<w where !columns[c].constant {
                        counts[r * w + c] += 1
                    }
                }
                for i in 0..<total.count { total[i] += shifted[i] }
            }
        }
        var mean = [Float](repeating: 0, count: w * h)
        for i in 0..<total.count where counts[i] > 0 { mean[i] = Float(total[i] / S(counts[i])) }
        return (total.map { Float($0) }, mean, counts, imaginary)
    }
}
