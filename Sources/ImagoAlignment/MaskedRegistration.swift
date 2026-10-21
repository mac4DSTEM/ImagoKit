//
//  MaskedRegistration.swift (ImagoAlignment)
//  scikit-image 0.26.0's masked phase_cross_correlation: Padfield's masked normalised cross-correlation
//  (`cross_correlate_masked`, mode "full") and the shift read from its maximum. Pixels outside a mask are set to zero and
//  do not count; translations whose overlap holds fewer than `overlapRatio` of the largest overlap are ignored.
//
//  THE STEPS, as scikit-image: arr1 = moving, arr2 = the flipped reference, both zero-padded to (M + R − 1) per axis (SciPy uses
//  the next fast length; with no wrap-around the correlation is the same up to rounding, so the kit uses the exact length);
//  numerator = ifft(F2 · F1) − ifft(F2m · F1) · ifft(F1m · F2) / n_overlap; denominators likewise from the squared images; the
//  correlation is numerator/denominator where denominator > 1000 · eps · max(denominator), clipped to [−1, 1], and 0 where the
//  overlap count (rounded, at least eps) is below the threshold. The shift is `−(mean(index of every maximum) − R + 1) + (M − R)/2`.
//  scikit-image returns the shift only (error and phase difference are NaN there); so does this.
//
//  EXACT TIES: scikit-image takes the mean of every index equal to the maximum. A correlation with exactly equal maxima
//  (a symmetric pair) decides on rounding noise, which differs between FFT lengths, so such a result is not reproducible.
//

import Foundation
import ImagoCore

extension PhaseCorrelation {
    /// The masked correlation surface, `(M.height + R.height − 1) × (M.width + R.width − 1)` row-major, as
    /// `cross_correlate_masked(moving, reference, movingMask, referenceMask, mode='full', axes=(0, 1))`.
    public static func maskedCorrelation(reference: Micrograph, moving: Micrograph, referenceMask: [Bool], movingMask: [Bool],
                                         overlapRatio: Double = 0.3, precision: Precision = .float32) throws -> (values: [Double], rows: Int, columns: Int) {
        switch precision {
        case .float32:
            let r = try masked(Float.self, reference, moving, referenceMask, movingMask, overlapRatio)
            return (r.values.map(Double.init), r.rows, r.columns)
        case .float64:
            return try masked(Double.self, reference, moving, referenceMask, movingMask, overlapRatio)
        }
    }

    /// `phase_cross_correlation(reference, moving, reference_mask=, moving_mask=, overlap_ratio=)`: the translation that registers
    /// `moving` onto `reference` (rows, columns), at whole pixels. `movingMask` defaults to `referenceMask` and then needs
    /// images of one size. Masks are true on valid pixels.
    public static func registerMasked(reference: Micrograph, moving: Micrograph, referenceMask: [Bool], movingMask: [Bool]? = nil,
                                      overlapRatio: Double = 0.3, precision: Precision = .float32) throws -> PixelShift {
        let mm: [Bool]
        if let movingMask { mm = movingMask } else {
            try checkSameSize(reference, moving)
            mm = referenceMask
        }
        let c = try maskedCorrelation(reference: reference, moving: moving, referenceMask: referenceMask, movingMask: mm,
                                      overlapRatio: overlapRatio, precision: precision)
        let top = c.values.max()!
        var sumR = 0.0, sumC = 0.0, count = 0.0
        for r in 0..<c.rows { for k in 0..<c.columns where c.values[r * c.columns + k] == top {
            sumR += Double(r); sumC += Double(k); count += 1
        } }
        let centerR = sumR / count - Double(reference.height) + 1
        let centerC = sumC / count - Double(reference.width) + 1
        return PixelShift(row: -centerR + Double(moving.height - reference.height) / 2,
                          column: -centerC + Double(moving.width - reference.width) / 2)
    }

    private static func checkSameSize(_ reference: Micrograph, _ moving: Micrograph) throws {
        guard reference.width == moving.width, reference.height == moving.height else {
            throw AlignmentError.sizeMismatch(index: 1, width: moving.width, height: moving.height,
                                              expectedWidth: reference.width, expectedHeight: reference.height)
        }
    }

    private static func masked<S: FFTSurfaceScalar>(_ type: S.Type, _ reference: Micrograph, _ moving: Micrograph,
                                                    _ referenceMask: [Bool], _ movingMask: [Bool], _ overlapRatio: Double) throws
        -> (values: [S], rows: Int, columns: Int) {
        guard overlapRatio >= 0, overlapRatio <= 1 else { throw AlignmentError.overlapRatio(overlapRatio) }
        guard referenceMask.contains(true), movingMask.contains(true) else { throw AlignmentError.emptyMask }
        guard referenceMask.count == reference.pixels.count else {
            throw AlignmentError.maskCount(expected: reference.pixels.count, actual: referenceMask.count)
        }
        guard movingMask.count == moving.pixels.count else {
            throw AlignmentError.maskCount(expected: moving.pixels.count, actual: movingMask.count)
        }
        let eps = S.ulpOfOne
        // arr1 = moving ("fixed" in scikit-image), arr2 = reference (flipped).
        let mh = moving.height, mw = moving.width, rh = reference.height, rw = reference.width
        let H = mh + rh - 1, W = mw + rw - 1
        func plane(_ values: (Int, Int) -> S, _ rows: Int, _ columns: Int, flipped: Bool = false) -> ComplexImage<S> {
            var re = [S](repeating: 0, count: H * W)
            for r in 0..<rows { for c in 0..<columns {
                re[r * W + c] = flipped ? values(rows - 1 - r, columns - 1 - c) : values(r, c)
            } }
            return ComplexImage(rows: H, columns: W, real: re)
        }
        func image(_ m: Micrograph, _ mask: [Bool]) -> (Int, Int) -> S {
            { r, c in mask[r * m.width + c] ? S(m.pixels[r * m.width + c]) : 0 }
        }
        let fixedValues = image(moving, movingMask), rotValues = image(reference, referenceMask)
        // A masked-in pixel must be finite; masked-out ones are zeroed first, as scikit-image does.
        for r in 0..<mh { for c in 0..<mw where movingMask[r * mw + c] && !moving.pixels[r * mw + c].isFinite {
            throw AlignmentError.nonFinitePixel(index: 1)
        } }
        for r in 0..<rh { for c in 0..<rw where referenceMask[r * rw + c] && !reference.pixels[r * rw + c].isFinite {
            throw AlignmentError.nonFinitePixel(index: 0)
        } }
        let fixedFFT = try S.fft2(plane(fixedValues, mh, mw), axes: .both, norm: .backward)
        let rotFFT = try S.fft2(plane(rotValues, rh, rw, flipped: true), axes: .both, norm: .backward)
        let fixedMaskFFT = try S.fft2(plane({ movingMask[$0 * mw + $1] ? 1 : 0 }, mh, mw), axes: .both, norm: .backward)
        let rotMaskFFT = try S.fft2(plane({ referenceMask[$0 * rw + $1] ? 1 : 0 }, rh, rw, flipped: true), axes: .both, norm: .backward)
        let fixedSqFFT = try S.fft2(plane({ let v = fixedValues($0, $1); return v * v }, mh, mw), axes: .both, norm: .backward)
        let rotSqFFT = try S.fft2(plane({ let v = rotValues($0, $1); return v * v }, rh, rw, flipped: true), axes: .both, norm: .backward)

        func product(_ a: ComplexImage<S>, _ b: ComplexImage<S>) throws -> [S] {
            var re = [S](repeating: 0, count: H * W), im = re
            for i in 0..<(H * W) {
                re[i] = a.re[i] * b.re[i] - a.im[i] * b.im[i]
                im[i] = a.re[i] * b.im[i] + a.im[i] * b.re[i]
            }
            return try S.ifft2(ComplexImage(rows: H, columns: W, re: re, im: im), axes: .both, norm: .backward).re
        }
        var overlap = try product(rotMaskFFT, fixedMaskFFT).map { max($0.rounded(.toNearestOrEven), eps) }
        let correlatedFixed = try product(rotMaskFFT, fixedFFT)
        let correlatedRot = try product(fixedMaskFFT, rotFFT)
        var numerator = try product(rotFFT, fixedFFT)
        var fixedDenom = try product(rotMaskFFT, fixedSqFFT)
        var movingDenom = try product(fixedMaskFFT, rotSqFFT)
        var denom = [S](repeating: 0, count: H * W)
        for i in 0..<(H * W) {
            numerator[i] -= correlatedFixed[i] * correlatedRot[i] / overlap[i]
            fixedDenom[i] = max(fixedDenom[i] - correlatedFixed[i] * correlatedFixed[i] / overlap[i], 0)
            movingDenom[i] = max(movingDenom[i] - correlatedRot[i] * correlatedRot[i] / overlap[i], 0)
            denom[i] = (fixedDenom[i] * movingDenom[i]).squareRoot()
        }
        let tol = 1000 * eps * (denom.map { abs($0) }.max() ?? 0)
        let threshold = S(overlapRatio) * (overlap.max() ?? 0)
        var out = [S](repeating: 0, count: H * W)
        for i in 0..<(H * W) where denom[i] > tol {
            out[i] = min(max(numerator[i] / denom[i], -1), 1)
        }
        for i in 0..<(H * W) where overlap[i] < threshold { out[i] = 0 }
        overlap.removeAll()
        return (out, H, W)
    }
}
