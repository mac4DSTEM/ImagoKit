//
//  AlignmentTypes.swift (ImagoAlignment)
//  The vocabulary of drift and stack alignment: a shift in pixels, the registration of one image onto another, the
//  options of the shift and the sum, and why a call is refused.
//
//  CONVENTION, scikit-image's: a shift is the translation to APPLY to the moving image so that it registers onto the
//  reference (`ImageShift.shift(moving, by: shift)`). Equivalently the moving image is the reference displaced by the
//  negated shift. Row first, then column; in pixels; positive row is toward larger row index (downward on the screen).
//

import Foundation
import ImagoCore

/// A translation in pixels: `row` along the vertical axis (y), `column` along the horizontal axis (x).
public nonisolated struct PixelShift: Sendable, Equatable, Hashable {
    public var row: Double
    public var column: Double

    public init(row: Double, column: Double) {
        self.row = row
        self.column = column
    }

    /// No translation.
    public static let zero = PixelShift(row: 0, column: 0)
}

/// How the cross-power spectrum is normalised before it is inverted. scikit-image's `normalization`.
public nonisolated enum Normalization: String, Sendable, Equatable, Hashable, CaseIterable {
    /// Phase correlation: every frequency has unit weight (`P / max(|P|, 100·eps)`, eps the precision's machine epsilon).
    /// Robust to a change of contrast, not to noise at frequencies without signal.
    case phase
    /// Plain cross-correlation (scikit-image's `normalization=None`).
    case unnormalized
}

/// The registration of one image onto another.
public nonisolated struct ImageRegistration: Sendable, Equatable {
    /// The translation to apply to the moving image so that it registers onto the reference, on the 1/`upsampleFactor`
    /// pixel grid; an axis of length 1 has shift 0.
    public let shift: PixelShift
    /// scikit-image's translation-invariant normalised RMS error, `√|1 − |CCmax|² / (src_amp · target_amp)|`. It is 0 for a
    /// copy of the reference up to a scale (`.unnormalized`), and near 1 with `.phase` (the correlation is then at most 1
    /// against the images' amplitudes). Where the images match closely it is ill-conditioned: the square root of a
    /// difference near 0, so its rounding noise is large (Float32: about 3e-4 at best); compare its square. Both branches of
    /// the refinement give the same value for one pair (the refined correlation and the amplitudes are both unnormalised).
    /// NaN when either image is identically zero (scikit-image warns and returns the same).
    public let error: Double
    /// The global phase difference, `atan2(Im CCmax, Re CCmax)` in radians (zero for images that are not negative).
    public let phaseDifference: Double
    public let upsampleFactor: Int
    public let normalization: Normalization
    public let precision: Precision

    public init(shift: PixelShift, error: Double, phaseDifference: Double, upsampleFactor: Int,
                normalization: Normalization, precision: Precision) {
        self.shift = shift
        self.error = error
        self.phaseDifference = phaseDifference
        self.upsampleFactor = upsampleFactor
        self.normalization = normalization
        self.precision = precision
    }
}

/// The interpolation order of a shift (SciPy's `order` 0 to 5; 2 to 5 are B-splines on the prefiltered coefficients).
public nonisolated enum ShiftOrder: Int, Sendable, Equatable, Hashable, CaseIterable {
    /// The nearest input pixel (the coordinate rounded half up after the mode's mapping).
    case nearest = 0
    /// Bilinear on the four neighbours.
    case linear = 1
    /// B-spline of degree 2 to 5 (SciPy's prefilter, `prefilter=True`); modes `gridConstant`, `nearest`, `reflect`, `mirror`
    /// and `gridWrap` only (`AlignmentError.splineMode` for `constant` and `wrap`).
    case quadratic = 2, cubic = 3, quartic = 4, quintic = 5
}

/// How a shift fills what lies outside the image: SciPy 1.18.1's modes, with their names.
public nonisolated enum ShiftMode: String, Sendable, Equatable, Hashable, CaseIterable {
    /// `cval` for every output pixel whose source coordinate lies outside [0, n − 1]; no interpolation toward `cval`.
    case constant
    /// The nearest edge pixel.
    case nearest
    /// SciPy's "wrap": the input repeats with period n − 1, so the last and the first pixel coincide (SciPy documents that
    /// which of the two is taken is not defined; the kit takes what SciPy takes).
    case wrap
    /// Half-sample symmetric: d c b a | a b c d | d c b a.
    case reflect
    /// Whole-sample symmetric: d c b | a b c d | c b a.
    case mirror
    /// `cval` beyond the edge, interpolated toward from the edge pixels (SciPy's "grid-constant").
    case gridConstant = "grid-constant"
    /// The input repeats with period n (SciPy's "grid-wrap").
    case gridWrap = "grid-wrap"
}

/// How `AlignedSum` shifts each frame.
public nonisolated enum AlignmentMethod: String, Sendable, Equatable, Hashable, CaseIterable {
    /// The exact shift of a periodic image (`ImageShift.fourierShift`); edges wrap, every pixel counts every frame.
    case fourier
    /// Bilinear (`ShiftOrder.linear`, `ShiftMode.constant`, `cval` 0); a pixel counts the frames whose shifted footprint
    /// covers it.
    case bilinear
}

/// What a drift series is registered against.
public nonisolated enum DriftReference: String, Sendable, Equatable, Hashable, CaseIterable {
    /// Frame 0.
    case first
    /// The mean of the unshifted frames.
    case mean
    /// The previous frame; the shifts are cumulated.
    case previous
}

/// Why an alignment call is refused.
public nonisolated enum AlignmentError: Error, Equatable, Sendable, LocalizedError {
    /// A stack with no frame.
    case emptyStack
    /// Image `index` (0 reference, 1 moving for a pair; the frame index in a stack) is not `expectedWidth × expectedHeight`.
    case sizeMismatch(index: Int, width: Int, height: Int, expectedWidth: Int, expectedHeight: Int)
    /// Image `index` holds a NaN or an infinity (a registration cannot be made from it).
    case nonFinitePixel(index: Int)
    /// The upsample factor is outside 1 ... `PhaseCorrelation.maximumUpsampleFactor`.
    case upsampleFactor(Int)
    /// A shift is NaN, infinite or larger than `ImageShift.maximumShift` in magnitude. `index` is the frame's (0 for a
    /// single shift).
    case invalidShift(index: Int)
    /// `shifts.count` differs from `frames.count`.
    case shiftCount(expected: Int, actual: Int)
    /// The Fourier transform of a `width × height` image could not be set up.
    case fftSetup(width: Int, height: Int)
    /// A spline order (2 to 5) was asked with the `constant` or `wrap` mode, whose spline coordinate rule is not ported.
    case splineMode(ShiftMode)
    /// A zoom factor or angle that is not finite, or an output size outside what a call produces.
    case invalidResampling(String)
    /// A mask holds `actual` values for an image of `expected` pixels.
    case maskCount(expected: Int, actual: Int)
    /// A mask is true nowhere: there is nothing to correlate.
    case emptyMask
    /// `overlapRatio` is outside [0, 1].
    case overlapRatio(Double)

    public var errorDescription: String? {
        switch self {
        case .emptyStack:
            "The stack has no frame."
        case .sizeMismatch(let index, let w, let h, let ew, let eh):
            "Image \(index) is \(w) × \(h) pixels; the images must all be \(ew) × \(eh)."
        case .nonFinitePixel(let index):
            "Image \(index) contains a pixel that is not a finite number."
        case .upsampleFactor(let u):
            "The upsample factor \(u) must be a whole number from 1 to \(PhaseCorrelation.maximumUpsampleFactor)."
        case .invalidShift(let index):
            "The shift of frame \(index) is not a finite number within ±\(Int(ImageShift.maximumShift)) pixels."
        case .shiftCount(let expected, let actual):
            "\(actual) shifts were given for \(expected) frames."
        case .fftSetup(let w, let h):
            "The Fourier transform of a \(w) × \(h) image could not be set up."
        case .splineMode(let mode):
            "The \(mode.rawValue) mode is not available at spline orders 2 to 5; use grid-constant, nearest, reflect, mirror or grid-wrap."
        case .invalidResampling(let why):
            "The resampling cannot be done: \(why)."
        case .emptyMask:
            "A mask has no valid pixel."
        case .overlapRatio(let r):
            "The overlap ratio \(r) must lie between 0 and 1."
        case .maskCount(let expected, let actual):
            "A mask holds \(actual) values for an image of \(expected) pixels."
        }
    }
}
