//
//  HRTEMFilter.swift (ImagoFilters)
//  The HRTEM filters — the average background subtraction filter (ABSF) and the Wiener filter after R. Kilaas (1998), as
//  D. R. G. Mitchell's DigitalMicrograph script `HRTEM_Filter.s` v4.0 describes them (registration-slice-1.md 1.3). A
//  functional re-implementation of the steps below: the script states no licence and nothing of it is copied (NOTICE).
//  Never compared with DigitalMicrograph.
//
//  THE STEPS, for a square, even n × n image I; r is a pixel's distance from (n/2, n/2), the DC bin of the centred
//  transform; every spectrum is the magnitude of the fftshifted forward FFT (NumPy's convention), every mask in that frame.
//    A1  E = 1/(1 + 0.414 (r/0.4n)²⁴); M = |FFT(I·E)|; S = the 3 × 3 median of M (edges replicated).
//    A2  t = 1; while d < delta and passes < cycles: A = rot(M); c = max((S − A)/S · 100, 0); the flags gain every pixel
//        with c > 100 − t and r < n/2; M takes A on every flagged pixel; d = 100 · flagged / n²; t += step. S is never
//        recomputed. rot(X): X sampled bilinearly at (n/2 + ρ cos θ, n/2 + ρ sin θ) (row, column) for 512 angles and
//        ρ = 0 … n/2 − 1, averaged over the angles into a profile P(ρ); rot(X) at a pixel is P linearly interpolated at r.
//    A3  M = rot(M); W = |FFT(I)| (the unwindowed image).
//    A4  a = max((W − M)/W, 0); R = round(n/2 · pct/100), half away from zero; where r < R, a = (1 − r/R) · 0.1 + 0.05;
//        w = max(a (W + M)/W, 0) — from the already clamped, tapered a.
//    A5  with smoothing, a and w × 1/(1 + 0.414 (r / (Ro · n/2))^(2 · order)).
//    A6  ABSF = real(IFFT(FFT(I) · a)), Wiener = real(IFFT(FFT(I) · w)).
//  Guards the script leaves to DigitalMicrograph: W = 0 gives a mask of 0 and S = 0 is never flagged; P beyond ρ = n/2 − 1
//  is held at its last value; a bilinear sample clamps to the image; odd and non-square sizes are refused.
//
//  THE DC: rot(M) at r = 0 samples only the centre, so the DC bin is never replaced and a(0) = 1 − |Σ I·E| / |Σ I|: the
//  ABSF output's mean is the input's times that, never zero (≈ 0.45 at the defaults). That is the method's behaviour, kept.
//
//  PRECISION: on the Float32 path the transforms are Float32 (FFT2D) on the mean-removed image, the DC carried in Double;
//  every spectrum, mask and the loop are Double. The Float64 path runs the same algebra through FFT2DDouble.
//

import Foundation
import ImagoCore
import ImagoFFT

/// Why the filters cannot run.
public nonisolated enum HRTEMFilterError: Error, Equatable, Sendable, LocalizedError {
    case notSquare(width: Int, height: Int)
    case oddSize(Int)
    case tooSmall(Int)
    case pixelCount(expected: Int, actual: Int)
    case nonFinitePixel
    case parameterOutOfRange(String)
    case fftSetup(Int)

    public var errorDescription: String? {
        switch self {
        case .notSquare(let w, let h):
            return "The HRTEM filters need a square image; this one is \(w) × \(h). Filter a centred square crop."
        case .oddSize(let n): return "The HRTEM filters need an even size; this image is \(n) × \(n)."
        case .tooSmall(let n): return "The HRTEM filters need at least 32 × 32 pixels; this image is \(n) × \(n)."
        case .pixelCount(let expected, let actual): return "Cannot filter: \(actual) pixels for an image of \(expected)."
        case .nonFinitePixel: return "The image holds a pixel that is not a finite number, so it cannot be filtered."
        case .parameterOutOfRange(let name): return "The filter's \(name) is outside the range the filter allows."
        case .fftSetup(let n): return "The Fourier transform of a \(n) × \(n) image could not be set up."
        }
    }
}

public nonisolated enum HRTEMFilter {

    /// The method's parameters with the script's defaults; out of range is refused, never clamped.
    public struct Parameters: Sendable, Equatable, Hashable {
        /// Stop once this percentage of the n² pixels is flagged, [0, 50].
        public var delta: Double = 5
        /// Threshold step per pass, percentage points, [1, 10].
        public var step: Double = 2
        /// The most passes, [1, 99].
        public var cycles: Int = 99
        /// The Butterworth low-pass on both masks.
        public var smoothing: Bool = true
        /// Butterworth order, [1, 20], and its half-point radius as a fraction of n/2, [0.01, 3].
        public var order: Int = 4
        public var ro: Double = 0.3
        /// The low-frequency taper's radius, percent of n/2, [0, 20].
        public var lowFrequencyPercent: Double = 0

        public init() {}
        public static let defaults = Parameters()

        /// The first parameter outside its range, by name.
        public var outOfRange: String? {
            if !(delta.isFinite && (0...50).contains(delta)) { return "delta" }
            if !(step.isFinite && (1...10).contains(step)) { return "step" }
            if !(1...99).contains(cycles) { return "cycles" }
            if !(1...20).contains(order) { return "order" }
            if !(ro.isFinite && (0.01...3).contains(ro)) { return "ro" }
            if !(lowFrequencyPercent.isFinite && (0...20).contains(lowFrequencyPercent)) { return "lowFrequencyPercent" }
            return nil
        }
    }

    /// Everything a filter run produced. Arrays are n × n, row-major; spectra and masks in the centred frame.
    public struct Result: Sendable {
        public let n: Int
        public let absf: SpectrumValues
        public let wiener: SpectrumValues
        /// A1: M before the loop, and its 3 × 3 median.
        public let initialBackground: [Double]
        public let median: [Double]
        /// Pass 1's c (zero outside the disk and where S = 0); empty if no pass ran.
        public let firstPassC: [Double]
        /// A3: the amorphous background M and the unwindowed spectrum W.
        public let background: [Double]
        public let spectrum: [Double]
        public let absfMask: [Double]
        public let wienerMask: [Double]
        public let flagged: [Bool]
        /// The pass (1-based) at which each pixel was first flagged; 0 where it never was.
        public let firstFlaggedPass: [Int]
        public let passes: Int
        public let flagsPerPass: [Int]
        public let finalDelta: Double
        public let parameters: Parameters
        public let precision: Precision
    }

    // MARK: Entry points

    /// Both filters of `image` (square, even, at least 32 pixels a side).
    public static func filter(_ image: Micrograph, parameters: Parameters = .defaults,
                              precision: Precision = .float32) throws -> Result {
        try filter(pixels: image.pixels, width: image.width, height: image.height, parameters: parameters, precision: precision)
    }

    /// The two outputs as images with `image`'s calibration (a Float64 result is rounded to Float32 pixels).
    public static func outputs(_ result: Result, like image: Micrograph) throws -> (absf: Micrograph, wiener: Micrograph) {
        func pixels(_ v: SpectrumValues) -> [Float] {
            switch v {
            case .float32(let p): p
            case .float64(let p): p.map { Float($0) }
            }
        }
        return (try Micrograph(width: result.n, height: result.n, pixels: pixels(result.absf), x: image.x, y: image.y),
                try Micrograph(width: result.n, height: result.n, pixels: pixels(result.wiener), x: image.x, y: image.y))
    }

    /// Both filters on n × n Float32 pixels. CPU-bound (several transforms and rotational averages): run it off the main
    /// actor.
    public static func filter(pixels: [Float], width: Int, height: Int, parameters: Parameters = .defaults,
                              precision: Precision = .float32) throws -> Result {
        guard width == height else { throw HRTEMFilterError.notSquare(width: width, height: height) }
        let n = width
        guard n % 2 == 0 else { throw HRTEMFilterError.oddSize(n) }
        guard n >= 32 else { throw HRTEMFilterError.tooSmall(n) }
        guard pixels.count == n * n else { throw HRTEMFilterError.pixelCount(expected: n * n, actual: pixels.count) }
        guard pixels.allSatisfy(\.isFinite) else { throw HRTEMFilterError.nonFinitePixel }
        if let name = parameters.outOfRange { throw HRTEMFilterError.parameterOutOfRange(name) }
        switch precision {
        case .float32:
            guard let fft = FFT2D(nx: n, ny: n) else { throw HRTEMFilterError.fftSetup(n) }
            return run(pixels: pixels, n: n, parameters: parameters, precision: precision, fft: fft)
        case .float64:
            guard let fft = FFT2DDouble(nx: n, ny: n) else { throw HRTEMFilterError.fftSetup(n) }
            return run(pixels: pixels, n: n, parameters: parameters, precision: precision, fft: fft)
        }
    }

    private static func run<F: FFT2DTransform>(pixels: [Float], n: Int, parameters: Parameters, precision: Precision,
                                               fft: F) -> Result {
        let count = n * n
        let half = n / 2
        var radius = [Double](repeating: 0, count: count)
        for row in 0..<n {
            for col in 0..<n {
                let dr = Double(row - half), dc = Double(col - half)
                radius[row * n + col] = (dr * dr + dc * dc).squareRoot()
            }
        }

        // A1. The windowed image is rounded to the transform's precision (Float32 on the fast path, exactly as the method's
        // Float32 reading does; kept in Double on the reference path) and carried as Double values.
        var windowed = [Double](repeating: 0, count: count)
        for i in 0..<count {
            let v = Double(pixels[i]) * edgeWindow(radius: radius[i], n: n)
            windowed[i] = F.Scalar.self == Float.self ? Double(Float(v)) : v
        }
        var m = centredMagnitude(windowed, n: n, fft: fft)
        let initialBackground = m
        let s = median3x3(m, n: n)

        // A2
        var flagged = [Bool](repeating: false, count: count)
        var firstFlaggedPass = [Int](repeating: 0, count: count)
        var flaggedCount = 0
        var threshold = 1.0
        var passes = 0
        var d = 0.0
        var flagsPerPass: [Int] = []
        var firstPassC: [Double] = []
        while d < parameters.delta && passes < parameters.cycles {
            let a = rotationalAverage(m, n: n)
            var c = passes == 0 ? [Double](repeating: 0, count: count) : []
            for i in 0..<count where radius[i] < Double(half) && s[i] > 0 {
                let ci = max((s[i] - a[i]) / s[i] * 100, 0)
                if passes == 0 { c[i] = ci }
                if !flagged[i], ci > 100 - threshold { flagged[i] = true; firstFlaggedPass[i] = passes + 1; flaggedCount += 1 }
            }
            for i in 0..<count where flagged[i] { m[i] = a[i] }
            if passes == 0 { firstPassC = c }
            d = 100 * Double(flaggedCount) / Double(count)
            passes += 1
            flagsPerPass.append(flaggedCount)
            threshold += parameters.step
        }

        // A3
        m = rotationalAverage(m, n: n)
        let w = centredMagnitude(pixels.map(Double.init), n: n, fft: fft)

        // A4, A5
        let taperRadius = (Double(half) * parameters.lowFrequencyPercent / 100).rounded()
        var absfMask = [Double](repeating: 0, count: count)
        var wienerMask = [Double](repeating: 0, count: count)
        for i in 0..<count {
            var a = w[i] > 0 ? max((w[i] - m[i]) / w[i], 0) : 0
            if radius[i] < taperRadius { a = lowFrequencyTaper(radius: radius[i], taperRadius: taperRadius) }
            var wiener = w[i] > 0 ? max(a * (w[i] + m[i]) / w[i], 0) : 0
            if parameters.smoothing {
                let lowPass = butterworth(radius: radius[i], ro: parameters.ro, order: parameters.order, n: n)
                a *= lowPass
                wiener *= lowPass
            }
            absfMask[i] = a
            wienerMask[i] = wiener
        }

        // A6
        let absf = applyMask(absfMask, to: pixels, n: n, fft: fft)
        let wiener = applyMask(wienerMask, to: pixels, n: n, fft: fft)
        return Result(n: n, absf: absf, wiener: wiener, initialBackground: initialBackground, median: s,
                      firstPassC: firstPassC, background: m, spectrum: w, absfMask: absfMask, wienerMask: wienerMask,
                      flagged: flagged, firstFlaggedPass: firstFlaggedPass, passes: passes, flagsPerPass: flagsPerPass,
                      finalDelta: d, parameters: parameters, precision: precision)
    }

    // MARK: The steps, public for tests and for callers who build on them

    /// A1's edge window at a pixel's radius r: 1/(1 + 0.414 (r/0.4n)²⁴).
    public static func edgeWindow(radius: Double, n: Int) -> Double {
        1 / (1 + 0.414 * pow(radius / (0.4 * Double(n)), 24))
    }

    /// A4's low-frequency taper inside r < R: (1 − r/R) · 0.1 + 0.05.
    public static func lowFrequencyTaper(radius: Double, taperRadius: Double) -> Double {
        (1 - radius / taperRadius) * 0.1 + 0.05
    }

    /// A5's Butterworth: 1/(1 + 0.414 (r / (Ro · n/2))^(2 · order)).
    public static func butterworth(radius: Double, ro: Double, order: Int, n: Int) -> Double {
        1 / (1 + 0.414 * pow(radius / (ro * Double(n / 2)), Double(2 * order)))
    }

    /// real(IFFT(FFT(pixels) · mask)), the mask in the centred frame (DC at (n/2, n/2)).
    public static func applyMask(_ mask: [Double], to pixels: [Float], n: Int,
                                 precision: Precision = .float32) throws -> SpectrumValues {
        guard mask.count == n * n, pixels.count == n * n else { throw HRTEMFilterError.pixelCount(expected: n * n, actual: pixels.count) }
        switch precision {
        case .float32:
            guard let fft = FFT2D(nx: n, ny: n) else { throw HRTEMFilterError.fftSetup(n) }
            return applyMask(mask, to: pixels, n: n, fft: fft)
        case .float64:
            guard let fft = FFT2DDouble(nx: n, ny: n) else { throw HRTEMFilterError.fftSetup(n) }
            return applyMask(mask, to: pixels, n: n, fft: fft)
        }
    }

    /// The profile P(ρ), ρ = 0 … n/2 − 1: X sampled bilinearly at (n/2 + ρ cos θ, n/2 + ρ sin θ) (row, column) for 512
    /// angles θ = 2πk/512 and averaged over them.
    public static func radialProfile(_ x: [Double], n: Int) -> [Double] {
        let half = n / 2
        let angles = 512
        var profile = [Double](repeating: 0, count: half)
        let cosines = (0..<angles).map { cos(2 * Double.pi * Double($0) / Double(angles)) }
        let sines = (0..<angles).map { sin(2 * Double.pi * Double($0) / Double(angles)) }
        for rho in 0..<half {
            var sum = 0.0
            for k in 0..<angles {
                sum += bilinear(x, n: n, row: Double(half) + Double(rho) * cosines[k],
                                col: Double(half) + Double(rho) * sines[k])
            }
            profile[rho] = sum / Double(angles)
        }
        return profile
    }

    /// rot(X): `radialProfile` interpolated linearly at each pixel's r, held at its last value beyond ρ = n/2 − 1.
    public static func rotationalAverage(_ x: [Double], n: Int) -> [Double] {
        let profile = radialProfile(x, n: n)
        let half = n / 2
        var out = [Double](repeating: 0, count: n * n)
        for row in 0..<n {
            for col in 0..<n {
                let dr = Double(row - half), dc = Double(col - half)
                let r = (dr * dr + dc * dc).squareRoot()
                let lower = Int(r)
                if lower >= half - 1 {
                    out[row * n + col] = profile[half - 1]
                } else {
                    let f = r - Double(lower)
                    out[row * n + col] = profile[lower] * (1 - f) + profile[lower + 1] * f
                }
            }
        }
        return out
    }

    // MARK: Private

    private static func bilinear(_ x: [Double], n: Int, row: Double, col: Double) -> Double {
        let r = min(max(row, 0), Double(n - 1)), c = min(max(col, 0), Double(n - 1))
        let r0 = Int(r), c0 = Int(c)
        let r1 = min(r0 + 1, n - 1), c1 = min(c0 + 1, n - 1)
        let fr = r - Double(r0), fc = c - Double(c0)
        return x[r0 * n + c0] * (1 - fr) * (1 - fc) + x[r0 * n + c1] * (1 - fr) * fc
            + x[r1 * n + c0] * fr * (1 - fc) + x[r1 * n + c1] * fr * fc
    }

    /// |FFT(pixels)|, fftshifted: the DC at (n/2, n/2). The mean is removed before the transform and the DC bin set in
    /// Double to |Σ pixels| — exact algebra; it keeps the DC out of the transform's rounding.
    private static func centredMagnitude<F: FFT2DTransform>(_ pixels: [Double], n: Int, fft: F) -> [Double] {
        let sum = pixels.reduce(0.0) { $0 + $1 }
        let mean = sum / Double(n * n)
        var re = pixels.map { F.Scalar($0 - mean) }
        var im = [F.Scalar](repeating: 0, count: n * n)
        fft.transform(re: &re, im: &im, forward: true, scaleInverse: true)
        var out = [Double](repeating: 0, count: n * n)
        let half = n / 2
        for u in 0..<n {
            let row = (u + half) % n
            for v in 0..<n {
                let col = (v + half) % n
                let a = Double(re[u * n + v]), b = Double(im[u * n + v])
                out[row * n + col] = (a * a + b * b).squareRoot()
            }
        }
        out[half * n + half] = abs(sum)
        return out
    }

    private static func applyMask<F: FFT2DTransform>(_ mask: [Double], to pixels: [Float], n: Int, fft: F) -> SpectrumValues {
        precondition(mask.count == n * n && pixels.count == n * n)
        // The mean through the DC in Double, the rest through the transforms.
        let mean = pixels.reduce(0.0) { $0 + Double($1) } / Double(n * n)
        var re = pixels.map { F.Scalar(Double($0) - mean) }
        var im = [F.Scalar](repeating: 0, count: n * n)
        fft.transform(re: &re, im: &im, forward: true, scaleInverse: true)
        let half = n / 2
        for u in 0..<n {
            let row = (u + half) % n
            for v in 0..<n {
                let factor = F.Scalar(mask[row * n + (v + half) % n])
                re[u * n + v] *= factor
                im[u * n + v] *= factor
            }
        }
        re[0] = 0; im[0] = 0
        fft.transform(re: &re, im: &im, forward: false, scaleInverse: true)
        let dc = mean * mask[half * n + half]
        if let floats = re as? [Float] { return .float32(floats.map { Float(Double($0) + dc) }) }
        return .float64(re.map { Double($0) + dc })
    }

    /// The 3 × 3 median, edges replicated.
    private static func median3x3(_ x: [Double], n: Int) -> [Double] {
        var out = [Double](repeating: 0, count: n * n)
        var window = [Double](repeating: 0, count: 9)
        for row in 0..<n {
            for col in 0..<n {
                var k = 0
                for dr in -1...1 {
                    let r = min(max(row + dr, 0), n - 1)
                    for dc in -1...1 {
                        window[k] = x[r * n + min(max(col + dc, 0), n - 1)]
                        k += 1
                    }
                }
                window.sort()
                out[row * n + col] = window[4]
            }
        }
        return out
    }
}
