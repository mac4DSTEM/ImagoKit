//
//  PowerSpectrum.swift (ImagoFFT)
//  The centred Fourier spectrum of an image or of a region of it — magnitude and power, with the frequency axes in the
//  image's reciprocal unit.
//
//  CONVENTIONS, NumPy's: `F = fftshift(fft2(x))`; the zero frequency at index n/2 (integer division) on each axis, for odd
//  and even n alike; the axis value at index i is (i − n/2) · (1 / (n · d)), d each axis's own scale, so a non-square or
//  anisotropic image has different spacings along x and y. Uncalibrated, d = 1 and the unit is "1/px". Nothing is
//  windowed, padded or mean-removed.
//

import Foundation
import ImagoCore

/// An array a product holds in the precision it was computed in.
public nonisolated enum SpectrumValues: Sendable, Equatable {
    case float32([Float])
    case float64([Double])

    public var count: Int {
        switch self {
        case .float32(let v): v.count
        case .float64(let v): v.count
        }
    }

    /// Element `i` as a Double (exact for either case).
    public subscript(i: Int) -> Double {
        switch self {
        case .float32(let v): Double(v[i])
        case .float64(let v): v[i]
        }
    }

    /// Every element as a Double.
    public var doubles: [Double] {
        switch self {
        case .float32(let v): v.map(Double.init)
        case .float64(let v): v
        }
    }

    /// The precision the values were computed in.
    public var precision: Precision {
        switch self {
        case .float32: .float32
        case .float64: .float64
        }
    }
}

/// Why a spectrum cannot be computed.
public nonisolated enum PowerSpectrumError: Error, Equatable, Sendable {
    case crop(PixelCrop)
    case fftSetup(width: Int, height: Int)
}

public nonisolated struct PowerSpectrum: Sendable {
    public let width: Int
    public let height: Int
    /// |F|, centred, row-major, x fastest; the zero frequency at (height/2, width/2).
    public let magnitude: SpectrumValues
    /// |F|², same layout.
    public let power: SpectrumValues
    /// The frequency at each column and row, in `unitsX` / `unitsY`.
    public let axisX: [Double]
    public let axisY: [Double]
    /// The frequency step along each axis, 1/(n·d) (the value NumPy's fftfreq multiplies by).
    public let spacingX: Double
    public let spacingY: Double
    /// "1/<unit>" of the image's axes ("nm" for an axis in "1/nm"), or "1/px" when an axis is uncalibrated.
    public let unitsX: String
    public let unitsY: String
    /// The region transformed, nil for the whole image. Its own size sets the frequency spacing.
    public let crop: PixelCrop?
    public let precision: Precision

    /// The spectrum of `image`, or of `crop` of it.
    public static func compute(_ image: Micrograph, crop: PixelCrop? = nil,
                               precision: Precision = .float32) throws -> PowerSpectrum {
        let source: Micrograph
        if let crop {
            guard let c = image.cropped(crop) else { throw PowerSpectrumError.crop(crop) }
            source = c
        } else {
            source = image
        }
        let w = source.width, h = source.height
        let magnitude: SpectrumValues, power: SpectrumValues
        switch precision {
        case .float32:
            guard let fft = FFT2D(nx: w, ny: h) else { throw PowerSpectrumError.fftSetup(width: w, height: h) }
            var re = source.pixels
            var im = [Float](repeating: 0, count: w * h)
            fft.transform(re: &re, im: &im, forward: true)
            let (m, p) = centred(re, im, w: w, h: h)
            (magnitude, power) = (.float32(m), .float32(p))
        case .float64:
            guard let fft = FFT2DDouble(nx: w, ny: h) else { throw PowerSpectrumError.fftSetup(width: w, height: h) }
            var re = source.pixels.map(Double.init)
            var im = [Double](repeating: 0, count: w * h)
            fft.transform(re: &re, im: &im, forward: true)
            let (m, p) = centred(re, im, w: w, h: h)
            (magnitude, power) = (.float64(m), .float64(p))
        }
        let (dx, ux) = LengthUnit.frequencySampling(source.x)
        let (dy, uy) = LengthUnit.frequencySampling(source.y)
        return PowerSpectrum(width: w, height: h, magnitude: magnitude, power: power,
                             axisX: frequencies(w, d: dx), axisY: frequencies(h, d: dy),
                             spacingX: 1.0 / (Double(w) * dx), spacingY: 1.0 / (Double(h) * dy),
                             unitsX: ux, unitsY: uy, crop: crop, precision: precision)
    }

    /// NumPy's `fftshift(fftfreq(n, d))`: the integer index times 1/(n·d), in that order of operations.
    public static func frequencies(_ n: Int, d: Double) -> [Double] {
        let val = 1.0 / (Double(n) * d)
        return (0..<n).map { Double($0 - n / 2) * val }
    }

    /// fftshift: output (r, c) takes input ((r + h − h/2) mod h, (c + w − w/2) mod w), i.e. index n/2 is zero.
    private static func centred<T: BinaryFloatingPoint>(_ re: [T], _ im: [T], w: Int, h: Int) -> ([T], [T]) {
        var magnitude = [T](repeating: 0, count: w * h)
        var power = [T](repeating: 0, count: w * h)
        for r in 0..<h {
            let sr = (r + h - h / 2) % h
            for c in 0..<w {
                let i = sr * w + (c + w - w / 2) % w
                let p = re[i] * re[i] + im[i] * im[i]
                power[r * w + c] = p
                magnitude[r * w + c] = p.squareRoot()
            }
        }
        return (magnitude, power)
    }
}
