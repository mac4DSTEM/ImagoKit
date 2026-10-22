//
//  RollingBall.swift (ImagoCore)
//  scikit-image 0.26.0's `restoration.rolling_ball` on a 2D image, with `ball_kernel` and `ellipsoid_kernel`: the background
//  intensity estimated by rolling a ball (or any kernel surface) under the image landscape; the apex of the ball is the result.
//
//  METHOD, as scikit-image: the kernel's surface heights K (+infinity outside the ball) become offsets D = K(centre) - K (infinity
//  stays infinity); the image is padded with +infinity by the kernel's half size; the background at a pixel is the minimum over the
//  kernel cells of (padded image + D). Computed in the image's own type (Double or Float; the kernel is built in Double and
//  converted, as scikit-image does with `kernel.astype(float_type)`). Results are held to `assert_allclose` dtype defaults.
//
//  DEVIATIONS. `nansafe` is not ported: NaN input is refused (`RestorationError.nonFinite`). 2D only; the cost is polynomial in the
//  radius (a 2D image is N^2 times the kernel area), as scikit-image's.
//

import Foundation

public nonisolated enum RestorationError: Error, Equatable, Sendable {
    /// Counts that do not match `width * height`, a non-positive dimension, or a kernel that is empty or not rows * columns.
    case shape
    case parameter(String)
    /// A NaN in the image (`nansafe` is not ported).
    case nonFinite
}

public nonisolated enum RollingBall {
    /// A kernel surface: `values[row * columns + column]` is the height of the ball at that offset, +infinity outside.
    public struct Kernel: Sendable, Equatable {
        public let rows: Int
        public let columns: Int
        public let values: [Double]

        public init(rows: Int, columns: Int, values: [Double]) throws {
            guard rows > 0, columns > 0, values.count == rows * columns else { throw RestorationError.shape }
            self.rows = rows; self.columns = columns; self.values = values
        }
    }

    /// The largest radius `ballKernel` builds (a (2r + 1) square grid of Doubles); `background(image:radius:)` has no such limit.
    public static let maximumRadius: Double = 8192

    /// `ball_kernel(radius, 2)`: the top half of a ball of `radius`, on a (2 ceil(radius) + 1) square grid.
    public static func ballKernel(radius: Double) throws -> Kernel {
        guard radius.isFinite, radius >= 0, radius <= maximumRadius else {
            throw RestorationError.parameter("radius must be between 0 and \(Int(maximumRadius)) pixels")
        }
        let half = Int(radius.rounded(.up)), side = 2 * half + 1
        var values = [Double](repeating: 0, count: side * side)
        for r in -half...half {
            for c in -half...half {
                let sumOfSquares = Double(r * r + c * c)
                var h = max(radius * radius - sumOfSquares, 0).squareRoot()
                if sumOfSquares.squareRoot() > radius { h = .infinity }
                values[(r + half) * side + c + half] = h
            }
        }
        return try Kernel(rows: side, columns: side, values: values)
    }

    /// `ellipsoid_kernel(shape, intensity)`: the top half of an ellipsoid with principal axes `shape` (rows, columns) and height `intensity`.
    public static func ellipsoidKernel(rows: Int, columns: Int, intensity: Double) throws -> Kernel {
        guard rows > 0, columns > 0, rows <= 2 * Int(maximumRadius), columns <= 2 * Int(maximumRadius), intensity.isFinite else {
            throw RestorationError.parameter("shape and intensity")
        }
        let semiR = max(rows / 2, 1), semiC = max(columns / 2, 1)
        let h = 2 * semiR + 1, w = 2 * semiC + 1
        var values = [Double](repeating: 0, count: h * w)
        for r in -semiR...semiR {
            for c in -semiC...semiC {
                let scaling = 1 - ((Double(r) / Double(semiR)) * (Double(r) / Double(semiR))
                    + (Double(c) / Double(semiC)) * (Double(c) / Double(semiC)))
                let surface = intensity * max(scaling, 0).squareRoot()
                values[(r + semiR) * w + c + semiC] = scaling < 0 ? .infinity : surface
            }
        }
        return try Kernel(rows: h, columns: w, values: values)
    }

    /// `rolling_ball(image, radius=)`. The ball's cells beyond the image can never land inside it, so only the cells within
    /// the image's size of the centre are visited; the centre height is the ball's own (`radius`), whatever the image size.
    public static func background<S: BinaryFloatingPoint & Sendable>(image: [S], rows height: Int, columns width: Int, radius: Double) throws -> [S] {
        guard radius.isFinite, radius >= 0 else { throw RestorationError.parameter("radius must be finite and not negative") }
        if radius <= maximumRadius { return try background(image: image, rows: height, columns: width, kernel: ballKernel(radius: radius)) }
        let (n, overflow) = width.multipliedReportingOverflow(by: height)
        guard width > 0, height > 0, !overflow, image.count == n else { throw RestorationError.shape }
        guard !image.contains(where: \.isNaN) else { throw RestorationError.nonFinite }
        let centre = S(radius)
        var cells: [(dr: Int, dc: Int, d: S)] = []
        for dr in -(height - 1)...(height - 1) {
            for dc in -(width - 1)...(width - 1) {
                let sumOfSquares = Double(dr) * Double(dr) + Double(dc) * Double(dc)
                if sumOfSquares.squareRoot() > radius { continue }
                cells.append((dr, dc, centre - S(max(radius * radius - sumOfSquares, 0).squareRoot())))
            }
        }
        return scan(image, width, height, cells)
    }

    /// `rolling_ball(image, kernel=)`.
    public static func background<S: BinaryFloatingPoint & Sendable>(image: [S], rows height: Int, columns width: Int, kernel: Kernel) throws -> [S] {
        let (n, overflow) = width.multipliedReportingOverflow(by: height)
        guard width > 0, height > 0, !overflow, image.count == n else { throw RestorationError.shape }
        guard !image.contains(where: \.isNaN) else { throw RestorationError.nonFinite }
        let centreR = kernel.rows / 2, centreC = kernel.columns / 2
        let centre = S(kernel.values[centreR * kernel.columns + centreC])
        // Offsets D = K(centre) - K in the image's type; +infinity cells are dropped (they would add infinity).
        var cells: [(dr: Int, dc: Int, d: S)] = []
        for r in 0..<kernel.rows {
            for c in 0..<kernel.columns {
                let k = kernel.values[r * kernel.columns + c]
                if k.isInfinite { continue }
                cells.append((r - centreR, c - centreC, centre - S(k)))
            }
        }
        return scan(image, width, height, cells)
    }

    private static func scan<S: BinaryFloatingPoint>(_ image: [S], _ width: Int, _ height: Int, _ cells: [(dr: Int, dc: Int, d: S)]) -> [S] {
        var out = [S](repeating: 0, count: width * height)
        for r in 0..<height {
            for c in 0..<width {
                var best = S.infinity
                for cell in cells {
                    let rr = r + cell.dr, cc = c + cell.dc
                    guard rr >= 0, rr < height, cc >= 0, cc < width else { continue }
                    let v = image[rr * width + cc] + cell.d
                    if v < best { best = v }
                }
                out[r * width + c] = best
            }
        }
        return out
    }
}
