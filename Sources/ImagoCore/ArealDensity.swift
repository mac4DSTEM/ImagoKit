//
//  ArealDensity.swift (ImagoCore)
//  The areal number density of a set of counted objects in a frame, with the Miles–Lantuéjoul edge correction, the length and width
//  summaries of the same objects, and the number density N_V = areal density / thickness with its σ propagated. Moved here from
//  mac4DSTEM's `PrecipitateStatistics.density`, with the same arithmetic in the same order.
//
//  THE RULE: no calibration, no density. `arealDensity` is nil whenever `pixelSize` is nil, not finite or not positive, and
//  `numberDensity` refuses a thickness that is not finite and positive. Nothing is ever computed against an assumed scale.
//
//  EDGE OBJECTS are excluded from the count and from the length and width summaries whether or not the caller accepted them (an
//  object crossing the frame edge is flagged, not counted); `edgeExcluded` says how many accepted objects that was. The density does
//  not inherit that loss: each counted object is weighted by Miles–Lantuéjoul's W·H / ((W − bx − 1)(H − by − 1)), the inverse
//  fraction of positions at which a bx × by pixel bounding box lies clear of the forbidden border row (unweighted, the density read
//  about 9 to 11 % low for T1 on a Thronsen-sized scan). The frame is the whole label image; `analysedPixels` (the area the
//  objects were looked for in, default the whole frame) is the denominator, so pixels that were not analysed are not edges.
//
//  VALIDATION: both results carry `validation: "none"`. The edge weight is a closed form checked against planted truth, not against a
//  published standard, so the product stays badged until a dataset with truth passes.
//

import Foundation

public nonisolated enum ArealDensityError: Error, Equatable, Sendable {
    /// `analysedPixels` is not in 0...(frame area).
    case analysedPixels
    /// A counted object's bounding box leaves no clear row or column in the frame: the labels and the frame disagree.
    case frameDoesNotFitObject(label: Int32)
    /// A thickness (or a sigma) that is not finite and positive (zero or more for a sigma), or an areal density below zero or not finite.
    case invalidValue(String)
}

extension ObjectStatistics {
    /// The count, the edge-corrected areal density and the length and width summaries of one object set.
    public struct ArealDensity: Sendable, Equatable {
        /// Accepted objects not touching the frame edge (the count).
        public let counted: Int
        /// Accepted objects excluded because they touch the frame edge.
        public let edgeExcluded: Int
        public let analysedPixels: Int
        public let pixelSize: Double?
        public let pixelUnit: String?
        /// Σ edge weights of the counted objects / (analysedPixels · pixelSize²); nil without a usable pixel size or with no analysed pixel.
        public let arealDensity: Double?
        /// Mean, median and mean-width of the counted objects, in pixels (`lengthPx`, `widthPx` of `Segmentation.ObjectMoments`); nil without one.
        public let meanLength: Double?
        public let medianLength: Double?
        public let meanWidth: Double?
        /// Always "none": see the file header.
        public var validation: String { "none" }
    }

    /// `PrecipitateStatistics.density`: the objects of `labels` (all, or the label numbers in `accepted`), the frame being the label image.
    public static func arealDensity(labels: Segmentation.LabelImage, accepted: Set<Int32>? = nil, analysedPixels: Int? = nil,
                                    pixelSize: Double?, pixelUnit: String? = nil) throws -> ArealDensity {
        let all = try Segmentation.objectMoments(labels)   // validates the label image before its size is multiplied
        let frameArea = labels.width * labels.height
        let analysed = analysedPixels ?? frameArea
        // Zero analysed pixels is allowed (a map with nothing indexed): the counts stay and the density is nil, as the app's was.
        guard analysed >= 0, analysed <= frameArea else { throw ArealDensityError.analysedPixels }
        let objects = all.filter { accepted?.contains($0.label) ?? true }
        let counted = objects.filter { !$0.touchesEdge }
        let excluded = objects.filter { $0.touchesEdge }

        let lengths = counted.map { $0.lengthPx }.sorted()
        let widths = counted.map { $0.widthPx }
        let meanLength = lengths.isEmpty ? nil : lengths.reduce(0, +) / Double(lengths.count)
        let medianLength: Double?
        if lengths.isEmpty {
            medianLength = nil
        } else if lengths.count % 2 == 1 {
            medianLength = lengths[lengths.count / 2]
        } else {
            medianLength = (lengths[lengths.count / 2 - 1] + lengths[lengths.count / 2]) / 2
        }
        let meanWidth = widths.isEmpty ? nil : widths.reduce(0, +) / Double(widths.count)

        var density: Double?
        if let pixelSize, pixelSize.isFinite, pixelSize > 0, analysed > 0 {
            var weighted = 0.0
            for o in counted { weighted += try edgeWeight(o, frameWidth: labels.width, frameHeight: labels.height) }
            density = weighted / (Double(analysed) * pixelSize * pixelSize)
        }
        return ArealDensity(counted: counted.count, edgeExcluded: excluded.count, analysedPixels: analysed, pixelSize: pixelSize,
                            pixelUnit: pixelUnit, arealDensity: density, meanLength: meanLength, medianLength: medianLength,
                            meanWidth: meanWidth)
    }

    /// Miles–Lantuéjoul: W·H / ((W − bx − 1)(H − by − 1)) for the object's pixel bounding box.
    public static func edgeWeight(_ object: Segmentation.ObjectMoments, frameWidth: Int, frameHeight: Int) throws -> Double {
        let freeX = frameWidth - (object.maxCol - object.minCol) - 1
        let freeY = frameHeight - (object.maxRow - object.minRow) - 1
        guard freeX > 0, freeY > 0 else { throw ArealDensityError.frameDoesNotFitObject(label: object.label) }
        return Double(frameWidth * frameHeight) / Double(freeX * freeY)
    }

    /// N_V = areal density / thickness, with the absolute sigmas that are given propagated in quadrature.
    public struct NumberDensity: Sendable, Equatable {
        public let value: Double
        /// nil when neither input carries a sigma.
        public let sigma: Double?
        public var validation: String { "none" }
    }

    /// `N_V = N_A / t`, σ(N_V) = N_V √((σ_A/N_A)² + (σ_t/t)²) over the sigmas given. Units are the caller's (1/nm² over nm is 1/nm³).
    public static func numberDensity(arealDensity: Double, arealDensitySigma: Double? = nil, thickness: Double,
                                     thicknessSigma: Double? = nil) throws -> NumberDensity {
        guard arealDensity.isFinite, arealDensity >= 0 else { throw ArealDensityError.invalidValue("areal density") }
        guard thickness.isFinite, thickness > 0 else { throw ArealDensityError.invalidValue("thickness") }
        for (name, sigma) in [("areal density sigma", arealDensitySigma), ("thickness sigma", thicknessSigma)] {
            if let sigma, !(sigma.isFinite && sigma >= 0) { throw ArealDensityError.invalidValue(name) }
        }
        let value = arealDensity / thickness
        guard arealDensitySigma != nil || thicknessSigma != nil else { return NumberDensity(value: value, sigma: nil) }
        // σ² = (σ_A / t)² + (N_A · σ_t / t²)²: the partial derivatives of N_A / t, over the sigmas that are given.
        let fromArea = (arealDensitySigma ?? 0) / thickness
        let fromThickness = arealDensity * (thicknessSigma ?? 0) / (thickness * thickness)
        let sigma = (fromArea * fromArea + fromThickness * fromThickness).squareRoot()
        return NumberDensity(value: value, sigma: sigma)
    }
}
