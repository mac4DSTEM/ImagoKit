// skimage.restoration.denoise_tv_chambolle / denoise_tv_bregman / denoise_nl_means against the pinned runs of
// tools/denoise/pins.py (scikit-image 0.26.0, NumPy 2.5.4, SciPy 1.18.1).
// Tolerance: scikit-image's wheel ships no tests, so the functions are held to
// assert_allclose's dtype defaults: rtol 1e-7 for Float64 and 1e-5 for Float32, atol 0 (|got - want| <= rtol * |want|).
import Foundation
import ImagoFilters
import Testing

/// numpy.testing.assert_allclose(got, want, rtol=rtol, atol=0): the worst relative difference, or nil when all agree.
private func worstRelative(_ got: [Double], _ want: [Double], rtol: Double) -> (rel: Double, abs: Double, at: Int)? {
    var worst: (Double, Double, Int)?
    for k in 0..<want.count {
        let diff = abs(got[k] - want[k])
        guard diff > rtol * abs(want[k]) else { continue }
        let rel = want[k] == 0 ? Double.infinity : diff / abs(want[k])
        if worst == nil || rel > worst!.0 { worst = (rel, diff, k) }
    }
    return worst.map { (rel: $0.0, abs: $0.1, at: $0.2) }
}

private func check(_ got: [Double], _ pin: DenoisePin) {
    let rtol = pin.c.dtype == "float32" ? 1e-5 : 1e-7
    #expect(got.count == pin.c.y.count, "\(pin)")
    guard got.count == pin.c.y.count else { return }
    let worst = worstRelative(got, pin.c.y, rtol: rtol)
    #expect(worst == nil, "\(pin): worst relative difference \(worst?.rel ?? 0), absolute \(worst?.abs ?? 0) at \(worst?.at ?? 0)")
}

@Test(arguments: DenoisePin.all("chambolle"))
func chambolleMatchesSciKitImage(_ pin: DenoisePin) throws {
    let p = pin.c.params
    let (h, w) = (pin.image.h, pin.image.w)
    let got: [Double]
    if pin.c.dtype == "float32" {
        got = try Denoise.totalVariationChambolle(pin.image.x.map(Float.init), rows: h, columns: w, weight: p.weight!, eps: p.eps!,
                                                  maxNumIter: p.max_num_iter!).map(Double.init)
    } else {
        got = try Denoise.totalVariationChambolle(pin.image.x, rows: h, columns: w, weight: p.weight!, eps: p.eps!, maxNumIter: p.max_num_iter!)
    }
    check(got, pin)
}

@Test(arguments: DenoisePin.all("bregman"))
func bregmanMatchesSciKitImage(_ pin: DenoisePin) throws {
    let p = pin.c.params
    let (h, w) = (pin.image.h, pin.image.w)
    let got: [Double]
    if pin.c.dtype == "float32" {
        got = try Denoise.totalVariationBregman(pin.image.x.map(Float.init), rows: h, columns: w, weight: p.weight!,
                                                maxNumIter: p.max_num_iter!, eps: p.eps!, isotropic: p.isotropic!).map(Double.init)
    } else {
        got = try Denoise.totalVariationBregman(pin.image.x, rows: h, columns: w, weight: p.weight!, maxNumIter: p.max_num_iter!,
                                                eps: p.eps!, isotropic: p.isotropic!)
    }
    check(got, pin)
}

@Test(arguments: DenoisePin.all("nlmeans"))
func nonLocalMeansMatchesSciKitImage(_ pin: DenoisePin) throws {
    let p = pin.c.params
    let (h, w) = (pin.image.h, pin.image.w)
    let got: [Double]
    if pin.c.dtype == "float32" {
        got = try Denoise.nonLocalMeans(pin.image.x.map(Float.init), rows: h, columns: w, patchSize: p.patch_size!,
                                        patchDistance: p.patch_distance!, h: p.h!, fastMode: p.fast_mode!, sigma: p.sigma!).map(Double.init)
    } else {
        got = try Denoise.nonLocalMeans(pin.image.x, rows: h, columns: w, patchSize: p.patch_size!, patchDistance: p.patch_distance!,
                                        h: p.h!, fastMode: p.fast_mode!, sigma: p.sigma!)
    }
    check(got, pin)
}

/// Records the largest differences per group (visible with `swift test --filter denoiseAgreementReport`); never fails.
@Test func denoiseAgreementReport() throws {
    var lines = [String]()
    for kind in ["chambolle", "bregman", "nlmeans"] {
        for dtype in ["float64", "float32"] {
            var maxRel = 0.0, maxAbs = 0.0, count = 0
            for pin in DenoisePin.all(kind) where pin.c.dtype == dtype {
                let p = pin.c.params, h = pin.image.h, w = pin.image.w
                let got: [Double]
                switch (kind, dtype) {
                case ("chambolle", "float32"):
                    got = try Denoise.totalVariationChambolle(pin.image.x.map(Float.init), rows: h, columns: w, weight: p.weight!, eps: p.eps!, maxNumIter: p.max_num_iter!).map(Double.init)
                case ("chambolle", _):
                    got = try Denoise.totalVariationChambolle(pin.image.x, rows: h, columns: w, weight: p.weight!, eps: p.eps!, maxNumIter: p.max_num_iter!)
                case ("bregman", "float32"):
                    got = try Denoise.totalVariationBregman(pin.image.x.map(Float.init), rows: h, columns: w, weight: p.weight!, maxNumIter: p.max_num_iter!, eps: p.eps!, isotropic: p.isotropic!).map(Double.init)
                case ("bregman", _):
                    got = try Denoise.totalVariationBregman(pin.image.x, rows: h, columns: w, weight: p.weight!, maxNumIter: p.max_num_iter!, eps: p.eps!, isotropic: p.isotropic!)
                case (_, "float32"):
                    got = try Denoise.nonLocalMeans(pin.image.x.map(Float.init), rows: h, columns: w, patchSize: p.patch_size!, patchDistance: p.patch_distance!, h: p.h!, fastMode: p.fast_mode!, sigma: p.sigma!).map(Double.init)
                default:
                    got = try Denoise.nonLocalMeans(pin.image.x, rows: h, columns: w, patchSize: p.patch_size!, patchDistance: p.patch_distance!, h: p.h!, fastMode: p.fast_mode!, sigma: p.sigma!)
                }
                count += 1
                for k in 0..<got.count {
                    let diff = abs(got[k] - pin.c.y[k])
                    maxAbs = max(maxAbs, diff)
                    if pin.c.y[k] != 0 { maxRel = max(maxRel, diff / abs(pin.c.y[k])) }
                }
            }
            lines.append("\(kind) \(dtype): \(count) pins, max relative \(maxRel), max absolute \(maxAbs)")
        }
    }
    print(lines.joined(separator: "\n"))
}
