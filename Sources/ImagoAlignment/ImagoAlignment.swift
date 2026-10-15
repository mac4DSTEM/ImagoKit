// ImagoAlignment — drift and stack alignment: phase cross-correlation with sub-pixel refinement
// (scikit-image's phase_cross_correlation), sub-pixel shifts (SciPy's ndimage.shift and fourier_shift), drift series and
// aligned stack sums (slice 2; its scope is registered in docs/registration-slice-1.md section 10). Empty until slice 2
// is registered.

/// The module's scope, stated where a reader of the API finds it.
public enum ImagoAlignmentScope {
    /// The registration that will define this module's numbers.
    public static let registration = "docs/registration-slice-2.md (not yet written)"
}
