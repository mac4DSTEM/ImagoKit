// ImagoAlignment — drift and stack alignment: phase cross-correlation with sub-pixel refinement (scikit-image's
// phase_cross_correlation), sub-pixel shifts (SciPy's ndimage.shift, orders 0 and 1, and fourier_shift), drift series and
// aligned stack sums. Registered in docs/registration-slice-2.md; empty until its code lands.

/// The module's scope, stated where a reader of the API finds it.
public enum ImagoAlignmentScope {
    /// The registration that defines this module's numbers.
    public static let registration = "docs/registration-slice-2.md"
}
