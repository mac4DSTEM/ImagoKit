// ImagoFilters — the HRTEM filters: average background subtraction (ABSF) and Wiener, after Kilaas (1998) as
// D. R. G. Mitchell's HRTEM_Filter script describes them (slice 1, docs/registration-slice-1.md 1.3). Later: bandpass
// and denoising filters against scikit-image and SciPy. Placeholder until slice 1's code lands.

/// The module's scope, stated where a reader of the API finds it.
public enum ImagoFiltersScope {
    /// The registration that defines this module's numbers.
    public static let registration = "docs/registration-slice-1.md"
}
