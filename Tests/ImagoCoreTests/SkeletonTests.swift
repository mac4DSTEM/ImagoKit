// The skeleton's one test (tier 3): every module links and names its registration. Slice 1's tests replace it.
import ImagoAlignment
import ImagoCore
import ImagoFFT
import ImagoFilters
import Testing

@Test func modulesNameTheirRegistration() {
    #expect(ImagoCoreScope.registration == "docs/registration-slice-1.md")
    #expect(ImagoFFTScope.registration == "docs/registration-slice-1.md")
    #expect(ImagoFiltersScope.registration == "docs/registration-slice-1.md")
    #expect(ImagoAlignmentScope.registration.hasPrefix("docs/registration-slice-2.md"))
}
