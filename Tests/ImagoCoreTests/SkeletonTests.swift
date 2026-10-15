// The placeholders' test (tier 3): the modules without code yet name their registration.
import ImagoAlignment
import ImagoFilters
import Testing

@Test func placeholdersNameTheirRegistration() {
    #expect(ImagoFiltersScope.registration == "docs/registration-slice-1.md")
    #expect(ImagoAlignmentScope.registration.hasPrefix("docs/registration-slice-2.md"))
}
