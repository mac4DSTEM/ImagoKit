// The placeholder's test (tier 3): the module without code yet names its registration.
import ImagoAlignment
import Testing

@Test func placeholderNamesItsRegistration() {
    #expect(ImagoAlignmentScope.registration.hasPrefix("docs/registration-slice-2.md"))
}
