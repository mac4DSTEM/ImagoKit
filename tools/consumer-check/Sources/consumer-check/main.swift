// ImagoKit consumer check: every symbol named here must be `public`. Slice 1 adds the spectrum, profile, spot, masked
// inverse and filter calls an app makes, each with a value it checks; the skeleton names each module's scope.
import Foundation
import ImagoAlignment
import ImagoCore
import ImagoFFT
import ImagoFilters

var checks = 0
var failures = 0
@MainActor func check(_ condition: Bool, _ label: String) {
    checks += 1
    if !condition { failures += 1; print("FAIL  \(label)") }
}

check(ImagoCoreScope.registration == "docs/registration-slice-1.md", "ImagoCore names its registration")
check(ImagoFFTScope.registration == "docs/registration-slice-1.md", "ImagoFFT names its registration")
check(ImagoFiltersScope.registration == "docs/registration-slice-1.md", "ImagoFilters names its registration")
check(ImagoAlignmentScope.registration.hasPrefix("docs/registration-slice-2.md"), "ImagoAlignment names its registration")

print("consumer-check summary: \(checks) checks, \(failures) failures")
if failures > 0 { exit(1) }
