# The name check (2026-10-15, at the repository's creation; tier 3)

The suite's standard: domain + `Kit`, modules prefixed by the kit's stem (`DiffractoKit` / `DiffractoCore`), in the Latin or
Italian register of the other kits (Spectro, Diffracto, Formato, Simulato, Cubo). Checked by one Haiku lane with web search,
2026-10-15 about 15:35 CEST, against: Apple's frameworks and APIs (developer.apple.com), the Swift Package Index search, GitHub
repository names (`api.github.com/search/repositories?q=<name>+in:name`), the field's software, trademark records reachable
from search (Justia; USPTO, EUIPO and WIPO were not searched directly), and PyPI, npm and CocoaPods for the exact name.

| Name | Apple | SPI / GitHub (Swift) | Field software | Trademark / commercial | Verdict |
|---|---|---|---|---|---|
| **ImagoKit** (chosen) | none (no framework, class or module "Imago") | SPI "imago": no package; GitHub "imagokit": 0 repositories; small unrelated Swift apps named Imago (nolanbrown/Imago, a DSLR camera app, 5 stars, inactive since 2020) | EPAM's Imago OCR (open-source GPL chemical-structure recognition from images; adjacent field, different product); Mayachitra "imago" (a 2009-era microscopy image manager, status unclear) | IMAGO marks exist, mostly class 42 (SaaS) and a photo agency; one abandoned 1999 class-9 filing; no live class-9 software mark seen | minor |
| MicrografoKit (runner-up) | none | GitHub: 0; web: nothing | none | none found | clear |
| PictorKit | none | GitHub: 0; web: nothing | none | none found | clear (less apt: "painter") |
| FiltroKit | none | SPI "filtro": 0; GitHub: 0 | none (FilterKit, an old CocoaPod, is a near miss) | none found | clear, but names only filtering |
| MicrographKit | none | SPI "micrograph": 0; GitHub: 0 | the word is generic in microscopy and part of Gatan's DigitalMicrograph | descriptive, weakly protectable | minor |
| ImmagineKit | none | GitHub: 0 | none | none found | clear, but reads as Italian for Apple's ImageKit |
| PixelKit | none | heestand-xyz/PixelKit (Swift/Metal "live graphics", 889 stars) and other PixelKit repositories | a Blender add-on "Pixel Kit" | the App Store app "PixelKit: Design & Dev Tools" | **blocking** |
| ImageKit | Apple's ImageKit (Quartz) | | | a commercial image CDN | **blocking** (excluded before the check) |

**Chosen: ImagoKit** (Latin *imago*, image; the suite's register). Modules: `ImagoCore`, `ImagoFFT`, `ImagoFilters`,
`ImagoAlignment`; later `ImagoStrain` (GPA) and `ImagoAtoms` (atom columns, GPL-3.0 if ported from Atomap). The minor clash is
EPAM's Imago OCR, which a search for "Imago" will also find; it is a C++ tool for chemical structure diagrams, not a Swift
package. **Runner-up: MicrografoKit**, clear on every check.

The coordinator was told the choice and the runner-up on 2026-10-15; the owner confirms or renames before the first push
(a direct USPTO/EUIPO search is the owner's, if wanted). Renaming before the first push is a mechanical `sed` over this
repository and the org folder's `References/ImagoKit` link.
