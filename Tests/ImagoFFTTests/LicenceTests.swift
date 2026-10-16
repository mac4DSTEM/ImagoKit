import Foundation
import Testing

@Test func packageLicenceIsGPL3() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    let licence = try String(contentsOf: root.appendingPathComponent("LICENSE"), encoding: .utf8)
    #expect(licence.contains("GNU GENERAL PUBLIC LICENSE"))
    #expect(licence.contains("Version 3, 29 June 2007"))
    #expect(licence.contains("END OF TERMS AND CONDITIONS"))
    let notice = try String(contentsOf: root.appendingPathComponent("NOTICE"), encoding: .utf8)
    #expect(notice.contains("Copyright (c) 2026 Paul Lobpreis"))
    #expect(notice.contains("Licensed under the GNU General Public License v3.0"))
    let readme = try String(contentsOf: root.appendingPathComponent("README.md"), encoding: .utf8)
    #expect(readme.contains("GPL-3.0 ([`LICENSE`](LICENSE))"))
}
