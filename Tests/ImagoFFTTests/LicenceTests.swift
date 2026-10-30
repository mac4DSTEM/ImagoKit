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
    #expect(readme.contains("GPL-3.0-only ([`LICENSE`](LICENSE))"))
}

@Test func everySourceFileStatesOneLicenceId() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    let sources = root.appendingPathComponent("Sources")
    let files = try #require(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
        .compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }
    #expect(files.count > 40)
    for file in files {
        let text = try String(contentsOf: file, encoding: .utf8)
        #expect(text.hasPrefix("// SPDX-License-Identifier: GPL-3.0-only\n"), "\(file.lastPathComponent)")
    }
}

@Test func theBSDNoticesOfThePortedPackagesAreShipped() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    let notice = try String(contentsOf: root.appendingPathComponent("NOTICE"), encoding: .utf8)
    for name in ["NumPy", "SciPy", "scikit-image"] {
        let text = try String(contentsOf: root.appendingPathComponent("Licenses/\(name).txt"), encoding: .utf8)
        #expect(text.contains("Redistribution and use in source and binary forms"), "\(name)")
        #expect(notice.contains("Licenses/\(name).txt"), "\(name)")
    }
}
