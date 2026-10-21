import Foundation

/// Where the committed fixtures live. They stay beside the test targets that first used them (the pin generators in
/// tools/ write there), so ImagoKitParityTests finds them here instead of through its own #filePath.
package enum FixtureDirs {
    /// The `Tests` folder, found from this file.
    package static let tests = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    /// The repository root.
    package static let repository = tests.deletingLastPathComponent()
    /// `Tests/<target>/Fixtures`.
    package static func fixtures(of target: String) -> URL {
        tests.appendingPathComponent(target).appendingPathComponent("Fixtures")
    }
}
