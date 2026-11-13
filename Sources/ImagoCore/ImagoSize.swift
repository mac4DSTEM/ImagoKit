//
//  ImagoSize.swift
//  ImagoCore
//
//  The checked pixel count of a rows × columns grid, for the guards that compare a buffer's count with it: a product that
//  overflows Int is no grid, where `rows * columns` trapped (macEMSuite/ImagoKit#78).
//

package nonisolated enum ImagoSize {
    /// rows · columns, or nil when either is negative or the product overflows Int.
    @inline(__always)
    package static func area(_ rows: Int, _ columns: Int) -> Int? {
        guard rows >= 0, columns >= 0 else { return nil }
        let (product, overflow) = rows.multipliedReportingOverflow(by: columns)
        return overflow ? nil : product
    }
}
