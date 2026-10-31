// Refusals of peak_local_max footprint= and labels= (no fixtures); the pinned comparisons are in ImagoKitParityTests.
import Foundation
import ImagoFilters
import Testing

@Suite struct PeakFootprintRefusalTests {
    @Test func badFootprintsAreRefused() {
        #expect(throws: FilterError.self) { try PeakFootprint(rows: 2, columns: 2, mask: [true, false, true]) }
        #expect(throws: FilterError.self) { try PeakFootprint(rows: 0, columns: 3, mask: []) }
        // SciPy refuses an all-zero footprint of more than one cell; one false cell takes the size-1 path.
        #expect(throws: FilterError.self) { try PeakFootprint(rows: 1, columns: 2, mask: [false, false]) }
        #expect((try? PeakFootprint(rows: 1, columns: 1, mask: [false])) != nil)
    }

    @Test func badLabelsAreRefused() {
        let a = [Double](repeating: 0, count: 16)
        let ones = [Int](repeating: 1, count: 16)
        #expect(throws: FilterError.self) { try PeakLocalMax.find(a, rows: 4, columns: 4, labels: [Int](repeating: 1, count: 15)) }
        #expect(throws: FilterError.self) { try PeakLocalMax.find(a, rows: 4, columns: 4, labels: ones, numPeaksPerLabel: -1) }
        #expect(throws: FilterError.self) {
            try PeakLocalMax.find(a, rows: 4, columns: 4, labels: ones, symmetry: SpectrumSymmetry(centreRow: 2, centreColumn: 2))
        }
    }
}
