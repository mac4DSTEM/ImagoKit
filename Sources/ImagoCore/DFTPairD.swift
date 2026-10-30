// SPDX-License-Identifier: GPL-3.0-only
//
//  DFTPairD.swift (ImagoCore)
//  A forward and an inverse `vDSP_DFT_zop` Double setup for one length. Accelerate's radix-2 Double FFT
//  (`vDSP_fft_zipD`, `vDSP_fft2d_zipD`) is 5 to 7 times slower than these plans from 128² up (measured 2026-10-21: 303 vs
//  55 µs at 128², 2.4 vs 0.32 ms at 256²), so the Double transforms use this wherever vDSP accepts the length (f·2ⁿ with
//  f in 1, 3, 5, 15) and keep the radix-2 call only where `init` returns nil. Out of place; the inverse is unnormalised,
//  like vDSP's. A setup is documented shareable across threads.
//

import Foundation
import Accelerate

nonisolated package final class DFTPairD: @unchecked Sendable {
    package let length: Int
    private let forwardSetup: vDSP_DFT_SetupD
    private let inverseSetup: vDSP_DFT_SetupD

    /// nil when vDSP has no plan for this length.
    package init?(length: Int) {
        guard length > 1,
              let forward = vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(length), .FORWARD) else { return nil }
        guard let inverse = vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(length), .INVERSE) else {
            vDSP_DFT_DestroySetupD(forward)
            return nil
        }
        self.length = length
        forwardSetup = forward
        inverseSetup = inverse
    }

    deinit {
        vDSP_DFT_DestroySetupD(forwardSetup)
        vDSP_DFT_DestroySetupD(inverseSetup)
    }

    package func execute(forward: Bool,
                         _ inputRe: UnsafePointer<Double>, _ inputIm: UnsafePointer<Double>,
                         _ outputRe: UnsafeMutablePointer<Double>, _ outputIm: UnsafeMutablePointer<Double>) {
        vDSP_DFT_ExecuteD(forward ? forwardSetup : inverseSetup, inputRe, inputIm, outputRe, outputIm)
    }
}
