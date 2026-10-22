import CoreML
import XCTest

@testable import FluidAudio

/// Tests for the bulk-copy paths in `CtcKeywordSpotter+Inference`: they must match
/// `MLMultiArray` subscript access exactly.
final class CtcInferenceBufferTests: XCTestCase {

    // MARK: - Audio input

    func testFillContiguousAudioMatchesSubscriptWritesAndZeroesPadding() throws {
        let samples = (0..<1_000).map { Float($0) / 997 - 0.5 }
        let count = 900

        for dataType in [MLMultiArrayDataType.float16, .float32] {
            let fast = try MLMultiArray(shape: [1, 1_024], dataType: dataType)
            fast.withUnsafeMutableBytes { buffer, _ in buffer.initializeMemory(as: UInt8.self, repeating: 0xFF) }
            let reference = try MLMultiArray(shape: [1, 1_024], dataType: dataType)
            for i in 0..<reference.count {
                reference[i] = NSNumber(value: i < count ? samples[i] : 0)
            }

            XCTAssertTrue(CtcKeywordSpotter.fillContiguousAudio(fast, with: samples, count: count))
            fast.withUnsafeBytes { fastBytes in
                reference.withUnsafeBytes { referenceBytes in
                    XCTAssertTrue(fastBytes.elementsEqual(referenceBytes), "dataType \(dataType.rawValue)")
                }
            }
        }
    }
}
