import CoreML
import XCTest

@testable import FluidAudio

/// Tests for the bulk-copy paths in `CtcKeywordSpotter+Inference`: they must match
/// `MLMultiArray` subscript access exactly, including padded (ANE-aligned) strides.
final class CtcInferenceBufferTests: XCTestCase {

    private let vocabSize = 5
    private let timeSteps = 7

    // MARK: - Helpers

    /// Allocate an array with explicit strides and fill the padding with 0xFF (NaN for floats),
    /// so reading padding instead of real elements fails the comparison.
    private func makeStridedArray(shape: [Int], strides: [Int], dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let elementSize = dataType == .float16 ? 2 : 4
        let byteCount = zip(shape, strides).map { ($0 - 1) * $1 }.reduce(1, +) * elementSize
        let pointer = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 64)
        pointer.initializeMemory(as: UInt8.self, repeating: 0xFF, count: byteCount)
        return try MLMultiArray(
            dataPointer: pointer,
            shape: shape.map { NSNumber(value: $0) },
            dataType: dataType,
            strides: strides.map { NSNumber(value: $0) },
            deallocator: { $0.deallocate() }
        )
    }

    private func fillLogits(_ array: MLMultiArray, timeAxis: Int, vocabAxis: Int) {
        var index = [NSNumber](repeating: 0, count: array.shape.count)
        for t in 0..<timeSteps {
            index[timeAxis] = NSNumber(value: t)
            for v in 0..<vocabSize {
                index[vocabAxis] = NSNumber(value: v)
                // Multiples of 1/8 are exact in Float16.
                array[index] = NSNumber(value: Float(t * vocabSize + v) / 8 - 2)
            }
        }
    }

    // MARK: - Logits

    func testFrameMajorLogitsMatchesSubscriptReadsForSupportedLayouts() throws {
        let layouts: [(shape: [Int], strides: [Int], timeAxis: Int, vocabAxis: Int)] = [
            // [1, V, 1, T] with time padded from 7 to 16
            ([1, vocabSize, 1, timeSteps], [vocabSize * 16, 16, 16, 1], 3, 1),
            // [1, V, 1, T] contiguous
            ([1, vocabSize, 1, timeSteps], [vocabSize * timeSteps, timeSteps, timeSteps, 1], 3, 1),
            // [1, T, V] with vocab padded from 5 to 8
            ([1, timeSteps, vocabSize], [timeSteps * 8, 8, 1], 1, 2),
        ]

        for dataType in [MLMultiArrayDataType.float16, .float32] {
            for layout in layouts {
                let array = try makeStridedArray(shape: layout.shape, strides: layout.strides, dataType: dataType)
                fillLogits(array, timeAxis: layout.timeAxis, vocabAxis: layout.vocabAxis)

                let fast = CtcKeywordSpotter.frameMajorLogits(
                    from: array, timeAxis: layout.timeAxis, vocabAxis: layout.vocabAxis)
                let reference = CtcKeywordSpotter.frameMajorLogitsBySubscript(
                    from: array, timeAxis: layout.timeAxis, vocabAxis: layout.vocabAxis)

                XCTAssertEqual(fast, reference, "dataType \(dataType.rawValue), strides \(layout.strides)")
            }
        }
    }

    func testFrameMajorLogitsRejectsUnsupportedLayouts() throws {
        let int32Array = try MLMultiArray(
            shape: [1, NSNumber(value: vocabSize), 1, NSNumber(value: timeSteps)], dataType: .int32)
        XCTAssertNil(CtcKeywordSpotter.frameMajorLogits(from: int32Array, timeAxis: 3, vocabAxis: 1))

        let noUnitStride = try makeStridedArray(
            shape: [1, vocabSize, 1, timeSteps], strides: [vocabSize * timeSteps * 2, timeSteps * 2, 1, 2],
            dataType: .float32)
        XCTAssertNil(CtcKeywordSpotter.frameMajorLogits(from: noUnitStride, timeAxis: 3, vocabAxis: 1))
    }

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
