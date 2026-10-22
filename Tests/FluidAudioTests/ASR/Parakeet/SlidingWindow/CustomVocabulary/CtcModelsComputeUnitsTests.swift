import AVFoundation
import XCTest

@testable import FluidAudio

/// The CTC MelSpectrogram returns near-zero features on the Neural Engine (macOS 27.2, M1 Pro), so the
/// encoder hears silence. `CtcModels.load` runs it on the CPU whatever the requested compute units.
@available(macOS 14.0, iOS 17.0, *)
final class CtcModelsComputeUnitsTests: XCTestCase {

    func testNeuralEngineLoadHearsSpeechLikeCpuLoad() async throws {
        let directory = CtcModels.defaultCacheDirectory()
        try XCTSkipUnless(CtcModels.modelsExist(at: directory), "CTC models not cached")
        let samples = Array(try Self.loadFixture().prefix(ASRConstants.maxModelSamples))

        let neuralEngine = try await CtcKeywordSpotter(models: CtcModels.load(from: directory))
            .computeLogProbs(for: samples).logProbs
        let cpu = try await CtcKeywordSpotter(models: CtcModels.load(from: directory, computeUnits: .cpuOnly))
            .computeLogProbs(for: samples).logProbs

        let differences: [Float] = zip(neuralEngine, cpu).flatMap { frames in
            zip(frames.0, frames.1).map { abs($0 - $1) }
        }
        let meanDifference = differences.reduce(0, +) / Float(differences.count)

        XCTAssertEqual(neuralEngine.count, cpu.count)
        XCTAssertGreaterThan(Self.speechFrames(cpu), 20)
        XCTAssertGreaterThan(Self.speechFrames(neuralEngine), Self.speechFrames(cpu) / 2)
        XCTAssertLessThan(meanDifference, 1)
    }

    private static func speechFrames(_ logProbs: [[Float]]) -> Int {
        logProbs.filter { frame in
            frame.indices.max { frame[$0] < frame[$1] } != ContextBiasingConstants.defaultBlankId
        }.count
    }

    private static func loadFixture() throws -> [Float] {
        guard
            let url = Bundle.module.url(forResource: "Fixtures/01-validation-request-21.4s", withExtension: "wav")
                ?? Bundle.module.url(forResource: "01-validation-request-21.4s", withExtension: "wav")
        else {
            throw XCTSkip("fixture missing from test bundle")
        }
        return try AudioConverter().resampleAudioFile(url)
    }
}
