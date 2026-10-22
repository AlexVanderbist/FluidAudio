import XCTest

@testable import FluidAudio

/// A spoken-form alias can sound unlike the term's spelling ("cube control" for "kubectl"). When the audio
/// says the alias, the term must still win the CTC comparison against the transcript's identical phrase.
@available(macOS 14.0, iOS 17.0, *)
final class VocabularyRescorerAliasScoringTests: XCTestCase {

    func testAliasHeardInTheAudioCountsAsEvidenceForTheTerm() async throws {
        let directory = CtcModels.defaultCacheDirectory()
        try XCTSkipUnless(CtcModels.modelsExist(at: directory), "CTC models not cached")
        let models = try await CtcModels.load(from: directory, computeUnits: .cpuOnly)
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(models: models),
            vocabulary: CustomVocabularyContext(terms: []),
            ctcModelDirectory: directory
        )
        let tokenizer = try await CtcTokenizer.load(from: directory)
        let logProbs = Self.logProbs(
            saying: tokenizer.encode("cube control"), vocabularySize: models.vocabulary.count + 1)

        func evaluate(matchedAlias: String?) -> VocabularyRescorer.CTCMatchResult {
            let candidate = VocabularyRescorer.CTCMatchCandidate(
                origin: .termCentricMultiWord,
                originalPhrase: "cube control",
                vocabTerm: "kubectl",
                matchedAlias: matchedAlias,
                vocabTokens: tokenizer.encode("kubectl"),
                similarity: 1,
                spanLength: 2,
                spanIndices: [0, 1],
                tokenRange: nil,
                spanStartTime: 0,
                spanEndTime: Double(logProbs.count) * Self.frameDuration
            )
            return rescorer.evaluateCTCMatch(
                candidate: candidate,
                logProbs: logProbs,
                frameDuration: Self.frameDuration,
                cbw: ContextBiasingConstants.defaultCbw,
                marginSeconds: 0
            )
        }

        XCTAssertTrue(evaluate(matchedAlias: "cube control").shouldReplace)
        XCTAssertFalse(evaluate(matchedAlias: nil).shouldReplace)
    }

    private static let frameDuration = 0.08

    /// Each token is hot for two frames, with a blank frame before and after it.
    private static func logProbs(saying tokens: [Int], vocabularySize: Int) -> [[Float]] {
        let blank = ContextBiasingConstants.defaultBlankId
        func frame(hot token: Int) -> [Float] {
            var frame = [Float](repeating: -12, count: vocabularySize)
            frame[token] = -0.05
            return frame
        }
        var frames = [frame(hot: blank)]
        for token in tokens {
            frames += [frame(hot: token), frame(hot: token), frame(hot: blank)]
        }
        return frames
    }
}
