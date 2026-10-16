import XCTest
@testable import Misago

/// The classifier on hand-made weights: readout only, so the verdict follows the calibrated readout bias.
final class ImageClassifierTests: XCTestCase {
    private func weights(bOut: [Float], calibration: VectorCalibration) -> HawkWeights {
        let classes = try! ClassList(others: ["a_home", "b_home"])
        let d = 8
        let inDim = RowScanEncoder.frameDim
        // No layers: logits = mean(RMSNorm(x W_in + b_in)) W_out + b_out, with W_in and W_out zero.
        return HawkWeights(
            inputDim: inDim, modelDim: d, rnnDim: 8, mlpDim: 8, heads: 1, convWidth: 4, convRight: 2, outputDim: 3,
            wIn: [Float](repeating: 0.0, count: inDim * d), bIn: [Float](repeating: 0.1, count: d), layers: [],
            finalNorm: [Float](repeating: 1.0, count: d), wOut: [Float](repeating: 0.0, count: d * 3), bOut: bOut,
            classNames: classes.names, calibrationScale: calibration.scale, calibrationBias: calibration.bias
        )
    }

    func testArgmaxAndThresholdFallback() throws {
        let image = ScreenImage()
        let confident = try ImageClassifier(weights: weights(bOut: [0.0, 3.0, 0.0], calibration: .identity(classCount: 3)), threshold: 0.5)
        let verdict = confident.classify(image)
        XCTAssertEqual(verdict.classIndex, 1)
        XCTAssertEqual(verdict.className, "a_home")
        XCTAssertFalse(verdict.isOther)
        XCTAssertGreaterThan(verdict.probability, 0.9)
        XCTAssertEqual(verdict.logits, [0.0, 3.0, 0.0])

        // A weak winner falls back to other when below the threshold.
        let weak = try ImageClassifier(weights: weights(bOut: [0.0, 0.2, 0.0], calibration: .identity(classCount: 3)), threshold: 0.5)
        let fallback = weak.classify(image)
        XCTAssertTrue(fallback.isOther)
        XCTAssertEqual(fallback.className, "other")
        XCTAssertEqual(fallback.probability, fallback.probabilities[0])

        // Calibration bias can move the decision.
        let biased = try ImageClassifier(weights: weights(bOut: [0.0, 0.2, 0.0], calibration: VectorCalibration(scale: 1.0, bias: [0.0, 0.0, 2.0])), threshold: 0.5)
        XCTAssertEqual(biased.classify(image).className, "b_home")
    }

    func testOutcomeFollowsClassNameSuffix() {
        func verdict(_ name: String) -> ImageVerdict {
            return ImageVerdict(classIndex: 1, className: name, probability: 0.9, probabilities: [0.1, 0.9], logits: [0, 1])
        }
        XCTAssertEqual(verdict("a_match_win").outcome, 2)
        XCTAssertEqual(verdict("a_match_lose").outcome, 1)
        XCTAssertEqual(verdict("a_match_draw").outcome, 0)
        XCTAssertEqual(verdict("a_home").outcome, -1)
        XCTAssertEqual(verdict("other").outcome, -1)
        XCTAssertEqual(ScreenReading(verdict: verdict("a_match_win"), reading: nil).outcome, 2)
    }

    func testInvalidClassNamesAreRejected() {
        var w = weights(bOut: [0.0, 0.0, 0.0], calibration: .identity(classCount: 3))
        w = HawkWeights(
            inputDim: w.inputDim, modelDim: w.modelDim, rnnDim: w.rnnDim, mlpDim: w.mlpDim, heads: w.heads, convWidth: w.convWidth,
            convRight: w.convRight, outputDim: w.outputDim, wIn: w.wIn, bIn: w.bIn, layers: w.layers, finalNorm: w.finalNorm,
            wOut: w.wOut, bOut: w.bOut, classNames: ["a", "b", "c"]
        )
        XCTAssertThrowsError(try ImageClassifier(weights: w))
    }
}
