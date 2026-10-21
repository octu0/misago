import XCTest
@testable import Misago

/// The classifier on hand-made weights: readout only, so the verdict follows the calibrated readout bias.
final class ImageClassifierTests: XCTestCase {
    private func weights(bOut: [Float], calibration: VectorCalibration, names: [String] = ["a_home", "b_home"], titles: [String] = []) -> HawkWeights {
        let classes = try! ClassList(names: ["other"] + names)
        let d = 8
        let inDim = RowScanEncoder.frameDim
        let outDim = classes.count + titles.count
        // No layers: logits = mean(RMSNorm(x W_in + b_in)) W_out + b_out, with W_in and W_out zero.
        return HawkWeights(
            inputDim: inDim, modelDim: d, rnnDim: 8, mlpDim: 8, heads: 1, convWidth: 4, convRight: 2, outputDim: outDim,
            wIn: [Float](repeating: 0.0, count: inDim * d), bIn: [Float](repeating: 0.1, count: d), layers: [],
            finalNorm: [Float](repeating: 1.0, count: d), wOut: [Float](repeating: 0.0, count: d * outDim), bOut: bOut,
            classNames: classes.names, calibrationScale: calibration.scale, calibrationBias: calibration.bias,
            titleNames: titles
        )
    }

    func testArgmaxWinsWithoutFallback() throws {
        let image = ScreenImage()
        let confident = try ImageClassifier(weights: weights(bOut: [0.0, 3.0, 0.0], calibration: .identity(classCount: 3)))
        let verdict = confident.classify(image)
        XCTAssertEqual(verdict.classIndex, 1)
        XCTAssertEqual(verdict.className, "a_home")
        XCTAssertTrue(verdict.recognized)
        XCTAssertGreaterThan(verdict.probability, 0.9)
        XCTAssertEqual(verdict.logits, [0.0, 3.0, 0.0])

        // A weak winner still wins: the probability is reported, not turned into `other`.
        let weak = try ImageClassifier(weights: weights(bOut: [0.0, 0.2, 0.0], calibration: .identity(classCount: 3)))
        let verdictWeak = weak.classify(image)
        XCTAssertEqual(verdictWeak.className, "a_home")
        XCTAssertLessThan(verdictWeak.probability, 0.5)
        XCTAssertEqual(verdictWeak.probability, verdictWeak.probabilities[1])

        // Calibration bias can move the decision.
        let biased = try ImageClassifier(weights: weights(bOut: [0.0, 0.2, 0.0], calibration: VectorCalibration(scale: 1.0, bias: [0.0, 0.0, 2.0])))
        XCTAssertEqual(biased.classify(image).className, "b_home")
    }

    func testRotationTagsAreMergedAndGiveTheOrientation() throws {
        let names = ["a_home", "b_home", "a_home@180", "b_home@180"]
        // a_home seen upside down is the most likely output; a_home also wins once rotations are summed.
        let classifier = try ImageClassifier(weights: weights(bOut: [0.0, 1.0, 0.0, 2.0, 0.0], calibration: .identity(classCount: 5), names: names))
        XCTAssertEqual(classifier.classes.names, ["other", "a_home", "b_home"])
        XCTAssertEqual(classifier.outputClasses.count, 5)
        let verdict = classifier.classify(ScreenImage())
        XCTAssertEqual(verdict.className, "a_home")
        XCTAssertEqual(verdict.rotation, 180)
        XCTAssertEqual(verdict.probabilities.count, 3)
        XCTAssertEqual(verdict.probabilities.reduce(0, +), 1.0, accuracy: 1e-5)
        XCTAssertEqual(verdict.probability, verdict.probabilities[1])
        XCTAssertEqual(verdict.logits.count, 5)

        // An upright winner keeps rotation 0; `other` always reports 0.
        let upright = try ImageClassifier(weights: weights(bOut: [0.0, 0.0, 3.0, 0.0, 1.0], calibration: .identity(classCount: 5), names: names))
        XCTAssertEqual(upright.classify(ScreenImage()).className, "b_home")
        XCTAssertEqual(upright.classify(ScreenImage()).rotation, 0)
        let other = try ImageClassifier(weights: weights(bOut: [5.0, 0.0, 0.0, 0.0, 0.0], calibration: .identity(classCount: 5), names: names))
        XCTAssertEqual(other.classify(ScreenImage()).rotation, 0)
    }

    func testTitlesAreIndependentAndGateTheDerivedValues() throws {
        // Screens other, a_match_win, b_home; titles a, b, other. The screen softmax favours a_match_win, but
        // title a is unlikely (logit −3): the image is not recognized, so no outcome is reported.
        let names = ["a_match_win", "b_home"]
        let unknown = try ImageClassifier(weights: weights(bOut: [0.0, 3.0, 0.0, -3.0, -4.0, -5.0], calibration: .identity(classCount: 3), names: names, titles: ["other", "a", "b"]))
        let verdict = unknown.classify(ScreenImage())
        XCTAssertEqual(verdict.className, "a_match_win")
        XCTAssertEqual(verdict.title, "a")
        XCTAssertEqual(verdict.titleProbability, 1.0 / (1.0 + exp(4.0)), accuracy: 1e-5)
        XCTAssertFalse(verdict.recognized)
        XCTAssertEqual(verdict.outcome, -1)
        XCTAssertEqual(verdict.titleProbabilities.count, 3)
        XCTAssertLessThan(verdict.titleProbabilities.values.max()!, 0.1)

        let known = try ImageClassifier(weights: weights(bOut: [0.0, 3.0, 0.0, -3.0, 4.0, -5.0], calibration: .identity(classCount: 3), names: names, titles: ["other", "a", "b"]))
        let recognized = known.classify(ScreenImage())
        XCTAssertTrue(recognized.recognized)
        XCTAssertEqual(recognized.outcome, 2)
        XCTAssertEqual(recognized.probabilities.count, 3)
        XCTAssertEqual(recognized.probabilities.reduce(0, +), 1.0, accuracy: 1e-5)
    }

    func testRotationTagParsing() {
        XCTAssertEqual(ClassList.parseRotation("a_home@180").base, "a_home")
        XCTAssertEqual(ClassList.parseRotation("a_home@180").rotation, 180)
        XCTAssertEqual(ClassList.parseRotation("a_home").rotation, 0)
        XCTAssertEqual(ClassList.parseRotation("a_home@x").base, "a_home@x")
        XCTAssertEqual(ClassList.rotated("a_home", 180), "a_home@180")
        XCTAssertEqual(ClassList.rotated("a_home", 0), "a_home")
        XCTAssertEqual(ClassList.rotated("a_home", -180), "a_home@180")
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

    func testBattleFollowsClassName() {
        func verdict(_ name: String) -> ImageVerdict {
            return ImageVerdict(classIndex: 1, className: name, probability: 0.9, probabilities: [0.1, 0.9], logits: [0, 1])
        }
        XCTAssertEqual(verdict("a_battle").battle, 1)
        XCTAssertEqual(verdict("a_battle_knockout").battle, 1)
        XCTAssertEqual(verdict("a_battle_hot_zone").battle, 1)
        XCTAssertEqual(verdict("a_battlelog").battle, 0)
        XCTAssertEqual(verdict("a_prebattle").battle, 0)
        XCTAssertEqual(verdict("a_replay").battle, 0)
        XCTAssertEqual(verdict("other").battle, 0)
        XCTAssertEqual(ScreenReading(verdict: verdict("a_battle_bounty"), reading: nil).battle, 1)
    }

    func testInvalidClassNamesAreRejected() {
        var w = weights(bOut: [0.0, 0.0, 0.0], calibration: .identity(classCount: 3))
        w = HawkWeights(
            inputDim: w.inputDim, modelDim: w.modelDim, rnnDim: w.rnnDim, mlpDim: w.mlpDim, heads: w.heads, convWidth: w.convWidth,
            convRight: w.convRight, outputDim: w.outputDim, wIn: w.wIn, bIn: w.bIn, layers: w.layers, finalNorm: w.finalNorm,
            wOut: w.wOut, bOut: w.bOut, classNames: ["a_home", "a_home", "b_home"]
        )
        XCTAssertThrowsError(try ImageClassifier(weights: w))  // duplicate names
    }
}
