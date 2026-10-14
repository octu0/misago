import XCTest
import MLX
@testable import Misago
@testable import MisagoTraining

/// Consistency between the MLX training model and the CPU inference network, and learnability on synthetic data.
/// Runs on the GPU because the linear scan is a Metal kernel.
final class HawkTests: XCTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        try MLXTestEnvironment.skipIfMetallibUnavailable()
        Device.setDefault(device: .gpu)
        MLXRandom.seed(7)
    }

    private func makeFeatures(frames: Int, dim: Int, phase: Int = 0) -> [[Float]] {
        var seq: [[Float]] = []
        var t = 0
        while t < frames {
            var frame = [Float](repeating: 0.0, count: dim)
            var d = 0
            while d < dim {
                frame[d] = 0.5 + 0.5 * sin(Float((t + phase) * 7 + d * 13) * 0.37)
                d += 1
            }
            seq.append(frame)
            t += 1
        }
        return seq
    }

    private let smallClasses = try! ClassList(others: ["a_home", "b_home"])

    private func smallNetwork() -> MLXHawkNetwork {
        return MLXHawkNetwork(inputDim: 12, modelDim: 16, rnnDim: 24, mlpDim: 32, heads: 3, convWidth: 4, convRight: 2, numLayers: 2, outputDim: 3)
    }

    private func assertClose(_ a: [Float], _ b: [Float], _ tolerance: Float, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, file: file, line: line)
        var worst: Float = 0.0
        var i = 0
        while i < min(a.count, b.count) {
            worst = max(worst, abs(a[i] - b[i]))
            i += 1
        }
        XCTAssertLessThan(worst, tolerance, file: file, line: line)
    }

    /// The CPU network reproduces the MLX classification logits, including the mean-then-project readout.
    func testForwardMatchesBetweenMLXAndCPU() {
        let net = smallNetwork()
        // Move trainable biases and norm gains away from their initial values so the test exercises them.
        let start = net.exportWeights(classes: smallClasses)
        let shifted = HawkWeights(
            inputDim: start.inputDim, modelDim: start.modelDim, rnnDim: start.rnnDim, mlpDim: start.mlpDim,
            heads: start.heads, convWidth: start.convWidth, convRight: start.convRight, outputDim: start.outputDim, wIn: start.wIn, bIn: start.bIn,
            layers: start.layers.map { l in
                HawkLayerWeights(
                    norm1: l.norm1.map { $0 * 1.3 }, wGate: l.wGate, wX: l.wX, convW: l.convW.map { $0 * 1.5 }, convB: l.convB.map { _ in 0.1 },
                    wA: l.wA, bA: l.bA.map { _ in 0.4 }, wI: l.wI, bI: l.bI.map { _ in -0.3 }, lambda: l.lambda, wO: l.wO,
                    norm2: l.norm2, wMLPGate: l.wMLPGate, wMLPUp: l.wMLPUp, wMLPDown: l.wMLPDown
                )
            },
            finalNorm: start.finalNorm.map { $0 * 0.8 }, wOut: start.wOut, bOut: start.bOut.map { _ in 0.05 }, classNames: smallClasses.names
        )
        net.importWeights(from: shifted)
        let frames = 37
        let features = makeFeatures(frames: frames, dim: 12)
        let logits = net(MLXArray(features.flatMap { $0 }, [1, frames, 12]))
        let perFrame = mean(net.frameLogits(MLXArray(features.flatMap { $0 }, [1, frames, 12])), axis: 1)
        eval(logits, perFrame)
        let expected = logits.asArray(Float.self)
        // Mean of per-frame logits equals the pooled readout.
        assertClose(expected, perFrame.asArray(Float.self), 1e-5)

        let cpu = HawkNetwork(weights: net.exportWeights(classes: smallClasses))
        assertClose(cpu.logits(features: features), expected, 2e-4)
        assertClose(cpu.logits(flatFeatures: features.flatMap { $0 }, frames: frames), expected, 2e-4)
    }

    /// Weights survive a JSON round trip and the calibration fields are kept.
    func testWeightsRoundTrip() throws {
        let net = smallNetwork()
        let weights = net.exportWeights(classes: smallClasses, calibration: VectorCalibration(scale: 0.7, bias: [0.0, -0.2, 0.4]))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("misago-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try weights.save(to: url)
        let loaded = try HawkWeights.load(from: url)
        XCTAssertEqual(loaded, weights)
        XCTAssertEqual(loaded.calibrationScale, 0.7)
        XCTAssertEqual(loaded.calibrationBias, [0.0, -0.2, 0.4])
        XCTAssertEqual(loaded.classNames, ["other", "a_home", "b_home"])
        let reloaded = MLXHawkNetwork(weights: loaded)
        XCTAssertEqual(reloaded.exportWeights(classes: smallClasses).wIn, weights.wIn)
    }

    /// The trainer separates three synthetic screen classes. The loss sits near ln 3 for the first 100 or so
    /// steps before dropping (and a learning rate of 0.005 without warmup collapses to a uniform output), so the
    /// budget is generous.
    func testOverfitsSyntheticScreens() {
        let classes = try! ClassList(others: ["a_home", "b_home"])
        let net = MLXHawkNetwork(classCount: classes.count)
        let trainer = MLXClassificationTrainer(network: net, learningRate: 0.002)
        var features: [[Float]] = []
        var labels: [Int] = []
        var i = 0
        while i < 18 {
            var pixels = [Float](repeating: 0.0, count: ScreenImage.pixelCount)
            var p = 0
            while p < pixels.count {
                pixels[p] = Float((p * 31 + i * 7) % 97) / 97.0 * 0.3
                p += 1
            }
            // Class 1 carries a bright band in the top rows, class 2 a bright band in the bottom rows.
            let label = i % 3
            if label == 1 {
                var x = 0
                while x < ScreenImage.width * 4 * 3 {
                    pixels[x] = 0.9
                    x += 1
                }
            }
            if label == 2 {
                var x = ScreenImage.pixelCount - ScreenImage.width * 4 * 3
                while x < ScreenImage.pixelCount {
                    pixels[x] = 0.9
                    x += 1
                }
            }
            features.append(pixels)
            labels.append(label)
            i += 1
        }
        var loss: Float = 0.0
        var step = 0
        while step < 300 {
            loss = trainer.trainBatch(features: features, labels: labels)
            step += 1
        }
        XCTAssertLessThan(loss, 0.1)
        let result = trainer.evaluateBatch(features: features, labels: labels)
        XCTAssertEqual(result.correct, 18)
        // The exported weights classify the same on the CPU.
        let classifier = try! ImageClassifier(weights: net.exportWeights(classes: classes))
        var correct = 0
        i = 0
        while i < features.count {
            let verdict = classifier.classify(try! ScreenImage(pixels: features[i]))
            if verdict.classIndex == labels[i] {
                correct += 1
            }
            XCTAssertEqual(verdict.className, classes[verdict.classIndex])
            XCTAssertEqual(verdict.probabilities.reduce(0.0, +), 1.0, accuracy: 1e-4)
            i += 1
        }
        XCTAssertEqual(correct, 18)
    }
}

extension HawkTests {
    /// Reference scan as an explicit time loop (differentiable through the unrolled graph).
    private func sequentialScan(_ a: MLXArray, _ x: MLXArray) -> MLXArray {
        var h = MLXArray.zeros([a.shape[0], a.shape[2]])
        var outs: [MLXArray] = []
        var t = 0
        while t < a.shape[1] {
            h = a[0..., t, 0...] * h + x[0..., t, 0...]
            outs.append(h)
            t += 1
        }
        return stacked(outs, axis: 1)
    }

    func testLinearScanMatchesSequentialForwardAndGradient() {
        let a = MLXRandom.uniform(low: 0.2, high: 0.99, [3, 41, 10])
        let x = MLXRandom.normal([3, 41, 10])
        let weight = MLXRandom.normal([3, 41, 10])
        func close(_ p: MLXArray, _ q: MLXArray, _ tol: Float) {
            eval(p, q)
            let u = p.asArray(Float.self)
            let v = q.asArray(Float.self)
            XCTAssertEqual(u.count, v.count)
            var worst: Float = 0.0
            var i = 0
            while i < u.count {
                worst = max(worst, abs(u[i] - v[i]))
                i += 1
            }
            XCTAssertLessThan(worst, tol)
        }
        close(MLXLinearScan.scan(a, x), sequentialScan(a, x), 1e-4)
        let kernelGrad = grad({ (arrays: [MLXArray]) -> MLXArray in
            sum(MLXLinearScan.scan(arrays[0], arrays[1]) * weight)
        })([a, x])
        let referenceGrad = grad({ (arrays: [MLXArray]) -> MLXArray in
            sum(self.sequentialScan(arrays[0], arrays[1]) * weight)
        })([a, x])
        close(kernelGrad[0], referenceGrad[0], 1e-3)
        close(kernelGrad[1], referenceGrad[1], 1e-3)
    }
}
