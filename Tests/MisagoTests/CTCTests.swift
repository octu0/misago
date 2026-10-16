import XCTest
import MLX
import MLXNN
@testable import Misago
@testable import MisagoTraining

final class CTCTests: XCTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        try MLXTestEnvironment.skipIfMetallibUnavailable()
        Device.setDefault(device: .gpu)
        MLXRandom.seed(3)
    }

    /// Brute-force CTC likelihood: sum over all alignments of length T that collapse to the labels.
    private func bruteForce(logProbs: [[Float]], labels: [Int], blank: Int) -> Double {
        let t = logProbs.count
        let v = logProbs[0].count
        var total = 0.0
        var path = [Int](repeating: 0, count: t)
        func collapse(_ p: [Int]) -> [Int] {
            var out: [Int] = []
            var prev = -1
            for s in p {
                if s != blank && s != prev {
                    out.append(s)
                }
                prev = s
            }
            return out
        }
        func recurse(_ i: Int, _ acc: Double) {
            if i == t {
                if collapse(path) == labels {
                    total += exp(acc)
                }
                return
            }
            var s = 0
            while s < v {
                path[i] = s
                recurse(i + 1, acc + Double(logProbs[i][s]))
                s += 1
            }
        }
        recurse(0, 0.0)
        return total
    }

    func testLossMatchesBruteForce() {
        let t = 5
        let v = 3
        let raw = MLXRandom.normal([2, t, v])
        eval(raw)
        let logProbs = logSoftmax(raw, axis: -1)
        eval(logProbs)
        let flat = logProbs.asArray(Float.self)
        var perSample: [[[Float]]] = []
        var b = 0
        while b < 2 {
            var rows: [[Float]] = []
            var i = 0
            while i < t {
                rows.append(Array(flat[((b * t + i) * v)..<((b * t + i + 1) * v)]))
                i += 1
            }
            perSample.append(rows)
            b += 1
        }
        let labels = [[1, 2, 1], [2, 2]]
        let targets = MLXCTCLoss.ExtendedTargets(targetsBatch: labels, frameCounts: [t, t], blankId: 0)
        let loss = MLXCTCLoss.loss(logits: raw, targets: targets)
        eval(loss)
        let expected = -(log(bruteForce(logProbs: perSample[0], labels: labels[0], blank: 0)) + log(bruteForce(logProbs: perSample[1], labels: labels[1], blank: 0))) / 2.0
        XCTAssertEqual(Double(loss.item(Float.self)), expected, accuracy: 1e-3)
    }

    /// The CPU stem reproduces the MLX convolution on random weights and a random strip.
    func testStemMatchesBetweenMLXAndCPU() {
        let model = MLXDigitModel()
        let weights = model.exportWeights()
        let strip = try! DigitStrip(pixels: (0..<DigitStrip.pixelCount).map { Float(($0 * 37) % 101) / 101.0 })
        let mlxFrames = model.frames(MLXArray(strip.pixels, [1, DigitStrip.height, DigitStrip.width, DigitStrip.channels]))
        eval(mlxFrames)
        let want = mlxFrames.asArray(Float.self)
        let got = ConvStem.frames(strip.pixels, weights: weights.stem)
        XCTAssertEqual(got.count, want.count)
        var worst: Float = 0.0
        var i = 0
        while i < got.count {
            worst = max(worst, abs(got[i] - want[i]))
            i += 1
        }
        XCTAssertLessThan(worst, 1e-4)
        // Whole reader: per-frame logits agree too.
        let logits = model.frameLogits(MLXArray(strip.pixels, [1, DigitStrip.height, DigitStrip.width, DigitStrip.channels]))
        eval(logits)
        let cpuLogits = try! DigitReader(weights: weights).network.frameLogits(flatFeatures: got, frames: ConvStemWeights.frameCount).flatMap { $0 }
        let wantLogits = logits.asArray(Float.self)
        worst = 0.0
        i = 0
        while i < cpuLogits.count {
            worst = max(worst, abs(cpuLogits[i] - wantLogits[i]))
            i += 1
        }
        XCTAssertLessThan(worst, 2e-4)
    }

    /// Synthetic strips: each digit d is a bright block whose color encodes d, placed left to right.
    /// The reader must learn to emit the digits in order.
    func testOverfitsSyntheticStrips() {
        func strip(_ digits: [Int], offset: Int) -> DigitStrip {
            var p = [Float](repeating: 0.05, count: DigitStrip.pixelCount)
            var k = 0
            while k < digits.count {
                let d = digits[k]
                let x0 = offset + k * 24
                var y = 10
                while y < 46 {
                    var x = x0
                    while x < x0 + 14 && x < DigitStrip.width {
                        let i = (y * DigitStrip.width + x) * 3
                        p[i] = Float(d) / 9.0
                        p[i + 1] = 1.0 - Float(d) / 9.0
                        p[i + 2] = Float(d % 3) / 2.0
                        x += 1
                    }
                    y += 1
                }
                k += 1
            }
            return try! DigitStrip(pixels: p)
        }
        var features: [[Float]] = []
        var labels: [String] = []
        let sequences: [[Int]] = [[1, 6, 0, 4], [2, 1, 3], [9, 5, 0], [7], [8, 0, 2, 4, 6], [5, 4, 3, 2, 1]]
        var i = 0
        while i < sequences.count {
            features.append(strip(sequences[i], offset: 4 + (i % 3) * 6).pixels)
            labels.append(sequences[i].map { String($0) }.joined())
            i += 1
        }
        let net = MLXDigitModel()
        let trainer = MLXCTCTrainer(network: net, learningRate: 0.002)
        var loss: Float = 0.0
        var step = 0
        while step < 1500 {
            loss = trainer.trainBatch(features: features, labels: labels)
            step += 1
        }
        XCTAssertLessThan(loss, 0.6)
        // Convergence on this toy task is not deterministic on the GPU (short sequences near the left edge are
        // the last to be learned); two thirds must be right. The real guarantee below is CPU/MLX agreement.
        let readings = trainer.read(features: features)
        var exact = 0
        var j = 0
        while j < readings.count {
            if readings[j].digits == labels[j] {
                exact += 1
            }
            j += 1
        }
        XCTAssertGreaterThanOrEqual(exact, 4, "\(readings.map { $0.digits })")
        // The CPU reader agrees with MLX, frame by frame and after decoding.
        let exported = net.exportWeights()
        let cpu = try! DigitReader(weights: exported)
        let mlxLogits = trainer.frameLogits(features: features)
        i = 0
        while i < sequences.count {
            let s = strip(sequences[i], offset: 4 + (i % 3) * 6)
            let cpuFrames = ConvStem.frames(s.pixels, weights: exported.stem)
            let cpuLogits = cpu.network.frameLogits(flatFeatures: cpuFrames, frames: ConvStemWeights.frameCount)
            var worst: Float = 0.0
            var t = 0
            while t < cpuLogits.count {
                var c = 0
                while c < cpuLogits[t].count {
                    worst = max(worst, abs(cpuLogits[t][c] - mlxLogits[i][t][c]))
                    c += 1
                }
                t += 1
            }
            XCTAssertLessThan(worst, 1e-2, "sample \(i) (\(labels[i])) worst frame logit difference")
            // Whatever MLX reads, the CPU reads the same.
            XCTAssertEqual(cpu.read(s).digits, readings[i].digits)
            i += 1
        }
    }
}
