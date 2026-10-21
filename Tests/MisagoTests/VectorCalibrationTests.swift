import XCTest
@testable import Misago

final class VectorCalibrationTests: XCTestCase {
    /// Labels drawn deterministically from softmax(0.5 · logits + [0, −1, 0.5]): the fit should recover those parameters.
    func testFitRecoversScaleAndBias() {
        let trueCalibration = VectorCalibration(scale: 0.5, bias: [0.0, -1.0, 0.5])
        var logits: [[Float]] = []
        var labels: [Int] = []
        var i = 0
        while i < 6000 {
            let row: [Float] = [Float((i * 37) % 200) / 10.0 - 10.0, Float((i * 91) % 200) / 10.0 - 10.0, Float((i * 53) % 200) / 10.0 - 10.0]
            let p = trueCalibration.probabilities(row)
            let u = Float((i * 7919) % 1000) / 1000.0
            var label = 0
            var cumulative: Float = 0.0
            var c = 0
            while c < p.count {
                cumulative += p[c]
                if u < cumulative {
                    label = c
                    break
                }
                label = c
                c += 1
            }
            logits.append(row)
            labels.append(label)
            i += 1
        }
        let fit = VectorCalibration.fit(logits: logits, labels: labels, classCount: 3)
        XCTAssertEqual(fit.scale, 0.5, accuracy: 0.05)
        XCTAssertEqual(fit.bias[0], 0.0)
        XCTAssertEqual(fit.bias[1], -1.0, accuracy: 0.2)
        XCTAssertEqual(fit.bias[2], 0.5, accuracy: 0.2)
    }

    /// Temperature scaling only: data drawn from softmax(0.5 · logits) gives scale 0.5 and no bias.
    func testFitScaleOnlyKeepsBiasZero() {
        let trueCalibration = VectorCalibration(scale: 0.5, bias: [0.0, 0.0, 0.0])
        var logits: [[Float]] = []
        var labels: [Int] = []
        var i = 0
        while i < 6000 {
            let row: [Float] = [Float((i * 37) % 200) / 10.0 - 10.0, Float((i * 91) % 200) / 10.0 - 10.0, Float((i * 53) % 200) / 10.0 - 10.0]
            let p = trueCalibration.probabilities(row)
            let u = Float((i * 7919) % 1000) / 1000.0
            var label = p.count - 1
            var cumulative: Float = 0.0
            var c = 0
            while c < p.count {
                cumulative += p[c]
                if u < cumulative {
                    label = c
                    break
                }
                c += 1
            }
            logits.append(row)
            labels.append(label)
            i += 1
        }
        let fit = VectorCalibration.fit(logits: logits, labels: labels, classCount: 3, fitBias: false)
        XCTAssertEqual(fit.scale, 0.5, accuracy: 0.05)
        XCTAssertEqual(fit.bias, [0.0, 0.0, 0.0])
    }

    /// Labels drawn from σ(0.5 · x − 1): Platt scaling recovers the scale and bias.
    func testPlattScalingRecoversScaleAndBias() {
        var logits: [Float] = []
        var targets: [Bool] = []
        var i = 0
        while i < 8000 {
            let x = Float((i * 37) % 400) / 20.0 - 10.0
            let p = 1.0 / (1.0 + exp(-(0.5 * x - 1.0)))
            let u = Float((i * 7919) % 1000) / 1000.0
            logits.append(x)
            targets.append(u < p)
            i += 1
        }
        let fit = PlattScaling.fit(logits: logits, targets: targets)
        XCTAssertEqual(fit.scale, 0.5, accuracy: 0.05)
        XCTAssertEqual(fit.bias, -1.0, accuracy: 0.15)
        XCTAssertEqual(PlattScaling.fit(logits: [1, 2], targets: [true, true]).scale, 1.0)
    }

    func testSeparableDataStaysFinite() {
        let logits: [[Float]] = [[3, -3], [2, -2], [1, -1], [-1, 1], [-2, 2], [-3, 3]]
        let labels = [0, 0, 0, 1, 1, 1]
        let fit = VectorCalibration.fit(logits: logits, labels: labels, classCount: 2)
        XCTAssertTrue(fit.scale.isFinite && fit.bias.allSatisfy { $0.isFinite })
        // Smoothed targets for 3 + 3 samples are 0.8, so the scale stays moderate and probabilities do not saturate.
        XCTAssertLessThan(fit.scale, 10.0)
        XCTAssertGreaterThan(fit.probabilities([3, -3])[0], fit.probabilities([1, -1])[0])
        XCTAssertGreaterThan(fit.probabilities([-3, 3])[1], 0.7)
        XCTAssertLessThan(fit.probabilities([-3, 3])[1], 0.95)
    }

    func testDegenerateInputsReturnIdentity() {
        XCTAssertEqual(VectorCalibration.fit(logits: [[1.0, 0.0]], labels: [1], classCount: 2), VectorCalibration.identity(classCount: 2))
        XCTAssertEqual(VectorCalibration.fit(logits: [[1.0], [2.0]], labels: [0, 0], classCount: 1), VectorCalibration.identity(classCount: 1))
    }

    func testProbabilitiesSumToOne() {
        let p = VectorCalibration(scale: 2.0, bias: [0.0, 1.0, -1.0]).probabilities([0.3, -0.2, 1.5])
        XCTAssertEqual(p.reduce(0.0, +), 1.0, accuracy: 1e-5)
        XCTAssertEqual(VectorCalibration.identity(classCount: 2).probabilities([0.0, 0.0]), [0.5, 0.5])
    }

    func testExpectedCalibrationError() {
        let rows: [[Float]] = [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]
        let perfect = VectorCalibration.expectedCalibrationError(probabilities: rows, labels: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        XCTAssertEqual(perfect.ece, 0.0, accuracy: 1e-6)
        let overconfident = VectorCalibration.expectedCalibrationError(probabilities: [[0.05, 0.95], [0.05, 0.95]], labels: [1, 0])
        XCTAssertEqual(overconfident.ece, 0.45, accuracy: 1e-6)
    }
}
