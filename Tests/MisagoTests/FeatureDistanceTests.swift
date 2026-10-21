import XCTest
@testable import Misago

final class FeatureDistanceTests: XCTestCase {
    /// Two clusters with a stretched shared covariance: a point far along the narrow axis is farther (in
    /// Mahalanobis terms) than one equally far along the wide axis, and both cluster members stay close.
    func testDistanceFollowsTheSharedCovariance() throws {
        var features: [[Float]] = []
        var labels: [Int] = []
        var state: UInt64 = 1
        func noise() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: state >> 11) % 2000) / 1000.0 - 1.0
        }
        for i in 0..<400 {
            let c = i % 2
            let center: Float = c == 0 ? -5 : 5
            features.append([center + 3.0 * noise(), 0.3 * noise()])
            labels.append(c)
        }
        let distance = try XCTUnwrap(FeatureDistance.fit(features: features, labels: labels, classCount: 2, shrinkage: 0.0))
        let nearMember = distance.distances([5.0, 0.0])
        XCTAssertLessThan(nearMember[1], 0.5)
        let alongWide = distance.distances([Float(8.0), 0.0])[1]
        let alongNarrow = distance.distances([5.0, 3.0])[1]
        XCTAssertGreaterThan(alongNarrow, 4 * alongWide)
    }

    func testLogisticRegressionRecoversTwoWeights() {
        var inputs: [[Float]] = []
        var targets: [Bool] = []
        for i in 0..<8000 {
            let a = Float((i * 37) % 400) / 40.0 - 5.0
            let b = Float((i * 91) % 400) / 40.0 - 5.0
            let p = 1.0 / (1.0 + exp(-(1.0 * a - 0.5 * b + 0.5)))
            inputs.append([a, b])
            targets.append(Float((i * 7919) % 1000) / 1000.0 < p)
        }
        let fit = LogisticRegression.fit(inputs: inputs, targets: targets)
        XCTAssertEqual(fit.weights[0], 1.0, accuracy: 0.12)
        XCTAssertEqual(fit.weights[1], -0.5, accuracy: 0.12)
        XCTAssertEqual(fit.bias, 0.5, accuracy: 0.15)
    }
}
