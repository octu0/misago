import Foundation

/// Result of classifying one image.
public struct ImageVerdict: Sendable, Equatable {
    /// Index into the class list; 0 is `other`.
    public let classIndex: Int
    public let className: String
    /// Calibrated probability of `classIndex`.
    public let probability: Float
    /// Calibrated probabilities of every class.
    public let probabilities: [Float]
    /// Raw logits.
    public let logits: [Float]

    public var isOther: Bool {
        return classIndex == 0
    }
}

/// Image → row-scan frames → Hawk → calibrated class probabilities.
///
/// The most probable class wins. When it is not `other` but its probability is below `threshold`, the
/// verdict falls back to `other` so that callers can treat low-confidence matches as "not recognized".
public final class ImageClassifier: @unchecked Sendable {
    public let network: HawkNetwork
    public let classes: ClassList
    public let threshold: Float

    public init(weights: HawkWeights, threshold: Float = 0.5) throws {
        self.network = HawkNetwork(weights: weights)
        self.classes = try ClassList(names: weights.classNames)
        self.threshold = threshold
    }

    public func classify(_ image: ScreenImage) -> ImageVerdict {
        let logits = network.logits(flatFeatures: RowScanEncoder.encodeFlat(image), frames: RowScanEncoder.frameCount)
        let probabilities = network.probabilities(logits: logits)
        var best = 0
        var c = 1
        while c < probabilities.count {
            if probabilities[best] < probabilities[c] {
                best = c
            }
            c += 1
        }
        if best != 0 && probabilities[best] < threshold {
            best = 0
        }
        return ImageVerdict(
            classIndex: best,
            className: classes[best],
            probability: probabilities[best],
            probabilities: probabilities,
            logits: logits
        )
    }
}
