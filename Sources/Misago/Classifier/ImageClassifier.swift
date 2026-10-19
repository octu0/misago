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

    /// Match outcome encoded in the class name: `_win` → 2, `_lose` → 1, `_draw` → 0, anything else −1
    /// (not a result screen). Datasets name their classes, so this convention keeps the code free of game names.
    public var outcome: Int {
        if className.hasSuffix("_win") {
            return 2
        }
        if className.hasSuffix("_lose") {
            return 1
        }
        if className.hasSuffix("_draw") {
            return 0
        }
        return -1
    }
}

/// Image → row-scan frames → Hawk → calibrated class probabilities. The most probable class wins; every
/// class, `other` included, is a screen of the target, and the probabilities are returned for callers that
/// want their own cut-off.
public final class ImageClassifier: @unchecked Sendable {
    public let network: HawkNetwork
    public let classes: ClassList
    public init(weights: HawkWeights) throws {
        self.network = HawkNetwork(weights: weights)
        self.classes = try ClassList(names: weights.classNames)
    }

    public func classify(_ image: ScreenImage) -> ImageVerdict {
        var input = image
        if let mask = network.weights.centerMask {
            input = image.masked(mask)
        }
        let logits = network.logits(flatFeatures: RowScanEncoder.encodeFlat(input), frames: RowScanEncoder.frameCount)
        let probabilities = network.probabilities(logits: logits)
        var best = 0
        var c = 1
        while c < probabilities.count {
            if probabilities[best] < probabilities[c] {
                best = c
            }
            c += 1
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
