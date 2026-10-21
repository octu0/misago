import Foundation

/// Result of classifying one image.
public struct ImageVerdict: Sendable, Equatable {
    /// Index into the class list (rotation tags merged).
    public let classIndex: Int
    public let className: String
    /// Calibrated probability of `classIndex`, summed over its rotations.
    public let probability: Float
    /// Calibrated probabilities of every class, each summed over its rotations.
    public let probabilities: [Float]
    /// Raw logits of every model output (one per class and rotation).
    public let logits: [Float]
    /// Clockwise rotation (degrees) that turns the image upright: the inverse of the most probable rotation tag
    /// of the winning class. 0 for `other` and for models trained without rotation tags.
    public let rotation: Int
    /// Title of the winning class and its independent probability (is this a screen of that title at all?).
    public let title: String
    public let titleProbability: Float
    /// Independent probability of every title; all near 0 for an image of nothing the model knows.
    public let titleProbabilities: [String: Float]

    public init(
        classIndex: Int, className: String, probability: Float, probabilities: [Float], logits: [Float], rotation: Int = 0,
        title: String? = nil, titleProbability: Float = 1.0, titleProbabilities: [String: Float] = [:]
    ) {
        self.classIndex = classIndex
        self.className = className
        self.probability = probability
        self.probabilities = probabilities
        self.logits = logits
        self.rotation = rotation
        self.title = title ?? ClassList.title(of: className)
        self.titleProbability = titleProbability
        self.titleProbabilities = titleProbabilities
    }

    /// The image is a screen of the winning class's title: its title probability is at least one half (the
    /// natural cut of an independent probability). Numbers, outcome and match state are only read when it is.
    public var recognized: Bool {
        return 0.5 <= titleProbability
    }

    /// Match outcome encoded in the class name: `_win` → 2, `_lose` → 1, `_draw` → 0, anything else −1
    /// (not a result screen). Datasets name their classes, so this convention keeps the code free of game names.
    public var outcome: Int {
        if recognized != true {
            return -1
        }
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

    /// 1 while a match is being played, 0 otherwise: class names ending in `_battle` or containing `_battle_`
    /// (`<title>_battle` for any in-match screen, `<title>_battle_<mode>` when the mode is known).
    public var battle: Int {
        if recognized != true {
            return 0
        }
        return className.hasSuffix("_battle") || className.contains("_battle_") ? 1 : 0
    }
}

/// Image → row-scan frames → Hawk → calibrated probabilities: an independent probability per title (near 0 for
/// anything the model was not trained on) and, among the screens, a softmax. The most probable screen wins;
/// its title probability says whether the image is a screen of that title at all.
///
/// Models may be trained on rotated copies tagged `<class>@<degrees>`: their outputs are merged per class, and
/// the tag of the winning class gives the image's orientation in the same pass.
public final class ImageClassifier: @unchecked Sendable {
    public let network: HawkNetwork
    /// Classes with rotation tags merged (what callers see).
    public let classes: ClassList
    /// Model outputs, rotation tags included.
    public let outputClasses: ClassList
    /// Per output: index into `classes` and the tagged rotation.
    private let outputMap: [(classIndex: Int, rotation: Int)]
    /// Independent outputs after the screens.
    public let titles: [String]

    public init(weights: HawkWeights) throws {
        self.network = HawkNetwork(weights: weights)
        self.outputClasses = try ClassList(names: weights.classNames)
        var baseNames: [String] = []
        var map: [(classIndex: Int, rotation: Int)] = []
        for name in weights.classNames {
            let parsed = ClassList.parseRotation(name)
            if let index = baseNames.firstIndex(of: parsed.base) {
                map.append((index, parsed.rotation))
            } else {
                baseNames.append(parsed.base)
                map.append((baseNames.count - 1, parsed.rotation))
            }
        }
        self.classes = try ClassList(names: baseNames)
        self.outputMap = map
        self.titles = weights.titleNames
    }

    public func classify(_ image: ScreenImage) -> ImageVerdict {
        var input = image
        if let mask = network.weights.centerMask {
            input = image.masked(mask)
        }
        let result = network.logitsAndPooled(flatFeatures: RowScanEncoder.encodeFlat(input), frames: RowScanEncoder.frameCount)
        let logits = result.logits
        let outputs = network.probabilities(logits: logits)
        let titleValues = network.titleProbabilities(logits: logits, pooled: result.pooled)
        var probabilities = [Float](repeating: 0.0, count: classes.count)
        var o = 0
        while o < outputs.count {
            probabilities[outputMap[o].classIndex] += outputs[o]
            o += 1
        }
        var best = 0
        var c = 1
        while c < probabilities.count {
            if probabilities[best] < probabilities[c] {
                best = c
            }
            c += 1
        }
        // Orientation: the most probable rotation tag of the winning class.
        var tag = 0
        var tagProbability: Float = -1.0
        o = 0
        while o < outputs.count {
            if outputMap[o].classIndex == best && tagProbability < outputs[o] {
                tagProbability = outputs[o]
                tag = outputMap[o].rotation
            }
            o += 1
        }
        var titleProbabilities: [String: Float] = [:]
        var t = 0
        while t < titles.count {
            titleProbabilities[titles[t]] = titleValues[t]
            t += 1
        }
        let title = ClassList.title(of: classes[best])
        // A model without title outputs vouches for every class it knows.
        let titleProbability = titles.isEmpty ? 1.0 : (titleProbabilities[title] ?? 0.0)
        return ImageVerdict(
            classIndex: best,
            className: classes[best],
            probability: probabilities[best],
            probabilities: probabilities,
            logits: logits,
            rotation: (360 - tag) % 360,
            title: title,
            titleProbability: titleProbability,
            titleProbabilities: titleProbabilities
        )
    }
}
