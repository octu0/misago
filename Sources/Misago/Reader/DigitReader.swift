import Foundation

/// Result of reading one number field.
public struct DigitReading: Sendable, Equatable {
    /// The digits read, possibly empty.
    public let digits: String
    /// Mean over emitted digits of the frame posterior at the emitting frame; 0 when nothing was emitted.
    public let confidence: Float

    /// The digits as an integer, or nil when empty or too long to represent.
    public var value: Int? {
        if digits.isEmpty || 18 < digits.count {
            return nil
        }
        return Int(digits)
    }
}

/// Strip → convolution stem → Hawk → greedy CTC decoding over the digit vocabulary.
public final class DigitReader: @unchecked Sendable {
    public let weights: DigitReaderWeights
    public let network: HawkNetwork

    public init(weights: DigitReaderWeights) throws {
        if weights.hawk.classNames != DigitVocabulary.names {
            throw ClassListError.invalidName(weights.hawk.classNames.joined(separator: ","))
        }
        if weights.hawk.inputDim != ConvStemWeights.frameDim {
            throw ScreenImageError.invalidSize(width: weights.hawk.inputDim, height: ConvStemWeights.frameDim)
        }
        self.weights = weights
        self.network = HawkNetwork(weights: weights.hawk)
    }

    public func read(_ strip: DigitStrip) -> DigitReading {
        let frames = ConvStem.frames(strip.pixels, weights: weights.stem)
        let logits = network.frameLogits(flatFeatures: frames, frames: ConvStemWeights.frameCount)
        return Self.greedyDecode(logits: logits)
    }

    /// Greedy CTC: argmax per frame, collapse repeats, drop blanks.
    public static func greedyDecode(logits: [[Float]]) -> DigitReading {
        var ids: [Int] = []
        var confidences: [Float] = []
        var previous = DigitVocabulary.blankId
        for frame in logits {
            let probs = VectorCalibration.softmax(frame)
            var best = 0
            var c = 1
            while c < probs.count {
                if probs[best] < probs[c] {
                    best = c
                }
                c += 1
            }
            if best != DigitVocabulary.blankId && best != previous {
                ids.append(best)
                confidences.append(probs[best])
            }
            previous = best
        }
        var confidence: Float = 0.0
        if confidences.isEmpty != true {
            confidence = confidences.reduce(0.0, +) / Float(confidences.count)
        }
        return DigitReading(digits: DigitVocabulary.string(of: ids), confidence: confidence)
    }
}
