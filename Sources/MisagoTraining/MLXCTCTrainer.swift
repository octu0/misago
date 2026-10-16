import Foundation
import Misago
import MLX
import MLXNN
import MLXOptimizers

/// CTC training of the digit reader (stem + Hawk). Every strip has the same size, so compiled steps are
/// keyed by batch size and padded label length.
public final class MLXCTCTrainer: @unchecked Sendable {
    public let network: MLXDigitModel
    public let optimizer: ArrayLRAdam
    public let clipNorm: Float

    public private(set) var lastGradientNorm: Float = 0.0

    private var compiledSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]
    private var compiledLogits: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    public static let cacheLimitBytes = 8 << 30

    public init(network: MLXDigitModel, learningRate: Float, clipNorm: Float = 5.0) {
        self.network = network
        self.optimizer = ArrayLRAdam(learningRate: learningRate)
        self.clipNorm = clipNorm
        MLX.Memory.cacheLimit = Self.cacheLimitBytes
    }

    public func updateLearningRate(_ lr: Float) {
        optimizer.learningRate = MLXArray(lr)
    }

    /// One CTC step over strip pixels `features` [B][56 * 320 * 3] with digit strings `labels` [B]. Returns the
    /// mean loss. Samples whose label is not digits-only are skipped.
    public func trainBatch(features: [[Float]], labels: [String]) -> Float {
        var validFeatures: [[Float]] = []
        var targets: [[Int]] = []
        var b = 0
        while b < min(features.count, labels.count) {
            if let ids = DigitVocabulary.ids(of: labels[b]) {
                validFeatures.append(features[b])
                targets.append(ids)
            }
            b += 1
        }
        guard let packed = pack(features: validFeatures) else {
            return 0.0
        }
        let frameCounts = [Int](repeating: ConvStemWeights.frameCount, count: packed.batchSize)
        let ext = MLXCTCLoss.ExtendedTargets(targetsBatch: targets, frameCounts: frameCounts, blankId: DigitVocabulary.blankId)
        let key = (packed.batchSize << 16) | ext.maxExtendedLength

        let lossFn = { (model: MLXDigitModel, arrays: [MLXArray]) -> [MLXArray] in
            let targets = MLXCTCLoss.ExtendedTargets(
                extTargets: arrays[1], skipMask: arrays[2], validMask: arrays[3],
                finalIndex1: arrays[4], finalIndex2: arrays[5], hasSecondFinal: arrays[6], inputLengths: arrays[7]
            )
            return [MLXCTCLoss.loss(logits: model.frameLogits(arrays[0]), targets: targets)]
        }
        let lg = valueAndGrad(model: network, lossFn)
        let maxNorm = clipNorm
        func step(arrays: [MLXArray]) -> [MLXArray] {
            let (lossValues, grads) = lg(self.network, Array(arrays[0..<8]))
            let (clipped, norm) = clipGradNorm(gradients: grads, maxNorm: maxNorm)
            let outer = self.optimizer.learningRate
            self.optimizer.learningRate = arrays[8]
            self.optimizer.update(model: self.network, gradients: clipped)
            self.optimizer.learningRate = outer
            return lossValues + [norm]
        }
        let stepFn: ([MLXArray]) -> [MLXArray]
        switch compiledSteps[key] {
        case .some(let cached):
            stepFn = cached
        case .none:
            let fn = compile(inputs: [network, optimizer], outputs: [network, optimizer], step)
            compiledSteps[key] = fn
            stepFn = fn
        }
        let lossValues = stepFn([packed.features] + ext.toArrays() + [optimizer.learningRate])
        eval(network, optimizer, lossValues)
        lastGradientNorm = lossValues[1].item(Float.self)
        return lossValues[0].item(Float.self)
    }

    /// Per-frame logits [B][T][V] without gradients.
    public func frameLogits(features: [[Float]]) -> [[[Float]]] {
        guard let packed = pack(features: features) else {
            return []
        }
        let fn: ([MLXArray]) -> [MLXArray]
        switch compiledLogits[packed.batchSize] {
        case .some(let cached):
            fn = cached
        case .none:
            let newFn = compile(inputs: [network]) { arrays in
                return [self.network.frameLogits(arrays[0])]
            }
            compiledLogits[packed.batchSize] = newFn
            fn = newFn
        }
        let out = fn([packed.features])[0]
        eval(out)
        let flat = out.asArray(Float.self)
        let frames = ConvStemWeights.frameCount
        let dim = DigitVocabulary.size
        var result: [[[Float]]] = []
        result.reserveCapacity(packed.batchSize)
        var b = 0
        while b < packed.batchSize {
            var rows: [[Float]] = []
            rows.reserveCapacity(frames)
            var t = 0
            while t < frames {
                let start = (b * frames + t) * dim
                rows.append(Array(flat[start..<(start + dim)]))
                t += 1
            }
            result.append(rows)
            b += 1
        }
        return result
    }

    /// Greedy decodes of a batch.
    public func read(features: [[Float]]) -> [DigitReading] {
        return frameLogits(features: features).map { DigitReader.greedyDecode(logits: $0) }
    }

    private func pack(features: [[Float]]) -> (features: MLXArray, batchSize: Int)? {
        let bSize = features.count
        if bSize == 0 {
            return nil
        }
        let perSample = DigitStrip.pixelCount
        var flat = [Float](repeating: 0.0, count: bSize * perSample)
        var b = 0
        while b < bSize {
            let seq = features[b]
            let n = min(perSample, seq.count)
            var i = 0
            while i < n {
                flat[b * perSample + i] = seq[i]
                i += 1
            }
            b += 1
        }
        return (MLXArray(flat, [bSize, DigitStrip.height, DigitStrip.width, DigitStrip.channels]), bSize)
    }
}

/// Levenshtein distance between two digit strings, for the character error rate.
public func editDistance(_ a: String, _ b: String) -> Int {
    let x = Array(a)
    let y = Array(b)
    if x.isEmpty {
        return y.count
    }
    if y.isEmpty {
        return x.count
    }
    var previous = Array(0...y.count)
    var current = [Int](repeating: 0, count: y.count + 1)
    var i = 1
    while i <= x.count {
        current[0] = i
        var j = 1
        while j <= y.count {
            let cost = x[i - 1] == y[j - 1] ? 0 : 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            j += 1
        }
        swap(&previous, &current)
        i += 1
    }
    return previous[y.count]
}
