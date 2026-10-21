import Foundation
import Misago
import MLX
import MLXNN
import MLXOptimizers

/// Training of the Hawk screen classifier. Every sample is the same fixed-length frame sequence, so the compiled
/// step is keyed by batch size only.
///
/// With titles, the outputs are the screens followed by one independent output per title. The loss is the
/// screen cross-entropy (over samples that have a screen; label −1 means none) plus a binary cross-entropy per
/// title (1 for the sample's own title, 0 for every other title and for samples labeled none).
public final class MLXClassificationTrainer: @unchecked Sendable {
    public let network: MLXHawkNetwork
    public let optimizer: ArrayLRAdam
    public let clipNorm: Float
    /// Share of the target spread uniformly over all classes, so logits stop growing once the data is fit.
    public let labelSmoothing: Float
    /// Number of screen outputs; the remaining outputs are titles.
    public let screenCount: Int
    /// [screenCount, titles] 0/1: the title of each screen. Nil without titles.
    private let titleMatrix: MLXArray?

    /// Gradient norm of the most recent batch, before clipping.
    public private(set) var lastGradientNorm: Float = 0.0

    private var compiledSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]
    private var compiledLogits: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    /// MLX buffer cache limit; one batch shape is reused for the whole run.
    public static let cacheLimitBytes = 8 << 30

    /// `titleOf[c]` is the title index of screen `c`; empty for a plain classifier (all outputs are screens).
    public init(network: MLXHawkNetwork, learningRate: Float, clipNorm: Float = 5.0, labelSmoothing: Float = 0.0, weightDecay: Float = 0.0, biasCorrection: Bool = false, titleOf: [Int] = []) {
        self.network = network
        self.screenCount = titleOf.isEmpty ? network.outputDim : titleOf.count
        if titleOf.isEmpty {
            self.titleMatrix = nil
        } else {
            let titleCount = network.outputDim - titleOf.count
            var m = [Float](repeating: 0.0, count: titleOf.count * titleCount)
            for (c, t) in titleOf.enumerated() {
                m[c * titleCount + t] = 1.0
            }
            self.titleMatrix = MLXArray(m, [titleOf.count, titleCount])
        }
        self.optimizer = ArrayLRAdam(learningRate: learningRate, biasCorrection: biasCorrection, weightDecay: weightDecay)
        self.clipNorm = clipNorm
        self.labelSmoothing = labelSmoothing
        MLX.Memory.cacheLimit = Self.cacheLimitBytes
    }

    public func updateLearningRate(_ lr: Float) {
        optimizer.learningRate = MLXArray(lr)
    }

    /// Restores saved weights and resets the Adam moments.
    public func rollback(to weights: HawkWeights) {
        network.importWeights(from: weights)
        optimizer.resetState()
        compiledSteps.removeAll()
        compiledLogits.removeAll()
        eval(network, optimizer)
    }

    /// One training step over `features` [B][T * inputDim] with class `labels` [B]. Returns the mean loss.
    public func trainBatch(features: [[Float]], labels: [Int]) -> Float {
        guard let packed = pack(features: features, labels: labels) else {
            return 0.0
        }
        let smoothing = labelSmoothing
        let screens = screenCount
        let titleMatrix = self.titleMatrix
        let lossFn = { (model: MLXHawkNetwork, arrays: [MLXArray]) -> [MLXArray] in
            let logits = model(arrays[0])
            guard let m = titleMatrix else {
                return [crossEntropy(logits: logits, targets: arrays[1], labelSmoothing: smoothing, reduction: .mean)]
            }
            let parts = split(logits, indices: [screens], axis: 1)
            let labels = arrays[1]
            let mask = (labels .>= 0).asType(Float.self)
            let safe = maximum(labels, MLXArray(Int32(0)))
            let perSample = crossEntropy(logits: parts[0], targets: safe, labelSmoothing: smoothing, reduction: .none)
            let screenLoss = (perSample * mask).sum() / maximum(mask.sum(), MLXArray(Float(1)))
            let oneHot = (expandedDimensions(safe, axis: -1) .== MLXArray(Int32(0) ..< Int32(screens))).asType(Float.self)
            let titleTargets = matmul(oneHot * expandedDimensions(mask, axis: -1), m)
            let z = parts[1]
            let titleLoss = (maximum(z, MLXArray(Float(0))) - z * titleTargets + log(1 + exp(-abs(z)))).mean()
            return [screenLoss + titleLoss]
        }
        let lg = valueAndGrad(model: network, lossFn)
        let maxNorm = clipNorm
        // The learning rate is passed as the last input so the compiled graph reads it as an argument, not a constant.
        func step(arrays: [MLXArray]) -> [MLXArray] {
            let (lossValues, grads) = lg(self.network, Array(arrays[0..<2]))
            let (clipped, norm) = clipGradNorm(gradients: grads, maxNorm: maxNorm)
            let outer = self.optimizer.learningRate
            self.optimizer.learningRate = arrays[2]
            self.optimizer.update(model: self.network, gradients: clipped)
            self.optimizer.learningRate = outer
            return lossValues + [norm]
        }
        let stepFn: ([MLXArray]) -> [MLXArray]
        switch compiledSteps[packed.batchSize] {
        case .some(let cached):
            stepFn = cached
        case .none:
            let fn = compile(inputs: [network, optimizer], outputs: [network, optimizer], step)
            compiledSteps[packed.batchSize] = fn
            stepFn = fn
        }
        let lossValues = stepFn([packed.features, packed.labels, optimizer.learningRate])
        eval(network, optimizer, lossValues)
        lastGradientNorm = lossValues[1].item(Float.self)
        return lossValues[0].item(Float.self)
    }

    /// Pooled features [B][modelDim] without gradients (for the feature distance).
    public func pooled(features: [[Float]]) -> [[Float]] {
        guard let packed = pack(features: features, labels: [Int](repeating: 0, count: features.count)) else {
            return []
        }
        let out = network.pooled(packed.features)
        eval(out)
        let flat = out.asArray(Float.self)
        let dim = network.modelDim
        return (0..<packed.batchSize).map { Array(flat[($0 * dim)..<(($0 + 1) * dim)]) }
    }

    /// Logits [B][outputDim] without gradients.
    public func logits(features: [[Float]]) -> [[Float]] {
        guard let packed = pack(features: features, labels: [Int](repeating: 0, count: features.count)) else {
            return []
        }
        let fn: ([MLXArray]) -> [MLXArray]
        switch compiledLogits[packed.batchSize] {
        case .some(let cached):
            fn = cached
        case .none:
            let newFn = compile(inputs: [network]) { arrays in
                return [self.network(arrays[0])]
            }
            compiledLogits[packed.batchSize] = newFn
            fn = newFn
        }
        let out = fn([packed.features])[0]
        eval(out)
        let flat = out.asArray(Float.self)
        let dim = network.outputDim
        var rows: [[Float]] = []
        rows.reserveCapacity(packed.batchSize)
        var b = 0
        while b < packed.batchSize {
            rows.append(Array(flat[(b * dim)..<((b + 1) * dim)]))
            b += 1
        }
        return rows
    }

    /// Mean loss and number of correct argmax predictions.
    public func evaluateBatch(features: [[Float]], labels: [Int]) -> (loss: Float, correct: Int) {
        let rows = logits(features: features)
        var lossSum: Float = 0.0
        var correct = 0
        var b = 0
        while b < rows.count {
            let logProbs = HawkNetwork.logSoftmax(rows[b])
            lossSum -= logProbs[labels[b]]
            var best = 0
            var c = 1
            while c < rows[b].count {
                if rows[b][best] < rows[b][c] {
                    best = c
                }
                c += 1
            }
            if best == labels[b] {
                correct += 1
            }
            b += 1
        }
        return (rows.isEmpty ? 0.0 : lossSum / Float(rows.count), correct)
    }

    /// Stacks flat frame sequences into [B, T, inputDim].
    private func pack(features: [[Float]], labels: [Int]) -> (features: MLXArray, labels: MLXArray, batchSize: Int)? {
        let bSize = min(features.count, labels.count)
        if bSize == 0 {
            return nil
        }
        let frames = RowScanEncoder.frameCount
        let inDim = network.inputDim
        let perSample = frames * inDim
        var flat = [Float](repeating: 0.0, count: bSize * perSample)
        var flatLabels = [Int32](repeating: 0, count: bSize)
        var b = 0
        while b < bSize {
            let seq = features[b]
            let n = min(perSample, seq.count)
            var i = 0
            while i < n {
                flat[b * perSample + i] = seq[i]
                i += 1
            }
            var label = labels[b]
            if label < -1 || screenCount <= label {
                label = -1
            }
            flatLabels[b] = Int32(label)
            b += 1
        }
        return (MLXArray(flat, [bSize, frames, inDim]), MLXArray(flatLabels), bSize)
    }
}
