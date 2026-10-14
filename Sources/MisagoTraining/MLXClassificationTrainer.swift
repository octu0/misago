import Foundation
import Misago
import MLX
import MLXNN
import MLXOptimizers

/// Cross-entropy training of the Hawk screen classifier. Every sample is the same fixed-length frame
/// sequence, so the compiled step is keyed by batch size only.
public final class MLXClassificationTrainer: @unchecked Sendable {
    public let network: MLXHawkNetwork
    public let optimizer: ArrayLRAdam
    public let clipNorm: Float

    /// Gradient norm of the most recent batch, before clipping.
    public private(set) var lastGradientNorm: Float = 0.0

    private var compiledSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]
    private var compiledLogits: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    /// MLX buffer cache limit; one batch shape is reused for the whole run.
    public static let cacheLimitBytes = 8 << 30

    public init(network: MLXHawkNetwork, learningRate: Float, clipNorm: Float = 5.0) {
        self.network = network
        self.optimizer = ArrayLRAdam(learningRate: learningRate)
        self.clipNorm = clipNorm
        MLX.GPU.set(cacheLimit: Self.cacheLimitBytes)
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
        let lossFn = { (model: MLXHawkNetwork, arrays: [MLXArray]) -> [MLXArray] in
            return [crossEntropy(logits: model(arrays[0]), targets: arrays[1], reduction: .mean)]
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
            if label < 0 || network.outputDim <= label {
                label = 0
            }
            flatLabels[b] = Int32(label)
            b += 1
        }
        return (MLXArray(flat, [bSize, frames, inDim]), MLXArray(flatLabels), bSize)
    }
}
