import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Adam first and second moments, and the number of updates so far (an array so compiled steps advance it).
public struct ArrayLRAdamState: Updatable {
    public let m: MLXArray
    public let v: MLXArray
    public let t: MLXArray

    public func innerState() -> [MLXArray] {
        return [m, v, t]
    }
}

/// Adam whose learning rate is an `MLXArray`, with optional bias correction and decoupled weight decay (AdamW).
///
/// MLX's `Adam` stores the learning rate as a `Float`, which `compile()` bakes into the traced graph,
/// so a learning-rate schedule would have no effect on a cached compiled step. Passing it as an array
/// input avoids that. Implements `Optimizer` directly because `OptimizerBase` has an internal initializer.
public final class ArrayLRAdam: Optimizer {
    public var learningRate: MLXArray
    public let betas: (Float, Float)
    public let eps: Float
    /// Divide the moments by (1 - beta^t): without it the first steps are about 3x larger than the learning rate.
    public let biasCorrection: Bool
    /// Decoupled weight decay per step, scaled by the learning rate; applied to matrices only (not biases or gains).
    public let weightDecay: Float

    private var stateStorage = NestedDictionary<String, ArrayLRAdamState>()

    public init(learningRate: Float, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8, biasCorrection: Bool = false, weightDecay: Float = 0.0) {
        self.learningRate = MLXArray(learningRate)
        self.betas = betas
        self.eps = eps
        self.biasCorrection = biasCorrection
        self.weightDecay = weightDecay
    }

    public func innerState() -> [MLXArray] {
        return stateStorage.flattenedValues().flatMap { $0.innerState() }
    }

    /// Zeroes the moments, keeping their shapes. Call after rolling back weights to discard stale gradient history.
    public func resetState() {
        stateStorage = stateStorage.mapValues { state in
            ArrayLRAdamState(m: MLXArray.zeros(like: state.m), v: MLXArray.zeros(like: state.v), t: MLXArray.zeros(like: state.t))
        }
    }

    public func update(model: Module, gradients: ModuleParameters) {
        let (b1, b2) = betas
        let lr = learningRate
        let epsValue = eps
        let correct = biasCorrection
        let decay = weightDecay
        let logB1 = Float(Foundation.log(Double(b1)))
        let logB2 = Float(Foundation.log(Double(b2)))
        let (newParameters, newStates) = gradients.mapValues(model.parameters(), stateStorage) { (gradient: MLXArray, parameter: MLXArray?, state: ArrayLRAdamState?) -> (MLXArray, ArrayLRAdamState?) in
            let p = parameter!
            let mPrev: MLXArray
            let vPrev: MLXArray
            let tPrev: MLXArray
            switch state {
            case .some(let s):
                mPrev = s.m
                vPrev = s.v
                tPrev = s.t
            case .none:
                mPrev = MLXArray.zeros(like: p)
                vPrev = MLXArray.zeros(like: p)
                tPrev = MLXArray(Float(0))
            }
            let t = tPrev + 1
            let m = b1 * mPrev + (1 - b1) * gradient
            let v = b2 * vPrev + (1 - b2) * square(gradient)
            var next: MLXArray
            if correct {
                let mHat = m / (1 - exp(t * logB1))
                let vHat = v / (1 - exp(t * logB2))
                next = p - lr * mHat / (sqrt(vHat) + epsValue)
            } else {
                next = p - lr * m / (sqrt(v) + epsValue)
            }
            if 0 < decay && 1 < p.ndim {
                next = next - lr * decay * p
            }
            return (next, ArrayLRAdamState(m: m, v: v, t: t))
        }
        stateStorage = newStates
        model.update(parameters: newParameters)
    }
}
