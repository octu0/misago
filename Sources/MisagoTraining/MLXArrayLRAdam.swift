import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Adam first and second moments.
public struct ArrayLRAdamState: Updatable {
    public let m: MLXArray
    public let v: MLXArray

    public func innerState() -> [MLXArray] {
        return [m, v]
    }
}

/// Adam without bias correction whose learning rate is an `MLXArray`.
///
/// MLX's `Adam` stores the learning rate as a `Float`, which `compile()` bakes into the traced graph,
/// so a learning-rate schedule would have no effect on a cached compiled step. Passing it as an array
/// input avoids that. Implements `Optimizer` directly because `OptimizerBase` has an internal initializer.
public final class ArrayLRAdam: Optimizer {
    public var learningRate: MLXArray
    public let betas: (Float, Float)
    public let eps: Float

    private var stateStorage = NestedDictionary<String, ArrayLRAdamState>()

    public init(learningRate: Float, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8) {
        self.learningRate = MLXArray(learningRate)
        self.betas = betas
        self.eps = eps
    }

    public func innerState() -> [MLXArray] {
        return stateStorage.flattenedValues().flatMap { $0.innerState() }
    }

    /// Zeroes the moments, keeping their shapes. Call after rolling back weights to discard stale gradient history.
    public func resetState() {
        stateStorage = stateStorage.mapValues { state in
            ArrayLRAdamState(m: MLXArray.zeros(like: state.m), v: MLXArray.zeros(like: state.v))
        }
    }

    public func update(model: Module, gradients: ModuleParameters) {
        let (b1, b2) = betas
        let lr = learningRate
        let epsValue = eps
        let (newParameters, newStates) = gradients.mapValues(model.parameters(), stateStorage) { (gradient: MLXArray, parameter: MLXArray?, state: ArrayLRAdamState?) -> (MLXArray, ArrayLRAdamState?) in
            let p = parameter!
            let mPrev: MLXArray
            let vPrev: MLXArray
            switch state {
            case .some(let s):
                mPrev = s.m
                vPrev = s.v
            case .none:
                mPrev = MLXArray.zeros(like: p)
                vPrev = MLXArray.zeros(like: p)
            }
            let m = b1 * mPrev + (1 - b1) * gradient
            let v = b2 * vPrev + (1 - b2) * square(gradient)
            let update = lr * m / (sqrt(v) + epsValue)
            return (p - update, ArrayLRAdamState(m: m, v: v))
        }
        stateStorage = newStates
        model.update(parameters: newParameters)
    }
}
