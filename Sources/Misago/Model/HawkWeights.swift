import Foundation

/// Default model shape for the screen classifier (about 1.07M parameters), shared by training and inference.
public enum HawkShape {
    public static let numLayers = 4
    public static let modelDim = 128
    public static let rnnDim = 192
    public static let mlpDim = 384
    /// Number of diagonal blocks in the RG-LRU gates.
    public static let heads = 4
    public static let convWidth = 4
    /// Frames below the current one seen by each layer's convolution. Offline images have no latency cost.
    public static let convRight = 2
}

/// Shape of the digit reader's Hawk (about 0.45M parameters) over the stem's 160 frames.
public enum DigitHawkShape {
    public static let numLayers = 3
    public static let modelDim = 96
    public static let rnnDim = 128
    public static let mlpDim = 288
    public static let heads = 4
    public static let convWidth = 4
    public static let convRight = 2
}

/// Weights of the Hawk model (the recurrent-only variant of Griffin, De et al. 2024), ported from haitaka.
/// Each layer is RMSNorm → recurrent block → residual, then RMSNorm → gated MLP → residual.
/// The recurrent block multiplies a GeLU branch with a linear → conv → RG-LRU branch, then projects back.
/// All matrices are row-major [input, output] (applied as x · W).
///
/// Classification reads out the mean over frames of the final RMSNorm output, projected once by `wOut`.
/// `classNames` names the `outputDim` classes (class 0 is `other`). `calibrationScale` and `calibrationBias`
/// map logits to calibrated probabilities (vector scaling): p = softmax(scale · logits + bias).
public struct HawkWeights: Sendable, Codable, Equatable {
    public let inputDim: Int
    public let modelDim: Int
    public let rnnDim: Int
    public let mlpDim: Int
    public let heads: Int
    public let convWidth: Int
    public let convRight: Int
    public let outputDim: Int

    public let wIn: [Float]          // [inputDim, modelDim]
    /// Input bias. Keeps the residual stream nonzero on all-black rows; otherwise each RMSNorm backward
    /// pass amplifies gradients by up to 1/√eps and deep stacks overflow.
    public let bIn: [Float]          // [modelDim]
    public let layers: [HawkLayerWeights]
    public let finalNorm: [Float]    // [modelDim]
    public let wOut: [Float]         // [modelDim, outputDim]
    public let bOut: [Float]         // [outputDim]
    public let classNames: [String]  // [outputDim]
    public var calibrationScale: Float
    public var calibrationBias: [Float]  // [outputDim]

    /// RG-LRU decay exponent c: a_t = σ(Λ)^(c·r_t).
    public static let decayExponent: Float = 8.0
    /// Epsilon inside RMSNorm, shared by the MLX trainer and the CPU network.
    public static let rmsNormEpsilon: Float = 1e-5
    /// Initial time constants (frames) spread log-uniformly across channels at r = 0.5.
    public static let initialTimeConstants: ClosedRange<Float> = 1.0...100.0

    public init(
        inputDim: Int,
        modelDim: Int,
        rnnDim: Int,
        mlpDim: Int,
        heads: Int,
        convWidth: Int,
        convRight: Int,
        outputDim: Int,
        wIn: [Float],
        bIn: [Float],
        layers: [HawkLayerWeights],
        finalNorm: [Float],
        wOut: [Float],
        bOut: [Float],
        classNames: [String],
        calibrationScale: Float = 1.0,
        calibrationBias: [Float]? = nil
    ) {
        self.inputDim = inputDim
        self.modelDim = modelDim
        self.rnnDim = rnnDim
        self.mlpDim = mlpDim
        self.heads = heads
        self.convWidth = convWidth
        self.convRight = convRight
        self.outputDim = outputDim
        self.wIn = wIn
        self.bIn = bIn
        self.layers = layers
        self.finalNorm = finalNorm
        self.wOut = wOut
        self.bOut = bOut
        self.classNames = classNames
        self.calibrationScale = calibrationScale
        self.calibrationBias = calibrationBias ?? [Float](repeating: 0.0, count: outputDim)
    }

    public var calibration: VectorCalibration {
        return VectorCalibration(scale: calibrationScale, bias: calibrationBias)
    }

    enum CodingKeys: String, CodingKey {
        case inputDim, modelDim, rnnDim, mlpDim, heads, convWidth, convRight, outputDim
        case wIn, bIn, layers, finalNorm, wOut, bOut, classNames, calibrationScale, calibrationBias
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inputDim = try c.decode(Int.self, forKey: .inputDim)
        modelDim = try c.decode(Int.self, forKey: .modelDim)
        rnnDim = try c.decode(Int.self, forKey: .rnnDim)
        mlpDim = try c.decode(Int.self, forKey: .mlpDim)
        heads = try c.decode(Int.self, forKey: .heads)
        convWidth = try c.decode(Int.self, forKey: .convWidth)
        convRight = try c.decode(Int.self, forKey: .convRight)
        outputDim = try c.decode(Int.self, forKey: .outputDim)
        wIn = try c.decode([Float].self, forKey: .wIn)
        bIn = try c.decode([Float].self, forKey: .bIn)
        layers = try c.decode([HawkLayerWeights].self, forKey: .layers)
        finalNorm = try c.decode([Float].self, forKey: .finalNorm)
        wOut = try c.decode([Float].self, forKey: .wOut)
        bOut = try c.decode([Float].self, forKey: .bOut)
        classNames = try c.decode([String].self, forKey: .classNames)
        calibrationScale = try c.decodeIfPresent(Float.self, forKey: .calibrationScale) ?? 1.0
        calibrationBias = try c.decodeIfPresent([Float].self, forKey: .calibrationBias) ?? [Float](repeating: 0.0, count: outputDim)
    }

    public var numLayers: Int {
        return layers.count
    }

    /// Convolution taps: `convWidth - 1` past, the current frame, and `convRight` future.
    public var convTaps: Int {
        return convWidth + convRight
    }

    public var headDim: Int {
        return rnnDim / heads
    }

    public var parameterCount: Int {
        var n = wIn.count + bIn.count + finalNorm.count + wOut.count + bOut.count
        for layer in layers {
            n += layer.parameterCount
        }
        return n
    }

    /// Initial Λ such that σ(Λ)^(c/2) = exp(-1/τ) for τ spread over `initialTimeConstants`.
    public static func initialLambda(rnnDim: Int) -> [Float] {
        let lo = logf(initialTimeConstants.lowerBound)
        let hi = logf(initialTimeConstants.upperBound)
        var values = [Float](repeating: 0.0, count: rnnDim)
        var n = 0
        while n < rnnDim {
            var frac: Float = 0.0
            if 1 < rnnDim {
                frac = Float(n) / Float(rnnDim - 1)
            }
            let tau = expf(lo + (hi - lo) * frac)
            let a = expf(-1.0 / (tau * decayExponent * 0.5))
            values[n] = logf(a / (1.0 - a))
            n += 1
        }
        return values
    }

    public func save(to url: URL) throws {
        let data = try JSONEncoder().encode(self)
        try data.write(to: url, options: .atomic)
    }

    public static func load(from url: URL) throws -> HawkWeights {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(HawkWeights.self, from: data)
    }
}

/// Weights of one Hawk layer.
public struct HawkLayerWeights: Sendable, Codable, Equatable {
    public let norm1: [Float]       // [modelDim]
    public let wGate: [Float]       // [modelDim, rnnDim] GeLU branch
    public let wX: [Float]          // [modelDim, rnnDim] recurrent branch
    public let convW: [Float]       // [convWidth + convRight, rnnDim]; row convWidth - 1 is the current frame, later rows are future
    public let convB: [Float]       // [rnnDim]
    public let wA: [Float]          // [heads, headDim, headDim] recurrence gate (block-diagonal)
    public let bA: [Float]          // [rnnDim]
    public let wI: [Float]          // [heads, headDim, headDim] input gate (block-diagonal)
    public let bI: [Float]          // [rnnDim]
    public let lambda: [Float]      // [rnnDim] logit Λ of the decay σ(Λ)
    public let wO: [Float]          // [rnnDim, modelDim]
    public let norm2: [Float]       // [modelDim]
    public let wMLPGate: [Float]    // [modelDim, mlpDim]
    public let wMLPUp: [Float]      // [modelDim, mlpDim]
    public let wMLPDown: [Float]    // [mlpDim, modelDim]

    public init(
        norm1: [Float], wGate: [Float], wX: [Float], convW: [Float], convB: [Float],
        wA: [Float], bA: [Float], wI: [Float], bI: [Float], lambda: [Float], wO: [Float],
        norm2: [Float], wMLPGate: [Float], wMLPUp: [Float], wMLPDown: [Float]
    ) {
        self.norm1 = norm1
        self.wGate = wGate
        self.wX = wX
        self.convW = convW
        self.convB = convB
        self.wA = wA
        self.bA = bA
        self.wI = wI
        self.bI = bI
        self.lambda = lambda
        self.wO = wO
        self.norm2 = norm2
        self.wMLPGate = wMLPGate
        self.wMLPUp = wMLPUp
        self.wMLPDown = wMLPDown
    }

    public var parameterCount: Int {
        return norm1.count + wGate.count + wX.count + convW.count + convB.count + wA.count + bA.count
            + wI.count + bI.count + lambda.count + wO.count + norm2.count + wMLPGate.count + wMLPUp.count + wMLPDown.count
    }
}
