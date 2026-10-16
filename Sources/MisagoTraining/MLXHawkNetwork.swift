import Foundation
import Misago
import MLX
import MLXNN

/// Linear recurrence h_t = a_t h_{t-1} + x_t (h_{-1} = 0) over [B, T, R], one Metal thread per channel.
/// The backward pass is a reverse-time scan kernel, so graph size does not grow with sequence length.
public enum MLXLinearScan {
    static let forwardKernel = MLXFast.metalKernel(
        name: "hawk_linear_scan_forward",
        inputNames: ["a", "x"],
        outputNames: ["h"],
        source: """
            uint gid = thread_position_in_grid.x;
            if (gid < uint(B * R)) {
                uint b = gid / R;
                uint r = gid - b * R;
                size_t base = size_t(b) * T * R + r;
                float state = 0.0f;
                for (int t = 0; t < T; ++t) {
                    size_t idx = base + size_t(t) * R;
                    state = a[idx] * state + x[idx];
                    h[idx] = state;
                }
            }
            """
    )

    /// δ_t = g_t + a_{t+1} δ_{t+1}, dx_t = δ_t, da_t = δ_t h_{t-1}.
    static let backwardKernel = MLXFast.metalKernel(
        name: "hawk_linear_scan_backward",
        inputNames: ["a", "h", "g"],
        outputNames: ["da", "dx"],
        source: """
            uint gid = thread_position_in_grid.x;
            if (gid < uint(B * R)) {
                uint b = gid / R;
                uint r = gid - b * R;
                size_t base = size_t(b) * T * R + r;
                float carry = 0.0f;
                for (int t = T - 1; 0 <= t; --t) {
                    size_t idx = base + size_t(t) * R;
                    float d = g[idx] + carry;
                    dx[idx] = d;
                    float previous = 0.0f;
                    if (0 < t) {
                        previous = h[idx - R];
                    }
                    da[idx] = d * previous;
                    carry = d * a[idx];
                }
            }
            """
    )

    static func launch(_ shape: [Int]) -> (template: [(String, any KernelTemplateArg)], grid: (Int, Int, Int), group: (Int, Int, Int)) {
        let threads = shape[0] * shape[2]
        return (
            template: [("B", shape[0]), ("T", shape[1]), ("R", shape[2])],
            grid: (threads, 1, 1),
            group: (min(256, threads), 1, 1)
        )
    }

    static func forward(_ a: MLXArray, _ x: MLXArray) -> MLXArray {
        let p = launch(a.shape)
        return forwardKernel(
            [a, x], template: p.template, grid: p.grid, threadGroup: p.group,
            outputShapes: [a.shape], outputDTypes: [.float32]
        )[0]
    }

    nonisolated(unsafe) static let function = CustomFunction {
        Forward { inputs in
            return [MLXLinearScan.forward(inputs[0], inputs[1])]
        }
        VJP { primals, cotangents in
            let a = primals[0]
            let h = MLXLinearScan.forward(a, primals[1])
            let p = launch(a.shape)
            return backwardKernel(
                [a, h, cotangents[0]], template: p.template, grid: p.grid, threadGroup: p.group,
                outputShapes: [a.shape, a.shape], outputDTypes: [.float32, .float32]
            )
        }
    }

    /// a, x: [B, T, R] (float32). Returns h: [B, T, R].
    public static func scan(_ a: MLXArray, _ x: MLXArray) -> MLXArray {
        return function([a.asType(.float32), x.asType(.float32)])[0]
    }
}

/// GeLU approximation x σ(1.702 x); must match the CPU inference path.
@inline(__always)
func hawkGelu(_ x: MLXArray) -> MLXArray {
    return x * sigmoid(1.702 * x)
}

@inline(__always)
func hawkRMSNorm(_ x: MLXArray, _ gain: MLXArray) -> MLXArray {
    return x * rsqrt(mean(x * x, axis: -1, keepDims: true) + HawkWeights.rmsNormEpsilon) * gain
}

/// One Hawk layer. See `HawkLayerWeights` for the meaning of each weight.
public final class MLXHawkLayer: Module, @unchecked Sendable {
    public var norm1: MLXArray
    public var wGate: MLXArray
    public var wX: MLXArray
    public var convW: MLXArray
    public var convB: MLXArray
    public var wA: MLXArray
    public var bA: MLXArray
    public var wI: MLXArray
    public var bI: MLXArray
    public var lambda: MLXArray
    public var wO: MLXArray
    public var norm2: MLXArray
    public var wMLPGate: MLXArray
    public var wMLPUp: MLXArray
    public var wMLPDown: MLXArray

    let heads: Int
    let convWidth: Int
    let convRight: Int

    init(modelDim: Int, rnnDim: Int, mlpDim: Int, heads: Int, convWidth: Int, convRight: Int, numLayers: Int) {
        self.heads = heads
        self.convWidth = convWidth
        self.convRight = convRight
        let headDim = rnnDim / heads
        func uniform(_ fanIn: Int, _ shape: [Int], scale: Float = 1.0) -> MLXArray {
            let bound = scale * sqrt(3.0 / Float(fanIn))
            return MLXRandom.uniform(low: -bound, high: bound, shape)
        }
        // Scale residual output projections by 1/√(2L) so the initial residual variance is independent of depth.
        let residualScale = 1.0 / sqrt(Float(2 * numLayers))
        self.norm1 = MLXArray.ones([modelDim])
        self.wGate = uniform(modelDim, [modelDim, rnnDim])
        self.wX = uniform(modelDim, [modelDim, rnnDim])
        self.convW = uniform(convWidth + convRight, [convWidth + convRight, rnnDim])
        self.convB = MLXArray.zeros([rnnDim])
        self.wA = uniform(headDim, [heads, headDim, headDim])
        self.bA = MLXArray.zeros([rnnDim])
        self.wI = uniform(headDim, [heads, headDim, headDim])
        self.bI = MLXArray.zeros([rnnDim])
        self.lambda = MLXArray(HawkWeights.initialLambda(rnnDim: rnnDim), [rnnDim])
        self.wO = uniform(rnnDim, [rnnDim, modelDim], scale: residualScale)
        self.norm2 = MLXArray.ones([modelDim])
        self.wMLPGate = uniform(modelDim, [modelDim, mlpDim])
        self.wMLPUp = uniform(modelDim, [modelDim, mlpDim])
        self.wMLPDown = uniform(mlpDim, [mlpDim, modelDim], scale: residualScale)
        super.init()
    }

    /// x: [B, T, modelDim] → [B, T, modelDim]. Every sequence has the same length, so there is no frame mask.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let frames = x.shape[1]
        let rnnDim = wX.shape[1]
        let headDim = rnnDim / heads

        let n = hawkRMSNorm(x, norm1)
        let branch = hawkGelu(matmul(n, wGate))
        let raw = matmul(n, wX)
        // Zero-pad convWidth - 1 frames before and convRight after; each frame sees
        // convWidth - 1 past frames, itself, and convRight future frames.
        let paddedRaw = concatenated(
            [MLXArray.zeros([batch, convWidth - 1, rnnDim]), raw, MLXArray.zeros([batch, convRight, rnnDim])], axis: 1)
        var u = convB + MLXArray.zeros([batch, frames, rnnDim])
        var k = 0
        while k < convWidth + convRight {
            u = u + paddedRaw[0..., k..<(k + frames), 0...] * convW[k]
            k += 1
        }
        let blocks = u.reshaped([batch, frames, heads, 1, headDim])
        let r = sigmoid(matmul(blocks, wA).reshaped([batch, frames, rnnDim]) + bA)
        let i = sigmoid(matmul(blocks, wI).reshaped([batch, frames, rnnDim]) + bI)
        // log a_t = c · r_t · log σ(Λ)
        let logA = (HawkWeights.decayExponent * r) * logSigmoid(lambda)
        let a = exp(logA)
        let input = sqrt(1.0 - exp(2.0 * logA) + 1e-6) * (i * u)
        let h = MLXLinearScan.scan(a, input)
        let mixed = x + matmul(branch * h, wO)

        let m = hawkRMSNorm(mixed, norm2)
        return mixed + matmul(hawkGelu(matmul(m, wMLPGate)) * matmul(m, wMLPUp), wMLPDown)
    }
}

/// Hawk model on MLX: input projection → layers → RMSNorm → mean over frames → readout.
public final class MLXHawkNetwork: Module, @unchecked Sendable {
    public let inputDim: Int
    public let modelDim: Int
    public let rnnDim: Int
    public let mlpDim: Int
    public let heads: Int
    public let convWidth: Int
    public let convRight: Int
    public let outputDim: Int

    public var wIn: MLXArray
    public var bIn: MLXArray
    public var layers: [MLXHawkLayer]
    public var finalNorm: MLXArray
    public var wOut: MLXArray
    public var bOut: MLXArray

    public init(
        inputDim: Int,
        modelDim: Int,
        rnnDim: Int,
        mlpDim: Int,
        heads: Int,
        convWidth: Int,
        convRight: Int,
        numLayers: Int,
        outputDim: Int
    ) {
        self.inputDim = inputDim
        self.modelDim = modelDim
        self.rnnDim = rnnDim
        self.mlpDim = mlpDim
        self.heads = heads
        self.convWidth = convWidth
        self.convRight = convRight
        self.outputDim = outputDim
        let inBound = sqrt(3.0 / Float(inputDim))
        self.wIn = MLXRandom.uniform(low: -inBound, high: inBound, [inputDim, modelDim])
        // Bias with RMS 0.1; see `HawkWeights.bIn`.
        let biasBound = 0.1 * sqrt(Float(3.0))
        self.bIn = MLXRandom.uniform(low: -biasBound, high: biasBound, [modelDim])
        var built: [MLXHawkLayer] = []
        var l = 0
        while l < numLayers {
            built.append(MLXHawkLayer(modelDim: modelDim, rnnDim: rnnDim, mlpDim: mlpDim, heads: heads, convWidth: convWidth, convRight: convRight, numLayers: numLayers))
            l += 1
        }
        self.layers = built
        self.finalNorm = MLXArray.ones([modelDim])
        let outBound = sqrt(3.0 / Float(modelDim))
        self.wOut = MLXRandom.uniform(low: -outBound, high: outBound, [modelDim, outputDim])
        self.bOut = MLXArray.zeros([outputDim])
        super.init()
    }

    /// Network with the default shape (`HawkShape`) on the row-scan input, one output per class.
    public convenience init(classCount: Int) {
        self.init(
            inputDim: RowScanEncoder.frameDim,
            modelDim: HawkShape.modelDim,
            rnnDim: HawkShape.rnnDim,
            mlpDim: HawkShape.mlpDim,
            heads: HawkShape.heads,
            convWidth: HawkShape.convWidth,
            convRight: HawkShape.convRight,
            numLayers: HawkShape.numLayers,
            outputDim: classCount
        )
    }

    public convenience init(weights: HawkWeights) {
        self.init(
            inputDim: weights.inputDim,
            modelDim: weights.modelDim,
            rnnDim: weights.rnnDim,
            mlpDim: weights.mlpDim,
            heads: weights.heads,
            convWidth: weights.convWidth,
            convRight: weights.convRight,
            numLayers: weights.numLayers,
            outputDim: weights.outputDim
        )
        importWeights(from: weights)
    }

    public var numLayers: Int {
        return layers.count
    }

    /// features: [B, T, inputDim] → classification logits [B, outputDim]: the final RMSNorm output averaged
    /// over frames, then projected once (the same as the mean of per-frame logits).
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        let x = hidden(features)
        let pooled = mean(hawkRMSNorm(x, finalNorm), axis: 1)
        return matmul(pooled, wOut) + bOut
    }

    /// Per-frame logits [B, T, outputDim], for inspection.
    public func frameLogits(_ features: MLXArray) -> MLXArray {
        return matmul(hawkRMSNorm(hidden(features), finalNorm), wOut) + bOut
    }

    /// Residual stream [B, T, modelDim] after all layers.
    public func hidden(_ features: MLXArray) -> MLXArray {
        var x = matmul(features, wIn) + bIn
        var l = 0
        while l < layers.count {
            x = layers[l](x)
            l += 1
        }
        return x
    }

    /// Loads weights through `update(parameters:)` rather than reassigning properties, so the module tree
    /// (and anything compiled against it) keeps the same parameter slots.
    public func importWeights(from w: HawkWeights) {
        let d = w.modelDim
        let r = w.rnnDim
        let m = w.mlpDim
        let hd = w.headDim
        var items: [(String, MLXArray)] = [
            ("wIn", MLXArray(w.wIn, [w.inputDim, d])),
            ("bIn", MLXArray(w.bIn, [d])),
            ("finalNorm", MLXArray(w.finalNorm, [d])),
            ("wOut", MLXArray(w.wOut, [d, w.outputDim])),
            ("bOut", MLXArray(w.bOut, [w.outputDim]))
        ]
        var l = 0
        while l < min(layers.count, w.layers.count) {
            let lw = w.layers[l]
            let p = "layers.\(l)."
            items.append((p + "norm1", MLXArray(lw.norm1, [d])))
            items.append((p + "wGate", MLXArray(lw.wGate, [d, r])))
            items.append((p + "wX", MLXArray(lw.wX, [d, r])))
            items.append((p + "convW", MLXArray(lw.convW, [w.convTaps, r])))
            items.append((p + "convB", MLXArray(lw.convB, [r])))
            items.append((p + "wA", MLXArray(lw.wA, [w.heads, hd, hd])))
            items.append((p + "bA", MLXArray(lw.bA, [r])))
            items.append((p + "wI", MLXArray(lw.wI, [w.heads, hd, hd])))
            items.append((p + "bI", MLXArray(lw.bI, [r])))
            items.append((p + "lambda", MLXArray(lw.lambda, [r])))
            items.append((p + "wO", MLXArray(lw.wO, [r, d])))
            items.append((p + "norm2", MLXArray(lw.norm2, [d])))
            items.append((p + "wMLPGate", MLXArray(lw.wMLPGate, [d, m])))
            items.append((p + "wMLPUp", MLXArray(lw.wMLPUp, [d, m])))
            items.append((p + "wMLPDown", MLXArray(lw.wMLPDown, [m, d])))
            l += 1
        }
        update(parameters: ModuleParameters.unflattened(items))
        eval(self)
    }

    public func exportWeights(classes: ClassList, calibration: VectorCalibration? = nil) -> HawkWeights {
        return exportWeights(classNames: classes.names, calibration: calibration)
    }

    /// Export with arbitrary output names (the digit vocabulary for the reader).
    public func exportWeights(classNames: [String], calibration: VectorCalibration? = nil) -> HawkWeights {
        precondition(classNames.count == outputDim, "name count must equal outputDim")
        eval(self)
        func values(_ x: MLXArray) -> [Float] {
            return x.asArray(Float.self)
        }
        let exported = layers.map { layer in
            HawkLayerWeights(
                norm1: values(layer.norm1), wGate: values(layer.wGate), wX: values(layer.wX),
                convW: values(layer.convW), convB: values(layer.convB),
                wA: values(layer.wA), bA: values(layer.bA), wI: values(layer.wI), bI: values(layer.bI),
                lambda: values(layer.lambda), wO: values(layer.wO), norm2: values(layer.norm2),
                wMLPGate: values(layer.wMLPGate), wMLPUp: values(layer.wMLPUp), wMLPDown: values(layer.wMLPDown)
            )
        }
        return HawkWeights(
            inputDim: inputDim,
            modelDim: modelDim,
            rnnDim: rnnDim,
            mlpDim: mlpDim,
            heads: heads,
            convWidth: convWidth,
            convRight: convRight,
            outputDim: outputDim,
            wIn: values(wIn),
            bIn: values(bIn),
            layers: exported,
            finalNorm: values(finalNorm),
            wOut: values(wOut),
            bOut: values(bOut),
            classNames: classNames,
            calibrationScale: calibration?.scale ?? 1.0,
            calibrationBias: calibration?.bias
        )
    }
}
