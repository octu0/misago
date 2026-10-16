import Foundation
import Misago
import MLX
import MLXNN

/// The digit reader on MLX: convolution stem → Hawk → per-frame logits.
public final class MLXDigitModel: Module, @unchecked Sendable {
    public var stemWeight: MLXArray   // [outChannels, 3, 3, inChannels] (MLX conv layout)
    public var stemBias: MLXArray     // [outChannels]
    public var hawk: MLXHawkNetwork

    public override init() {
        let oc = ConvStemWeights.outChannels
        let ic = ConvStemWeights.inChannels
        let k = ConvStemWeights.kernel
        let fanIn = k * k * ic
        let bound = sqrt(3.0 / Float(fanIn))
        self.stemWeight = MLXRandom.uniform(low: -bound, high: bound, [oc, k, k, ic])
        self.stemBias = MLXArray.zeros([oc])
        self.hawk = MLXHawkNetwork(
            inputDim: ConvStemWeights.frameDim,
            modelDim: DigitHawkShape.modelDim,
            rnnDim: DigitHawkShape.rnnDim,
            mlpDim: DigitHawkShape.mlpDim,
            heads: DigitHawkShape.heads,
            convWidth: DigitHawkShape.convWidth,
            convRight: DigitHawkShape.convRight,
            numLayers: DigitHawkShape.numLayers,
            outputDim: DigitVocabulary.size
        )
        super.init()
    }

    public convenience init(weights: DigitReaderWeights) {
        self.init()
        importWeights(from: weights)
    }

    /// strips: [B, 56, 320, 3] → Hawk frames [B, 160, 448]. Output column ox becomes frame ox holding [row][channel].
    public func frames(_ strips: MLXArray) -> MLXArray {
        let batch = strips.shape[0]
        let conv = conv2d(strips, stemWeight, stride: .init(ConvStemWeights.stride), padding: .init(ConvStemWeights.padding)) + stemBias
        let activated = hawkGelu(conv)  // [B, rows, cols, oc]
        return activated.transposed(0, 2, 1, 3).reshaped([batch, ConvStemWeights.frameCount, ConvStemWeights.frameDim])
    }

    /// strips: [B, 56, 320, 3] → per-frame logits [B, 160, vocabulary].
    public func frameLogits(_ strips: MLXArray) -> MLXArray {
        return hawk.frameLogits(frames(strips))
    }

    public func importWeights(from w: DigitReaderWeights) {
        let oc = ConvStemWeights.outChannels
        let ic = ConvStemWeights.inChannels
        let k = ConvStemWeights.kernel
        // CPU layout [ky][kx][ic][oc] → MLX [oc][ky][kx][ic].
        let cpu = MLXArray(w.stem.weight, [k, k, ic, oc])
        update(parameters: ModuleParameters.unflattened([
            ("stemWeight", cpu.transposed(3, 0, 1, 2)),
            ("stemBias", MLXArray(w.stem.bias, [oc]))
        ]))
        hawk.importWeights(from: w.hawk)
        eval(self)
    }

    public func exportWeights() -> DigitReaderWeights {
        eval(self)
        // MLX [oc][ky][kx][ic] → CPU [ky][kx][ic][oc].
        let cpu = stemWeight.transposed(1, 2, 3, 0)
        eval(cpu)
        let stem = ConvStemWeights(weight: cpu.asArray(Float.self), bias: stemBias.asArray(Float.self))
        return DigitReaderWeights(stem: stem, hawk: hawk.exportWeights(classNames: DigitVocabulary.names))
    }
}
