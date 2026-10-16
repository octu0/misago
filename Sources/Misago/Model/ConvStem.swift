import Foundation

/// Weights of the digit reader's front end: one 3 x 3 convolution with stride 2 over the strip, followed by
/// GeLU. Every output column becomes one Hawk frame holding all rows and channels, so the recurrence runs
/// left to right over 160 frames and the convolution gives it tolerance to small vertical shifts.
/// Weight layout: `[kernelY][kernelX][inChannel][outChannel]` (out channels contiguous for SIMD).
public struct ConvStemWeights: Sendable, Codable, Equatable {
    public static let kernel = 3
    public static let stride = 2
    public static let padding = 1
    public static let inChannels = DigitStrip.channels
    public static let outChannels = 16
    public static let outputRows = (DigitStrip.height + 2 * padding - kernel) / stride + 1      // 28
    public static let outputColumns = (DigitStrip.width + 2 * padding - kernel) / stride + 1    // 160
    /// Hawk frames per strip and values per frame.
    public static let frameCount = outputColumns
    public static let frameDim = outputRows * outChannels                                        // 448

    public let weight: [Float]  // [3, 3, inChannels, outChannels]
    public let bias: [Float]    // [outChannels]

    public init(weight: [Float], bias: [Float]) {
        self.weight = weight
        self.bias = bias
    }

    public var parameterCount: Int {
        return weight.count + bias.count
    }
}

/// CPU forward pass of the stem: strip pixels → [frameCount * frameDim] Hawk input.
public enum ConvStem {
    public static func frames(_ pixels: [Float], weights w: ConvStemWeights) -> [Float] {
        let inW = DigitStrip.width
        let inH = DigitStrip.height
        let ic = ConvStemWeights.inChannels
        let oc = ConvStemWeights.outChannels
        let k = ConvStemWeights.kernel
        let stride = ConvStemWeights.stride
        let pad = ConvStemWeights.padding
        let rows = ConvStemWeights.outputRows
        let cols = ConvStemWeights.outputColumns
        var out = [Float](repeating: 0.0, count: cols * rows * oc)
        precondition(oc == 16, "stem kernel assumes 16 output channels")
        pixels.withUnsafeBufferPointer { pp in
        w.weight.withUnsafeBufferPointer { wp in
        w.bias.withUnsafeBufferPointer { bp in
        out.withUnsafeMutableBufferPointer { op in
            let p = pp.baseAddress!
            let wgt = wp.baseAddress!
            let bias0 = UnsafeRawPointer(bp.baseAddress!).loadUnaligned(as: SIMD8<Float>.self)
            let bias1 = UnsafeRawPointer(bp.baseAddress! + 8).loadUnaligned(as: SIMD8<Float>.self)
            let o = op.baseAddress!
            let gk = SIMD8<Float>(repeating: 1.702)
            var oy = 0
            while oy < rows {
                var ox = 0
                while ox < cols {
                    var acc0 = bias0
                    var acc1 = bias1
                    var ky = 0
                    while ky < k {
                        let iy = oy * stride + ky - pad
                        if 0 <= iy && iy < inH {
                            var kx = 0
                            while kx < k {
                                let ix = ox * stride + kx - pad
                                if 0 <= ix && ix < inW {
                                    let pixel = p + (iy * inW + ix) * ic
                                    let wBase = wgt + ((ky * k + kx) * ic) * oc
                                    var c = 0
                                    while c < ic {
                                        let v = SIMD8<Float>(repeating: pixel[c])
                                        let wc = wBase + c * oc
                                        acc0 = acc0.addingProduct(v, UnsafeRawPointer(wc).loadUnaligned(as: SIMD8<Float>.self))
                                        acc1 = acc1.addingProduct(v, UnsafeRawPointer(wc + 8).loadUnaligned(as: SIMD8<Float>.self))
                                        c += 1
                                    }
                                }
                                kx += 1
                            }
                        }
                        ky += 1
                    }
                    // GeLU (x σ(1.702 x)), then frame ox gets [oy][oc].
                    let g0 = acc0 * SIMDMath.sigmoid(gk * acc0)
                    let g1 = acc1 * SIMDMath.sigmoid(gk * acc1)
                    let dst = o + (ox * rows + oy) * oc
                    UnsafeMutableRawPointer(dst).storeBytes(of: g0, as: SIMD8<Float>.self)
                    UnsafeMutableRawPointer(dst + 8).storeBytes(of: g1, as: SIMD8<Float>.self)
                    ox += 1
                }
                oy += 1
            }
        }}}}
        return out
    }
}

/// Weights of the whole digit reader: the stem and the Hawk over its frames.
public struct DigitReaderWeights: Sendable, Codable, Equatable {
    public let stem: ConvStemWeights
    public let hawk: HawkWeights

    public init(stem: ConvStemWeights, hawk: HawkWeights) {
        self.stem = stem
        self.hawk = hawk
    }

    public var parameterCount: Int {
        return stem.parameterCount + hawk.parameterCount
    }

    public func save(to url: URL) throws {
        try JSONEncoder().encode(self).write(to: url, options: .atomic)
    }

    public static func load(from url: URL) throws -> DigitReaderWeights {
        return try JSONDecoder().decode(DigitReaderWeights.self, from: try Data(contentsOf: url))
    }
}
