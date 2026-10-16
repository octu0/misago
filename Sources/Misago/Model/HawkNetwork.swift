import Foundation

/// CPU inference for the Hawk model over a whole, fixed-length frame sequence (one image).
/// Linear layers run as matrix products over all frames (`Matrix.gemm`, pure Swift SIMD); gates and the
/// recurrence use SIMD8. Nothing here depends on Apple frameworks, so it builds on Linux as well.
/// `modelDim`, `rnnDim` and `mlpDim` must be multiples of 8.
///
/// Unlike haitaka's streaming network, every frame is available up front, so each layer pads the sequence
/// with `convWidth - 1` zero frames before and `convRight` zero frames after and runs the convolution in one pass.
/// This matches the MLX training graph exactly.
public final class HawkNetwork: @unchecked Sendable {
    public let weights: HawkWeights
    /// Per layer, c · log σ(Λ), so that log a_t = r_t · logDecay.
    private let logDecay: [[Float]]

    public init(weights: HawkWeights) {
        precondition(weights.modelDim % 8 == 0 && weights.rnnDim % 8 == 0 && weights.mlpDim % 8 == 0, "widths must be multiples of 8")
        self.weights = weights
        self.logDecay = weights.layers.map { layer in
            layer.lambda.map { lambda -> Float in
                // Numerically stable log σ(Λ).
                var logSigmoid = -log1pf(expf(-lambda))
                if lambda < 0.0 {
                    logSigmoid = lambda - log1pf(expf(lambda))
                }
                return HawkWeights.decayExponent * logSigmoid
            }
        }
    }

    /// Classification logits [outputDim] for one frame sequence: mean over frames of the final RMSNorm
    /// output, projected once by `wOut`. Equal to the mean of per-frame logits.
    public func logits(features: [[Float]]) -> [Float] {
        let frames = features.count
        let inDim = weights.inputDim
        var chunk = [Float](repeating: 0.0, count: frames * inDim)
        var t = 0
        while t < frames {
            let row = features[t]
            let base = t * inDim
            var d = 0
            while d < min(inDim, row.count) {
                chunk[base + d] = row[d]
                d += 1
            }
            t += 1
        }
        return logits(flatFeatures: chunk, frames: frames)
    }

    /// Same as `logits(features:)` with the frames already concatenated ([frames * inputDim]).
    public func logits(flatFeatures chunk: [Float], frames: Int) -> [Float] {
        let w = weights
        let d = w.modelDim
        if frames <= 0 {
            return w.bOut
        }
        let normed = normedHidden(flatFeatures: chunk, frames: frames)
        // Mean of the normed frames, then one projection: logits are linear in the normed vector.
        var pooled = [Float](repeating: 0.0, count: d)
        normed.withUnsafeBufferPointer { np in
            pooled.withUnsafeMutableBufferPointer { pp in
                let n = np.baseAddress!
                let p = pp.baseAddress!
                var f = 0
                while f < frames {
                    var c = 0
                    while c < d {
                        Self.store(Self.load(p + c) + Self.load(n + f * d + c), p + c)
                        c += 8
                    }
                    f += 1
                }
                let inv = SIMD8<Float>(repeating: 1.0 / Float(frames))
                var c = 0
                while c < d {
                    Self.store(Self.load(p + c) * inv, p + c)
                    c += 8
                }
            }
        }
        var out = w.bOut
        Self.gemm(pooled, w.wOut, &out, m: 1, n: w.outputDim, k: d, beta: 1.0)
        return out
    }

    /// Per-frame logits [frames][outputDim], for sequence labelling (CTC).
    public func frameLogits(flatFeatures chunk: [Float], frames: Int) -> [[Float]] {
        let w = weights
        let d = w.modelDim
        if frames <= 0 {
            return []
        }
        let normed = normedHidden(flatFeatures: chunk, frames: frames)
        var logits = [Float](repeating: 0.0, count: frames * w.outputDim)
        var f = 0
        while f < frames {
            var c = 0
            while c < w.outputDim {
                logits[f * w.outputDim + c] = w.bOut[c]
                c += 1
            }
            f += 1
        }
        Self.gemm(normed, w.wOut, &logits, m: frames, n: w.outputDim, k: d, beta: 1.0)
        var rows: [[Float]] = []
        rows.reserveCapacity(frames)
        f = 0
        while f < frames {
            rows.append(Array(logits[(f * w.outputDim)..<((f + 1) * w.outputDim)]))
            f += 1
        }
        return rows
    }

    /// Input projection, all layers and the final RMSNorm: [frames, modelDim].
    private func normedHidden(flatFeatures chunk: [Float], frames: Int) -> [Float] {
        FloatEnvironment.prepareForInference()
        let w = weights
        let d = w.modelDim
        var x = [Float](repeating: 0.0, count: frames * d)
        x.withUnsafeMutableBufferPointer { xp in
            w.bIn.withUnsafeBufferPointer { bp in
                var f = 0
                while f < frames {
                    (xp.baseAddress! + f * d).update(from: bp.baseAddress!, count: d)
                    f += 1
                }
            }
        }
        Self.gemm(chunk, w.wIn, &x, m: frames, n: d, k: w.inputDim, beta: 1.0)
        var l = 0
        while l < w.numLayers {
            x = layer(l, x, frames: frames)
            l += 1
        }
        var normed = [Float](repeating: 0.0, count: frames * d)
        Self.rmsNorm(x, w.finalNorm, &normed, frames: frames, width: d)
        return normed
    }

    /// Calibrated class probabilities from classification logits.
    public func probabilities(logits: [Float]) -> [Float] {
        return weights.calibration.probabilities(logits)
    }

    /// One layer over the whole sequence; x is [frames, modelDim] and the result replaces it.
    private func layer(_ l: Int, _ xIn: [Float], frames: Int) -> [Float] {
        let w = weights
        let d = w.modelDim
        let r = w.rnnDim
        let m = w.mlpDim
        let hd = w.headDim
        let taps = w.convTaps
        let lw = w.layers[l]
        var x = xIn

        var normed = [Float](repeating: 0.0, count: frames * d)
        var branch = [Float](repeating: 0.0, count: frames * r)
        var z = [Float](repeating: 0.0, count: frames * r)
        var u = [Float](repeating: 0.0, count: frames * r)
        var keepPre = [Float](repeating: 0.0, count: frames * r)
        var admitPre = [Float](repeating: 0.0, count: frames * r)
        var mlpGate = [Float](repeating: 0.0, count: frames * m)
        var mlpUp = [Float](repeating: 0.0, count: frames * m)

        Self.rmsNorm(x, lw.norm1, &normed, frames: frames, width: d)
        Self.gemm(normed, lw.wGate, &branch, m: frames, n: r, k: d)
        Self.geluInPlace(&branch)
        Self.gemm(normed, lw.wX, &z, m: frames, n: r, k: d)
        // Zero padding: convWidth - 1 frames before, convRight after.
        var padded = [Float](repeating: 0.0, count: (frames + taps - 1) * r)
        padded.replaceSubrange(((w.convWidth - 1) * r)..<((w.convWidth - 1) * r + frames * r), with: z)
        Self.conv(padded, lw.convW, lw.convB, &u, frames: frames, width: r, taps: taps)
        // Block-diagonal gate projections.
        var h = 0
        while h < w.heads {
            let offset = h * hd * hd
            u.withUnsafeBufferPointer { ub in
                lw.wA.withUnsafeBufferPointer { wa in
                    keepPre.withUnsafeMutableBufferPointer { out in
                        Self.gemmRaw(ub.baseAddress! + h * hd, r, wa.baseAddress! + offset, hd, out.baseAddress! + h * hd, r, m: frames, n: hd, k: hd, beta: 0.0)
                    }
                }
                lw.wI.withUnsafeBufferPointer { wi in
                    admitPre.withUnsafeMutableBufferPointer { out in
                        Self.gemmRaw(ub.baseAddress! + h * hd, r, wi.baseAddress! + offset, hd, out.baseAddress! + h * hd, r, m: frames, n: hd, k: hd, beta: 0.0)
                    }
                }
            }
            h += 1
        }
        scan(layer: l, frames: frames, u: u, keepPre: keepPre, admitPre: admitPre, branch: &branch)
        Self.gemm(branch, lw.wO, &x, m: frames, n: d, k: r, beta: 1.0)

        Self.rmsNorm(x, lw.norm2, &normed, frames: frames, width: d)
        Self.gemm(normed, lw.wMLPGate, &mlpGate, m: frames, n: m, k: d)
        Self.gemm(normed, lw.wMLPUp, &mlpUp, m: frames, n: m, k: d)
        Self.geluInPlace(&mlpGate)
        Self.multiplyInPlace(&mlpGate, mlpUp)
        Self.gemm(mlpGate, lw.wMLPDown, &x, m: frames, n: d, k: m, beta: 1.0)
        return x
    }

    /// RG-LRU from a zero state: a = exp(r · c · log σ(Λ)), h = a h + √(1 - a²) (i u). Multiplies h into the GeLU branch in place.
    private func scan(layer: Int, frames: Int, u: [Float], keepPre: [Float], admitPre: [Float], branch: inout [Float]) {
        let lw = weights.layers[layer]
        let r = weights.rnnDim
        let one = SIMD8<Float>(repeating: 1.0)
        let eps = SIMD8<Float>(repeating: 1e-6)
        logDecay[layer].withUnsafeBufferPointer { decay in
        lw.bA.withUnsafeBufferPointer { bA in
        lw.bI.withUnsafeBufferPointer { bI in
        u.withUnsafeBufferPointer { ub in
        keepPre.withUnsafeBufferPointer { kp in
        admitPre.withUnsafeBufferPointer { ap in
        branch.withUnsafeMutableBufferPointer { out in
            let dp = decay.baseAddress!
            let bap = bA.baseAddress!
            let bip = bI.baseAddress!
            let up = ub.baseAddress!
            let kpp = kp.baseAddress!
            let app = ap.baseAddress!
            let op = out.baseAddress!
            var c = 0
            while c < r {
                let logA = Self.load(dp + c)
                let keepBias = Self.load(bap + c)
                let admitBias = Self.load(bip + c)
                var state = SIMD8<Float>(repeating: 0.0)
                var t = 0
                while t < frames {
                    let idx = t * r + c
                    let keep = SIMDMath.sigmoid(Self.load(kpp + idx) + keepBias)
                    let admit = SIMDMath.sigmoid(Self.load(app + idx) + admitBias)
                    let a = SIMDMath.exp(keep * logA)
                    let scale = (one - a * a + eps).squareRoot()
                    state = a * state + scale * (admit * Self.load(up + idx))
                    Self.store(state * Self.load(op + idx), op + idx)
                    t += 1
                }
                c += 8
            }
        }}}}}}}
    }

    // MARK: - Helpers

    /// c = a · b (+ beta · c), row-major: a [m, k], b [k, n], c [m, n].
    static func gemm(_ a: [Float], _ b: [Float], _ c: inout [Float], m: Int, n: Int, k: Int, beta: Float = 0.0) {
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                c.withUnsafeMutableBufferPointer { cp in
                    gemmRaw(ap.baseAddress!, k, bp.baseAddress!, n, cp.baseAddress!, n, m: m, n: n, k: k, beta: beta)
                }
            }
        }
    }

    static func gemmRaw(_ a: UnsafePointer<Float>, _ lda: Int, _ b: UnsafePointer<Float>, _ ldb: Int,
                        _ c: UnsafeMutablePointer<Float>, _ ldc: Int, m: Int, n: Int, k: Int, beta: Float) {
        Matrix.gemm(a, lda, b, ldb, c, ldc, m: m, n: n, k: k, beta: beta)
    }

    /// RMSNorm per frame; `width` is a multiple of 8.
    static func rmsNorm(_ x: [Float], _ gain: [Float], _ out: inout [Float], frames: Int, width: Int) {
        x.withUnsafeBufferPointer { xp in
        gain.withUnsafeBufferPointer { gp in
        out.withUnsafeMutableBufferPointer { op in
            let xb = xp.baseAddress!
            let g = gp.baseAddress!
            let o = op.baseAddress!
            var f = 0
            while f < frames {
                let row = xb + f * width
                var acc = SIMD8<Float>(repeating: 0.0)
                var c = 0
                while c < width {
                    let v = load(row + c)
                    acc = acc.addingProduct(v, v)
                    c += 8
                }
                let scale = SIMD8<Float>(repeating: 1.0 / sqrtf(acc.sum() / Float(width) + HawkWeights.rmsNormEpsilon))
                let dst = o + f * width
                c = 0
                while c < width {
                    store(load(row + c) * scale * load(g + c), dst + c)
                    c += 8
                }
                f += 1
            }
        }}}
    }

    /// Sigmoid approximation of GeLU: x σ(1.702 x). Below −20 the value is under 1e-14 and is set to zero so
    /// no denormals are produced.
    static func geluInPlace(_ x: inout [Float]) {
        let k = SIMD8<Float>(repeating: 1.702)
        let cutoff = SIMD8<Float>(repeating: -20.0)
        let zero = SIMD8<Float>(repeating: 0.0)
        x.withUnsafeMutableBufferPointer { p in
            let base = p.baseAddress!
            var i = 0
            while i + 8 <= p.count {
                let v = load(base + i)
                store((v * SIMDMath.sigmoid(k * v)).replacing(with: zero, where: v .< cutoff), base + i)
                i += 8
            }
            while i < p.count {
                p[i] = p[i] / (1.0 + expf(-1.702 * p[i]))
                i += 1
            }
        }
    }

    @inline(__always)
    static func load(_ p: UnsafePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func load(_ p: UnsafeMutablePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func store(_ v: SIMD8<Float>, _ p: UnsafeMutablePointer<Float>) {
        UnsafeMutableRawPointer(p).storeBytes(of: v, as: SIMD8<Float>.self)
    }

    static func multiplyInPlace(_ x: inout [Float], _ y: [Float]) {
        let n = min(x.count, y.count)
        x.withUnsafeMutableBufferPointer { xp in
            y.withUnsafeBufferPointer { yp in
                let a = xp.baseAddress!
                let b = yp.baseAddress!
                var i = 0
                while i + 8 <= n {
                    store(load(a + i) * load(b + i), a + i)
                    i += 8
                }
                while i < n {
                    a[i] *= b[i]
                    i += 1
                }
            }
        }
    }

    /// Depthwise convolution along frames. `buffer` holds [frames + taps - 1, width]; output frame t reads rows t ..< t + taps.
    static func conv(_ buffer: [Float], _ convW: [Float], _ convB: [Float], _ out: inout [Float], frames: Int, width: Int, taps: Int) {
        buffer.withUnsafeBufferPointer { bp in
        convW.withUnsafeBufferPointer { wp in
        convB.withUnsafeBufferPointer { cb in
        out.withUnsafeMutableBufferPointer { op in
            let b = bp.baseAddress!
            let w = wp.baseAddress!
            let bias = cb.baseAddress!
            let o = op.baseAddress!
            var t = 0
            while t < frames {
                var c = 0
                while c < width {
                    var acc = load(bias + c)
                    var j = 0
                    while j < taps {
                        acc += load(w + j * width + c) * load(b + (t + j) * width + c)
                        j += 1
                    }
                    store(acc, o + t * width + c)
                    c += 8
                }
                t += 1
            }
        }}}}
    }

    public static func logSoftmax(_ logits: [Float]) -> [Float] {
        var peak = -Float.greatestFiniteMagnitude
        for v in logits {
            if peak < v {
                peak = v
            }
        }
        var total: Float = 0.0
        for v in logits {
            total += expf(v - peak)
        }
        let norm = peak + logf(total)
        return logits.map { $0 - norm }
    }
}
