import Foundation
import MLX
import MLXNN

/// CTC loss built from MLX operations.
///
/// Only the log-space forward recursion (alpha) is implemented; automatic differentiation
/// supplies the backward pass, so logits never leave the GPU.
public enum MLXCTCLoss {
    /// Finite stand-in for -inf, so logAddExp never evaluates inf - inf.
    static let negativeInfinity: Float = -1e30

    /// Batched CTC extended label sequences: blanks interleaved with labels, length 2U + 1.
    public struct ExtendedTargets {
        public let extTargets: MLXArray     // [B, Lmax] Int32
        public let skipMask: MLXArray       // [B, Lmax] 1.0 where the s-2 → s skip is allowed
        public let validMask: MLXArray      // [B, Lmax] 1.0 at valid positions
        public let finalIndex1: MLXArray    // [B] Int32  L_b - 1
        public let finalIndex2: MLXArray    // [B] Int32  L_b - 2 (clamped to 0)
        public let hasSecondFinal: MLXArray // [B] 1.0 when 2 <= L_b
        public let inputLengths: MLXArray   // [B] Int32
        public let maxExtendedLength: Int

        public init(
            targetsBatch: [[Int]],
            frameCounts: [Int],
            blankId: Int
        ) {
            let batchSize = targetsBatch.count
            var extendedLengths = [Int](repeating: 0, count: batchSize)
            var maxLen = 1
            var b = 0
            while b < batchSize {
                let u = targetsBatch[b].count
                extendedLengths[b] = (u * 2) + 1
                if maxLen < extendedLengths[b] {
                    maxLen = extendedLengths[b]
                }
                b += 1
            }
            // Round up to a multiple of 32 to limit distinct shapes: compiled steps retrace for every
            // (frames, extended length) pair. Padding has validMask = 0 and does not affect the loss.
            maxLen = ((maxLen + 31) / 32) * 32
            self.maxExtendedLength = maxLen

            var flatExt = [Int32](repeating: Int32(blankId), count: batchSize * maxLen)
            var flatSkip = [Float](repeating: 0.0, count: batchSize * maxLen)
            var flatValid = [Float](repeating: 0.0, count: batchSize * maxLen)
            var idx1 = [Int32](repeating: 0, count: batchSize)
            var idx2 = [Int32](repeating: 0, count: batchSize)
            var hasSecond = [Float](repeating: 0.0, count: batchSize)
            var lengths = [Int32](repeating: 0, count: batchSize)

            b = 0
            while b < batchSize {
                let labels = targetsBatch[b]
                let lLen = extendedLengths[b]
                let base = b * maxLen

                // blank, l0, blank, l1, ..., blank
                var u = 0
                while u < labels.count {
                    flatExt[base + (u * 2) + 1] = Int32(labels[u])
                    u += 1
                }

                var sPos = 0
                while sPos < lLen {
                    flatValid[base + sPos] = 1.0
                    // Skip from s-2 is allowed only when s is a label that differs from the label at s-2.
                    if 2 <= sPos {
                        let cur = flatExt[base + sPos]
                        let prev2 = flatExt[base + sPos - 2]
                        if cur != Int32(blankId) && cur != prev2 {
                            flatSkip[base + sPos] = 1.0
                        }
                    }
                    sPos += 1
                }

                idx1[b] = Int32(lLen - 1)
                if 2 <= lLen {
                    idx2[b] = Int32(lLen - 2)
                    hasSecond[b] = 1.0
                }
                lengths[b] = Int32(frameCounts[b])
                b += 1
            }

            self.extTargets = MLXArray(flatExt, [batchSize, maxLen])
            self.skipMask = MLXArray(flatSkip, [batchSize, maxLen])
            self.validMask = MLXArray(flatValid, [batchSize, maxLen])
            self.finalIndex1 = MLXArray(idx1, [batchSize])
            self.finalIndex2 = MLXArray(idx2, [batchSize])
            self.hasSecondFinal = MLXArray(hasSecond, [batchSize])
            self.inputLengths = MLXArray(lengths, [batchSize])
        }

        public init(
            extTargets: MLXArray,
            skipMask: MLXArray,
            validMask: MLXArray,
            finalIndex1: MLXArray,
            finalIndex2: MLXArray,
            hasSecondFinal: MLXArray,
            inputLengths: MLXArray
        ) {
            self.extTargets = extTargets
            self.skipMask = skipMask
            self.validMask = validMask
            self.finalIndex1 = finalIndex1
            self.finalIndex2 = finalIndex2
            self.hasSecondFinal = hasSecondFinal
            self.inputLengths = inputLengths
            self.maxExtendedLength = extTargets.shape[1]
        }

        public func toArrays() -> [MLXArray] {
            return [
                extTargets,
                skipMask,
                validMask,
                finalIndex1,
                finalIndex2,
                hasSecondFinal,
                inputLengths,
            ]
        }
    }

    /// Shifts alpha right by k along s, filling the head with `negativeInfinity`.
    static func shiftedRight(_ alpha: MLXArray, by k: Int) -> MLXArray {
        let batchSize = alpha.shape[0]
        let length = alpha.shape[1]
        if length <= k {
            return full([batchSize, length], values: MLXArray(negativeInfinity), type: Float.self)
        }
        let head = full([batchSize, k], values: MLXArray(negativeInfinity), type: Float.self)
        return concatenated([head, alpha[0..., 0..<(length - k)]], axis: 1)
    }

    /// Mean CTC negative log-likelihood over the batch.
    ///
    /// - Parameters:
    ///   - logits: [B, T, V]
    ///   - targets: extended label sequences
    public static func loss(
        logits: MLXArray,
        targets: ExtendedTargets
    ) -> MLXArray {
        let seqLen = logits.shape[1]
        let logProbs = logSoftmax(logits, axis: -1)

        // Log-probabilities at each extended label position for frame t, [B, Lmax].
        func emission(at t: Int) -> MLXArray {
            let step = logProbs[0..., t, 0...]
            return takeAlong(step, targets.extTargets, axis: 1)
        }

        let invalid = targets.validMask .<= 0.0

        // At t = 0 only s = 0 and s = 1 are reachable.
        var alpha = emission(at: 0)
        let lengthRange = MLXArray(Array(0..<targets.maxExtendedLength).map { Int32($0) },
                                   [1, targets.maxExtendedLength])
        let startReachable = lengthRange .<= MLXArray(Int32(1))
        alpha = which(startReachable, alpha, MLXArray(negativeInfinity))
        alpha = which(invalid, MLXArray(negativeInfinity), alpha)

        var t = 1
        while t < seqLen {
            let stay = alpha
            let fromPrev = shiftedRight(alpha, by: 1)
            let skipRaw = shiftedRight(alpha, by: 2)
            let skip = which(MLXArray(0.0) .< targets.skipMask, skipRaw, MLXArray(negativeInfinity))

            var next = logAddExp(logAddExp(stay, fromPrev), skip) + emission(at: t)
            next = which(invalid, MLXArray(negativeInfinity), next)

            // Freeze alpha past each sequence's own length.
            let active = MLXArray(Int32(t)) .< targets.inputLengths
            alpha = which(active.expandedDimensions(axis: 1), next, alpha)
            t += 1
        }

        // Valid endings: the final blank or the final label.
        let last1 = takeAlong(alpha, targets.finalIndex1.expandedDimensions(axis: 1), axis: 1)
            .squeezed(axis: 1)
        let last2Raw = takeAlong(alpha, targets.finalIndex2.expandedDimensions(axis: 1), axis: 1)
            .squeezed(axis: 1)
        let last2 = which(MLXArray(0.0) .< targets.hasSecondFinal, last2Raw, MLXArray(negativeInfinity))

        let logLikelihood = logAddExp(last1, last2)
        // Infeasible alignments (too few frames for the labels) keep the -1e30 sentinel.
        // Exclude them from the mean so they do not blow up the loss and gradients.
        let feasible = logLikelihood .> MLXArray(negativeInfinity * 0.5)
        let safe = which(feasible, logLikelihood, MLXArray(Float(0.0)))
        let count = maximum(sum(feasible.asType(.float32)), MLXArray(Float(1.0)))
        return -sum(safe) / count
    }
}
