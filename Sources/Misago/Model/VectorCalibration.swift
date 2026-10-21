import Foundation

/// Vector scaling for multi-class logits: p = softmax(scale · logits + bias), with one shared scale
/// (inverse temperature) and one bias per class. Fitted by minimizing the negative log-likelihood on
/// held-out data with damped Newton steps; the problem is convex in (scale, bias).
public struct VectorCalibration: Sendable, Equatable {
    public let scale: Float
    public let bias: [Float]

    public init(scale: Float, bias: [Float]) {
        self.scale = scale
        self.bias = bias
    }

    public static func identity(classCount: Int) -> VectorCalibration {
        return VectorCalibration(scale: 1.0, bias: [Float](repeating: 0.0, count: classCount))
    }

    public var classCount: Int {
        return bias.count
    }

    /// Calibrated class probabilities.
    public func probabilities(_ logits: [Float]) -> [Float] {
        var z = [Float](repeating: 0.0, count: logits.count)
        var i = 0
        while i < logits.count {
            z[i] = scale * logits[i] + (i < bias.count ? bias[i] : 0.0)
            i += 1
        }
        return Self.softmax(z)
    }

    public static func softmax(_ z: [Float]) -> [Float] {
        var peak = -Float.greatestFiniteMagnitude
        for v in z {
            if peak < v {
                peak = v
            }
        }
        var total: Float = 0.0
        var out = z.map { v -> Float in
            let e = expf(v - peak)
            total += e
            return e
        }
        var i = 0
        while i < out.count {
            out[i] /= total
            i += 1
        }
        return out
    }

    /// Fits (scale, bias) to logits [N][K] and labels [N]. Bias of class 0 is fixed at zero (the softmax is
    /// invariant to a shared offset). Targets are smoothed as in Platt (1999): the true class gets
    /// (n_y + 1) / (n_y + 2) and the rest is spread over the other classes, so perfectly separable data does not
    /// drive the scale to infinity. Returns the identity when there are fewer than two samples or a single class.
    /// With `fitBias` false only the scale is fitted (temperature scaling) and every bias stays zero: per-class
    /// biases fitted on a small held-out set follow its few samples of the rare classes.
    public static func fit(logits: [[Float]], labels: [Int], classCount: Int, iterations: Int = 100, fitBias: Bool = true) -> VectorCalibration {
        let n = min(logits.count, labels.count)
        let k = classCount
        if n < 2 || k < 2 {
            return identity(classCount: k)
        }
        var classCounts = [Int](repeating: 0, count: k)
        var i0 = 0
        while i0 < n {
            if 0 <= labels[i0] && labels[i0] < k {
                classCounts[labels[i0]] += 1
            }
            i0 += 1
        }
        func target(_ c: Int, _ y: Int) -> Double {
            let top = (Double(classCounts[y]) + 1.0) / (Double(classCounts[y]) + 2.0)
            if c == y {
                return top
            }
            return (1.0 - top) / Double(k - 1)
        }
        // Parameters θ = [scale, bias_1 ... bias_{K-1}] (bias_0 = 0), or [scale] alone.
        let dim = fitBias ? k : 1
        var theta = [Double](repeating: 0.0, count: dim)
        theta[0] = 1.0

        func probs(_ theta: [Double], _ row: [Float]) -> [Double] {
            var z = [Double](repeating: 0.0, count: k)
            var peak = -Double.greatestFiniteMagnitude
            var c = 0
            while c < k {
                z[c] = theta[0] * Double(row[c]) + (0 < c && fitBias ? theta[c] : 0.0)
                peak = max(peak, z[c])
                c += 1
            }
            var total = 0.0
            c = 0
            while c < k {
                z[c] = exp(z[c] - peak)
                total += z[c]
                c += 1
            }
            c = 0
            while c < k {
                z[c] /= total
                c += 1
            }
            return z
        }
        func nll(_ theta: [Double]) -> Double {
            var total = 0.0
            var i = 0
            while i < n {
                let p = probs(theta, logits[i])
                var c = 0
                while c < k {
                    total -= target(c, labels[i]) * log(max(p[c], 1e-300))
                    c += 1
                }
                i += 1
            }
            return total
        }

        var current = nll(theta)
        var it = 0
        while it < iterations {
            // Gradient g and Hessian H of the NLL. With u_c = ∂z_c/∂θ (u_c[0] = logit_c, u_c[j] = [c == j]),
            // g = Σ_i Σ_c (p_c − [c == y]) u_c and H = Σ_i Σ_c p_c u_c u_cᵀ − (Σ_c p_c u_c)(Σ_c p_c u_c)ᵀ.
            var g = [Double](repeating: 0.0, count: dim)
            var h = [Double](repeating: 0.0, count: dim * dim)
            var i = 0
            while i < n {
                let row = logits[i]
                let p = probs(theta, row)
                var mean = [Double](repeating: 0.0, count: dim)
                var c = 0
                while c < k {
                    var u = [Double](repeating: 0.0, count: dim)
                    u[0] = Double(row[c])
                    if 0 < c && fitBias {
                        u[c] = 1.0
                    }
                    let d = p[c] - target(c, labels[i])
                    var a = 0
                    while a < dim {
                        g[a] += d * u[a]
                        mean[a] += p[c] * u[a]
                        var b = 0
                        while b < dim {
                            h[a * dim + b] += p[c] * u[a] * u[b]
                            b += 1
                        }
                        a += 1
                    }
                    c += 1
                }
                var a = 0
                while a < dim {
                    var b = 0
                    while b < dim {
                        h[a * dim + b] -= mean[a] * mean[b]
                        b += 1
                    }
                    a += 1
                }
                i += 1
            }
            var a = 0
            while a < dim {
                h[a * dim + a] += 1e-6
                a += 1
            }
            guard let step = solve(h, g, dim) else {
                break
            }
            // Damped Newton: halve the step until the NLL decreases.
            var stepScale = 1.0
            var accepted = false
            var tries = 0
            while tries < 30 {
                var candidate = theta
                var j = 0
                while j < dim {
                    candidate[j] -= stepScale * step[j]
                    j += 1
                }
                let value = nll(candidate)
                if value <= current {
                    theta = candidate
                    current = value
                    accepted = true
                    break
                }
                stepScale *= 0.5
                tries += 1
            }
            var largest = 0.0
            for s in step {
                largest = max(largest, abs(stepScale * s))
            }
            if accepted != true || largest < 1e-9 {
                break
            }
            it += 1
        }
        var bias = [Float](repeating: 0.0, count: k)
        var c = 1
        while fitBias && c < k {
            bias[c] = Float(theta[c])
            c += 1
        }
        return VectorCalibration(scale: Float(theta[0]), bias: bias)
    }

    /// Solves H x = g by Gaussian elimination with partial pivoting; nil when singular.
    private static func solve(_ hIn: [Double], _ gIn: [Double], _ n: Int) -> [Double]? {
        var h = hIn
        var g = gIn
        var col = 0
        while col < n {
            var pivot = col
            var row = col + 1
            while row < n {
                if abs(h[pivot * n + col]) < abs(h[row * n + col]) {
                    pivot = row
                }
                row += 1
            }
            if abs(h[pivot * n + col]) < 1e-18 {
                return nil
            }
            if pivot != col {
                var j = 0
                while j < n {
                    h.swapAt(col * n + j, pivot * n + j)
                    j += 1
                }
                g.swapAt(col, pivot)
            }
            row = col + 1
            while row < n {
                let factor = h[row * n + col] / h[col * n + col]
                var j = col
                while j < n {
                    h[row * n + j] -= factor * h[col * n + j]
                    j += 1
                }
                g[row] -= factor * g[col]
                row += 1
            }
            col += 1
        }
        var x = [Double](repeating: 0.0, count: n)
        var row = n - 1
        while 0 <= row {
            var sum = g[row]
            var j = row + 1
            while j < n {
                sum -= h[row * n + j] * x[j]
                j += 1
            }
            x[row] = sum / h[row * n + row]
            row -= 1
        }
        return x
    }

    /// Expected calibration error over equal-width confidence bins (confidence = top probability,
    /// correct = argmax equals the label), and the per-bin (confidence, accuracy, count).
    public static func expectedCalibrationError(probabilities: [[Float]], labels: [Int], bins: Int = 10) -> (ece: Float, bins: [(confidence: Float, accuracy: Float, count: Int)]) {
        let n = min(probabilities.count, labels.count)
        var sumConf = [Float](repeating: 0.0, count: bins)
        var sumAcc = [Float](repeating: 0.0, count: bins)
        var counts = [Int](repeating: 0, count: bins)
        var i = 0
        while i < n {
            var best = 0
            var c = 1
            while c < probabilities[i].count {
                if probabilities[i][best] < probabilities[i][c] {
                    best = c
                }
                c += 1
            }
            let confidence = probabilities[i].isEmpty ? 0.0 : probabilities[i][best]
            var bin = Int(confidence * Float(bins))
            if bins <= bin {
                bin = bins - 1
            }
            sumConf[bin] += confidence
            sumAcc[bin] += best == labels[i] ? 1.0 : 0.0
            counts[bin] += 1
            i += 1
        }
        var ece: Float = 0.0
        var rows: [(confidence: Float, accuracy: Float, count: Int)] = []
        var b = 0
        while b < bins {
            if 0 < counts[b] {
                let conf = sumConf[b] / Float(counts[b])
                let acc = sumAcc[b] / Float(counts[b])
                ece += abs(conf - acc) * Float(counts[b]) / Float(max(1, n))
                rows.append((confidence: conf, accuracy: acc, count: counts[b]))
            } else {
                rows.append((confidence: 0.0, accuracy: 0.0, count: 0))
            }
            b += 1
        }
        return (ece, rows)
    }
}
