import Foundation

/// How far a pooled feature vector lies from everything the classifier was trained on: the Mahalanobis distance
/// to the nearest class mean under a shared (within-class) covariance. Images unlike any training image land
/// far from every mean, even when the classifier's outputs happen to be confident.
///
/// Stored whitened: `whitening` is L⁻¹ for the Cholesky factor Σ = L Lᵀ, so z = L⁻¹ h has identity covariance
/// and the distance is ‖z − ν_c‖ with ν_c = L⁻¹ μ_c. Inference costs one d × d product and one difference per
/// class.
public struct FeatureDistance: Sendable, Codable, Equatable {
    public let dim: Int
    /// L⁻¹, row-major [dim, dim].
    public let whitening: [Float]
    /// Whitened class means [classes][dim], in the order of the screen outputs.
    public let means: [[Float]]

    public init(dim: Int, whitening: [Float], means: [[Float]]) {
        self.dim = dim
        self.whitening = whitening
        self.means = means
    }

    /// Distance from `feature` to each class mean.
    public func distances(_ feature: [Float]) -> [Float] {
        var z = [Float](repeating: 0.0, count: dim)
        var r = 0
        while r < dim {
            var s: Float = 0.0
            var c = 0
            while c <= r {
                s += whitening[r * dim + c] * feature[c]
                c += 1
            }
            z[r] = s
            r += 1
        }
        return means.map { mean in
            var s: Float = 0.0
            var i = 0
            while i < dim {
                let e = z[i] - mean[i]
                s += e * e
                i += 1
            }
            return s.squareRoot()
        }
    }

    /// Fits the class means and the shared covariance of `features` [n][dim] grouped by `labels` (labels < 0
    /// are skipped). `shrinkage` blends the covariance with a scaled identity so that it stays invertible.
    public static func fit(features: [[Float]], labels: [Int], classCount: Int, shrinkage: Double = 0.05) -> FeatureDistance? {
        guard let dim = features.first?.count, 0 < dim else {
            return nil
        }
        var sums = [[Double]](repeating: [Double](repeating: 0.0, count: dim), count: classCount)
        var counts = [Int](repeating: 0, count: classCount)
        for (f, label) in zip(features, labels) where 0 <= label && label < classCount {
            for i in 0..<dim {
                sums[label][i] += Double(f[i])
            }
            counts[label] += 1
        }
        let means: [[Double]] = (0..<classCount).map { c in counts[c] == 0 ? sums[c] : sums[c].map { $0 / Double(counts[c]) } }
        var cov = [Double](repeating: 0.0, count: dim * dim)
        var n = 0
        for (f, label) in zip(features, labels) where 0 <= label && label < classCount {
            let m = means[label]
            for a in 0..<dim {
                let ea = Double(f[a]) - m[a]
                for b in 0...a {
                    cov[a * dim + b] += ea * (Double(f[b]) - m[b])
                }
            }
            n += 1
        }
        if n < 2 {
            return nil
        }
        var trace = 0.0
        for a in 0..<dim {
            for b in 0...a {
                cov[a * dim + b] /= Double(n - 1)
                cov[b * dim + a] = cov[a * dim + b]
            }
            trace += cov[a * dim + a]
        }
        let ridge = max(1e-6, trace / Double(dim))
        for a in 0..<dim {
            for b in 0..<dim {
                cov[a * dim + b] *= (1.0 - shrinkage)
            }
            cov[a * dim + a] += shrinkage * ridge
        }
        // Cholesky Σ = L Lᵀ, then L⁻¹ by forward substitution.
        var l = [Double](repeating: 0.0, count: dim * dim)
        for i in 0..<dim {
            for j in 0...i {
                var s = cov[i * dim + j]
                for k in 0..<j {
                    s -= l[i * dim + k] * l[j * dim + k]
                }
                if i == j {
                    if s <= 0 {
                        return nil
                    }
                    l[i * dim + i] = s.squareRoot()
                } else {
                    l[i * dim + j] = s / l[j * dim + j]
                }
            }
        }
        var inv = [Double](repeating: 0.0, count: dim * dim)
        for col in 0..<dim {
            for i in col..<dim {
                var s = i == col ? 1.0 : 0.0
                for k in col..<i {
                    s -= l[i * dim + k] * inv[k * dim + col]
                }
                inv[i * dim + col] = s / l[i * dim + i]
            }
        }
        let whitened: [[Float]] = means.map { m in
            (0..<dim).map { r in
                var s = 0.0
                for c in 0...r {
                    s += inv[r * dim + c] * m[c]
                }
                return Float(s)
            }
        }
        return FeatureDistance(dim: dim, whitening: inv.map { Float($0) }, means: whitened)
    }
}

/// Logistic regression p = σ(w · x + b) on a few inputs, fitted by damped Newton steps on the binary
/// negative log-likelihood with Platt-smoothed targets ((n₊ + 1) / (n₊ + 2) and 1 / (n₋ + 2)).
public enum LogisticRegression {
    public static func fit(inputs: [[Float]], targets: [Bool], iterations: Int = 100) -> (weights: [Float], bias: Float) {
        let n = min(inputs.count, targets.count)
        let k = inputs.first?.count ?? 0
        let positives = targets.prefix(n).filter { $0 }.count
        let negatives = n - positives
        if positives == 0 || negatives == 0 || k == 0 {
            // Nothing to separate: keep the first input as is.
            var identity = [Float](repeating: 0.0, count: k)
            if 0 < k {
                identity[0] = 1.0
            }
            return (identity, 0.0)
        }
        let high = (Double(positives) + 1.0) / (Double(positives) + 2.0)
        let low = 1.0 / (Double(negatives) + 2.0)
        let dim = k + 1
        var theta = [Double](repeating: 0.0, count: dim)
        theta[0] = 1.0
        func z(_ t: [Double], _ i: Int) -> Double {
            var s = t[k]
            for j in 0..<k {
                s += t[j] * Double(inputs[i][j])
            }
            return s
        }
        func loss(_ t: [Double]) -> Double {
            var total = 0.0
            for i in 0..<n {
                let v = z(t, i)
                let y = targets[i] ? high : low
                total += max(v, 0) - v * y + log(1 + exp(-abs(v)))
            }
            return total
        }
        var current = loss(theta)
        var it = 0
        while it < iterations {
            var g = [Double](repeating: 0.0, count: dim)
            var h = [Double](repeating: 0.0, count: dim * dim)
            for i in 0..<n {
                let p = 1.0 / (1.0 + exp(-z(theta, i)))
                let d = p - (targets[i] ? high : low)
                let w = p * (1 - p)
                var x = inputs[i].map { Double($0) }
                x.append(1.0)
                for a in 0..<dim {
                    g[a] += d * x[a]
                    for b in 0..<dim {
                        h[a * dim + b] += w * x[a] * x[b]
                    }
                }
            }
            for a in 0..<dim {
                h[a * dim + a] += 1e-6
            }
            guard let step = solve(h, g, dim) else {
                break
            }
            var scale = 1.0
            var accepted = false
            var tries = 0
            while tries < 30 {
                let candidate = (0..<dim).map { theta[$0] - scale * step[$0] }
                let value = loss(candidate)
                if value <= current {
                    theta = candidate
                    current = value
                    accepted = true
                    break
                }
                scale *= 0.5
                tries += 1
            }
            if accepted != true || (step.map { abs($0 * scale) }.max() ?? 0) < 1e-9 {
                break
            }
            it += 1
        }
        return (theta.prefix(k).map { Float($0) }, Float(theta[k]))
    }

    /// Solves H x = g by Gaussian elimination with partial pivoting; nil when singular.
    private static func solve(_ hIn: [Double], _ gIn: [Double], _ n: Int) -> [Double]? {
        var a = hIn
        var b = gIn
        for col in 0..<n {
            var pivot = col
            for r in (col + 1)..<max(col + 1, n) where abs(a[r * n + col]) > abs(a[pivot * n + col]) {
                pivot = r
            }
            if abs(a[pivot * n + col]) < 1e-15 {
                return nil
            }
            if pivot != col {
                for c in 0..<n {
                    a.swapAt(col * n + c, pivot * n + c)
                }
                b.swapAt(col, pivot)
            }
            for r in (col + 1)..<max(col + 1, n) {
                let f = a[r * n + col] / a[col * n + col]
                for c in col..<n {
                    a[r * n + c] -= f * a[col * n + c]
                }
                b[r] -= f * b[col]
            }
        }
        var x = [Double](repeating: 0.0, count: n)
        for r in stride(from: n - 1, through: 0, by: -1) {
            var s = b[r]
            for c in (r + 1)..<max(r + 1, n) {
                s -= a[r * n + c] * x[c]
            }
            x[r] = s / a[r * n + r]
        }
        return x
    }
}
