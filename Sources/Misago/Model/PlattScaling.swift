import Foundation

/// Platt scaling of one independent output: p = σ(scale · logit + bias), fitted by minimizing the binary
/// negative log-likelihood on held-out data with damped Newton steps. Targets are smoothed as in Platt (1999),
/// (n₊ + 1) / (n₊ + 2) and 1 / (n₋ + 2), so separable data keeps a finite scale.
public enum PlattScaling {
    public static func fit(logits: [Float], targets: [Bool], iterations: Int = 100) -> (scale: Float, bias: Float) {
        let n = min(logits.count, targets.count)
        let positives = targets.prefix(n).filter { $0 }.count
        let negatives = n - positives
        if positives == 0 || negatives == 0 {
            return (1.0, 0.0)
        }
        let high = (Double(positives) + 1.0) / (Double(positives) + 2.0)
        let low = 1.0 / (Double(negatives) + 2.0)
        func loss(_ a: Double, _ b: Double) -> Double {
            var total = 0.0
            var i = 0
            while i < n {
                let z = a * Double(logits[i]) + b
                let y = targets[i] ? high : low
                // −[y log σ(z) + (1 − y) log(1 − σ(z))], written stably.
                total += max(z, 0) - z * y + log(1 + exp(-abs(z)))
                i += 1
            }
            return total
        }
        var a = 1.0
        var b = 0.0
        var current = loss(a, b)
        var it = 0
        while it < iterations {
            var ga = 0.0, gb = 0.0, haa = 0.0, hab = 0.0, hbb = 0.0
            var i = 0
            while i < n {
                let x = Double(logits[i])
                let p = 1.0 / (1.0 + exp(-(a * x + b)))
                let d = p - (targets[i] ? high : low)
                let w = p * (1 - p)
                ga += d * x
                gb += d
                haa += w * x * x
                hab += w * x
                hbb += w
                i += 1
            }
            haa += 1e-6
            hbb += 1e-6
            let det = haa * hbb - hab * hab
            if abs(det) < 1e-18 {
                break
            }
            let stepA = (hbb * ga - hab * gb) / det
            let stepB = (haa * gb - hab * ga) / det
            var scale = 1.0
            var accepted = false
            var tries = 0
            while tries < 30 {
                let na = a - scale * stepA
                let nb = b - scale * stepB
                let value = loss(na, nb)
                if value <= current {
                    a = na
                    b = nb
                    current = value
                    accepted = true
                    break
                }
                scale *= 0.5
                tries += 1
            }
            if accepted != true || max(abs(scale * stepA), abs(scale * stepB)) < 1e-9 {
                break
            }
            it += 1
        }
        return (Float(a), Float(b))
    }
}
