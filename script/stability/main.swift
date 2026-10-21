import Foundation
import Misago

// Compares classifiers trained the same way with different seeds on one held-out dataset. Images whose predicted
// class differs between the models, or whose probability for the true class spreads by more than --spread, are
// "unstable": the result depends on the seed. None unstable means the training is reproducible enough.

var evalPath = ""
var weightPaths: [String] = []
var maxSpread: Float = 0.3

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    func value() -> String? {
        if (argIdx + 1) < args.count {
            argIdx += 1
            return args[argIdx]
        }
        return nil
    }
    switch arg {
    case "-h", "--help":
        print("Usage: stability --eval <eval.msgo> --weights <a.json> --weights <b.json> ... [--spread 0.3]")
        print("  --spread <0-1>  Largest tolerated difference of the true-class probability between models (default 0.3)")
        exit(0)
    case "--eval":
        if let v = value() { evalPath = v }
    case "--weights":
        if let v = value() { weightPaths.append(v) }
    case "--spread":
        if let v = value(), let x = Float(v) { maxSpread = x }
    default:
        print("error: unknown argument \(arg)")
        exit(1)
    }
    argIdx += 1
}
if evalPath.isEmpty || weightPaths.count < 2 {
    print("error: specify --eval and at least two --weights. See stability --help.")
    exit(1)
}

let reader: DatasetReader
do {
    reader = try DatasetReader(url: URL(fileURLWithPath: evalPath))
} catch {
    print("error: cannot read dataset \(evalPath): \(error)")
    exit(1)
}
var paths = [String](repeating: "", count: reader.count)
if let data = try? Data(contentsOf: URL(fileURLWithPath: evalPath + ".meta.json")),
   let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
   let samples = meta["samples"] as? [[String: Any]], samples.count == reader.count {
    paths = samples.map { ($0["path"] as? String) ?? "" }
}

let labels = reader.labels
var features: [[Float]] = []
features.reserveCapacity(reader.count)
var i = 0
while i < reader.count {
    features.append(RowScanEncoder.encodeFlat(try! reader.sample(at: i).image))
    i += 1
}

/// Calibrated probabilities [model][sample][class], computed the same way `classify` does.
var probabilities: [[[Float]]] = []
for path in weightPaths {
    guard let weights = try? HawkWeights.load(from: URL(fileURLWithPath: path)), weights.classNames == reader.classes.names else {
        print("error: cannot read \(path) or its classes differ from the dataset")
        exit(1)
    }
    let network = HawkNetwork(weights: weights)
    let classCount = reader.classes.count
    var flat = [Float](repeating: 0, count: features.count * classCount)
    flat.withUnsafeMutableBufferPointer { buffer in
        let base = buffer.baseAddress!
        DispatchQueue.concurrentPerform(iterations: features.count) { k in
            let probs = network.probabilities(logits: network.logits(flatFeatures: features[k], frames: RowScanEncoder.frameCount))
            var c = 0
            while c < classCount {
                base[k * classCount + c] = probs[c]
                c += 1
            }
        }
    }
    probabilities.append((0..<features.count).map { Array(flat[($0 * classCount)..<(($0 + 1) * classCount)]) })
}

func argmax(_ v: [Float]) -> Int {
    var best = 0
    var c = 1
    while c < v.count {
        if v[best] < v[c] {
            best = c
        }
        c += 1
    }
    return best
}

// Screens only: samples labeled none (no class) are left out.
let labeled = (0..<labels.count).filter { 0 <= labels[$0] }
let predictions = probabilities.map { $0.map(argmax) }
for (m, path) in weightPaths.enumerated() {
    let correct = labeled.filter { predictions[m][$0] == labels[$0] }.count
    print("\(path): \(correct)/\(labeled.count) correct")
}
var a = 0
while a < weightPaths.count {
    var b = a + 1
    while b < weightPaths.count {
        let same = labeled.filter { predictions[a][$0] == predictions[b][$0] }.count
        print("agreement \(a)-\(b): \(same)/\(labeled.count) (\(String(format: "%.2f", Float(same) * 100 / Float(max(1, labeled.count))))%)")
        b += 1
    }
    a += 1
}

var unstable = 0
for k in labeled {
    let classesSeen = Set(predictions.map { $0[k] })
    let truth = probabilities.map { $0[k][labels[k]] }
    let spread = (truth.max() ?? 0) - (truth.min() ?? 0)
    if 1 < classesSeen.count || maxSpread < spread {
        unstable += 1
        let values = truth.map { String(format: "%.2f", $0) }.joined(separator: " ")
        print("  unstable \(paths[k].isEmpty ? "#\(k)" : paths[k])  label \(reader.classes[labels[k]])  p(label) \(values)")
    }
}
print("unstable: \(unstable)/\(labeled.count) (spread > \(maxSpread) or differing predictions)")
print(unstable == 0 ? "verdict: stable across seeds" : "verdict: results depend on the seed")
