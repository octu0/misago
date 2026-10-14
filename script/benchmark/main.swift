import Foundation
import Misago

setbuf(stdout, nil)

// Measures CPU inference speed of the screen classifier (single thread).
// Usage: benchmark [--weights <model.json>] [--iterations 200]
// Without weights, a model of the default shape with pseudo-random weights is used.

var weightsPath = ""
var iterations = 200
var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    switch args[argIdx] {
    case "-h", "--help":
        print("Usage: benchmark [--weights <model.json>] [--iterations 200]")
        exit(0)
    case "-w", "--weights":
        if (argIdx + 1) < args.count {
            argIdx += 1
            weightsPath = args[argIdx]
        }
    case "--iterations":
        if (argIdx + 1) < args.count {
            argIdx += 1
            if let v = Int(args[argIdx]) {
                iterations = max(1, v)
            }
        }
    default:
        print("error: unknown argument \(args[argIdx])")
        exit(1)
    }
    argIdx += 1
}

/// Deterministic pseudo-random weights of the default shape with three classes, for timing without a trained model.
func syntheticWeights() -> HawkWeights {
    let classes = try! ClassList(others: ["a_home", "b_home"])
    var state: UInt64 = 0x9E3779B97F4A7C15
    func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(Int64(bitPattern: state >> 11) % 2_000_000) / 1_000_000.0 - 1.0
    }
    func uniform(_ count: Int, _ fanIn: Int) -> [Float] {
        let bound = sqrt(3.0 / Float(fanIn))
        return (0..<count).map { _ in next() * bound }
    }
    let inDim = RowScanEncoder.frameDim
    let d = HawkShape.modelDim
    let r = HawkShape.rnnDim
    let m = HawkShape.mlpDim
    let hd = r / HawkShape.heads
    let taps = HawkShape.convWidth + HawkShape.convRight
    var layers: [HawkLayerWeights] = []
    var l = 0
    while l < HawkShape.numLayers {
        layers.append(HawkLayerWeights(
            norm1: [Float](repeating: 1.0, count: d), wGate: uniform(d * r, d), wX: uniform(d * r, d),
            convW: uniform(taps * r, taps), convB: [Float](repeating: 0.0, count: r),
            wA: uniform(HawkShape.heads * hd * hd, hd), bA: [Float](repeating: 0.0, count: r),
            wI: uniform(HawkShape.heads * hd * hd, hd), bI: [Float](repeating: 0.0, count: r),
            lambda: HawkWeights.initialLambda(rnnDim: r), wO: uniform(r * d, r),
            norm2: [Float](repeating: 1.0, count: d), wMLPGate: uniform(d * m, d), wMLPUp: uniform(d * m, d), wMLPDown: uniform(m * d, m)
        ))
        l += 1
    }
    return HawkWeights(
        inputDim: inDim, modelDim: d, rnnDim: r, mlpDim: m, heads: HawkShape.heads,
        convWidth: HawkShape.convWidth, convRight: HawkShape.convRight, outputDim: classes.count,
        wIn: uniform(inDim * d, inDim), bIn: uniform(d, 300), layers: layers,
        finalNorm: [Float](repeating: 1.0, count: d), wOut: uniform(d * classes.count, d), bOut: [Float](repeating: 0.0, count: classes.count),
        classNames: classes.names
    )
}

let weights: HawkWeights
if weightsPath.isEmpty {
    weights = syntheticWeights()
    print("weights: synthetic (default shape)")
} else {
    do {
        weights = try HawkWeights.load(from: URL(fileURLWithPath: weightsPath))
        print("weights: \(weightsPath)")
    } catch {
        print("error: cannot read weights \(weightsPath): \(error)")
        exit(1)
    }
}
print("Hawk: input \(weights.inputDim), layers \(weights.numLayers), width \(weights.modelDim), RNN width \(weights.rnnDim), MLP width \(weights.mlpDim), classes \(weights.outputDim), parameters \(weights.parameterCount)")
print("sequence: \(RowScanEncoder.frameCount) frames x \(RowScanEncoder.frameDim)")

let classifier = try! ImageClassifier(weights: weights)
var pixels = [Float](repeating: 0.0, count: ScreenImage.pixelCount)
var i = 0
while i < pixels.count {
    pixels[i] = Float((i * 7919) % 256) / 255.0
    i += 1
}
let image = try! ScreenImage(pixels: pixels)
// Warm up.
var checksum: Float = 0.0
checksum += classifier.classify(image).probability
var times: [Double] = []
var it = 0
while it < iterations {
    let start = CFAbsoluteTimeGetCurrent()
    checksum += classifier.classify(image).probability
    times.append((CFAbsoluteTimeGetCurrent() - start) * 1000.0)
    it += 1
}
times.sort()
let mean = times.reduce(0.0, +) / Double(times.count)
let median = times[times.count / 2]
let p95 = times[min(times.count - 1, Int(Double(times.count) * 0.95))]
print("\(iterations) iterations, single thread: mean \(String(format: "%.3f", mean)) ms, median \(String(format: "%.3f", median)) ms, p95 \(String(format: "%.3f", p95)) ms, min \(String(format: "%.3f", times[0])) ms (checksum \(checksum))")
