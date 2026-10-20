import Foundation
import Misago
import MisagoTraining
import MLX

setbuf(stdout, nil)

print("==================================================")
print("=== misago train / eval ===")
print("==================================================")

/// Training defaults. Edit these values directly instead of adding command-line options.
enum Defaults {
    static let lrMax: Float = 0.002
    static let lrMin: Float = 0.0002
    static let warmupStepDivisor = 20
    static let minWarmupSteps = 20
    static let batchSize = 64
    static let clipNorm: Float = 5.0
    /// Samples compared between MLX and the CPU network after training.
    static let parityCheckSamples = 8
}

var epochs = 20
var batchSize = Defaults.batchSize
var trainPath = ""
var evalPath = ""
var exportWeightsPath: String? = nil
var importWeightsPath: String? = nil
var evalDumpPath = ""
var centerMask: [Float]? = nil
var seed: UInt64 = 9

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
        print("Usage: train --train <train.msgo> [--eval <eval.msgo>] [options]")
        print("Options:")
        print("  --train <path>             Training dataset (prepare output) [required]")
        print("  --eval <path>              Held-out dataset: model selection and probability calibration")
        print("  -e, --epochs <Int>         Number of epochs (default: 20; 0 = eval only)")
        print("  -b, --batch-size <Int>     Mini-batch size (default: \(Defaults.batchSize))")
        print("  --seed <Int>               Random seed (default: 9)")
        print("  --export-weights <path>    Output path for the trained weights (JSON)")
        print("  --import-weights <path>    Initial weights (eval only with -e 0)")
        print("  --eval-dump <jsonl>        Write per-sample eval results (with source paths when <eval>.meta.json exists)")
        print("  --center-mask x0,y0,x1,y1  Record the mask the datasets were built with, so inference applies it too")
        exit(0)
    case "--train":
        if let v = value() { trainPath = v }
    case "--eval":
        if let v = value() { evalPath = v }
    case "-e", "--epochs":
        if let v = value(), let n = Int(v) { epochs = max(0, n) }
    case "-b", "--batch-size":
        if let v = value(), let n = Int(v) { batchSize = max(1, n) }
    case "--seed":
        if let v = value(), let n = UInt64(v) { seed = n }
    case "--export-weights":
        if let v = value() { exportWeightsPath = v }
    case "--import-weights":
        if let v = value() { importWeightsPath = v }
    case "--eval-dump":
        if let v = value() { evalDumpPath = v }
    case "--center-mask":
        if let v = value() {
            let parts = v.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
            if parts.count == 4 {
                centerMask = parts
            }
        }
    default:
        print("error: unknown argument \(arg)")
        exit(1)
    }
    argIdx += 1
}
if trainPath.isEmpty {
    print("error: specify --train <train.msgo>. See train --help.")
    exit(1)
}

func openDataset(_ path: String) -> DatasetReader {
    do {
        return try DatasetReader(url: URL(fileURLWithPath: path))
    } catch {
        print("error: cannot read dataset \(path): \(error)")
        exit(1)
    }
}
func describe(_ name: String, _ reader: DatasetReader) {
    let counts = reader.classCounts.enumerated().map { "\(reader.classes[$0.offset]) \($0.element)" }.joined(separator: ", ")
    print("\(name): \(reader.count) samples (\(counts))")
}
let trainSet = openDataset(trainPath)
let classes = trainSet.classes
describe("train", trainSet)
var evalSet: DatasetReader? = nil
if evalPath.isEmpty != true {
    let reader = openDataset(evalPath)
    if reader.classes != classes {
        print("error: eval classes \(reader.classes.names) differ from train classes \(classes.names)")
        exit(1)
    }
    describe("eval", reader)
    evalSet = reader
}

var importedWeights: HawkWeights? = nil
if let path = importWeightsPath {
    guard let w = try? HawkWeights.load(from: URL(fileURLWithPath: path)) else {
        print("error: cannot read weights \(path)")
        exit(1)
    }
    guard w.inputDim == RowScanEncoder.frameDim, w.classNames == classes.names else {
        print("error: weights do not match the input (\(w.inputDim)/\(RowScanEncoder.frameDim)) or classes (\(w.classNames)/\(classes.names))")
        exit(1)
    }
    importedWeights = w
    print("import weights: \(path) (\(w.parameterCount) parameters)")
}
if epochs == 0 && importedWeights == nil {
    print("error: -e 0 needs --import-weights")
    exit(1)
}

/// Flat frame sequences and labels for a list of sample indices.
func batch(_ reader: DatasetReader, _ indices: [Int]) -> (features: [[Float]], labels: [Int]) {
    var features: [[Float]] = []
    var labels: [Int] = []
    features.reserveCapacity(indices.count)
    for i in indices {
        let sample = try! reader.sample(at: i)
        features.append(RowScanEncoder.encodeFlat(sample.image))
        labels.append(sample.label)
    }
    return (features, labels)
}

/// Logits for every sample of a dataset, in order.
@MainActor func allLogits(_ trainer: MLXClassificationTrainer, _ reader: DatasetReader) -> [[Float]] {
    var out: [[Float]] = []
    out.reserveCapacity(reader.count)
    var start = 0
    while start < reader.count {
        let end = min(reader.count, start + batchSize)
        let b = batch(reader, Array(start..<end))
        out.append(contentsOf: trainer.logits(features: b.features))
        start = end
    }
    return out
}

struct EvalResult {
    let loss: Float
    let logits: [[Float]]
    let labels: [Int]

    /// Accuracy and confusion matrix [label][predicted] under a calibration (identity = raw logits).
    func summary(calibration: VectorCalibration) -> (accuracy: Float, confusion: [[Int]], probabilities: [[Float]]) {
        var correct = 0
        var confusion = [[Int]](repeating: [Int](repeating: 0, count: classes.count), count: classes.count)
        var probabilities: [[Float]] = []
        probabilities.reserveCapacity(logits.count)
        var i = 0
        while i < logits.count {
            let probs = calibration.probabilities(logits[i])
            var predicted = 0
            var c = 1
            while c < probs.count {
                if probs[predicted] < probs[c] {
                    predicted = c
                }
                c += 1
            }
            if predicted == labels[i] {
                correct += 1
            }
            confusion[labels[i]][predicted] += 1
            probabilities.append(probs)
            i += 1
        }
        return (Float(correct) / Float(max(1, logits.count)), confusion, probabilities)
    }
}

@MainActor func evaluate(_ trainer: MLXClassificationTrainer, _ reader: DatasetReader) -> EvalResult {
    let logits = allLogits(trainer, reader)
    let labels = reader.labels
    var lossSum: Float = 0.0
    var i = 0
    while i < logits.count {
        lossSum -= HawkNetwork.logSoftmax(logits[i])[labels[i]]
        i += 1
    }
    return EvalResult(loss: lossSum / Float(max(1, logits.count)), logits: logits, labels: labels)
}

/// Accuracy, confusion and ECE, all judged on the calibrated probabilities (the same rule `classify` uses).
func report(_ name: String, _ r: EvalResult, calibration: VectorCalibration) {
    let s = r.summary(calibration: calibration)
    print("  \(name): loss \(String(format: "%.4f", r.loss)), accuracy \(String(format: "%.2f", s.accuracy * 100))% (\(r.logits.count) samples, calibrated)")
    print("    confusion [label -> predicted]:")
    var a = 0
    while a < classes.count {
        let row = s.confusion[a].enumerated().map { "\(classes[$0.offset]) \($0.element)" }.joined(separator: ", ")
        print("      \(classes[a]) -> \(row)")
        a += 1
    }
    let ece = VectorCalibration.expectedCalibrationError(probabilities: s.probabilities, labels: r.labels)
    let bias = calibration.bias.map { String(format: "%.3f", $0) }.joined(separator: ", ")
    print("    ECE \(String(format: "%.4f", ece.ece)) (scale \(String(format: "%.3f", calibration.scale)), bias [\(bias)])")
}

MLXRandom.seed(seed)
let mlxNetwork = MLXHawkNetwork(classCount: classes.count)
if let w = importedWeights {
    mlxNetwork.importWeights(from: w)
}
let trainer = MLXClassificationTrainer(network: mlxNetwork, learningRate: Defaults.lrMax, clipNorm: Defaults.clipNorm)
print("Hawk: input \(RowScanEncoder.frameDim) x \(RowScanEncoder.frameCount) frames, layers \(HawkShape.numLayers), width \(HawkShape.modelDim), RNN width \(HawkShape.rnnDim), MLP width \(HawkShape.mlpDim), gate blocks \(HawkShape.heads), conv \(HawkShape.convWidth)+\(HawkShape.convRight), classes \(classes.count)")
print("  parameters \(mlxNetwork.exportWeights(classes: classes).parameterCount), batch \(batchSize), epochs \(epochs), seed \(seed)")

var bestWeights: HawkWeights = importedWeights ?? mlxNetwork.exportWeights(classes: classes)
var calibration = bestWeights.calibration

if 0 < epochs {
    print("\n--- Training ---")
    let scheduler = CosineLRScheduler(lrMax: Defaults.lrMax, lrMin: Defaults.lrMin)
    let stepsPerEpoch = (trainSet.count + batchSize - 1) / batchSize
    let totalSteps = max(1, epochs * stepsPerEpoch)
    let warmupSteps = min(max(Defaults.minWarmupSteps, totalSteps / Defaults.warmupStepDivisor), max(1, totalSteps / 2))
    print("  learning rate \(Defaults.lrMax) -> \(Defaults.lrMin), steps \(totalSteps) (warmup \(warmupSteps)), grad clip \(Defaults.clipNorm)")
    // The batch order comes from the seed too, so one seed always gives the same run.
    var rng = SplitMix64(seed: seed)
    var indices = Array(0..<trainSet.count)
    // Selection: highest eval accuracy, ties broken by lower eval loss. Loss alone picks very early epochs
    // when the frequent classes converge first and the rare, diverse ones are still underfit; calibration
    // is fitted afterwards, so a somewhat larger loss costs nothing.
    var bestScore: (accuracy: Float, loss: Float) = (-1.0, Float.greatestFiniteMagnitude)
    var bestEpoch = 0
    var globalStep = 0
    let trainStart = CFAbsoluteTimeGetCurrent()
    var ep = 1
    while ep <= epochs {
        let epStart = CFAbsoluteTimeGetCurrent()
        indices.shuffle(using: &rng)
        var lossSum: Float = 0.0
        var steps = 0
        var start = 0
        while start < indices.count {
            let end = min(indices.count, start + batchSize)
            let b = batch(trainSet, Array(indices[start..<end]))
            globalStep += 1
            trainer.updateLearningRate(scheduler.learningRate(step: globalStep, totalSteps: totalSteps, warmupSteps: warmupSteps))
            let loss = trainer.trainBatch(features: b.features, labels: b.labels)
            if loss.isFinite != true || trainer.lastGradientNorm.isFinite != true {
                print("    ⚠ loss \(loss) / grad norm \(trainer.lastGradientNorm) (step \(globalStep))")
            }
            lossSum += loss
            steps += 1
            start = end
        }
        let trainLoss = lossSum / Float(max(1, steps))
        var line = "epoch \(ep)/\(epochs): train loss \(String(format: "%.4f", trainLoss))"
        var score: (accuracy: Float, loss: Float) = (0.0, trainLoss)
        if let evalSet = evalSet {
            let r = evaluate(trainer, evalSet)
            let accuracy = r.summary(calibration: .identity(classCount: classes.count)).accuracy
            line += ", eval loss \(String(format: "%.4f", r.loss)), eval accuracy \(String(format: "%.2f", accuracy * 100))%"
            score = (accuracy, r.loss)
        }
        line += " (\(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - epStart)) s, grad norm \(String(format: "%.2f", trainer.lastGradientNorm)))"
        if bestScore.accuracy < score.accuracy || (bestScore.accuracy == score.accuracy && score.loss < bestScore.loss) {
            bestScore = score
            bestEpoch = ep
            bestWeights = mlxNetwork.exportWeights(classes: classes)
            line += " *"
        }
        print(line)
        ep += 1
    }
    print("  best epoch \(bestEpoch) (accuracy \(String(format: "%.2f", bestScore.accuracy * 100))%, loss \(String(format: "%.4f", bestScore.loss))), \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - trainStart) / 60.0)) min")
    mlxNetwork.importWeights(from: bestWeights)
}

print("\n--- Evaluation ---")
if let evalSet = evalSet {
    let r = evaluate(trainer, evalSet)
    if 0 < epochs {
        // The held-out set also selects the epoch, so this calibration is slightly optimistic; a separate set would be cleaner.
        calibration = VectorCalibration.fit(logits: r.logits, labels: r.labels, classCount: classes.count)
        print("  calibration fitted on eval: scale \(String(format: "%.3f", calibration.scale))")
    }
    report("eval", r, calibration: calibration)
    if evalDumpPath.isEmpty != true {
        var paths: [String] = []
        if let metaData = try? Data(contentsOf: URL(fileURLWithPath: evalPath + ".meta.json")),
           let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any],
           let samples = meta["samples"] as? [[String: Any]] {
            paths = samples.map { ($0["path"] as? String) ?? "" }
        }
        var lines: [String] = []
        let s = r.summary(calibration: calibration)
        var i = 0
        while i < r.logits.count {
            let probs = s.probabilities[i]
            var predicted = 0
            var c = 1
            while c < probs.count {
                if probs[predicted] < probs[c] {
                    predicted = c
                }
                c += 1
            }
            var row: [String: Any] = [
                "index": i,
                "label": classes[r.labels[i]],
                "predicted": classes[predicted],
                "correct": predicted == r.labels[i],
                "probabilities": probs.map { Double(String(format: "%.4f", $0)) ?? 0.0 }
            ]
            if i < paths.count {
                row["path"] = paths[i]
            }
            if let data = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]), let text = String(data: data, encoding: .utf8) {
                lines.append(text)
            }
            i += 1
        }
        do {
            try (lines.joined(separator: "\n") + "\n").write(toFile: evalDumpPath, atomically: true, encoding: .utf8)
            print("  eval dump: \(evalDumpPath)")
        } catch {
            print("error: cannot write \(evalDumpPath): \(error)")
        }
    }
}
let trainResult = evaluate(trainer, trainSet)
report("train", trainResult, calibration: calibration)

// The exported weights must give the same logits on the CPU path.
var exported = mlxNetwork.exportWeights(classes: classes, calibration: calibration)
exported.centerMask = centerMask ?? importedWeights?.centerMask
let cpu = HawkNetwork(weights: exported)
var worst: Float = 0.0
var k = 0
while k < min(Defaults.parityCheckSamples, trainSet.count) {
    let sample = try! trainSet.sample(at: k)
    let got = cpu.logits(flatFeatures: RowScanEncoder.encodeFlat(sample.image), frames: RowScanEncoder.frameCount)
    let want = trainResult.logits[k]
    var c = 0
    while c < want.count {
        worst = max(worst, abs(got[c] - want[c]))
        c += 1
    }
    k += 1
}
print("  MLX vs CPU logits: max abs diff \(String(format: "%.2e", worst)) over \(k) samples")

if let path = exportWeightsPath {
    let url = URL(fileURLWithPath: path)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    do {
        try exported.save(to: url)
        print("exported weights: \(path) (\(exported.parameterCount) parameters)")
    } catch {
        print("error: cannot write weights \(path): \(error)")
        exit(1)
    }
}
