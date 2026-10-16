import Foundation
import Misago
import MisagoTraining
import MLX

setbuf(stdout, nil)

print("==================================================")
print("=== misago train-digits / eval ===")
print("==================================================")

enum Defaults {
    static let lrMax: Float = 0.002
    static let lrMin: Float = 0.0002
    static let warmupStepDivisor = 20
    static let minWarmupSteps = 20
    static let batchSize = 64
    static let clipNorm: Float = 5.0
    static let parityCheckSamples = 8
}

var epochs = 30
var batchSize = Defaults.batchSize
var trainPaths: [String] = []
var evalPath = ""
var exportWeightsPath: String? = nil
var importWeightsPath: String? = nil
var evalDumpPath = ""
var seed: UInt64 = 7

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
        print("Usage: train-digits --train <train.msgd> [--train <more.msgd>] [--eval <eval.msgd>] [options]")
        print("Options:")
        print("  --train <path>             Training dataset (repeatable; datasets are concatenated)")
        print("  -e, --epochs <Int>         Number of epochs (default: 30; 0 = eval only)")
        print("  -b, --batch-size <Int>     Mini-batch size (default: \(Defaults.batchSize))")
        print("  --seed <Int>               Random seed (default: 7)")
        print("  --export-weights <path>    Output path for the trained weights (JSON)")
        print("  --import-weights <path>    Initial weights (eval only with -e 0)")
        print("  --eval-dump <jsonl>        Write per-sample eval results (with source paths when <eval>.meta.json exists)")
        exit(0)
    case "--train":
        if let v = value() { trainPaths.append(v) }
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
    default:
        print("error: unknown argument \(arg)")
        exit(1)
    }
    argIdx += 1
}
if trainPaths.isEmpty {
    print("error: specify --train <train.msgd>. See train-digits --help.")
    exit(1)
}

/// Several dataset files read as one.
final class DatasetUnion: @unchecked Sendable {
    let readers: [DigitDatasetReader]
    let offsets: [Int]
    let count: Int
    init(readers: [DigitDatasetReader]) {
        self.readers = readers
        var offsets: [Int] = []
        var total = 0
        for r in readers {
            offsets.append(total)
            total += r.count
        }
        self.offsets = offsets
        self.count = total
    }
    private func locate(_ index: Int) -> (DigitDatasetReader, Int) {
        var r = readers.count - 1
        while 0 < r && index < offsets[r] {
            r -= 1
        }
        return (readers[r], index - offsets[r])
    }
    func sample(at index: Int) throws -> DigitSample {
        let (reader, local) = locate(index)
        return try reader.sample(at: local)
    }
    var allDigits: [String] {
        return readers.flatMap { $0.allDigits }
    }
}

func openDataset(_ path: String) -> DigitDatasetReader {
    do {
        return try DigitDatasetReader(url: URL(fileURLWithPath: path))
    } catch {
        print("error: cannot read dataset \(path): \(error)")
        exit(1)
    }
}
func describe(_ name: String, count: Int, digits: [String]) {
    var lengths: [Int: Int] = [:]
    for d in digits {
        lengths[d.count, default: 0] += 1
    }
    let summary = lengths.keys.sorted().map { "\($0) digits: \(lengths[$0]!)" }.joined(separator: ", ")
    print("\(name): \(count) samples (\(summary))")
}
let trainSet = DatasetUnion(readers: trainPaths.map { openDataset($0) })
describe("train", count: trainSet.count, digits: trainSet.allDigits)
var evalSet: DatasetUnion? = nil
if evalPath.isEmpty != true {
    let reader = DatasetUnion(readers: [openDataset(evalPath)])
    describe("eval", count: reader.count, digits: reader.allDigits)
    evalSet = reader
}

var importedWeights: DigitReaderWeights? = nil
if let path = importWeightsPath {
    guard let w = try? DigitReaderWeights.load(from: URL(fileURLWithPath: path)) else {
        print("error: cannot read weights \(path)")
        exit(1)
    }
    guard w.hawk.inputDim == ConvStemWeights.frameDim, w.hawk.classNames == DigitVocabulary.names else {
        print("error: weights do not match the stem output or the digit vocabulary")
        exit(1)
    }
    importedWeights = w
    print("import weights: \(path) (\(w.parameterCount) parameters)")
}
if epochs == 0 && importedWeights == nil {
    print("error: -e 0 needs --import-weights")
    exit(1)
}

func batch(_ reader: DatasetUnion, _ indices: [Int]) -> (features: [[Float]], labels: [String]) {
    var features: [[Float]] = []
    var labels: [String] = []
    for i in indices {
        let sample = try! reader.sample(at: i)
        features.append(sample.strip.pixels)
        labels.append(sample.digits)
    }
    return (features, labels)
}

struct EvalResult {
    let readings: [DigitReading]
    let labels: [String]
    var exact: Int {
        var n = 0
        var i = 0
        while i < readings.count {
            if readings[i].digits == labels[i] {
                n += 1
            }
            i += 1
        }
        return n
    }
    /// Character error rate: edit distance over label characters.
    var cer: Float {
        var distance = 0
        var total = 0
        var i = 0
        while i < readings.count {
            distance += editDistance(readings[i].digits, labels[i])
            total += max(1, labels[i].count)
            i += 1
        }
        return Float(distance) / Float(max(1, total))
    }
}

@MainActor func evaluate(_ trainer: MLXCTCTrainer, _ reader: DatasetUnion) -> EvalResult {
    var readings: [DigitReading] = []
    var start = 0
    while start < reader.count {
        let end = min(reader.count, start + batchSize)
        readings.append(contentsOf: trainer.read(features: batch(reader, Array(start..<end)).features))
        start = end
    }
    return EvalResult(readings: readings, labels: reader.allDigits)
}

func report(_ name: String, _ r: EvalResult) {
    print("  \(name): exact \(r.exact)/\(r.labels.count) (\(String(format: "%.2f", Float(r.exact) / Float(max(1, r.labels.count)) * 100))%), CER \(String(format: "%.4f", r.cer))")
}

MLXRandom.seed(seed)
let mlxNetwork = MLXDigitModel()
if let w = importedWeights {
    mlxNetwork.importWeights(from: w)
}
let trainer = MLXCTCTrainer(network: mlxNetwork, learningRate: Defaults.lrMax, clipNorm: Defaults.clipNorm)
print("stem: 3x3 stride 2, \(ConvStemWeights.outChannels) channels -> \(ConvStemWeights.frameCount) frames x \(ConvStemWeights.frameDim)")
print("Hawk: layers \(DigitHawkShape.numLayers), width \(DigitHawkShape.modelDim), RNN width \(DigitHawkShape.rnnDim), MLP width \(DigitHawkShape.mlpDim), vocabulary \(DigitVocabulary.size)")
print("  parameters \(mlxNetwork.exportWeights().parameterCount), batch \(batchSize), epochs \(epochs), seed \(seed)")

var bestWeights: DigitReaderWeights = importedWeights ?? mlxNetwork.exportWeights()

if 0 < epochs {
    print("\n--- Training ---")
    let scheduler = CosineLRScheduler(lrMax: Defaults.lrMax, lrMin: Defaults.lrMin)
    let stepsPerEpoch = (trainSet.count + batchSize - 1) / batchSize
    let totalSteps = max(1, epochs * stepsPerEpoch)
    let warmupSteps = min(max(Defaults.minWarmupSteps, totalSteps / Defaults.warmupStepDivisor), max(1, totalSteps / 2))
    print("  learning rate \(Defaults.lrMax) -> \(Defaults.lrMin), steps \(totalSteps) (warmup \(warmupSteps)), grad clip \(Defaults.clipNorm)")
    var rng = SystemRandomNumberGenerator()
    var indices = Array(0..<trainSet.count)
    // Selection: highest eval exact-match count, ties broken by lower CER (train loss when there is no eval set).
    var bestScore: (exact: Int, cer: Float) = (-1, Float.greatestFiniteMagnitude)
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
        var score: (exact: Int, cer: Float) = (0, trainLoss)
        if let evalSet = evalSet {
            let r = evaluate(trainer, evalSet)
            line += ", eval exact \(r.exact)/\(r.labels.count), CER \(String(format: "%.4f", r.cer))"
            score = (r.exact, r.cer)
        }
        line += " (\(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - epStart)) s, grad norm \(String(format: "%.2f", trainer.lastGradientNorm)))"
        if bestScore.exact < score.exact || (bestScore.exact == score.exact && score.cer < bestScore.cer) {
            bestScore = score
            bestEpoch = ep
            bestWeights = mlxNetwork.exportWeights()
            line += " *"
        }
        print(line)
        ep += 1
    }
    print("  best epoch \(bestEpoch), \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - trainStart) / 60.0)) min")
    mlxNetwork.importWeights(from: bestWeights)
}

print("\n--- Evaluation ---")
if let evalSet = evalSet {
    let r = evaluate(trainer, evalSet)
    report("eval", r)
    if evalDumpPath.isEmpty != true {
        var paths: [String] = []
        if let metaData = try? Data(contentsOf: URL(fileURLWithPath: evalPath + ".meta.json")),
           let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any],
           let samples = meta["samples"] as? [[String: Any]] {
            paths = samples.map { ($0["path"] as? String) ?? "" }
        }
        var lines: [String] = []
        var i = 0
        while i < r.readings.count {
            var row: [String: Any] = [
                "index": i, "label": r.labels[i], "read": r.readings[i].digits,
                "correct": r.readings[i].digits == r.labels[i],
                "confidence": Double(String(format: "%.4f", r.readings[i].confidence)) ?? 0.0
            ]
            if i < paths.count {
                row["path"] = paths[i]
            }
            if let data = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]), let text = String(data: data, encoding: .utf8) {
                lines.append(text)
            }
            i += 1
        }
        try? (lines.joined(separator: "\n") + "\n").write(toFile: evalDumpPath, atomically: true, encoding: .utf8)
        print("  eval dump: \(evalDumpPath)")
    }
}
let trainResult = evaluate(trainer, trainSet)
report("train", trainResult)

// The exported weights must read the same on the CPU path.
let exported = mlxNetwork.exportWeights()
let cpu = try! DigitReader(weights: exported)
var mismatches = 0
var k = 0
while k < min(Defaults.parityCheckSamples, trainSet.count) {
    let sample = try! trainSet.sample(at: k)
    if cpu.read(sample.strip).digits != trainResult.readings[k].digits {
        mismatches += 1
    }
    k += 1
}
print("  MLX vs CPU greedy decode: \(mismatches) mismatches over \(k) samples")

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
