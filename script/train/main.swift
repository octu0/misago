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
    static let labelSmoothing: Float = 0.0
    static let weightDecay: Float = 0.0
    static let biasCorrection = false
    /// Calibration fits per-class biases too (vector scaling) instead of the temperature alone.
    static let calibrationBias = false
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
    print("\(name): \(reader.count) samples (\(counts), none \(reader.noneCount))")
}
let trainSet = openDataset(trainPath)
let classes = trainSet.classes
/// Independent title outputs follow the screens (the title of `<title>_<screen>` is the part before "_").
let titles = classes.titles
let titleOf = classes.names.map { titles.firstIndex(of: ClassList.title(of: $0))! }
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

/// Held-out outputs split into screens (samples with a class) and titles (every sample).
struct EvalResult {
    /// Screen logits and labels of the samples that have a class.
    let screenLogits: [[Float]]
    let labels: [Int]
    /// Sample index (in file order) of each screen row.
    let screenRows: [Int]
    /// Title logits [sample][title] and the label of every sample (−1 = none).
    let titleLogits: [[Float]]
    let allLabels: [Int]
    /// Raw screen NLL.
    let loss: Float
    /// Full logits (screens then titles), for the parity check.
    let logits: [[Float]]
    /// Per sample and title, the feature distance to the title's nearest training class (empty until fitted).
    var titleDistances: [[Float]] = []

    /// Accuracy and confusion matrix [label][predicted] under a calibration (identity = raw logits).
    func summary(calibration: VectorCalibration) -> (accuracy: Float, confusion: [[Int]], probabilities: [[Float]]) {
        var correct = 0
        var confusion = [[Int]](repeating: [Int](repeating: 0, count: classes.count), count: classes.count)
        var probabilities: [[Float]] = []
        probabilities.reserveCapacity(screenLogits.count)
        var i = 0
        while i < screenLogits.count {
            let probs = calibration.probabilities(screenLogits[i])
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
        return (Float(correct) / Float(max(1, screenLogits.count)), confusion, probabilities)
    }

    /// Whether sample `i` belongs to title `t`.
    func isTitle(_ i: Int, _ t: Int) -> Bool {
        return 0 <= allLabels[i] && titleOf[allLabels[i]] == t
    }

    /// Per-title Platt scaling fitted on these samples.
    func fitTitleCalibration() -> [Float] {
        var out: [Float] = []
        var t = 0
        while t < titles.count {
            let fit = PlattScaling.fit(logits: titleLogits.map { $0[t] }, targets: (0..<allLabels.count).map { isTitle($0, t) })
            out.append(fit.scale)
            out.append(fit.bias)
            t += 1
        }
        return out
    }

    func titleProbability(_ i: Int, _ t: Int, _ calibration: [Float], _ distanceWeights: [Float] = []) -> Float {
        var z = calibration[2 * t] * titleLogits[i][t] + calibration[2 * t + 1]
        if t < distanceWeights.count && i < titleDistances.count {
            z += distanceWeights[t] * titleDistances[i][t]
        }
        return 1.0 / (1.0 + exp(-z))
    }

    /// Mean binary NLL of the title outputs under a calibration.
    func titleLoss(_ calibration: [Float]) -> Float {
        if titles.isEmpty || allLabels.isEmpty {
            return 0.0
        }
        var total: Float = 0.0
        var i = 0
        while i < allLabels.count {
            var t = 0
            while t < titles.count {
                let p = titleProbability(i, t, calibration)
                total -= log(max(isTitle(i, t) ? p : 1.0 - p, 1e-12))
                t += 1
            }
            i += 1
        }
        return total / Float(allLabels.count * titles.count)
    }
}

@MainActor func evaluate(_ trainer: MLXClassificationTrainer, _ reader: DatasetReader) -> EvalResult {
    let logits = allLogits(trainer, reader)
    let allLabels = reader.labels
    var screenLogits: [[Float]] = []
    var labels: [Int] = []
    var rows: [Int] = []
    var titleLogits: [[Float]] = []
    var lossSum: Float = 0.0
    var i = 0
    while i < logits.count {
        let screens = Array(logits[i].prefix(classes.count))
        titleLogits.append(Array(logits[i].dropFirst(classes.count)))
        if 0 <= allLabels[i] {
            screenLogits.append(screens)
            labels.append(allLabels[i])
            rows.append(i)
            lossSum -= HawkNetwork.logSoftmax(screens)[allLabels[i]]
        }
        i += 1
    }
    return EvalResult(
        screenLogits: screenLogits, labels: labels, screenRows: rows, titleLogits: titleLogits, allLabels: allLabels,
        loss: lossSum / Float(max(1, labels.count)), logits: logits
    )
}

/// Pooled features of every sample of a dataset, in order.
@MainActor func allPooled(_ trainer: MLXClassificationTrainer, _ reader: DatasetReader) -> [[Float]] {
    var out: [[Float]] = []
    out.reserveCapacity(reader.count)
    var start = 0
    while start < reader.count {
        let end = min(reader.count, start + batchSize)
        out.append(contentsOf: trainer.pooled(features: batch(reader, Array(start..<end)).features))
        start = end
    }
    return out
}

/// Per sample and title, the distance to the nearest class mean of that title.
func titleDistances(_ pooled: [[Float]], _ distance: FeatureDistance) -> [[Float]] {
    return pooled.map { feature in
        let perClass = distance.distances(feature)
        var out = [Float](repeating: Float.greatestFiniteMagnitude, count: titles.count)
        for (c, d) in perClass.enumerated() {
            out[titleOf[c]] = min(out[titleOf[c]], d)
        }
        return out
    }
}

/// Accuracy, confusion and ECE, all judged on the calibrated probabilities (the same rule `classify` uses), then
/// the title outputs: mean probability on the title's own samples, on other titles' samples and on none.
func report(_ name: String, _ r: EvalResult, calibration: VectorCalibration, titleCalibration: [Float], distanceWeights: [Float] = []) {
    let s = r.summary(calibration: calibration)
    print("  \(name): loss \(String(format: "%.4f", r.loss)), accuracy \(String(format: "%.2f", s.accuracy * 100))% (\(r.labels.count) samples with a class, calibrated)")
    print("    confusion [label -> predicted]:")
    var a = 0
    while a < classes.count {
        let row = s.confusion[a].enumerated().filter { 0 < $0.element }.map { "\(classes[$0.offset]) \($0.element)" }.joined(separator: ", ")
        print("      \(classes[a]) -> \(row)")
        a += 1
    }
    let ece = VectorCalibration.expectedCalibrationError(probabilities: s.probabilities, labels: r.labels)
    print("    ECE \(String(format: "%.4f", ece.ece)) (scale \(String(format: "%.3f", calibration.scale)))")
    var t = 0
    while t < titles.count {
        var own: [Float] = [], others: [Float] = [], none: [Float] = []
        var i = 0
        while i < r.allLabels.count {
            let p = r.titleProbability(i, t, titleCalibration, distanceWeights)
            if r.isTitle(i, t) {
                own.append(p)
            } else if r.allLabels[i] < 0 {
                none.append(p)
            } else {
                others.append(p)
            }
            i += 1
        }
        func mean(_ v: [Float]) -> String {
            return v.isEmpty ? "-" : String(format: "%.3f", v.reduce(0, +) / Float(v.count))
        }
        let ownCorrect = own.filter { 0.5 <= $0 }.count
        let noneCorrect = none.filter { $0 < 0.5 }.count
        print("    title \(titles[t]): own mean \(mean(own)) (≥0.5: \(ownCorrect)/\(own.count)), none mean \(mean(none)) (<0.5: \(noneCorrect)/\(none.count)), other titles mean \(mean(others)), max on none \(none.max().map { String(format: "%.3f", $0) } ?? "-") (scale \(String(format: "%.3f", titleCalibration[2 * t])), bias \(String(format: "%.3f", titleCalibration[2 * t + 1]))\(t < distanceWeights.count ? ", distance weight \(String(format: "%.3f", distanceWeights[t]))" : ""))")
        t += 1
    }
}

MLXRandom.seed(seed)
let mlxNetwork = MLXHawkNetwork(classCount: classes.count + titles.count)
if let w = importedWeights {
    mlxNetwork.importWeights(from: w)
}
let trainer = MLXClassificationTrainer(
    network: mlxNetwork, learningRate: Defaults.lrMax, clipNorm: Defaults.clipNorm,
    labelSmoothing: Defaults.labelSmoothing, weightDecay: Defaults.weightDecay, biasCorrection: Defaults.biasCorrection,
    titleOf: titleOf
)
print("Hawk: input \(RowScanEncoder.frameDim) x \(RowScanEncoder.frameCount) frames, layers \(HawkShape.numLayers), width \(HawkShape.modelDim), RNN width \(HawkShape.rnnDim), MLP width \(HawkShape.mlpDim), gate blocks \(HawkShape.heads), conv \(HawkShape.convWidth)+\(HawkShape.convRight), classes \(classes.count), titles \(titles.joined(separator: ", "))")
print("  parameters \(mlxNetwork.exportWeights(classes: classes, titles: titles).parameterCount), batch \(batchSize), epochs \(epochs), seed \(seed)")

var bestWeights: HawkWeights = importedWeights ?? mlxNetwork.exportWeights(classes: classes, titles: titles)
var calibration = bestWeights.calibration
var titleCalibration = bestWeights.titleCalibration
var featureDistance: FeatureDistance? = bestWeights.featureDistance
var distanceWeights: [Float] = bestWeights.titleDistanceWeight

if 0 < epochs {
    print("\n--- Training ---")
    let scheduler = CosineLRScheduler(lrMax: Defaults.lrMax, lrMin: Defaults.lrMin)
    let stepsPerEpoch = (trainSet.count + batchSize - 1) / batchSize
    let totalSteps = max(1, epochs * stepsPerEpoch)
    let warmupSteps = min(max(Defaults.minWarmupSteps, totalSteps / Defaults.warmupStepDivisor), max(1, totalSteps / 2))
    print("  learning rate \(Defaults.lrMax) -> \(Defaults.lrMin), steps \(totalSteps) (warmup \(warmupSteps)), grad clip \(Defaults.clipNorm), label smoothing \(Defaults.labelSmoothing), weight decay \(Defaults.weightDecay), bias correction \(Defaults.biasCorrection)")
    // The batch order comes from the seed too, so one seed always gives the same run.
    var rng = SplitMix64(seed: seed)
    var indices = Array(0..<trainSet.count)
    // Selection: lowest eval loss after temperature scaling, ties broken by higher eval accuracy. Accuracy moves
    // in steps of one sample on a small held-out set, so choosing by it picks epochs almost at random.
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
            // Loss after temperature scaling: logits keep growing once the training set is fit, which raises the
            // raw loss even while the ranking still improves; a fitted temperature removes that scale. The title
            // outputs add their binary loss after Platt scaling, for the same reason.
            let temperature = VectorCalibration.fit(logits: r.screenLogits, labels: r.labels, classCount: classes.count, fitBias: false)
            var calibratedLoss: Float = 0.0
            var i = 0
            while i < r.screenLogits.count {
                calibratedLoss -= log(max(temperature.probabilities(r.screenLogits[i])[r.labels[i]], 1e-12))
                i += 1
            }
            calibratedLoss /= Float(max(1, r.screenLogits.count))
            let titleLoss = r.titleLoss(r.fitTitleCalibration())
            line += ", eval loss \(String(format: "%.4f", r.loss)) (calibrated \(String(format: "%.4f", calibratedLoss)), titles \(String(format: "%.4f", titleLoss))), eval accuracy \(String(format: "%.2f", accuracy * 100))%"
            score = (accuracy, calibratedLoss + titleLoss)
        }
        line += " (\(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - epStart)) s, grad norm \(String(format: "%.2f", trainer.lastGradientNorm)))"
        if score.loss < bestScore.loss || (score.loss == bestScore.loss && bestScore.accuracy < score.accuracy) {
            bestScore = score
            bestEpoch = ep
            bestWeights = mlxNetwork.exportWeights(classes: classes, titles: titles)
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
    var r = evaluate(trainer, evalSet)
    if 0 < epochs {
        // The held-out set also selects the epoch, so this calibration is slightly optimistic; a separate set would be cleaner.
        calibration = VectorCalibration.fit(logits: r.screenLogits, labels: r.labels, classCount: classes.count, fitBias: Defaults.calibrationBias)
        titleCalibration = r.fitTitleCalibration()
        print("  calibration fitted on eval: scale \(String(format: "%.3f", calibration.scale))")
    }
    report("eval", r, calibration: calibration, titleCalibration: titleCalibration)
    if 0 < epochs && titles.isEmpty != true {
        // Feature distance: class means and shared covariance of the training features (samples with a class),
        // then per title a logistic regression on [title logit, distance] fitted on the held-out samples.
        let trainPooled = allPooled(trainer, trainSet)
        if let distance = FeatureDistance.fit(features: trainPooled, labels: trainSet.labels, classCount: classes.count) {
            featureDistance = distance
            r.titleDistances = titleDistances(allPooled(trainer, evalSet), distance)
            var weights: [Float] = []
            var t = 0
            while t < titles.count {
                let inputs = (0..<r.allLabels.count).map { [r.titleLogits[$0][t], r.titleDistances[$0][t]] }
                let fit = LogisticRegression.fit(inputs: inputs, targets: (0..<r.allLabels.count).map { r.isTitle($0, t) })
                titleCalibration[2 * t] = fit.weights[0]
                titleCalibration[2 * t + 1] = fit.bias
                weights.append(fit.weights[1])
                t += 1
            }
            distanceWeights = weights
            print("  feature distance fitted on \(trainPooled.count) training samples; titles with distance:")
            report("eval", r, calibration: calibration, titleCalibration: titleCalibration, distanceWeights: distanceWeights)
        } else {
            print("  feature distance: covariance not invertible, skipped")
        }
    }
    if evalDumpPath.isEmpty != true {
        var paths: [String] = []
        if let metaData = try? Data(contentsOf: URL(fileURLWithPath: evalPath + ".meta.json")),
           let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any],
           let samples = meta["samples"] as? [[String: Any]] {
            paths = samples.map { ($0["path"] as? String) ?? "" }
        }
        var lines: [String] = []
        let s = r.summary(calibration: calibration)
        var screenRow = [Int: Int]()
        for (k, row) in r.screenRows.enumerated() {
            screenRow[row] = k
        }
        var i = 0
        while i < r.allLabels.count {
            var row: [String: Any] = [
                "index": i,
                "label": r.allLabels[i] < 0 ? "none" : classes[r.allLabels[i]],
                "titles": Dictionary(uniqueKeysWithValues: titles.enumerated().map { ($0.element, Double(String(format: "%.4f", r.titleProbability(i, $0.offset, titleCalibration, distanceWeights))) ?? 0.0) })
            ]
            if let k = screenRow[i] {
                let probs = s.probabilities[k]
                var predicted = 0
                var c = 1
                while c < probs.count {
                    if probs[predicted] < probs[c] {
                        predicted = c
                    }
                    c += 1
                }
                row["predicted"] = classes[predicted]
                row["correct"] = predicted == r.allLabels[i]
                row["probabilities"] = probs.map { Double(String(format: "%.4f", $0)) ?? 0.0 }
            }
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
var trainResult = evaluate(trainer, trainSet)
if let distance = featureDistance, distanceWeights.isEmpty != true {
    trainResult.titleDistances = titleDistances(allPooled(trainer, trainSet), distance)
}
report("train", trainResult, calibration: calibration, titleCalibration: titleCalibration, distanceWeights: distanceWeights)

// The exported weights must give the same logits on the CPU path.
var exported = mlxNetwork.exportWeights(classes: classes, titles: titles, calibration: calibration, titleCalibration: titleCalibration)
exported.centerMask = centerMask ?? importedWeights?.centerMask
exported.featureDistance = featureDistance
exported.titleDistanceWeight = distanceWeights
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
