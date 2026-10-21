import Foundation
import Misago

setbuf(stdout, nil)

// Runs the whole pipeline on image files: classify, crop the class's number field, read the digits.
// Usage: read --weights <classifier.json> --digit-weights <digits.json> --regions <regions.json> <image>...
// Prints one JSON object per line: class, probability, number (-1 when none), digits, confidence, ms.

var weightsPath = ""
var digitWeightsPath = ""
var regionsPath = ""
var profile = false
var showFrames = false
var imagePaths: [String] = []
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
        print("Usage: read --weights <classifier.json> --digit-weights <digits.json> --regions <regions.json> [--profile] <image>...")
        exit(0)
    case "-w", "--weights":
        if let v = value() { weightsPath = v }
    case "--digit-weights":
        if let v = value() { digitWeightsPath = v }
    case "--regions":
        if let v = value() { regionsPath = v }
    case "--profile":
        profile = true
    case "--frames":
        showFrames = true
    default:
        if arg.hasPrefix("-") {
            print("error: unknown argument \(arg)")
            exit(1)
        }
        imagePaths.append(arg)
    }
    argIdx += 1
}
if weightsPath.isEmpty || digitWeightsPath.isEmpty || regionsPath.isEmpty || imagePaths.isEmpty {
    print("error: specify --weights, --digit-weights, --regions and at least one image. See read --help.")
    exit(1)
}
let reader: ScreenReader
do {
    let classifier = try ImageClassifier(weights: try HawkWeights.load(from: URL(fileURLWithPath: weightsPath)))
    let digits = try DigitReader(weights: try DigitReaderWeights.load(from: URL(fileURLWithPath: digitWeightsPath)))
    let regions = try RegionTable.load(from: URL(fileURLWithPath: regionsPath))
    reader = ScreenReader(classifier: classifier, digitReader: digits, regions: regions)
} catch {
    print("error: cannot load models: \(error)")
    exit(1)
}
for path in imagePaths {
    do {
        let decoded = try loadRGB(path: path)
        let start = CFAbsoluteTimeGetCurrent()
        let result = try reader.read(width: decoded.width, height: decoded.height, rgb: decoded.rgb)
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        var object: [String: Any] = [
            "path": path,
            "class": result.verdict.className,
            "probability": Double(String(format: "%.4f", result.verdict.probability)) ?? 0.0,
            "title": result.verdict.title,
            "title_probability": Double(String(format: "%.4f", result.verdict.titleProbability)) ?? 0.0,
            "titles": result.verdict.titleProbabilities.mapValues { Double(String(format: "%.4f", $0)) ?? 0.0 },
            "number": result.number,
            "outcome": result.outcome,
            "battle": result.battle,
            "rotation": result.rotation,
            "ms": Double(String(format: "%.2f", ms)) ?? 0.0
        ]
        if let reading = result.reading {
            object["digits"] = reading.digits
            object["confidence"] = Double(String(format: "%.4f", reading.confidence)) ?? 0.0
        }
        if showFrames, let strip = try reader.strip(
            try reader.landscape(width: decoded.width, height: decoded.height, rgb: decoded.rgb),
            verdict: result.verdict, rotation: result.rotation
        ) {
            // Per-frame CTC argmax over the strip, "-" for blank: shows where digits are emitted and dropped.
            let frames = ConvStem.frames(strip.pixels, weights: reader.digitReader.weights.stem)
            let logits = reader.digitReader.network.frameLogits(flatFeatures: frames, frames: ConvStemWeights.frameCount)
            var sequence = ""
            var topProbability = ""
            for frame in logits {
                let probs = VectorCalibration.softmax(frame)
                var best = 0
                var c = 1
                while c < probs.count {
                    if probs[best] < probs[c] {
                        best = c
                    }
                    c += 1
                }
                sequence.append(best == 0 ? "-" : Character(String(best - 1)))
                topProbability.append(probs[best] < 0.6 ? "?" : (probs[best] < 0.9 ? "~" : " "))
            }
            object["frames"] = sequence
            object["frames_confidence"] = topProbability
        }
        if profile {
            // Stage timings, repeated so one-off costs do not dominate.
            var normalizeMs = 0.0, classifyMs = 0.0, extractMs = 0.0, readMs = 0.0
            let runs = 5
            var r = 0
            while r < runs {
                var t = CFAbsoluteTimeGetCurrent()
                let input = try reader.landscape(width: decoded.width, height: decoded.height, rgb: decoded.rgb)
                normalizeMs += (CFAbsoluteTimeGetCurrent() - t) * 1000.0
                t = CFAbsoluteTimeGetCurrent()
                let oriented = reader.classify(input)
                classifyMs += (CFAbsoluteTimeGetCurrent() - t) * 1000.0
                t = CFAbsoluteTimeGetCurrent()
                let strip = try reader.strip(input, verdict: oriented.verdict, rotation: oriented.rotation)
                extractMs += (CFAbsoluteTimeGetCurrent() - t) * 1000.0
                if let strip = strip {
                    t = CFAbsoluteTimeGetCurrent()
                    _ = reader.digitReader.read(strip)
                    readMs += (CFAbsoluteTimeGetCurrent() - t) * 1000.0
                }
                r += 1
            }
            let n = Double(runs)
            object["profile_ms"] = ["normalize": normalizeMs / n, "classify": classifyMs / n, "extract": extractMs / n, "read": readMs / n].mapValues { Double(String(format: "%.2f", $0)) ?? 0.0 }
        }
        if let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), let text = String(data: data, encoding: .utf8) {
            print(text)
        }
    } catch {
        let object: [String: Any] = ["path": path, "error": "\(error)"]
        if let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), let text = String(data: data, encoding: .utf8) {
            print(text)
        }
    }
}
